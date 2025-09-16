from numpy.testing import assert_allclose

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import numpy as np
from skimage.measure import blur_effect
from libertem.api import Context
from libertem.udf.sum import SumUDF
import optax
import jax.numpy as jnp

from microscope_calibration.util.stem_overfocus_sim import project
from microscope_calibration.udf.stem_overfocus import OverfocusUDF
from microscope_calibration.common.model import (
    Parameters4DSTEM, PixelYX, DescanError, Model4DSTEM
)
from microscope_calibration.util.optimize import optimize, make_overfocus_loss_function


def test_optimize():
    scan_rotation = np.pi/2
    flip_y = True
    detector_rotation = 0.

    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 1.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 16
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2 + 0.00314157, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_y=flip_y,
        # Simulate detector larger than object to avoid clipping at the borders
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=detector_rotation,
        # Descan error designed to give whole pixel shifts
        descan_error=DescanError(
            offpxi=detector_pixel_pitch,
            offpyi=detector_pixel_pitch * 2,
            offsxi=-1 * detector_pixel_pitch/camera_length,
            offsyi=-2 * detector_pixel_pitch/camera_length,
            pxo_pxi=2 * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=3 * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=-3 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=-4 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        )
    )
    obj = np.zeros((obj_half_size * 2, obj_half_size * 2))
    obj[obj_half_size, obj_half_size] = 1
    sim = project(
        obj,
        scan_shape=(2*obj_half_size, 2*obj_half_size),
        detector_shape=(4*obj_half_size, 4*obj_half_size),
        sim_params=params
    )
    ctx = Context.make_with('inline')
    ds = ctx.load('memory', data=sim)
    udf = OverfocusUDF(overfocus_params={'params': params})
    make_new_params, loss = make_overfocus_loss_function(
        params=params,
        ctx=ctx,
        dataset=ds,
        overfocus_udf=udf,
    )
    res = optimize(loss=loss)
    res_params = make_new_params(res.x)
    assert_allclose(res_params.scan_rotation, params.scan_rotation, atol=0.1)
    assert_allclose(res_params.overfocus, params.overfocus, rtol=0.1)

    valdict = {'val': False}

    def callback(args, new_params, udf_results, current_loss):
        if valdict['val']:
            pass
        else:
            valdict['val'] = True
            assert_allclose(args, [0, 0])
            assert params == new_params
            assert_allclose(udf_results[0]['shifted_sum'].data.astype(bool), obj.astype(bool))

    make_new_params, loss = make_overfocus_loss_function(
        params=params,
        ctx=ctx,
        dataset=ds,
        overfocus_udf=udf,
        callback=callback,
        blur_function=blur_effect,
        extra_udfs=(SumUDF(), ),
        plots=(),
    )
    res = optimize(
        loss=loss, minimizer_kwargs={'method': 'SLSQP'},
        bounds=[(-10, 10), (-10, 10)],
    )
    res_params = make_new_params(res.x)
    assert_allclose(res_params.scan_rotation, params.scan_rotation, atol=0.1)
    assert_allclose(res_params.overfocus, params.overfocus, rtol=0.1)


def test_descan_error():
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 1.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 16
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2 + 0.00314157, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=0., y=0.),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=2*obj_half_size, y=2*obj_half_size),
        detector_rotation=0.,
        descan_error=DescanError(
            sxo_pyi=1 * detector_pixel_pitch/scan_pixel_pitch,
            syo_pxi=1 * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=-2 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=-1 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        )
    )
    test_positions = jnp.array((
        (0, 0),
        (100, 0),
        (0, 100)
    ))

    target_px = []
    for scan_y, scan_x in test_positions:
        target_model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=scan_y, x=scan_x))
        target_ray = target_model.make_source_ray(source_dx=0, source_dy=0).ray
        target_res = target_model.trace(target_ray)
        target_px.append((
            target_res['detector'].sampling['detector_px'].x,
            target_res['detector'].sampling['detector_px'].y,
        ))

    target_px = jnp.array(target_px)

    @jax.jit
    def loss(args):
        sxo_pyi, syo_pxi, sxo_pxi, syo_pyi = args
        opt_params = params.derive(descan_error=DescanError(
            sxo_pyi=sxo_pyi,
            syo_pxi=syo_pxi,
            sxo_pxi=sxo_pxi,
            syo_pyi=syo_pyi,
        ))
        res = []
        for scan_y, scan_x in test_positions:
            opt_model = Model4DSTEM.build(params=opt_params, scan_pos=PixelYX(y=scan_y, x=scan_x))
            opt_ray = opt_model.make_source_ray(source_dx=0, source_dy=0).ray
            opt_res = opt_model.trace(opt_ray)
            res.append((
                opt_res['detector'].sampling['detector_px'].x,
                opt_res['detector'].sampling['detector_px'].y,
            ))
        return jnp.linalg.norm(jnp.array(res) - target_px)

    start = jnp.zeros(4)
    correct = jnp.array((
        1 * detector_pixel_pitch/scan_pixel_pitch,
        1 * detector_pixel_pitch/scan_pixel_pitch,
        -2 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        -1 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
    ))

    assert_allclose(loss(correct), 0.)
    assert not np.allclose(loss(start), 0)

    solver = optax.lbfgs()
    optargs = start.copy()
    opt_state = solver.init(optargs)
    value_and_grad = optax.value_and_grad_from_state(loss)

    @jax.jit
    def optstep(optargs, opt_state):

        value, grad = value_and_grad(optargs, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, optargs, value=value, grad=grad, value_fn=loss
        )
        optargs = optax.apply_updates(optargs, updates)
        return optargs, opt_state

    for i in range(10):
        print(f'Objective function: {loss(optargs)}, distance {optargs - correct}')
        optargs, opt_state = optstep(optargs, opt_state)
    print(f'Objective function: {loss(optargs)}, distance {optargs - correct}')
    assert_allclose(optargs, correct)
