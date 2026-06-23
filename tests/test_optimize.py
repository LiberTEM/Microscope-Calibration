from numpy.testing import assert_allclose
import pytest

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import numpy as np
from skimage.measure import blur_effect
from libertem.api import Context
from libertem.udf.sum import SumUDF
from libertem.udf.com import CoMUDF, RegressionOptions
import optax
import jax.numpy as jnp

from microscope_calibration.util.stem_overfocus_sim import project
from microscope_calibration.udf.stem_overfocus import OverfocusUDF
from microscope_calibration.common.model import (
    Parameters4DSTEM, PixelYX, DescanError, trace
)
from microscope_calibration.util.optimize import (
    optimize, make_overfocus_loss_function,
    solve_camera_length, solve_scan_pixel_pitch,
    solve_full_descan_error, normalize_descan_error,
    solve_tilt_descan_error, _tilt_descan,
    solve_tilt_descan_error_points,
)


def test_optimize():
    scan_rotation = np.pi/2
    flip_factor = -1.
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
        flip_factor=flip_factor,
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
            assert_allclose(udf_results[0]['backprojected_sum'].data.astype(bool), obj.astype(bool))

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
        flip_factor=1.,
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
        target_res = trace(
            params=params, scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0, source_dx=0)
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
            opt_res = trace(
                params=opt_params, scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0, source_dx=0)
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


def test_camera_length():
    # Determine camera length from a known diffraction angle in radians,
    # corresponding detector pixel offset, and detector pixel pitch
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 0.01
    camera_length = 1.234
    propagation_distance = overfocus + camera_length
    obj_half_size = 16
    # This is known, e.g. from crystal structure, diffraction order and
    # wavelength
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2 + 0.00314157, propagation_distance)
    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=0.,
        flip_factor=1.,
        detector_center=PixelYX(x=2*obj_half_size, y=2*obj_half_size),
    )
    # This is observed on the detector
    px_radius = jnp.tan(angle) * propagation_distance / detector_pixel_pitch

    res, residual = solve_camera_length(
        # Start with a negative value on purpose
        ref_params=params.derive(camera_length=-2*camera_length),
        diffraction_angle=angle,
        radius_px=px_radius,
    )
    assert_allclose(res.camera_length, propagation_distance)
    assert_allclose(residual, 0., atol=1e-12)


def test_scan_pixel_pitch():
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 0.01
    camera_length = 1.234
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
        scan_rotation=0.,
        flip_factor=1.,
        detector_center=PixelYX(x=2*obj_half_size, y=2*obj_half_size),
    )

    point_1 = PixelYX(1., 2.)
    point_2 = PixelYX(7., 9.)
    distance = np.linalg.norm(np.array(point_2) - np.array(point_1)) * scan_pixel_pitch

    res, residual = solve_scan_pixel_pitch(
        ref_params=params.derive(scan_pixel_pitch=.3543),
        point_1=point_1,
        point_2=point_2,
        physical_distance=distance,
    )
    assert_allclose(res.scan_pixel_pitch, scan_pixel_pitch)
    assert_allclose(residual, 0., atol=1e-12)


@pytest.mark.parametrize(
    'scan_rotation, flip_factor, detector_rotation', [
        (-np.pi, 1., np.pi/7),
        (0., -1., 0.),
        (np.pi/7*3, -1., -np.pi/3)
    ]
)
@pytest.mark.parametrize(
    'descans', (
        np.zeros(12),
        np.linspace(-1, 1, 12),
        # alternating -0.5, and 0.5
        (np.full(12, -1) ** np.array(range(12))) * 0.25,
        # Alternating mishmash
        (np.full(12, -1) ** np.array(range(12))) * np.linspace(-1, 1, 12) % 0.11,
    )
)
def test_full_descan_error(scan_rotation, flip_factor, detector_rotation, descans):
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = scan_pixel_pitch
    overfocus = 0.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 8
    # Small epsilon to combat aliasing
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2*2 + 0.001, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_factor=flip_factor,
        detector_center=PixelYX(x=obj_half_size*8-2, y=obj_half_size*8+1),
        detector_rotation=detector_rotation,
        descan_error=DescanError(
            offpxi=descans[0] * detector_pixel_pitch,
            offpyi=descans[1] * detector_pixel_pitch,
            offsxi=-descans[2] * detector_pixel_pitch/camera_length,
            offsyi=-descans[3] * detector_pixel_pitch/camera_length,
            pxo_pxi=descans[4] * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=descans[5] * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pxi=-descans[6] * detector_pixel_pitch/scan_pixel_pitch,
            pxo_pyi=-descans[7] * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=descans[8] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=descans[9] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pxi=-descans[10] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            sxo_pyi=-descans[11] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        ),
    )

    # we simulate a vacuum reference scan
    obj = np.ones((2*obj_half_size, 2*obj_half_size))
    sims = {}
    for cl in (1, 2, 3):
        sims[cl] = project(
            image=obj,
            detector_shape=(16*obj_half_size, 16*obj_half_size),
            scan_shape=(2*obj_half_size, 2*obj_half_size),
            sim_params=params.derive(camera_length=cl),
        )

    # Calculate CoM regressions with LiberTEM
    ctx = Context.make_with('inline')
    udf = CoMUDF.with_params(
        regression=RegressionOptions.SUBTRACT_LINEAR,
        cy=params.detector_center.y,
        cx=params.detector_center.x,
    )
    regs = {}
    for (cl, sim) in sims.items():
        ds = ctx.load('memory', data=sim)
        res = ctx.run_udf(dataset=ds, udf=udf)
        regs[cl] = res['regression'].raw_data

    # The regressions from CoM suffer from imprecision due to aliasing
    # To check optimization accuracy and precision we calculate what the exact
    # values of there regressions should be
    exact_regs = {}
    for cl in sims.keys():
        exact_params = params.derive(
            camera_length=cl
        )
        res_0 = trace(params=exact_params, scan_pos=PixelYX(y=0., x=0.), source_dy=0., source_dx=0.)
        res_y = trace(params=exact_params, scan_pos=PixelYX(y=1., x=0.), source_dy=0., source_dx=0.)
        res_x = trace(params=exact_params, scan_pos=PixelYX(y=0., x=1.), source_dy=0., source_dx=0.)
        dy = res_0['detector'].sampling['detector_px'].y - params.detector_center.y
        dx = res_0['detector'].sampling['detector_px'].x - params.detector_center.x
        dydy = (
            res_y['detector'].sampling['detector_px'].y
            - res_0['detector'].sampling['detector_px'].y
        )
        dxdy = (
            res_y['detector'].sampling['detector_px'].x
            - res_0['detector'].sampling['detector_px'].x
        )
        dydx = (
            res_x['detector'].sampling['detector_px'].y
            - res_0['detector'].sampling['detector_px'].y
        )
        dxdx = (
            res_x['detector'].sampling['detector_px'].x
            - res_0['detector'].sampling['detector_px'].x
        )

        reg = np.array((
            (dy, dx),
            (dydy, dxdy),
            (dydx, dxdx)
        ))
        exact_regs[cl] = reg

    # We make sure the exact results approximate the results obtained with CoM.
    # 1-5 % of a pixel is about as good as the approximation gets
    for cl in sims.keys():
        assert_allclose(regs[cl], exact_regs[cl], rtol=5e-2, atol=5e-2)

    opt_res, residual = solve_full_descan_error(
        ref_params=params.derive(
            descan_error=DescanError(),
        ),
        regressions=exact_regs,
    )

    assert_allclose(params.descan_error, opt_res.descan_error, atol=1e-11)
    assert_allclose(residual, 0., atol=1e-11)


def test_normalize_descan(random_params):
    print(random_params)
    normalized, residual = normalize_descan_error(random_params)
    assert_allclose(residual, 0, atol=1e-11)

    for cl in (0.1, 3):
        for sy in (0, 1):
            for sx in (-1, 3):
                print(cl, sy, sx)
                pr = random_params.derive(
                    camera_length=cl,
                )
                pn = normalized.derive(
                    camera_length=cl,
                )
                ref = trace(params=pr, scan_pos=PixelYX(y=sy, x=sx), source_dy=0., source_dx=0.)
                norm = trace(params=pn, scan_pos=PixelYX(y=sy, x=sx), source_dy=0., source_dx=0.)
                assert_allclose(
                    ref['detector'].sampling['detector_px'].x,
                    norm['detector'].sampling['detector_px'].x,
                    atol=1e-12
                )
                assert_allclose(
                    ref['detector'].sampling['detector_px'].y,
                    norm['detector'].sampling['detector_px'].y,
                    atol=1e-12
                )


@pytest.mark.parametrize(
    'scan_rotation, flip_factor, detector_rotation', [
        (-np.pi, 1., np.pi/7),
        (0., -1., 0.),
        (np.pi/7*3, -1., -np.pi/3)
    ]
)
@pytest.mark.parametrize(
    'descans', (
        np.zeros(12),
        np.linspace(-1, 1, 12),
        # alternating -0.5, and 0.5
        (np.full(12, -1) ** np.array(range(12))) * 0.25,
        # Alternating mishmash
        (np.full(12, -1) ** np.array(range(12))) * np.linspace(-1, 1, 12) % 0.11,
    )
)
def test_tilt_descan_error(scan_rotation, flip_factor, detector_rotation, descans):
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = scan_pixel_pitch
    overfocus = 0.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 8
    # Small epsilon to combat aliasing
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2*2 + 0.001, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_factor=flip_factor,
        detector_center=PixelYX(x=obj_half_size*8+2, y=obj_half_size*8-1),
        detector_rotation=detector_rotation,
        descan_error=DescanError(
            offpxi=descans[0] * detector_pixel_pitch,
            offpyi=descans[1] * detector_pixel_pitch,
            offsxi=-descans[2] * detector_pixel_pitch/camera_length,
            offsyi=-descans[3] * detector_pixel_pitch/camera_length,
            pxo_pxi=descans[4] * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=descans[5] * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pxi=-descans[6] * detector_pixel_pitch/scan_pixel_pitch,
            pxo_pyi=-descans[7] * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=descans[8] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=descans[9] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pxi=-descans[10] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            sxo_pyi=-descans[11] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        ),
    )

    # we simulate a vacuum reference scan
    obj = np.ones((2*obj_half_size, 2*obj_half_size))
    sim = project(
            image=obj,
            detector_shape=(16*obj_half_size, 16*obj_half_size),
            scan_shape=(2*obj_half_size, 2*obj_half_size),
            sim_params=params,
        )

    # Calculate CoM regressions with LiberTEM
    ctx = Context.make_with('inline')
    udf = CoMUDF.with_params(
        regression=RegressionOptions.SUBTRACT_LINEAR,
        cy=params.detector_center.y,
        cx=params.detector_center.x,
    )
    ds = ctx.load('memory', data=sim)
    res = ctx.run_udf(dataset=ds, udf=udf)
    reg = res['regression'].raw_data

    # The regressions from CoM suffer from imprecision due to aliasing
    # To check optimization accuracy and precision we calculate what the exact
    # values of there regressions should be
    res_0 = trace(params=params, scan_pos=PixelYX(y=0., x=0.), source_dy=0., source_dx=0.)
    res_y = trace(params=params, scan_pos=PixelYX(y=1., x=0.), source_dy=0., source_dx=0.)
    res_x = trace(params=params, scan_pos=PixelYX(y=0., x=1.), source_dy=0., source_dx=0.)
    dy = res_0['detector'].sampling['detector_px'].y - params.detector_center.y
    dx = res_0['detector'].sampling['detector_px'].x - params.detector_center.x
    dydy = (
        res_y['detector'].sampling['detector_px'].y
        - res_0['detector'].sampling['detector_px'].y
    )
    dxdy = (
        res_y['detector'].sampling['detector_px'].x
        - res_0['detector'].sampling['detector_px'].x
    )
    dydx = (
        res_x['detector'].sampling['detector_px'].y
        - res_0['detector'].sampling['detector_px'].y
    )
    dxdx = (
        res_x['detector'].sampling['detector_px'].x
        - res_0['detector'].sampling['detector_px'].x
    )

    exact_reg = np.array((
        (dy, dx),
        (dydy, dxdy),
        (dydx, dxdx)
    ))

    # We make sure the exact results approximate the results obtained with CoM.
    # 1-5 % of a pixel is about as good as the approximation gets
    assert_allclose(reg, exact_reg, rtol=5e-2, atol=5e-2)

    opt_res, residual = solve_tilt_descan_error(
        ref_params=params.derive(
            descan_error=_tilt_descan(de=params.descan_error, y=np.zeros(6)),
        ),
        regression=exact_reg,
    )
    assert_allclose(residual, 0., atol=1e-11)
    for key in ('pxo_pxi', 'pxo_pyi', 'pyo_pxi', 'pyo_pyi', 'offpxi', 'offpyi',
                'sxo_pxi', 'syo_pyi', 'syo_pxi', 'sxo_pyi'):
        print(key)
        assert_allclose(
            getattr(params.descan_error, key),
            getattr(opt_res.descan_error, key),
            atol=1e-11
        )


@pytest.mark.parametrize(
    'scan_rotation, flip_factor, detector_rotation', [
        (0., 1., 0.),
        (np.pi/7*3, -1., -np.pi/3)
    ]
)
@pytest.mark.parametrize(
    'descans', (
        np.zeros(12),
        # Alternating mishmash
        (np.full(12, -1) ** np.array(range(12))) * np.linspace(-1, 1, 12) % 0.11,
    )
)
@pytest.mark.parametrize(
    'scan_pos, works', (
        (tuple(), False),
        (((0., 0.),), False),
        (((0., 0.), (0., 1.), (1., 0.)), True),
        (((1., 2.), (3., 5.), (7., 11.), (13., 17.)), True),
    )
)
def test_tilt_descan_error_points(
        scan_rotation, flip_factor, detector_rotation, descans, scan_pos, works):
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = scan_pixel_pitch
    overfocus = 0.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 8
    # Small epsilon to combat aliasing
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2*2 + 0.001, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_factor=flip_factor,
        detector_center=PixelYX(x=obj_half_size*8+2, y=obj_half_size*8-1),
        detector_rotation=detector_rotation,
        descan_error=DescanError(
            offpxi=descans[0] * detector_pixel_pitch,
            offpyi=descans[1] * detector_pixel_pitch,
            offsxi=-descans[2] * detector_pixel_pitch/camera_length,
            offsyi=-descans[3] * detector_pixel_pitch/camera_length,
            pxo_pxi=descans[4] * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=descans[5] * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pxi=-descans[6] * detector_pixel_pitch/scan_pixel_pitch,
            pxo_pyi=-descans[7] * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=descans[8] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=descans[9] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pxi=-descans[10] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            sxo_pyi=-descans[11] * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        ),
    )

    points = []
    for (scan_y, scan_x) in scan_pos:
        res = trace(
            params=params,
            scan_pos=PixelYX(y=scan_y, x=scan_x),
            source_dx=0., source_dy=0.,
        )
        detector_center = res['detector'].sampling['detector_px']
        points.append((scan_y, scan_x, detector_center.y, detector_center.x))
    opt_res, residual = solve_tilt_descan_error_points(
        ref_params=params.derive(
            # Blank out the tilt parts of the descan error
            descan_error=_tilt_descan(de=params.descan_error, y=np.zeros(6)),
        ),
        points=points,
    )
    default_attrs = ('pxo_pxi', 'pxo_pyi', 'pyo_pxi', 'pyo_pyi', 'offpxi', 'offpyi')
    opt_attrs = ('sxo_pxi', 'syo_pyi', 'syo_pxi', 'sxo_pyi')
    if works:
        attrs = default_attrs + opt_attrs
        # For some reason less accurate than in other tests
        assert_allclose(residual, 0., atol=1e-8)
    else:
        attrs = default_attrs
    for key in attrs:
        print(key)
        assert_allclose(
            getattr(params.descan_error, key),
            getattr(opt_res.descan_error, key),
            # For some reason less accurate than in other tests
            atol=1e-8
        )
        assert isinstance(getattr(params.descan_error, key), float)
        assert isinstance(getattr(opt_res.descan_error, key), float)
