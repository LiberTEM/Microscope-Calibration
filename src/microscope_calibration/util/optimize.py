import numpy as np
from scipy. optimize import shgo
from skimage.measure import blur_effect
from typing import TYPE_CHECKING, Callable, Optional
from collections.abc import Iterable

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
import optax

from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM, PixelYX

if TYPE_CHECKING:
    from libertem.udf.base import UDF
    from libertem.api import Context
    from libertem.io.dataset.base import DataSet

    from microscope_calibration.udf.stem_overfocus import OverfocusUDF


def make_overfocus_loss_function(
        params: Parameters4DSTEM,
        ctx: 'Context',
        dataset: 'DataSet',
        overfocus_udf: 'OverfocusUDF',
        blur_function: Optional[Callable] = None,
        extra_udfs: Iterable['UDF'] = (),
        callback: Optional[Callable] = None,
        **kwargs):
    '''
    Build a parameter mapping and loss function to optimize

    This maps the :code:`scan_rotation` and :code:`overfocus` parameters
    for :class:`OverfocusUDF` in such a way that the starting value is at
    0 and a sensible interval for optimization is :code`[-10, 10]` based
    on the starting parameters passed as an argument.

    It also constructs a loss function that updates the parameters of
    the :class:`OverfocusUDF` passed as parameter :code:`overfocus_udf`,
    runs it with the specified context, dataset and extra parameters,
    and then returns the blur effect of the image.

    Parameters
    ----------
    params
        Starting value for optimization
    ctx
        Run UDF using this Context
    dataset
        Run UDF with this dataset
    overfocus_udf
        Instance of :class:`OverfocusUDF` to use. Updating and using an existing instance
        allows using live plots.
    blur_function
        Function to calculate the blur of the :code:`shifted_sum` result buffer.
        By default, :code:`skimage.measure.blur_effect` is used.
    extra_udfs
        Iterable of other UDFs to run alongside
    callback
        Function that is called for each loop with the loss function
        arguments, current parameters, UDF results and current loss
    **kwargs
        Extra arguments to pass to the :func:`~libertem.api.Context.run_udf` call,
        such as plots.

    Returns
    -------
    make_new_params
        Function that updates a copy of the passed input parameters
        with an argument vector of the loss function. Since this vector
        is mapped into a [-10, 10] interval, this function is created to
        invert the mapping. This can be used
        to create parameters from the minimization result.
    loss
        Function that can be called with :func:`scipy.optimize.minimize`. Starting value is
        [0, 0], sensible range is ([-10, 10], [-10, 10]).
    '''
    # Rotate and scale the angle so that the optimizer works between +-10,
    # corresponding to +- 5 deg
    rotation_diff = params.scan_rotation * 180 / np.pi
    rotation_scale = 1
    # Values to shift and scale the overfocus so that the optimizer works between +-10
    overfocus_diff = params.overfocus
    overfocus_scale = 40 / np.abs(params.overfocus)

    if blur_function is None:
        blur_function = blur_effect

    def make_new_params(args) -> Parameters4DSTEM:
        '''
        Map parameters from +-10 to original range

        Parameters
        ----------

        args
            (scan_rotation, overfocused) mapped to +- 10
        '''
        transformed_rotation, transformed_overfocus = args
        rotation = transformed_rotation / rotation_scale + rotation_diff
        overfocus = transformed_overfocus / overfocus_scale + overfocus_diff
        return params.derive(
            overfocus=overfocus,
            scan_rotation=rotation / 180 * np.pi,
        )

    def loss(args) -> float:
        '''
        Loss function for optimizer to call

        This calls the UDF with the appropriate parameters
        and returns the blur as a metric. It can perform additional actions
        per iteration through the :code:`callback`, :code:`extra_udfs` and
        :code:`kwargs` parameters of the surrounding function.

        Parameters
        ----------

        args
            (scan_rotation, overfocused) mapped to +-10 range
        '''
        params = make_new_params(args)
        # Hack to make parameter update work
        overfocus_udf.params.overfocus_params['params'] = params
        res = ctx.run_udf(dataset=dataset, udf=[overfocus_udf] + list(extra_udfs), **kwargs)
        blur = blur_function(res[0]['shifted_sum'].data)
        if callback is not None:
            callback(args, overfocus_udf.params.overfocus_params['params'], res, blur)
        return blur

    return make_new_params, loss


def optimize(loss, bounds=None, minimizer_kwargs=None, **kwargs):
    '''
    Convenience function to call :func:`scipy.optimize.shgo`

    This calls :func:`scipy.optimize.shgo` with sensible bounds and minimizer
    method for a loss function created with
    :func:`make_overfocus_loss_function`. Additional kwargs are passed to
    :func:`scipy.optimize.shgo`.
    '''
    if bounds is None:
        bounds = [(-10, 10), (-10, 10)]

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'COBYLA'}
    res = shgo(
        func=loss,
        bounds=bounds,
        minimizer_kwargs=minimizer_kwargs,
        **kwargs
    )
    return res


def _solve(start: jnp.array, loss: Callable[[jnp.array], float], limit=1e-12):
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
        return optargs, opt_state, jnp.linalg.norm(updates)

    while True:
        optargs, opt_state, change = optstep(optargs, opt_state)
        if change < limit:
            break

    return optargs


# FIXME include wavelength calculation etc for more practical
# input parameters
def solve_camera_length(ref_params: Parameters4DSTEM, diffraction_angle, radius_px):
    test_dx = jnp.tan(diffraction_angle)

    @jax.jit
    def loss(optargs):
        opt_params = ref_params.derive(
            camera_length=optargs[0],
            overfocus=0.
        )
        opt_model = Model4DSTEM.build(params=opt_params, scan_pos=PixelYX(y=0., x=0.))
        opt_ray_1 = opt_model.make_source_ray(source_dx=test_dx, source_dy=0.).ray
        opt_res_1 = opt_model.trace(opt_ray_1)
        opt_ray_2 = opt_model.make_source_ray(source_dx=-test_dx, source_dy=0.).ray
        opt_res_2 = opt_model.trace(opt_ray_2)
        px_1 = opt_res_1['detector'].sampling['detector_px']
        px_2 = opt_res_2['detector'].sampling['detector_px']
        distance = jnp.linalg.norm(jnp.array(px_2) - jnp.array(px_1))
        return jnp.abs(distance - 2*radius_px)

    start = jnp.array((ref_params.camera_length, ))
    opt_res = _solve(
        start=start,
        loss=loss,
    )
    # The loss function has minima at camera_length and -camera_length.
    # we take the positive side since a negative camera length doesn't make sense
    # for a classical TEM, only for reflection.
    return ref_params.derive(
        camera_length=jnp.abs(opt_res[0]),
    )


def solve_scan_pixel_pitch(
        ref_params: Parameters4DSTEM,
        point_1: PixelYX, point_2: PixelYX,
        physical_distance: float):
    @jax.jit
    def loss(optargs):
        opt_params = ref_params.derive(
            scan_pixel_pitch=optargs[0],
            overfocus=0.
        )
        opt_model_1 = Model4DSTEM.build(params=opt_params, scan_pos=point_1)
        opt_ray_1 = opt_model_1.make_source_ray(source_dx=0., source_dy=0.).ray
        opt_res_1 = opt_model_1.trace(opt_ray_1)
        opt_model_2 = Model4DSTEM.build(params=opt_params, scan_pos=point_2)
        opt_ray_2 = opt_model_2.make_source_ray(source_dx=0., source_dy=0.).ray
        opt_res_2 = opt_model_2.trace(opt_ray_2)
        dx = opt_res_2['specimen'].ray.x - opt_res_1['specimen'].ray.x
        dy = opt_res_2['specimen'].ray.y - opt_res_1['specimen'].ray.y
        opt_distance = jnp.linalg.norm(jnp.array((dy, dx)))
        return jnp.abs(opt_distance - physical_distance)

    start = jnp.array((ref_params.scan_pixel_pitch, ))
    opt_res = _solve(
        start=start,
        loss=loss,
    )
    # The loss function has minima at scan_pixel_pitch and -scan_pixel_pitch. we
    # take the positive side since the inversion can be better expressed with a
    # scan rotation.
    return ref_params.derive(
        scan_pixel_pitch=jnp.abs(opt_res[0]),
    )
