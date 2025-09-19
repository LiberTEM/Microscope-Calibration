from typing import NamedTuple

import numpy as np
from scipy. optimize import shgo
from skimage.measure import blur_effect
from typing import TYPE_CHECKING, Callable, Optional
from collections.abc import Iterable

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
import optimistix

from microscope_calibration.common.model import Parameters4DSTEM, PixelYX, DescanError, trace

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


class _CLArgs(NamedTuple):
    ref_params: Parameters4DSTEM
    test_dx: float
    radius_px: float


@jax.jit
def _cl_loss(y, args: _CLArgs):
    opt_params = args.ref_params.derive(
        camera_length=y[0],
        overfocus=0.
    )
    opt_res_1 = trace(
        opt_params, scan_pos=PixelYX(y=0., x=0.), source_dx=args.test_dx, source_dy=0.)
    opt_res_2 = trace(
        opt_params, scan_pos=PixelYX(y=0., x=0.), source_dx=-args.test_dx, source_dy=0.)
    px_1 = opt_res_1['detector'].sampling['detector_px']
    px_2 = opt_res_2['detector'].sampling['detector_px']
    distance = jnp.linalg.norm(jnp.array(px_2) - jnp.array(px_1))
    return distance - 2*args.radius_px


# FIXME include wavelength calculation etc for more practical
# input parameters
def solve_camera_length(ref_params: Parameters4DSTEM, diffraction_angle, radius_px):
    args = _CLArgs(
        radius_px=radius_px,
        test_dx=jnp.tan(diffraction_angle),
        ref_params=ref_params
    )
    start = jnp.array((ref_params.camera_length, ))
    opt_res = optimistix.least_squares(
        fn=_cl_loss,
        args=args,
        solver=optimistix.BFGS(atol=1e-12, rtol=1e-12),
        y0=start
    )
    residual = _cl_loss(opt_res.value, args)
    # The loss function has minima at camera_length and -camera_length.
    # we take the positive side since a negative camera length doesn't make sense
    # for a classical TEM, only for reflection.
    return ref_params.derive(
        camera_length=jnp.abs(opt_res.value[0]),
    ), residual


class _SPPArgs(NamedTuple):
    ref_params: Parameters4DSTEM
    point_1: PixelYX
    point_2: PixelYX
    physical_distance: float


@jax.jit
def _spp_loss(y, args: _SPPArgs):
    opt_params = args.ref_params.derive(
        scan_pixel_pitch=y[0],
        overfocus=0.
    )
    opt_res_1 = trace(opt_params, scan_pos=args.point_1, source_dx=0., source_dy=0.)
    opt_res_2 = trace(opt_params, scan_pos=args.point_2, source_dx=0., source_dy=0.)
    dx = opt_res_2['specimen'].ray.x - opt_res_1['specimen'].ray.x
    dy = opt_res_2['specimen'].ray.y - opt_res_1['specimen'].ray.y
    opt_distance = jnp.linalg.norm(jnp.array((dy, dx)))
    return opt_distance - args.physical_distance


def solve_scan_pixel_pitch(
        ref_params: Parameters4DSTEM,
        point_1: PixelYX, point_2: PixelYX,
        physical_distance: float):

    args = _SPPArgs(
        ref_params=ref_params,
        point_1=point_1,
        point_2=point_2,
        physical_distance=physical_distance
    )
    start = jnp.array((ref_params.scan_pixel_pitch, ))
    opt_res = optimistix.least_squares(
        fn=_spp_loss,
        args=args,
        solver=optimistix.BFGS(atol=1e-12, rtol=1e-12),
        y0=start
    )
    residual = _spp_loss(opt_res.value, args)
    # The loss function has minima at scan_pixel_pitch and -scan_pixel_pitch. we
    # take the positive side since the inversion can be better expressed with a
    # scan rotation.
    return ref_params.derive(
        scan_pixel_pitch=jnp.abs(opt_res.value[0]),
    ), residual


# As returned by CoMUDF in the 'regression' buffer with
# RegressionOptions.SUBTRACT_LINEAR This allows preliminary calibration of
# descan error for a single camera length by adjusting constant tilt offset and
# tilt as a function of scan.
CoMRegression = np.ndarray[tuple[3, 2], np.floating]


# Type specification for dictionary where keys are calibrated camera lengths and
# values regression specifiers. This allows full calibration of descan error if at
# least two different camera lengths are provided.
CoMRegressions = dict[float, CoMRegression]


class _DEFullArgs(NamedTuple):
    # Aligned with the CoM regression coordinate system.
    # Currently only tested for no scan rotation and no flip_y
    aligned_params: Parameters4DSTEM
    regressions: CoMRegressions


@jax.jit
def _de_full_loss(y, args: _DEFullArgs):
    de = DescanError(*y)
    distances = []
    for cl, reg in args.regressions.items():
        opt_params = args.aligned_params.derive(
            camera_length=cl,
            descan_error=de,
        )
        for scan_y in (0., 1.):
            for scan_x in (0., 1.):
                dy = reg[0, 0]
                dx = reg[0, 1]
                dydy = reg[1, 0]
                dxdy = reg[1, 1]
                dydx = reg[2, 0]
                dxdx = reg[2, 1]
                det_y = opt_params.detector_center.y + (dy + dydy*scan_y + dydx*scan_x)
                det_x = opt_params.detector_center.x + (dx + dxdy*scan_y + dxdx*scan_x)
                res = trace(
                    opt_params, scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=0., source_dy=0.)
                distances.extend((
                    det_y - res['detector'].sampling['detector_px'].y,
                    det_x - res['detector'].sampling['detector_px'].x,
                ))
    return jnp.array(distances)


def solve_full_descan_error(ref_params: Parameters4DSTEM, regressions: CoMRegressions):
    # Caveat: scan and detector center of ref_params and of regressions should
    # match.

    # Align coordinate system directions with native CoM coordinate
    # system without corrections
    aligned_params = ref_params.derive(
        flip_y=False,
        scan_rotation=0.,
        detector_rotation=0.,
    )
    args = _DEFullArgs(
        aligned_params=aligned_params,
        regressions=regressions,
    )

    # Start with a small epsilon to prevent NaN results of yet unknown origin
    # for some parameter combinations
    start = jnp.full(shape=(len(DescanError()), ), fill_value=1e-6)
    opt_res = optimistix.least_squares(
        fn=_de_full_loss,
        args=args,
        solver=optimistix.BFGS(atol=1e-12, rtol=1e-12),
        y0=start
    )
    residual = _de_full_loss(opt_res.value, args)

    # Bring descan error back to original coordinate system
    res_params = aligned_params.derive(
        descan_error=DescanError(*opt_res.value)
    ).adjust_scan_rotation(
        ref_params.scan_rotation
    ).adjust_detector_rotation(
        ref_params.detector_rotation
    ).adjust_flip_y(
        ref_params.flip_y
    )

    return res_params, residual


class _NormArgs(NamedTuple):
    ref_params: Parameters4DSTEM


def _zero_const(de: DescanError) -> DescanError:
    return DescanError(
        pxo_pxi=de.pxo_pxi,
        pxo_pyi=de.pxo_pyi,
        pyo_pxi=de.pyo_pxi,
        pyo_pyi=de.pyo_pyi,
        sxo_pxi=de.sxo_pxi,
        sxo_pyi=de.sxo_pyi,
        syo_pxi=de.syo_pxi,
        syo_pyi=de.syo_pyi,
        offpxi=0.,
        offpyi=0.,
        offsxi=0.,
        offsyi=0.,
    )


@jax.jit
def _norm_loss(y, args: _NormArgs):
    distances = []
    scy, scx, dcy, dcx = y
    de_new = _zero_const(args.ref_params.descan_error)
    for cl in (0, 1, 2):
        opt_params = args.ref_params.derive(
            camera_length=cl,
            descan_error=de_new,
            scan_center=PixelYX(y=scy, x=scx),
            detector_center=PixelYX(y=dcy, x=dcx),
        )
        ref_params = args.ref_params.derive(
            camera_length=cl,
        )
        for scan_y in (0., 1.,):
            for scan_x in (0., 1.):
                res = trace(
                    opt_params, scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0., source_dx=0.)
                ref = trace(
                    ref_params, scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0., source_dx=0.)
                distances.append((
                    res['detector'].sampling['detector_px'].y
                    - ref['detector'].sampling['detector_px'].y,
                    res['detector'].sampling['detector_px'].x
                    - ref['detector'].sampling['detector_px'].x
                ))
    return jnp.array(distances)


def normalize_descan_error(ref_params: Parameters4DSTEM):
    args = _NormArgs(
        ref_params=ref_params,
    )
    start = jnp.array((
        ref_params.scan_center.y,
        ref_params.scan_center.x,
        ref_params.detector_center.y,
        ref_params.detector_center.x
    ))
    opt_res = optimistix.least_squares(
        fn=_norm_loss,
        args=args,
        solver=optimistix.BFGS(atol=1e-12, rtol=1e-12),
        y0=start
    )
    residual = _norm_loss(opt_res.value, args)
    scy, scx, dcy, dcx = opt_res.value
    res_params = ref_params.derive(
        scan_center=PixelYX(y=scy, x=scx),
        detector_center=PixelYX(y=dcy, x=dcx),
        descan_error=_zero_const(ref_params.descan_error)
    )
    return res_params, residual


class _DETiltArgs(NamedTuple):
    # Aligned with the CoM regression coordinate system.
    # Currently only tested for no scan rotation and no flip_y
    aligned_params: Parameters4DSTEM
    regression: CoMRegression


def _tilt_descan(de: DescanError, y) -> DescanError:
    return DescanError(
        pxo_pxi=de.pxo_pxi,
        pxo_pyi=de.pxo_pyi,
        pyo_pxi=de.pyo_pxi,
        pyo_pyi=de.pyo_pyi,
        sxo_pxi=y[0],
        sxo_pyi=y[1],
        syo_pxi=y[2],
        syo_pyi=y[3],
        offpxi=de.offpxi,
        offpyi=de.offpyi,
        offsxi=y[4],
        offsyi=y[5],
    )


@jax.jit
def _de_tilt_loss(y, args: _DETiltArgs):
    opt_params = args.aligned_params.derive(
        descan_error=_tilt_descan(de=args.aligned_params.descan_error, y=y)
    )

    distances = []
    reg = args.regression
    for scan_y in (0., 1.):
        for scan_x in (0., 1.):
            dy = reg[0, 0]
            dx = reg[0, 1]
            dydy = reg[1, 0]
            dxdy = reg[1, 1]
            dydx = reg[2, 0]
            dxdx = reg[2, 1]
            det_y = opt_params.detector_center.y + (dy + dydy*scan_y + dydx*scan_x)
            det_x = opt_params.detector_center.x + (dx + dxdy*scan_y + dxdx*scan_x)
            res = trace(
                opt_params, scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=0., source_dy=0.)
            distances.extend((
                det_y - res['detector'].sampling['detector_px'].y,
                det_x - res['detector'].sampling['detector_px'].x,
            ))
    return jnp.array(distances)


def solve_tilt_descan_error(ref_params: Parameters4DSTEM, regression: CoMRegression):
    # Caveat: scan and detector center of ref_params and of regressions should
    # match.

    # Align coordinate system directions with native CoM coordinate
    # system without corrections
    # We make sure that the offset-based descan error components are preserved
    aligned_params = ref_params.adjust_flip_y(
        flip_y=False,
    ).adjust_scan_rotation(
        scan_rotation=0.,
    ).adjust_detector_rotation(
        detector_rotation=0.,
    )
    args = _DETiltArgs(
        aligned_params=aligned_params,
        regression=regression,
    )

    # Start with a small epsilon to prevent NaN results of yet unknown origin
    # for some parameter combinations
    start = jnp.full(shape=(6, ), fill_value=1e-6)
    opt_res = optimistix.least_squares(
        fn=_de_tilt_loss,
        args=args,
        solver=optimistix.BFGS(atol=1e-12, rtol=1e-12),
        y0=start
    )
    residual = _de_tilt_loss(opt_res.value, args)

    # Bring descan error back to original coordinate system
    res_params = aligned_params.derive(
        descan_error=_tilt_descan(aligned_params.descan_error, opt_res.value)
    ).adjust_detector_rotation(
        ref_params.detector_rotation
    ).adjust_scan_rotation(
        ref_params.scan_rotation
    ).adjust_flip_y(
        ref_params.flip_y
    )

    return res_params, residual
