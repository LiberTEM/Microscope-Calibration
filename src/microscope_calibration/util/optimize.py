from typing import NamedTuple, TYPE_CHECKING

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
# ruff: disable[E402]
import numpy as np
from scipy.optimize import shgo
from skimage.measure import blur_effect
from collections.abc import Callable
from collections.abc import Iterable

import jax.numpy as jnp
import optimistix

from microscope_calibration.common.model import (
    DescanError,
    Model4DSTEM,
    PixelYX,
)
from microscope_calibration.common.stem_overfocus import get_diffraction_pixel_radius
# ruff: enable[E402]

if TYPE_CHECKING:
    from libertem.api import Context
    from libertem.io.dataset.base import DataSet
    from libertem.udf.base import UDF

    from microscope_calibration.udf.stem_overfocus import OverfocusUDF


def _numeric(model: Model4DSTEM, modules) -> Model4DSTEM:
    """
    :code:`model` as instance of :code:`type(model).lambdify(modules)`, see
    :meth:`Model4DSTEM.lambdify`
    """
    return type(model).lambdify(modules).from_model(model)


def check_float64():
    """
    Check that JAX uses float64

    Electron optics calculations combine lengths in the range of picometers,
    e.g. the electron wavelength, with lengths in the range of meters, e.g. the
    camera length. The precision of float32 is not sufficient for that. JAX
    uses float32 by default. This package activates float64 on import with
    :code:`jax.config.update("jax_enable_x64", True)`, but it can be
    deactivated again by other code.

    Raises
    ------
    RuntimeError
        If JAX doesn't use float64
    """
    test = jnp.array((1, 2, 3), dtype=jnp.float64)
    if test.dtype != jnp.float64:
        raise RuntimeError(
            f"No float64 support activated in JAX, got {test.dtype}. Downcasting to "
            "float32 leads to inaccuracies that are much larger than permissible for "
            "electron optics calculations. Activate it with "
            '`jax.config.update("jax_enable_x64", True)` before using JAX.'
        )


def _least_squares(fn, args, y0, tol: float = 1e-12, **kwargs):
    """
    Minimize the sum of squares of :code:`fn(y, args)` starting from :code:`y0`

    Raises :class:`RuntimeError` if JAX doesn't use float64, see
    :func:`check_float64`. Additional :code:`kwargs` are passed to
    :func:`optimistix.least_squares`.

    Returns
    -------
    solution, residual
        Optimization result and :code:`fn(y, args)` for it
    """
    check_float64()
    opt_res = optimistix.least_squares(
        fn=fn, args=args, solver=optimistix.BFGS(atol=tol, rtol=tol), y0=y0, **kwargs
    )
    return opt_res, fn(opt_res.value, args)


def make_overfocus_loss_function(
    model: Model4DSTEM,
    ctx: Context,
    dataset: DataSet,
    overfocus_udf: OverfocusUDF,
    blur_function: Callable | None = None,
    extra_udfs: Iterable[UDF] = (),
    callback: Callable | None = None,
    **kwargs,
):
    """
    Build a parameter mapping and loss function to optimize

    This maps the :code:`detector_rotation` and :code:`overfocus` parameters
    for :class:`OverfocusUDF` in such a way that the starting value is at
    0 and a sensible interval for optimization is :code:`[-10, 10]` based
    on the starting parameters passed as an argument: The detector rotation
    changes by +-10 deg, and the overfocus by +-25 % of its starting value.

    It also constructs a loss function that updates the parameters of
    the :class:`OverfocusUDF` passed as parameter :code:`overfocus_udf`,
    runs it with the specified context, dataset and extra parameters,
    and then returns the blur effect of the image.

    Parameters
    ----------
    model
        Starting value for optimization. Its overfocus must not be zero.
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
    make_new_model
        Function that updates a copy of the passed input parameters
        with an argument vector of the loss function. Since this vector
        is mapped into a [-10, 10] interval, this function is created to
        invert the mapping. This can be used
        to create parameters from the minimization result.
    loss
        Function that can be called with :func:`scipy.optimize.minimize`. Starting value is
        [0, 0], sensible range is ([-10, 10], [-10, 10]).
    """
    if model.overfocus == 0:
        raise ValueError("The overfocus of the starting model must not be zero.")
    # Rotate and scale the angle so that the optimizer works between +-10,
    # corresponding to +- 10 deg
    rotation_diff = model.detector_rotation * 180 / np.pi
    rotation_scale = 1
    # Values to shift and scale the overfocus so that the optimizer works between +-10
    overfocus_diff = model.overfocus
    overfocus_scale = 40 / np.abs(model.overfocus)

    if blur_function is None:
        blur_function = blur_effect

    def make_new_model(args) -> Model4DSTEM:
        """
        Map parameters from +-10 to original range

        Parameters
        ----------

        args
            (detector_rotation, overfocus) mapped to +-10
        """
        transformed_rotation, transformed_overfocus = args
        rotation = transformed_rotation / rotation_scale + rotation_diff
        overfocus = transformed_overfocus / overfocus_scale + overfocus_diff
        new_model = (
            _numeric(model, np)
            .derive(overfocus=overfocus)
            .adjust_detector_rotation(detector_rotation=rotation / 180 * np.pi)
        )
        return type(model).from_model(new_model)

    def loss(args) -> float:
        """
        Loss function for optimizer to call

        This calls the UDF with the appropriate parameters
        and returns the blur as a metric. It can perform additional actions
        per iteration through the :code:`callback`, :code:`extra_udfs` and
        :code:`kwargs` parameters of the surrounding function.

        Parameters
        ----------

        args
            (detector_rotation, overfocus) mapped to +-10
        """
        model = make_new_model(args)
        # Hack to make parameter update work
        overfocus_udf.params.overfocus_model["model"] = model
        res = ctx.run_udf(dataset=dataset, udf=[overfocus_udf] + list(extra_udfs), **kwargs)
        blur = blur_function(res[0]["backprojected_sum"].data)
        if callback is not None:
            callback(args, overfocus_udf.params.overfocus_model["model"], res, blur)
        return blur

    return make_new_model, loss


def optimize(loss, bounds=None, minimizer_kwargs=None, **kwargs):
    """
    Convenience function to call :func:`scipy.optimize.shgo`

    This calls :func:`scipy.optimize.shgo` with sensible bounds and minimizer
    method for a loss function created with
    :func:`make_overfocus_loss_function`. Additional kwargs are passed to
    :func:`scipy.optimize.shgo`.
    """
    if bounds is None:
        bounds = [(-10, 10), (-10, 10)]

    if minimizer_kwargs is None:
        minimizer_kwargs = {"method": "COBYLA"}
    res = shgo(func=loss, bounds=bounds, minimizer_kwargs=minimizer_kwargs, **kwargs)
    return res


class _CLArgs(NamedTuple):
    ref_model: Model4DSTEM
    twotheta: float
    radius_px: float


@jax.jit
def _cl_loss(y, args: _CLArgs):
    opt_model = args.ref_model.derive(camera_length=y[0])
    radius = get_diffraction_pixel_radius(model=opt_model, twotheta=args.twotheta)
    return jnp.array((radius - args.radius_px,))


def solve_camera_length(
    ref_model: Model4DSTEM, twotheta: float, radius_px: float
) -> tuple[Model4DSTEM, jax.Array]:
    """
    Fit the camera length to the radius of a diffraction ring

    Parameters
    ----------
    ref_model
        Starting value. The other parameters are kept.
    twotheta
        Diffraction angle 2θ in radians, e.g. from
        :func:`microscope_calibration.util.diffraction.get_twothetas`
    radius_px
        Observed distance between primary beam and diffracted beam on the
        detector in pixels, see
        :func:`~microscope_calibration.common.stem_overfocus.get_diffraction_pixel_radius`

    Returns
    -------
    model, residual
        Fitted model and the residual of the radius in pixels
    """
    args = _CLArgs(ref_model=ref_model, twotheta=twotheta, radius_px=radius_px)
    start = jnp.array((ref_model.camera_length,))
    opt_res, _ = _least_squares(fn=_cl_loss, args=args, y0=start)
    # The loss function has minima at camera_length and -camera_length.
    # we take the positive side since a negative camera length doesn't make sense
    # for a classical TEM, only for reflection.
    camera_length = jnp.abs(opt_res.value[0])
    residual = _cl_loss(jnp.array((camera_length,)), args)
    return ref_model.derive(camera_length=camera_length).normalize_types(), residual


class _SPPArgs(NamedTuple):
    ref_model: Model4DSTEM
    point_1: PixelYX
    point_2: PixelYX
    physical_distance: float


@jax.jit
def _spp_loss(y, args: _SPPArgs):
    opt_model = args.ref_model.derive(scan_pixel_pitch=y[0], overfocus=0.0)
    opt_res_1 = opt_model.trace(scan_pos=args.point_1, source_dx=0.0, source_dy=0.0)
    opt_res_2 = opt_model.trace(scan_pos=args.point_2, source_dx=0.0, source_dy=0.0)
    dx = opt_res_2["specimen"].ray.x - opt_res_1["specimen"].ray.x
    dy = opt_res_2["specimen"].ray.y - opt_res_1["specimen"].ray.y
    opt_distance = jnp.linalg.norm(jnp.array((dy, dx)))
    return opt_distance - args.physical_distance


def solve_scan_pixel_pitch(
    ref_model: Model4DSTEM,
    point_1: PixelYX,
    point_2: PixelYX,
    physical_distance: float,
) -> tuple[Model4DSTEM, jax.Array]:
    """
    Fit the scan pixel pitch to the known distance between two scan positions

    Parameters
    ----------
    ref_model
        Starting value. The other parameters are kept.
    point_1, point_2
        Scan positions in pixels
    physical_distance
        Distance between the two scan positions on the specimen in m

    Returns
    -------
    model, residual
        Fitted model and the residuals of the fit in m
    """
    args = _SPPArgs(
        ref_model=_numeric(ref_model, jnp),
        point_1=point_1,
        point_2=point_2,
        physical_distance=physical_distance,
    )
    start = jnp.array((ref_model.scan_pixel_pitch,))
    opt_res, residual = _least_squares(fn=_spp_loss, args=args, y0=start)
    # The loss function has minima at scan_pixel_pitch and -scan_pixel_pitch. we
    # take the positive side since the inversion can be better expressed with a
    # scan rotation.
    return ref_model.derive(
        scan_pixel_pitch=jnp.abs(opt_res.value[0]),
    ).normalize_types(), residual


# As returned by CoMUDF in the 'regression' buffer with
# RegressionOptions.SUBTRACT_LINEAR This allows preliminary calibration of
# descan error for a single camera length by adjusting constant tilt offset and
# tilt as a function of scan.
CoMRegression = np.ndarray[tuple[3, 2], np.floating]


# Type specification for dictionary where keys are calibrated camera lengths and
# values regression specifiers. This allows full calibration of descan error if at
# least two different camera lengths are provided.
CoMRegressions = dict[float, CoMRegression]


def _regression_residuals(model: Model4DSTEM, reg: CoMRegression) -> list:
    """
    Deviation of the central ray on the detector from a CoM regression, in
    pixels, at the corners of a scan pixel
    """
    (dy, dx), (dydy, dxdy), (dydx, dxdx) = reg
    distances = []
    for scan_y in (0.0, 1.0):
        for scan_x in (0.0, 1.0):
            det_y = model.detector_center.y + (dy + dydy * scan_y + dydx * scan_x)
            det_x = model.detector_center.x + (dx + dxdy * scan_y + dxdx * scan_x)
            res = model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=0.0, source_dy=0.0)
            distances.extend(
                (
                    det_y - res["detector"].sampling["detector_px"].y,
                    det_x - res["detector"].sampling["detector_px"].x,
                )
            )
    return distances


class _DEFullArgs(NamedTuple):
    # Aligned with the CoM regression coordinate system
    aligned_model: Model4DSTEM
    regressions: CoMRegressions


@jax.jit
def _de_full_loss(y, args: _DEFullArgs):
    de = DescanError(*y)
    distances = []
    for cl, reg in args.regressions.items():
        opt_model = args.aligned_model.derive(
            camera_length=cl,
            descan_error=de,
        )
        distances.extend(_regression_residuals(opt_model, reg))
    return jnp.array(distances)


def solve_full_descan_error(
    ref_model: Model4DSTEM, regressions: CoMRegressions
) -> tuple[Model4DSTEM, jax.Array]:
    """
    Fit all components of the descan error to CoM regressions

    Parameters
    ----------
    ref_model
        Model with the same scan and detector center as the regressions. The
        other parameters are kept.
    regressions
        CoM regressions of the central beam in detector pixels for at least
        two different camera lengths, see :data:`CoMRegressions`

    Returns
    -------
    model, residual
        Fitted model and the residuals of the fit in detector pixels
    """
    # Caveat: scan and detector center of ref_model and of regressions should
    # match.

    # Align coordinate system directions with native CoM coordinate
    # system without corrections
    aligned_model = _numeric(ref_model, np).derive(
        flip_factor=1.0,
        scan_rotation=0.0,
        detector_rotation=0.0,
    )
    args = _DEFullArgs(
        aligned_model=_numeric(aligned_model, jnp),
        regressions=regressions,
    )

    # Start with a small epsilon to prevent NaN results of yet unknown origin
    # for some parameter combinations
    start = jnp.full(shape=(len(DescanError()),), fill_value=1e-6)
    opt_res, residual = _least_squares(fn=_de_full_loss, args=args, y0=start)

    # Bring descan error back to original coordinate system
    res_model = (
        aligned_model.derive(descan_error=DescanError(*opt_res.value))
        .adjust_scan_rotation(ref_model.scan_rotation)
        .adjust_detector_rotation(ref_model.detector_rotation)
        .adjust_flip_factor(ref_model.flip_factor)
    )

    return type(ref_model).from_model(res_model).normalize_types(), residual


class _NormArgs(NamedTuple):
    ref_model: Model4DSTEM


def _zero_const(de: DescanError) -> DescanError:
    return de._replace(offpxi=0.0, offpyi=0.0, offsxi=0.0, offsyi=0.0)


@jax.jit
def _norm_loss(y, args: _NormArgs):
    distances = []
    scy, scx, dcy, dcx = y
    de_new = _zero_const(args.ref_model.descan_error)
    for cl in (0, 1, 2):
        opt_model = args.ref_model.derive(
            camera_length=cl,
            descan_error=de_new,
            scan_center=PixelYX(y=scy, x=scx),
            detector_center=PixelYX(y=dcy, x=dcx),
        )
        ref_model = args.ref_model.derive(
            camera_length=cl,
        )
        for scan_y in (
            0.0,
            1.0,
        ):
            for scan_x in (0.0, 1.0):
                scan_pos = PixelYX(y=scan_y, x=scan_x)
                res = opt_model.trace(scan_pos=scan_pos, source_dy=0.0, source_dx=0.0)
                ref = ref_model.trace(scan_pos=scan_pos, source_dy=0.0, source_dx=0.0)
                distances.append(
                    (
                        res["detector"].sampling["detector_px"].y
                        - ref["detector"].sampling["detector_px"].y,
                        res["detector"].sampling["detector_px"].x
                        - ref["detector"].sampling["detector_px"].x,
                    )
                )
    return jnp.array(distances)


def normalize_descan_error(ref_model: Model4DSTEM) -> tuple[Model4DSTEM, jax.Array]:
    """
    Express the constant offsets of the descan error as scan and detector center

    The descan error is overparameterized: Its constant offsets have the same
    effect on the central beam as shifts of the scan and detector center.
    This returns an equivalent model with the constant offsets set to zero.

    Returns
    -------
    model, residual
        Fitted model and the residuals of the fit in detector pixels
    """
    args = _NormArgs(
        ref_model=_numeric(ref_model, jnp),
    )
    start = jnp.array(
        (
            ref_model.scan_center.y,
            ref_model.scan_center.x,
            ref_model.detector_center.y,
            ref_model.detector_center.x,
        )
    )
    opt_res, residual = _least_squares(fn=_norm_loss, args=args, y0=start)
    scy, scx, dcy, dcx = opt_res.value
    res_model = ref_model.derive(
        scan_center=PixelYX(y=scy, x=scx),
        detector_center=PixelYX(y=dcy, x=dcx),
        descan_error=_zero_const(ref_model.descan_error),
    ).normalize_types()
    return res_model, residual


class _DETiltArgs(NamedTuple):
    # Aligned with the CoM regression coordinate system
    aligned_model: Model4DSTEM
    regression: CoMRegression


def _tilt_descan(de: DescanError, y) -> DescanError:
    """
    Replace the tilt components of :code:`de` with :code:`y`
    """
    sxo_pxi, sxo_pyi, syo_pxi, syo_pyi, offsxi, offsyi = y
    return de._replace(
        sxo_pxi=sxo_pxi,
        sxo_pyi=sxo_pyi,
        syo_pxi=syo_pxi,
        syo_pyi=syo_pyi,
        offsxi=offsxi,
        offsyi=offsyi,
    )


@jax.jit
def _de_tilt_loss(y, args: _DETiltArgs):
    opt_model = args.aligned_model.derive(
        descan_error=_tilt_descan(de=args.aligned_model.descan_error, y=y)
    )

    return jnp.array(_regression_residuals(opt_model, args.regression))


def solve_tilt_descan_error(
    ref_model: Model4DSTEM, regression: CoMRegression
) -> tuple[Model4DSTEM, jax.Array]:
    """
    Fit the tilt components of the descan error to a CoM regression

    This allows a preliminary calibration of the descan error for a single
    camera length. The position components of the descan error are kept.

    Parameters
    ----------
    ref_model
        Model with the same scan and detector center as the regression. The
        other parameters are kept.
    regression
        CoM regression of the central beam in detector pixels, see
        :data:`CoMRegression`

    Returns
    -------
    model, residual
        Fitted model and the residuals of the fit in detector pixels
    """
    # Caveat: scan and detector center of ref_model and of regressions should
    # match.

    # Align coordinate system directions with native CoM coordinate
    # system without corrections
    # We make sure that the offset-based descan error components are preserved
    aligned_model = (
        _numeric(ref_model, np)
        .adjust_flip_factor(
            flip_factor=1.0,
        )
        .adjust_scan_rotation(
            scan_rotation=0.0,
        )
        .adjust_detector_rotation(
            detector_rotation=0.0,
        )
    )
    args = _DETiltArgs(
        aligned_model=_numeric(aligned_model, jnp),
        regression=regression,
    )

    # Start with a small epsilon to prevent NaN results of yet unknown origin
    # for some parameter combinations
    start = jnp.full(shape=(6,), fill_value=1e-6)
    opt_res, residual = _least_squares(fn=_de_tilt_loss, args=args, y0=start, max_steps=10000)

    # Bring descan error back to original coordinate system
    res_model = (
        aligned_model.derive(descan_error=_tilt_descan(aligned_model.descan_error, opt_res.value))
        .adjust_detector_rotation(ref_model.detector_rotation)
        .adjust_scan_rotation(ref_model.scan_rotation)
        .adjust_flip_factor(ref_model.flip_factor)
    )

    return type(ref_model).from_model(res_model).normalize_types(), residual


class _DETiltPointArgs(NamedTuple):
    model: Model4DSTEM
    # [(scan_y, scan_x, detector_cy, detector_cx) * n]
    points: jax.Array


@jax.jit
def _de_tilt_point_loss(y, args: _DETiltPointArgs):
    opt_model = args.model.derive(descan_error=_tilt_descan(de=args.model.descan_error, y=y))

    distances = []
    for scan_y, scan_x, det_y, det_x in args.points:
        res = opt_model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=0.0, source_dy=0.0)
        distances.extend(
            (
                det_y - res["detector"].sampling["detector_px"].y,
                det_x - res["detector"].sampling["detector_px"].x,
            )
        )
    return jnp.array(distances)


def solve_tilt_descan_error_points(
    ref_model: Model4DSTEM, points: jax.Array
) -> tuple[Model4DSTEM, jax.Array]:
    """
    Fit the tilt components of the descan error to central beam positions

    Like :func:`solve_tilt_descan_error`, but with the position of the
    central beam at individual scan positions instead of a regression.

    Parameters
    ----------
    ref_model
        Starting value. The other parameters are kept.
    points
        Array with at least three rows :code:`(scan_y, scan_x, detector_y,
        detector_x)` in pixels

    Returns
    -------
    model, residual
        Fitted model and the residuals of the fit in detector pixels
    """
    points = jnp.asarray(points, dtype=jnp.float64).reshape((-1, 4))
    # Six unknowns, two equations per point
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points to fit the descan error, got {len(points)}.")
    args = _DETiltPointArgs(
        model=_numeric(ref_model, jnp),
        points=points,
    )

    # Start with a small epsilon to prevent NaN results of yet unknown origin
    # for some parameter combinations
    start = jnp.full(shape=(6,), fill_value=1e-6)
    opt_res, residual = _least_squares(
        fn=_de_tilt_point_loss,
        args=args,
        y0=start,
        # FIXME Doesn't reach 1e-12 like the others, for unknown reasons
        tol=1e-11,
        # FIXME needs more steps than others, for unknown reasons
        max_steps=10000,
    )
    # Write the new tilts and offsets into the previous descan error
    new_descan = _tilt_descan(ref_model.descan_error, opt_res.value)
    res_model = ref_model.derive(descan_error=new_descan).normalize_types()

    return res_model, residual


class _CoordPointArgs(NamedTuple):
    model: Model4DSTEM
    # [(scan_y, scan_x, specimen_y (px), specimen_x (px), detector_y, detector_x) * n]
    points: jax.Array
    flip_factor: float


def _coords_point_modelupdate(
    model: Model4DSTEM, overfocus, detector_rotation, flip_factor
) -> Model4DSTEM:
    return (
        model.derive(
            overfocus=overfocus,
        )
        .adjust_detector_rotation(
            detector_rotation=detector_rotation,
        )
        .adjust_flip_factor(
            flip_factor=flip_factor,
        )
    )


@jax.jit
def _coords_point_loss(y, args: _CoordPointArgs):
    detector_rotation, overfocus = y[:2]
    # We need to co-optimize the source tilts that hit the detector pixel for
    # each point set because we can't easily calculate that directly
    tilts = y[2:]

    assert len(tilts) == 2 * len(args.points)

    opt_model = _coords_point_modelupdate(
        model=args.model,
        overfocus=overfocus,
        detector_rotation=detector_rotation,
        flip_factor=args.flip_factor,
    )

    detector_distances = []
    specimen_distances = []
    for i, (scan_y, scan_x, spec_y, spec_x, det_y, det_x) in enumerate(args.points):
        dy, dx = tilts[2 * i : 2 * i + 2]
        res = opt_model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=dx, source_dy=dy)
        detector_distances.extend(
            (
                res["detector"].sampling["detector_px"].y - det_y,
                res["detector"].sampling["detector_px"].x - det_x,
            )
        )
        specimen_distances.extend(
            (
                res["specimen"].sampling["scan_px"].y - spec_y,
                res["specimen"].sampling["scan_px"].x - spec_x,
            )
        )

    return jnp.concatenate((jnp.array(detector_distances), jnp.array(specimen_distances)))


def solve_coords_points(ref_model: Model4DSTEM, points: jax.Array) -> tuple[Model4DSTEM, jax.Array]:
    """
    Fit overfocus, detector rotation and flip factor to matched points

    The flip factor is either 1 or -1. Both are fitted, and the better fit is returned.

    Parameters
    ----------
    ref_model
        Starting values. The descan error is adjusted to compensate the
        changes, see :meth:`Model4DSTEM.adjust_detector_rotation`.
    points
        Array with rows :code:`(scan_y, scan_x, specimen_y, specimen_x,
        detector_y, detector_x)` in pixels: A ray through scan position
        :code:`(scan_y, scan_x)` that passes the specimen at the given scan
        pixel coordinates hits the detector at the given detector pixel.

    Returns
    -------
    model, residual
        Fitted model and the residual of the fitted points in pixels
    """
    points = jnp.asarray(points, dtype=jnp.float64).reshape((-1, 6))
    if len(points) < 1:
        raise ValueError("Need at least one point.")
    start = jnp.array([ref_model.detector_rotation, ref_model.overfocus] + [0.0, 0.0] * len(points))
    results = []
    for flip_factor in (1.0, -1.0):
        args = _CoordPointArgs(
            model=_numeric(ref_model, jnp), points=points, flip_factor=flip_factor
        )
        opt_res, residual = _least_squares(
            fn=_coords_point_loss,
            args=args,
            y0=start,
            max_steps=10000,
            # Can fail to converge for the wrong flip factor
            throw=False,
        )
        results.append((float(jnp.sum(residual**2)), flip_factor, opt_res, residual))
    _, flip_factor, opt_res, residual = min(results, key=lambda r: r[0])
    if not bool(opt_res.result == optimistix.RESULTS.successful):
        raise RuntimeError(f"Fit did not converge: {optimistix.RESULTS[opt_res.result]}")
    # Write the new values into the previous parameters
    detector_rotation, overfocus = opt_res.value[:2]
    res_model = _coords_point_modelupdate(
        model=_numeric(ref_model, np),
        overfocus=overfocus,
        detector_rotation=detector_rotation,
        flip_factor=flip_factor,
    )

    return type(ref_model).from_model(res_model).normalize_types(), residual


class _HitSpecimenArgs(NamedTuple):
    model: Model4DSTEM
    scan_pos: PixelYX
    specimen_px: PixelYX


@jax.jit
def _hit_specimen_loss(y, args: _HitSpecimenArgs):
    dy, dx = y
    opt_model = args.model

    res = opt_model.trace(scan_pos=args.scan_pos, source_dx=dx, source_dy=dy)
    specimen_px = res["specimen"].sampling["scan_px"]
    return jnp.array((specimen_px.y - args.specimen_px.y, specimen_px.x - args.specimen_px.x))


class SlopeYX(NamedTuple):
    dy: float
    dx: float


def solve_hit_specimen(
    model: Model4DSTEM, scan_pos: PixelYX, specimen_px: PixelYX
) -> tuple[SlopeYX, jax.Array]:
    """
    Find the source tilt of the ray that passes the specimen at a given point

    Parameters
    ----------
    model
        Model to trace
    scan_pos
        Scan position in pixels
    specimen_px
        Point on the specimen in scan pixel coordinates

    Returns
    -------
    slope, residual
        Source tilt and the residual on the specimen in scan pixels
    """
    args = _HitSpecimenArgs(
        model=_numeric(model, jnp),
        scan_pos=scan_pos,
        specimen_px=specimen_px,
    )

    start = jnp.array([0.0, 0.0])
    opt_res, residual = _least_squares(fn=_hit_specimen_loss, args=args, y0=start, max_steps=10000)
    dy, dx = opt_res.value
    return SlopeYX(dy=float(dy), dx=float(dx)), residual
