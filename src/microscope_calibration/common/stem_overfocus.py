from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
# ruff: disable[E402]

import jax.numpy as jnp
import numba
import numpy as np
import sympy as sym

from microscope_calibration.common.model import (
    CoordXY,
    DescanError,
    Model4DSTEM,
    PixelYX,
    Scanner,
    lambdify_trace_for,
)
from microscope_calibration.util.sympy import lambdify, symbol_maker
# ruff: enable[E402]

# Define here to facilitate mocking in order to test
# the code that checks for float64 support in JAX as a probable cause
# for discrepancies
target_dtype = jnp.float64

CoordMappingT = Callable[[CoordXY], PixelYX]


class FittingError(RuntimeError):
    pass


trace = lambdify_trace_for(jnp)


def _do_lstsq(input_samples, output_samples):
    output_samples = np.array(output_samples)
    input_samples = np.array(input_samples)

    x, residuals, rank, s = np.linalg.lstsq(input_samples, output_samples)

    # FIXME include test also based on singular values
    if len(residuals) != output_samples.shape[1]:
        raise FittingError("Mismatch between residuals and samples, likely ill-posed or sth.")
    # Confirm that the solution is exact, in particular that
    # the model is linear
    assert rank == input_samples.shape[1]

    no_residuals = np.allclose(residuals, 0.0, rtol=1e-12, atol=1e-11)

    test_samples = np.empty_like(output_samples)
    for i in range(len(input_samples)):
        test_samples[i] = input_samples[i] @ x

    reproduced = np.allclose(output_samples, test_samples, rtol=1e-9, atol=1e-9)

    if not (no_residuals and reproduced):
        test = jnp.array((1, 2, 3), dtype=jnp.float64)
        if test.dtype != target_dtype:
            raise RuntimeError(
                f"No float64 support activated in JAX. Downcasting to {test.dtype} is "
                "leading to inaccuracies that are "
                "much larger than permissible for electron optics calculations."
            )
        else:
            if not no_residuals:
                raise RuntimeError(
                    f"Model seems not linear: Residuals {residuals} exceeding tolerance of 1e-12"
                )
            # Currently not sure if this can be reached in tests without also having residuals
            elif not reproduced:  # pragma: no cover
                raise RuntimeError(  # pragma: no cover
                    f"Discrepancies between model output {output_samples} "  # pragma: no cover
                    f"and equivalent linear transformation {test_samples} "  # pragma: no cover
                    "exceed tolerance of 1e-12."  # pragma: no cover
                )  # pragma: no cover
            else:  # pragma: no cover
                raise RuntimeError(  # pragma: no cover
                    "If this code is reached, logic is broken."  # pragma: no cover
                )  # pragma: no cover

    return x


def _derive_affine_map(inputs, outputs, equations, unknowns):
    """
    Derive the matrix of an affine map from :code:`inputs` to :code:`outputs`

    The outputs are expressions of the inputs and of unknowns, e.g. the source
    tilt. The unknowns are eliminated by solving :code:`equations`, expressions
    that have to be zero, for them.

    Returns
    -------
    matrix
        Nested tuple with one row per input plus a row for the constant
        offset, and one column per output plus a constant column, so that
        :code:`(*inputs, 1) @ matrix == (*outputs, 1)`.
    """
    solutions = sym.solve(equations, unknowns, dict=True)
    if len(solutions) != 1 or set(solutions[0]) != set(unknowns):
        raise ValueError(f"No unique solution for {unknowns}: {solutions}")
    (solution,) = solutions
    outputs = [sym.expand(sym.sympify(o).subs(solution)) for o in outputs]
    # Clean up factors like 1.0 from Ray._one
    outputs = [
        o.xreplace({f: sym.Integer(int(f)) for f in o.atoms(sym.Float) if float(f).is_integer()})
        for o in outputs
    ]
    forbidden = set(inputs) | set(unknowns)
    rows = []
    for inp in inputs:
        row = []
        for o in outputs:
            coeff = sym.diff(o, inp)
            if coeff.free_symbols & forbidden:
                raise ValueError(f"Map is not affine, d({o})/d({inp}) = {coeff}")
            row.append(coeff)
        rows.append((*row, 0))
    constants = [o.subs({inp: 0 for inp in inputs}) for o in outputs]
    for c in constants:
        if c.free_symbols & forbidden:
            raise ValueError(f"Map is not affine, offset {c}")
    rows.append((*constants, 1))
    return tuple(rows)


def _derive_backward_matrix():
    """
    Derive the backward transformation matrix from :meth:`Model4DSTEM.trace`

    Maps scan position in scan pixel coordinates and physical specimen
    coordinates to detector pixel coordinates and source tilt. The source tilt
    is eliminated by solving for the given specimen position.
    """
    model = symbol_maker(Model4DSTEM, postfix="model", recurse_for=(PixelYX, DescanError))
    scan_y, scan_x, tilt_y, tilt_x, spec_y, spec_x = sym.symbols(
        "scan_y scan_x tilt_y tilt_x spec_y spec_x"
    )
    res = model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=tilt_y, source_dx=tilt_x)
    spec = res["specimen"].ray
    det = res["detector"].sampling["detector_px"]
    matrix = _derive_affine_map(
        inputs=(scan_y, scan_x, spec_y, spec_x),
        outputs=(det.y, det.x, tilt_y, tilt_x),
        equations=(spec.y - spec_y, spec.x - spec_x),
        unknowns=(tilt_y, tilt_x),
    )
    return {"model": model}, matrix


# Functions that are derived symbolically from Model4DSTEM.trace() and rendered
# into _generated.py, see microscope_calibration.common.generate
GENERATED_FUNCTIONS = {
    "backward_matrix": _derive_backward_matrix,
}


@lambdify(modules=np, recurse_for=(PixelYX, Model4DSTEM, DescanError))
def _backward_matrix_physical(model: Model4DSTEM):
    # Imported here so that the generator can run even if
    # _generated.py is missing or outdated
    from microscope_calibration.common import _generated

    return _generated.backward_matrix(model=model)


def _affine_coefficients(mapping: CoordMappingT) -> tuple[np.ndarray, np.ndarray]:
    """
    Matrix :code:`a` and offset :code:`c` of an affine mapping from physical
    coordinates :code:`(y, x)` to pixel coordinates :code:`(y, x)`, i.e.
    :code:`mapping(p) == a @ p + c`.

    The mapping is evaluated with SymPy symbols to obtain the coefficients.
    """
    y, x = sym.symbols("y x")
    res = mapping(CoordXY(x=x, y=y))
    a = np.empty((2, 2))
    c = np.empty(2)
    for i, out in enumerate((res.y, res.x)):
        out = sym.expand(sym.sympify(out))
        for j, inp in enumerate((y, x)):
            coeff = sym.diff(out, inp)
            if coeff.free_symbols:
                raise ValueError(f"Coordinate mapping is not affine: {out}")
            a[i, j] = float(coeff)
        c[i] = float(out.subs({y: 0, x: 0}))
    return a, c


def _specimen_transform(mapping: CoordMappingT) -> np.ndarray:
    """
    Matrix that converts the input row :code:`(scan_y, scan_x, px_y, px_x, 1)`
    with pixel coordinates from :code:`mapping` to the input row with the
    corresponding physical coordinates :code:`(scan_y, scan_x, y, x, 1)`.
    """
    a, c = _affine_coefficients(mapping)
    try:
        a_inv = np.linalg.inv(a)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Coordinate mapping is singular: {a}") from e
    transform = np.eye(5)
    transform[2:4, 2:4] = a_inv.T
    transform[4, 2:4] = -(a_inv @ c)
    return transform


def _evaluate_matrix(f, *args) -> np.ndarray:
    """
    Evaluate a numeric matrix function, converting division by zero
    into :class:`FittingError`
    """
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.array(f(*args), dtype=np.float64)
    except ZeroDivisionError as e:
        raise FittingError("Transformation is singular") from e


def _check_footprint(matrix: np.ndarray, semiconv: float, min_footprint_px: float):
    """
    Check that a matrix that maps pixel coordinates in some plane to source
    tilt is well-defined

    The rows 2 and 3 of :code:`matrix` are the pixel coordinates in that plane,
    columns 2 and 3 the source tilt. Their inverse is the change of pixel
    coordinates with source tilt. Scaled by the semi-convergence angle, the
    smallest singular value of that is the smallest semi-axis of the beam's
    footprint in that plane in pixels.

    If the footprint is (close to) zero, the transformation is singular, e.g.
    in focus, and pixel coordinates don't determine the tilt.
    """
    if not np.all(np.isfinite(matrix)):
        raise FittingError(f"Transformation is singular: {matrix}")
    with np.errstate(divide="ignore"):
        footprint = semiconv / np.linalg.norm(matrix[2:4, 2:4], ord=2)
    if not footprint >= min_footprint_px:
        raise FittingError(
            f"Beam footprint of {footprint} px is smaller than min_footprint_px="
            f"{min_footprint_px}, the transformation is (close to) singular."
        )


def get_backward_transformation_matrix(
    rec_model: Model4DSTEM,
    specimen_to_image: CoordMappingT | None = None,
    min_footprint_px: float = 1e-6,
):
    """
    Calculate a transformation matrix that maps from scan position in scan pixel
    coordinates and specimen pixel coordinates to detector coordinates in pixel
    coordinates and tilt of the ray at the source.

    Using a matrix multiplication instead of solving for ray solutions for each
    pixel greatly improves performance.

    The detector positions from the output can be used to pick the right value
    from a detector frame to superimpose an image of the specimen resp. an image
    of the beam-shaping aperture. The tilt can be used to determine if the beam
    passes through through the microscope or if it is blocked by the
    beam-shaping aperture.

    The matrix is derived symbolically from :meth:`Model4DSTEM.trace` for
    physical specimen coordinates, see :func:`_derive_backward_matrix` and
    :mod:`microscope_calibration.common.generate`. It is combined with the
    affine mapping :code:`specimen_to_image` from physical specimen coordinates
    to specimen pixel coordinates.

    Parameters
    ----------
    rec_model
        Model of the reconstruction
    specimen_to_image
        Affine mapping from physical specimen coordinates to specimen pixel
        coordinates. It is evaluated with SymPy symbols to determine its
        coefficients. By default, scan pixel coordinates of :code:`rec_model`.
    min_footprint_px
        Minimum smallest semi-axis of the beam's footprint in the specimen
        plane in specimen pixels, given the semi-convergence angle. Below
        that, the transformation is considered (close to) singular, e.g. in
        focus.

    Raises
    ------
    FittingError
        If the transformation is singular or close to singular.
    """
    if specimen_to_image is None:
        specimen_to_image = rec_model.real_to_scan
    matrix = _specimen_transform(specimen_to_image) @ _evaluate_matrix(
        _backward_matrix_physical, rec_model
    )
    _check_footprint(matrix, semiconv=float(rec_model.semiconv), min_footprint_px=min_footprint_px)
    return matrix


# Separate functions spun out to facilitate re-use of coordinate calculations
# for other purposes
@numba.njit(inline="always", cache=True)
def project_tilt_y(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 2]
        + scan_x * mat[1, 2]
        + image_y * mat[2, 2]
        + image_x * mat[3, 2]
        + mat[4, 2]
    )


@numba.njit(inline="always", cache=True)
def project_tilt_x(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 3]
        + scan_x * mat[1, 3]
        + image_y * mat[2, 3]
        + image_x * mat[3, 3]
        + mat[4, 3]
    )


@numba.njit(inline="always", cache=True)
def project_det_y(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 0]
        + scan_x * mat[1, 0]
        + image_y * mat[2, 0]
        + image_x * mat[3, 0]
        + mat[4, 0]
    )


@numba.njit(inline="always", cache=True)
def project_det_x(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 1]
        + scan_x * mat[1, 1]
        + image_y * mat[2, 1]
        + image_x * mat[3, 1]
        + mat[4, 1]
    )


@numba.njit(cache=True)
def project_frame_backwards(frame, source_semiconv, mat, scan_y, scan_x, image_out):
    limit = np.abs(np.tan(source_semiconv)) ** 2
    for image_y in range(image_out.shape[0]):
        for image_x in range(image_out.shape[1]):
            # Manually unrolled matrix-vector product to allow skipping before
            # calculating all values and facilitate auto-vectorization of the
            # loop

            # _one = (
            #       scan_y * mat[0, 4] + scan_x * mat[1, 4]
            #       + det_y * mat[2, 4] + det_x * mat[3, 4] + mat[4, 4]
            # )
            # assert np.allclose(_one, 1)
            tilt_y = project_tilt_y(image_y, image_x, scan_y, scan_x, mat)
            tilt_x = project_tilt_x(image_y, image_x, scan_y, scan_x, mat)
            if np.abs(tilt_y) ** 2 + np.abs(tilt_x) ** 2 < limit:
                det_y = project_det_y(image_y, image_x, scan_y, scan_x, mat)
                det_x = project_det_x(image_y, image_x, scan_y, scan_x, mat)
                det_y = int(np.round(det_y))
                det_x = int(np.round(det_x))
                if det_y >= 0 and det_y < frame.shape[0] and det_x >= 0 and det_x < frame.shape[1]:
                    image_out[image_y, image_x] += frame[det_y, det_x]


def get_detector_correction_matrix(rec_model: Model4DSTEM, ref_model: Model4DSTEM | None = None):
    """
    Calculate a transformation matrix that maps from scan position in scan pixel
    coordinates and output detector pixel coordinates in a reference system to
    detector pixel coordinates in the reconstruction system and tilt of the ray
    at the source.

    If no reference parameters are specified, it derives the reference from the
    reconstruction parameters by setting scan rotation and descan error to 0,
    and flip_factor to 1..

    Using a matrix multiplication instead of solving for ray solutions for each
    pixel greatly improves performance.

    The detector positions from the output can be used to pick the right value
    from a detector frame to superimpose an image of the beam-shaping aperture
    or create corrected detector frames for further processing. The tilt can be
    used to determine if the beam passes through through the microscope or if it
    is blocked by the beam-shaping aperture, or to calculate virtual detector
    images etc.

    It may be possible to derive this matrix from partial derivatives of the
    model. However, this is postponed for now since this matrix mixes input and
    output values with respect of the model, so one may have to work with a
    combination of forward and inverse derivatives.

    For the time being this method traces a number of sample rays and deduces
    the mapping matrix from these samples.
    """

    # scan position y/x, source tilt y/x
    test_parameters = np.array(
        (
            [0.0, 0.0, 0.0, 0.0],
            [100.0, 100.0, 0.0, 0.0],
            [-100.0, 100.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
        )
    )

    input_samples = []
    output_samples = []

    @lambdify(modules=np)
    def get_sample(scan_pos_x, scan_pos_y, source_dy, source_dx):
        scan_pos = PixelYX(x=scan_pos_x, y=scan_pos_y)
        res = rec_model.trace(
            scan_pos=scan_pos,
            source_dy=source_dy,
            source_dx=source_dx,
        )

        ref_res = ref_model.trace(
            scan_pos=scan_pos,
            source_dy=source_dy,
            source_dx=source_dx,
        )

        input_sample = (
            scan_pos.y,
            scan_pos.x,
            ref_res["detector"].sampling["detector_px"].y,
            ref_res["detector"].sampling["detector_px"].x,
            1.0,
        )
        output_sample = (
            res["detector"].sampling["detector_px"].y,
            res["detector"].sampling["detector_px"].x,
            source_dy,
            source_dx,
            1.0,
        )
        return (input_sample, output_sample)

    if ref_model is None:
        ref_model = rec_model.derive(
            scan_rotation=0.0,
            flip_factor=1.0,
            descan_error=DescanError(),
            detector_rotation=rec_model.scan_rotation,
        )

    for test_param_raw in test_parameters:
        # We are paranoid and confirm that the model is linear
        for factor in (1.0, 2.0):
            test_param = test_param_raw * factor
            input_sample, output_sample = get_sample(*test_param)
            output_samples.append(output_sample)
            input_samples.append(input_sample)

    return _do_lstsq(input_samples, output_samples)


# Separate functions spun out to facilitate re-use of coordinate calculations
# for other purposes, such as corrected virtual detectors
@numba.njit(inline="always", cache=True)
def corrected_det_y(det_corr_y, det_corr_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 0]
        + scan_x * mat[1, 0]
        + det_corr_y * mat[2, 0]
        + det_corr_x * mat[3, 0]
        + mat[4, 0]
    )


@numba.njit(inline="always", cache=True)
def corrected_det_x(det_corr_y, det_corr_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 1]
        + scan_x * mat[1, 1]
        + det_corr_y * mat[2, 1]
        + det_corr_x * mat[3, 1]
        + mat[4, 1]
    )


@numba.njit(cache=True)
def correct_frame(frame, mat, scan_y, scan_x, detector_out):
    for det_corr_y in range(detector_out.shape[0]):
        for det_corr_x in range(detector_out.shape[1]):
            # Manually unrolled matrix-vector product to allow skipping before
            # calculating all values and facilitate auto-vectorization of the
            # loop
            det_y = corrected_det_y(det_corr_y, det_corr_x, scan_y, scan_x, mat)
            det_x = corrected_det_x(det_corr_y, det_corr_x, scan_y, scan_x, mat)
            det_y = int(np.round(det_y))
            det_x = int(np.round(det_x))
            if det_y >= 0 and det_y < frame.shape[0] and det_x >= 0 and det_x < frame.shape[1]:
                detector_out[det_corr_y, det_corr_x] += frame[det_y, det_x]


@lambdify(recurse_for=(PixelYX, Model4DSTEM, DescanError), modules=jnp)
def trace_with_diffractor(model: Model4DSTEM, scan_pos: PixelYX, source_dy, source_dx, twotheta):
    diffractor = Scanner(
        z=model.overfocus,
        scan_pos_x=0.0,
        scan_pos_y=0.0,
        scan_tilt_x=sym.tan(twotheta),
    )
    return model.trace(
        scan_pos=scan_pos, source_dx=source_dx, source_dy=source_dy, specimen=diffractor
    )


@jax.jit
def get_diffraction_pixel_radius(model: Model4DSTEM, twotheta: float):
    center_res = trace(
        model,
        scan_pos=PixelYX(0.0, 0.0),
        source_dx=0.0,
        source_dy=0.0,
    )
    diff_res = trace_with_diffractor(
        model=model,
        scan_pos=PixelYX(0.0, 0.0),
        source_dx=0.0,
        source_dy=0.0,
        twotheta=twotheta,
    )
    return jnp.linalg.norm(
        jnp.array(center_res["detector"].sampling["detector_px"])
        - jnp.array(diff_res["detector"].sampling["detector_px"])
    )


@jax.jit
def get_primary_beam_radius(model: Model4DSTEM):
    center_res = trace(
        model,
        scan_pos=PixelYX(0.0, 0.0),
        source_dx=0.0,
        source_dy=0.0,
    )
    border_res = trace(
        model,
        scan_pos=PixelYX(0.0, 0.0),
        source_dx=model.semiconv,
        source_dy=0.0,
    )
    return jnp.linalg.norm(
        jnp.array(center_res["detector"].sampling["detector_px"])
        - jnp.array(border_res["detector"].sampling["detector_px"])
    )


@jax.jit
def ring_radii(model: Model4DSTEM, twothetas):
    pixel_radii = jnp.array(
        [get_diffraction_pixel_radius(model=model, twotheta=twotheta) for twotheta in twothetas]
    )
    beam_radius = get_primary_beam_radius(model)
    ri = jnp.maximum(pixel_radii - beam_radius, 0)
    ro = pixel_radii + beam_radius
    return ri, ro


@jax.jit
def get_center(model: Model4DSTEM, scan_pos: PixelYX):
    center_res = trace(
        model,
        scan_pos=scan_pos,
        source_dx=0.0,
        source_dy=0.0,
    )
    y = center_res["detector"].sampling["detector_px"].y
    x = center_res["detector"].sampling["detector_px"].x
    return PixelYX(
        y=y,
        x=x,
    )
