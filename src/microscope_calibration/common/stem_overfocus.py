from collections.abc import Callable
from functools import cache

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
    NonlinearModelError,
    PixelYX,
    Scanner,
    check_pytree,
    lambdify_trace_for,
    symbolic_model,
    warn_runtime_derivation,
)
from microscope_calibration.util.sympy import lambdify, lambdify_tree
# ruff: enable[E402]

CoordMappingT = Callable[[CoordXY], PixelYX]


class FittingError(RuntimeError):
    pass


trace = lambdify_trace_for(jnp)


def _derive_affine_map(inputs, outputs):
    """
    Derive the matrix of an affine map from :code:`inputs` to :code:`outputs`

    Raises :class:`NonlinearModelError` if the outputs are not affine in the
    inputs.

    Returns
    -------
    matrix
        Nested tuple with one row per input plus a row for the constant
        offset, and one column per output plus a constant column, so that
        :code:`(*inputs, 1) @ matrix == (*outputs, 1)`.
    """
    outputs = [sym.expand(sym.sympify(o)) for o in outputs]
    # Clean up factors like 1.0 from Ray._one
    outputs = [
        o.xreplace({f: sym.Integer(int(f)) for f in o.atoms(sym.Float) if float(f).is_integer()})
        for o in outputs
    ]
    inputs_set = set(inputs)
    rows = []
    for inp in inputs:
        row = []
        for o in outputs:
            coeff = sym.diff(o, inp)
            if coeff.free_symbols & inputs_set:
                raise NonlinearModelError(f"Model is not linear, d({o})/d({inp}) = {coeff}")
            row.append(coeff)
        rows.append((*row, 0))
    constants = [o.subs({inp: 0 for inp in inputs}) for o in outputs]
    for c in constants:
        if c.free_symbols & inputs_set:
            raise NonlinearModelError(f"Model is not linear, offset {c}")
    rows.append((*constants, 1))
    return tuple(rows)


def _derive_ray_transfer_matrix(model_type: type[Model4DSTEM] = Model4DSTEM):
    """
    Derive the ray transfer matrix of a model from :meth:`Model4DSTEM.trace`

    The matrix maps scan position in scan pixel coordinates and source tilt
    to physical specimen coordinates and detector pixel coordinates:

    .. code-block:: text

        (scan_y, scan_x, tilt_y, tilt_x, 1) @ matrix
            == (specimen_y, specimen_x, detector_y, detector_x, 1)

    This is the only part that is derived symbolically and generated. The
    transformation matrices for back projection, forward projection and
    detector correction are calculated from it with :func:`_eliminate_tilt`.

    Notes
    -----
    Alternatively, the transformation matrices themselves can be derived
    fully symbolically by eliminating the tilt with SymPy. For example, for the
    backward transformation matrix from scan pixel coordinates and physical
    specimen coordinates to detector pixel coordinates and tilt:

    .. code-block:: python

        res = model.trace(
            scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=tilt_y, source_dx=tilt_x
        )
        spec = res["specimen"].ray
        det = res["detector"].sampling["detector_px"]
        (solution,) = sym.solve(
            (spec.y - spec_y, spec.x - spec_x), (tilt_y, tilt_x), dict=True
        )
        outputs = [
            sym.expand(sym.trigsimp(o).subs(solution))
            for o in (det.y, det.x, tilt_y, tilt_x)
        ]
        # Coefficients of the affine map
        matrix = _derive_affine_map(
            inputs=(scan_y, scan_x, spec_y, spec_x), outputs=outputs
        )

    The forward transformation matrix solves for the detector position
    instead, and the detector correction matrix solves for the detector
    position of a reference model and substitutes the tilt into the
    reconstruction model.

    This works and gives the same results, but the expressions become large
    since the inverse of the 2x2 tilt system is multiplied into all entries.
    For the detector correction matrix, both models' rotations, flips and
    descan errors appear in each entry. Even with common subexpression
    elimination and simplification of the solved tilt, the generated code
    is about 11k characters with expressions of up to 3.5k characters for
    the detector correction matrix, 3.3k characters for the forward matrix
    and 3.8k characters for the backward matrix, and the derivation takes
    about 15 s. In comparison, the generated ray transfer matrix is about
    2.4k characters with expressions of up to about 530 characters, and the
    derivation takes 0.2 s. Eliminating the tilt numerically from the ray
    transfer matrix is exact linear algebra on a 2x2 matrix, so it doesn't
    reintroduce the sampling and least-squares fitting that the symbolic
    derivation replaced.
    """
    model = symbolic_model(model_type, postfix="model")
    scan_y, scan_x, tilt_y, tilt_x = sym.symbols("scan_y scan_x tilt_y tilt_x")
    res = model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=tilt_y, source_dx=tilt_x)
    spec = res["specimen"].ray
    det = res["detector"].sampling["detector_px"]
    matrix = _derive_affine_map(
        inputs=(scan_y, scan_x, tilt_y, tilt_x),
        outputs=(spec.y, spec.x, det.y, det.x),
    )
    return {"model": model}, matrix


# Functions that are derived symbolically from Model4DSTEM.trace() and rendered
# into _generated.py, see microscope_calibration.common.generate
GENERATED_FUNCTIONS = {
    "ray_transfer_matrix": _derive_ray_transfer_matrix,
}


@cache
def _ray_transfer_matrix_numeric(model_type: type[Model4DSTEM]):
    """
    Numeric ray transfer matrix function for a model class

    Uses generated code if available for the class, see
    :meth:`Model4DSTEM._generated_module`. Otherwise, it is derived from the
    class's :meth:`~Model4DSTEM.trace` at runtime.
    """
    module = model_type._generated_module()
    if module is None:
        warn_runtime_derivation("ray_transfer_matrix", model_type)
        args, matrix = _derive_ray_transfer_matrix(model_type)
        model = args["model"]
    else:
        model = symbolic_model(model_type, postfix="model")
        matrix = module.ray_transfer_matrix(model=model)
    check_pytree(model)
    return lambdify_tree(model, matrix, modules=np)


def ray_transfer_matrix(
    model: Model4DSTEM, specimen_to_image: CoordMappingT | None = None
) -> np.ndarray:
    """
    Ray transfer matrix of :code:`model`

    .. code-block:: text

        (scan_y, scan_x, tilt_y, tilt_x, 1) @ matrix
            == (specimen_y, specimen_x, detector_y, detector_x, 1)

    Scan and detector positions are in pixel coordinates, the specimen
    position in pixel coordinates according to :code:`specimen_to_image`,
    and tilt is the source tilt in radians.

    The matrix is derived symbolically from :meth:`Model4DSTEM.trace`, see
    :func:`_derive_ray_transfer_matrix` and
    :mod:`microscope_calibration.common.generate`.

    Parameters
    ----------
    model
        Model to trace
    specimen_to_image
        Affine mapping from physical specimen coordinates to specimen pixel
        coordinates. It is evaluated with SymPy symbols to determine its
        coefficients. By default, scan pixel coordinates of :code:`model`.

    For subclasses without generated code, the matrix is derived from their
    :code:`trace()` on first use instead, see :class:`Model4DSTEM`.

    Raises
    ------
    FittingError
        If the matrix can't be calculated, e.g. with zero pixel pitch.
    NonlinearModelError
        If the :code:`trace()` of a subclass without generated code is not
        linear.
    """
    f = _ray_transfer_matrix_numeric(type(model))
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            matrix = np.array(f(model), dtype=np.float64)
    except ZeroDivisionError as e:
        raise FittingError(f"Ray transfer matrix is singular for {model}") from e
    if not np.all(np.isfinite(matrix)):
        raise FittingError(f"Ray transfer matrix is not finite for {model}: {matrix}")
    if specimen_to_image is None:
        specimen_to_image = model.real_to_scan
    a, c = _affine_coefficients(specimen_to_image)
    # Convert physical specimen coordinates to pixel coordinates
    output_transform = np.eye(5)
    output_transform[0:2, 0:2] = a.T
    output_transform[4, 0:2] = c
    return matrix @ output_transform


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


def _eliminate_tilt(
    plane: np.ndarray, output: np.ndarray, semiconv: float, min_footprint_px: float
) -> np.ndarray:
    """
    Combine two affine maps of scan position and source tilt to a map from
    scan position and position in a plane to an output position and source tilt

    The source tilt is eliminated by solving for the position in the plane.

    Parameters
    ----------
    plane, output
        5x2 matrices with :code:`(scan_y, scan_x, tilt_y, tilt_x, 1) @ plane
        == (plane_y, plane_x)` in pixel coordinates and correspondingly for
        :code:`output`, i.e. columns of a ray transfer matrix.
    semiconv
        Semi-convergence angle of the beam
    min_footprint_px
        Minimum smallest semi-axis of the beam's footprint in the plane, see
        below.

    Returns
    -------
    matrix
        5x5 matrix with :code:`(scan_y, scan_x, plane_y, plane_x, 1) @ matrix
        == (output_y, output_x, tilt_y, tilt_x, 1)`

    Raises
    ------
    FittingError
        The rows of :code:`plane` for tilt are the change of the position in
        the plane with source tilt. Scaled by the semi-convergence angle, the
        smallest singular value of that is the smallest semi-axis of the
        beam's footprint in the plane in pixels. If the footprint is smaller
        than :code:`min_footprint_px`, the transformation is considered (close
        to) singular, e.g. in focus, since the position in the plane doesn't
        determine the tilt.
    """
    p_scan, p_tilt, p_const = plane[0:2], plane[2:4], plane[4]
    o_scan, o_tilt, o_const = output[0:2], output[2:4], output[4]
    footprint = semiconv * np.min(np.linalg.svd(p_tilt, compute_uv=False))
    if not footprint >= min_footprint_px:
        raise FittingError(
            f"Beam footprint of {footprint} px is smaller than min_footprint_px="
            f"{min_footprint_px}, the transformation is (close to) singular."
        )
    # plane = scan @ p_scan + tilt @ p_tilt + p_const
    # -> tilt = (plane - scan @ p_scan - p_const) @ inv(p_tilt)
    p_tilt_inv = np.linalg.inv(p_tilt)
    matrix = np.zeros((5, 5))
    matrix[0:2, 0:2] = o_scan - p_scan @ p_tilt_inv @ o_tilt
    matrix[0:2, 2:4] = -p_scan @ p_tilt_inv
    matrix[2:4, 0:2] = p_tilt_inv @ o_tilt
    matrix[2:4, 2:4] = p_tilt_inv
    matrix[4, 0:2] = o_const - p_const @ p_tilt_inv @ o_tilt
    matrix[4, 2:4] = -p_const @ p_tilt_inv
    matrix[4, 4] = 1
    return matrix


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

    The matrix is calculated from the :func:`ray_transfer_matrix` of
    :code:`rec_model` by eliminating the source tilt, see
    :func:`_eliminate_tilt`.

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
    transfer = ray_transfer_matrix(rec_model, specimen_to_image=specimen_to_image)
    return _eliminate_tilt(
        plane=transfer[:, 0:2],
        output=transfer[:, 2:4],
        semiconv=float(rec_model.semiconv),
        min_footprint_px=min_footprint_px,
    )


@numba.njit(inline="always", cache=True)
def affine_column(scan_y, scan_x, in_y, in_x, mat, column):
    """
    Column :code:`column` of :code:`(scan_y, scan_x, in_y, in_x, 1) @ mat`

    Calculating single columns of the matrix-vector product allows skipping
    the rest early and facilitates auto-vectorization of loops.
    """
    return (
        scan_y * mat[0, column]
        + scan_x * mat[1, column]
        + in_y * mat[2, column]
        + in_x * mat[3, column]
        + mat[4, column]
    )


@numba.njit(cache=True)
def project_frame_backwards(frame, source_semiconv, mat, scan_y, scan_x, image_out):
    limit = np.abs(np.tan(source_semiconv)) ** 2
    for image_y in range(image_out.shape[0]):
        for image_x in range(image_out.shape[1]):
            tilt_y = affine_column(scan_y, scan_x, image_y, image_x, mat, 2)
            tilt_x = affine_column(scan_y, scan_x, image_y, image_x, mat, 3)
            if np.abs(tilt_y) ** 2 + np.abs(tilt_x) ** 2 < limit:
                det_y = int(np.round(affine_column(scan_y, scan_x, image_y, image_x, mat, 0)))
                det_x = int(np.round(affine_column(scan_y, scan_x, image_y, image_x, mat, 1)))
                if det_y >= 0 and det_y < frame.shape[0] and det_x >= 0 and det_x < frame.shape[1]:
                    image_out[image_y, image_x] += frame[det_y, det_x]


def get_detector_correction_matrix(
    rec_model: Model4DSTEM,
    ref_model: Model4DSTEM | None = None,
    min_footprint_px: float = 1e-6,
):
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

    The matrix is calculated from the :func:`ray_transfer_matrix` of both
    models by eliminating the source tilt, see :func:`_eliminate_tilt`.

    Parameters
    ----------
    rec_model
        Model of the reconstruction
    ref_model
        Model of the reference system
    min_footprint_px
        Minimum smallest semi-axis of the beam's footprint in the detector
        plane of the reference model in pixels, given the semi-convergence
        angle of :code:`rec_model`. Below that, the transformation is
        considered (close to) singular, e.g. with the detector in a crossover.

    Raises
    ------
    FittingError
        If the transformation is singular or close to singular.
    """
    if ref_model is None:
        ref_model = rec_model.derive(
            scan_rotation=0.0,
            flip_factor=1.0,
            descan_error=DescanError(),
            detector_rotation=rec_model.scan_rotation,
        )
    return _eliminate_tilt(
        plane=ray_transfer_matrix(ref_model)[:, 2:4],
        output=ray_transfer_matrix(rec_model)[:, 2:4],
        semiconv=float(rec_model.semiconv),
        min_footprint_px=min_footprint_px,
    )


# Separate functions to re-use coordinate calculations for other purposes,
# such as corrected virtual detectors
@numba.njit(inline="always", cache=True)
def corrected_det_y(det_corr_y, det_corr_x, scan_y, scan_x, mat):
    return affine_column(scan_y, scan_x, det_corr_y, det_corr_x, mat, 0)


@numba.njit(inline="always", cache=True)
def corrected_det_x(det_corr_y, det_corr_x, scan_y, scan_x, mat):
    return affine_column(scan_y, scan_x, det_corr_y, det_corr_x, mat, 1)


@numba.njit(cache=True)
def correct_frame(frame, mat, scan_y, scan_x, detector_out):
    for det_corr_y in range(detector_out.shape[0]):
        for det_corr_x in range(detector_out.shape[1]):
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
