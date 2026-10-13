import jax

jax.config.update("jax_enable_x64", True)  # noqa fmt: skip
# ruff: disable[E402]

import numba
import numpy as np

from microscope_calibration.common.model import Model4DSTEM
from microscope_calibration.common.stem_overfocus import (
    CoordMappingT,
    _eliminate_tilt,
    affine_column,
    ray_transfer_matrix,
)
# ruff: enable[E402]


def smiley(size):
    """
    Smiley face test object from https://doi.org/10.1093/micmic/ozad021
    """
    obj = np.ones((size, size), dtype=np.complex64)
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]

    outline = (((y * 1.2) ** 2 + x**2) > (110 / 256 * size) ** 2) & (
        ((y * 1.2) ** 2 + x**2) < (120 / 256 * size) ** 2
    )
    obj[outline] = 0.0

    left_eye = ((y + 40 / 256 * size) ** 2 + (x + 40 / 256 * size) ** 2) < (20 / 256 * size) ** 2
    obj[left_eye] = 0
    right_eye = (np.abs(y + 40 / 256 * size) < 15 / 256 * size) & (
        np.abs(x - 40 / 256 * size) < 30 / 256 * size
    )
    obj[right_eye] = 0

    nose = (y + 20 / 256 * size + x > 0) & (x < 0) & (y < 10 / 256 * size)

    obj[nose] = (0.05j * x + 0.05j * y)[nose]

    mouth = (
        (((y * 1) ** 2 + x**2) > (50 / 256 * size) ** 2)
        & (((y * 1) ** 2 + x**2) < (70 / 256 * size) ** 2)
        & (y > 20 / 256 * size)
    )

    obj[mouth] = 0

    tongue = (
        ((y - 50 / 256 * size) ** 2 + (x - 50 / 256 * size) ** 2) < (20 / 256 * size) ** 2
    ) & ((y**2 + x**2) > (70 / 256 * size) ** 2)
    obj[tongue] = 0

    # This wave modulation introduces a strong signature in the diffraction pattern
    # that allows to confirm the correct scale and orientation.
    signature_wave = np.exp(1j * (3 * y + 7 * x) * 2 * np.pi / size)

    obj += 0.3 * signature_wave - 0.3

    obj = np.abs(obj)
    return obj


def get_forward_transformation_matrix(
    sim_model: Model4DSTEM,
    specimen_to_image: CoordMappingT | None = None,
    min_footprint_px: float = 1e-6,
):
    """
    Calculate a transformation matrix that maps from scan position in scan pixel
    coordinates and detector pixel coordinates to specimen coordinates in scan
    pixel coordinates and tilt of the ray at the source.

    Using a matrix multiplication instead of solving for ray solutions for each
    pixel greatly improves performance.

    The input values for that matrix correspond to the pixel indices in a 4D
    STEM dataset.

    The specimen position from the output can be used to pick the right value
    from an object. The tilt can be used to determine if the beam passes through
    through the microscope or if it is blocked by the beam-shaping aperture.

    The matrix is calculated from the
    :func:`~microscope_calibration.common.stem_overfocus.ray_transfer_matrix`
    of :code:`sim_model` by eliminating the source tilt.

    Parameters
    ----------
    sim_model
        Model of the simulation
    specimen_to_image
        Affine mapping from physical specimen coordinates to specimen pixel
        coordinates. It is evaluated with SymPy symbols to determine its
        coefficients. By default, scan pixel coordinates of :code:`sim_model`.
    min_footprint_px
        Minimum smallest semi-axis of the beam's footprint in the detector
        plane in pixels, given the semi-convergence angle. Below that, the
        transformation is considered (close to) singular, e.g. with the
        detector in a crossover.

    Raises
    ------
    FittingError
        If the transformation is singular or close to singular.
    """
    transfer = ray_transfer_matrix(sim_model, specimen_to_image=specimen_to_image)
    return _eliminate_tilt(
        plane=transfer[:, 2:4],
        output=transfer[:, 0:2],
        semiconv=float(sim_model.semiconv),
        min_footprint_px=min_footprint_px,
    )


@numba.njit(cache=True)
def project_frame_forward(obj, source_semiconv, mat, scan_y, scan_x, out):
    limit = np.abs(np.tan(source_semiconv)) ** 2
    for det_y in range(out.shape[0]):
        for det_x in range(out.shape[1]):
            tilt_y = affine_column(scan_y, scan_x, det_y, det_x, mat, 2)
            tilt_x = affine_column(scan_y, scan_x, det_y, det_x, mat, 3)
            if np.abs(tilt_y) ** 2 + np.abs(tilt_x) ** 2 < limit:
                spec_y = affine_column(scan_y, scan_x, det_y, det_x, mat, 0)
                spec_x = affine_column(scan_y, scan_x, det_y, det_x, mat, 1)
                spec_y = int(np.round(spec_y))
                spec_x = int(np.round(spec_x))
                if spec_y >= 0 and spec_y < obj.shape[0] and spec_x >= 0 and spec_x < obj.shape[1]:
                    out[det_y, det_x] = obj[spec_y, spec_x]
            else:
                out[det_y, det_x] = 0.0


def project(
    image,
    scan_shape,
    detector_shape,
    sim_model: Model4DSTEM,
    specimen_to_image: CoordMappingT | None = None,
):
    result = np.zeros(tuple(scan_shape) + tuple(detector_shape), dtype=image.dtype)
    mat = get_forward_transformation_matrix(
        sim_model=sim_model, specimen_to_image=specimen_to_image
    )
    for scan_y in range(result.shape[0]):
        for scan_x in range(result.shape[1]):
            project_frame_forward(
                obj=image,
                source_semiconv=sim_model.semiconv,
                mat=mat,
                scan_y=scan_y,
                scan_x=scan_x,
                out=result[scan_y, scan_x],
            )
    return result
