from typing import Callable, Optional

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
import numpy as np
import numba

from microscope_calibration.common.model import (
    Parameters4DSTEM, PixelYX, CoordXY, DescanError, trace
)


# Define here to facilitate mocking in order to test
# the code that checks for float64 support in JAX as a probable cause
# for discrepancies
target_dtype = jnp.float64

CoordMappingT = Callable[[CoordXY], PixelYX]


def _do_lstsq(input_samples, output_samples):
    output_samples = np.array(output_samples)
    input_samples = np.array(input_samples)

    x, residuals, rank, s = np.linalg.lstsq(input_samples, output_samples)

    # FIXME include test also based on singular values
    assert len(residuals) == output_samples.shape[1]
    # Confirm that the solution is exact, in particular that
    # the model is linear
    assert rank == input_samples.shape[1]

    no_residuals = np.allclose(residuals, 0., rtol=1e-12, atol=1e-11)

    test_samples = np.empty_like(output_samples)
    for i in range(len(input_samples)):
        test_samples[i] = input_samples[i] @ x

    reproduced = np.allclose(
        output_samples,
        test_samples,
        rtol=1e-10,
        atol=1e-10
    )

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
            elif not reproduced:                                             # pragma: no cover
                raise RuntimeError(                                          # pragma: no cover
                    f"Discrepancies between model output {output_samples} "  # pragma: no cover
                    f"and equivalent linear transformation {test_samples} "  # pragma: no cover
                    "exceed tolerance of 1e-12."                             # pragma: no cover
                )                                                            # pragma: no cover
            else:                                                            # pragma: no cover
                raise RuntimeError(                                          # pragma: no cover
                    "If this code is reached, logic is broken."              # pragma: no cover
                    )                                                        # pragma: no cover

    return x


def get_backward_transformation_matrix(
        rec_params: Parameters4DSTEM, specimen_to_image: Optional[CoordMappingT] = None):
    '''
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

    It may be possible to derive this matrix from partial derivatives of the
    model. However, this is postponed for now since this matrix mixes input and
    output values with respect of the model, so one may have to work with a
    combination of forward and inverse derivatives.

    For the time being this method traces a number of sample rays and deduces
    the mapping matrix from these samples.
    '''

    # scan position y/x, source tilt y/x
    test_parameters = np.array((
        [0., 0., 0., 0.],
        [100., 100., 0., 0.],
        [-100., 100., 0., 0.],
        [10., 0., 0., 0.],
        [0., 10., 0., 0.],
        [0., 0., 0.1, 0.],
        [0., 0., 0., 0.1],
        [1., 1., 1., 1.],
        [1., 2., 3., 4.],
    ))

    input_samples = []
    output_samples = []

    for test_param_raw in test_parameters:
        # We are paranoid and confirm that the model is linear
        for factor in (1., 2.):
            test_param = test_param_raw * factor
            scan_pos = PixelYX(x=test_param[0], y=test_param[1])
            source_dy = test_param[2]
            source_dx = test_param[3]
            res = trace(
                params=rec_params,
                scan_pos=scan_pos,
                source_dy=source_dy,
                source_dx=source_dx
            )

            if specimen_to_image is None:
                spec_px = res['specimen'].sampling['scan_px']
            else:
                spec_px = specimen_to_image(CoordXY(
                    x=res['specimen'].ray.x,
                    y=res['specimen'].ray.y
                ))
            input_sample = (
                scan_pos.y,
                scan_pos.x,
                spec_px.y,
                spec_px.x,
                1.
            )
            output_sample = (
                res['detector'].sampling['detector_px'].y,
                res['detector'].sampling['detector_px'].x,
                source_dy,
                source_dx,
                1.,
            )
            output_samples.append(output_sample)
            input_samples.append(input_sample)

    return _do_lstsq(input_samples, output_samples)


# Separate functions spun out to facilitate re-use of coordinate calculations
# for other purposes
@numba.njit(inline='always', cache=True)
def project_tilt_y(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 2] + scan_x * mat[1, 2]
        + image_y * mat[2, 2] + image_x * mat[3, 2] + mat[4, 2]
    )


@numba.njit(inline='always', cache=True)
def project_tilt_x(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 3] + scan_x * mat[1, 3]
        + image_y * mat[2, 3] + image_x * mat[3, 3] + mat[4, 3]
    )


@numba.njit(inline='always', cache=True)
def project_det_y(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 0] + scan_x * mat[1, 0]
        + image_y * mat[2, 0] + image_x * mat[3, 0] + mat[4, 0]
    )


@numba.njit(inline='always', cache=True)
def project_det_x(image_y, image_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 1] + scan_x * mat[1, 1]
        + image_y * mat[2, 1] + image_x * mat[3, 1] + mat[4, 1]
    )


@numba.njit(cache=True)
def project_frame_backwards(frame, source_semiconv, mat, scan_y, scan_x, image_out):
    limit = np.abs(np.tan(source_semiconv))**2
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
            if np.abs(tilt_y)**2 + np.abs(tilt_x)**2 < limit:
                det_y = project_det_y(image_y, image_x, scan_y, scan_x, mat)
                det_x = project_det_x(image_y, image_x, scan_y, scan_x, mat)
                det_y = int(np.round(det_y))
                det_x = int(np.round(det_x))
                if det_y >= 0 and det_y < frame.shape[0] and det_x >= 0 and det_x < frame.shape[1]:
                    image_out[image_y, image_x] += frame[det_y, det_x]


def get_detector_correction_matrix(
        rec_params: Parameters4DSTEM, ref_params: Optional[Parameters4DSTEM] = None):
    '''
    Calculate a transformation matrix that maps from scan position in scan pixel
    coordinates and output detector pixel coordinates in a reference system to
    detector pixel coordinates in the reconstruction system and tilt of the ray
    at the source.

    If no reference parameters are specified, it derives the reference from the
    reconstruction parameters by setting scan rotation and descan error to 0,
    and flip_y to False.

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
    '''

    # scan position y/x, source tilt y/x
    test_parameters = np.array((
        [0., 0., 0., 0.],
        [100., 100., 0., 0.],
        [-100., 100., 0., 0.],
        [10., 0., 0., 0.],
        [0., 10., 0., 0.],
        [0., 0., 0.1, 0.],
        [0., 0., 0., 0.1],
        [1., 1., 1., 1.],
        [1., 2., 3., 4.],
    ))

    input_samples = []
    output_samples = []

    if ref_params is None:
        ref_params = rec_params.derive(
            scan_rotation=0.,
            flip_y=False,
            descan_error=DescanError(),
            detector_rotation=rec_params.scan_rotation,
        )

    for test_param_raw in test_parameters:
        # We are paranoid and confirm that the model is linear
        for factor in (1., 2.):
            test_param = test_param_raw * factor
            scan_pos = PixelYX(x=test_param[0], y=test_param[1])
            source_dy = test_param[2]
            source_dx = test_param[3]
            res = trace(
                params=rec_params,
                scan_pos=scan_pos,
                source_dy=source_dy,
                source_dx=source_dx
            )

            ref_res = trace(
                params=ref_params,
                scan_pos=scan_pos,
                source_dy=source_dy,
                source_dx=source_dx
            )

            input_sample = (
                scan_pos.y,
                scan_pos.x,
                ref_res['detector'].sampling['detector_px'].y,
                ref_res['detector'].sampling['detector_px'].x,
                1.
            )
            output_sample = (
                res['detector'].sampling['detector_px'].y,
                res['detector'].sampling['detector_px'].x,
                source_dy,
                source_dx,
                1.,
            )
            output_samples.append(output_sample)
            input_samples.append(input_sample)

    return _do_lstsq(input_samples, output_samples)


# Separate functions spun out to facilitate re-use of coordinate calculations
# for other purposes, such as corrected virtual detectors
@numba.njit(inline='always', cache=True)
def corrected_det_y(det_corr_y, det_corr_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 0] + scan_x * mat[1, 0]
        + det_corr_y * mat[2, 0] + det_corr_x * mat[3, 0] + mat[4, 0]
    )


@numba.njit(inline='always', cache=True)
def corrected_det_x(det_corr_y, det_corr_x, scan_y, scan_x, mat):
    return (
        scan_y * mat[0, 1] + scan_x * mat[1, 1]
        + det_corr_y * mat[2, 1] + det_corr_x * mat[3, 1] + mat[4, 1]
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
            if (det_y >= 0 and det_y < frame.shape[0]
                    and det_x >= 0 and det_x < frame.shape[1]):
                detector_out[det_corr_y, det_corr_x] += frame[det_y, det_x]
