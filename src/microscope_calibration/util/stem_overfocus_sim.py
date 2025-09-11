from typing import Callable, Optional

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
import numpy as np
import numba

from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM, PixelYX, CoordXY


# Define here to facilitate mocking in order to test
# the code that checks for float64 support in JAX as a probable cause
# for discrepancies
target_dtype = jnp.float64


def smiley(size):
    '''
    Smiley face test object from https://doi.org/10.1093/micmic/ozad021
    '''
    obj = np.ones((size, size), dtype=np.complex64)
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]

    outline = (((y*1.2)**2 + x**2) > (110/256*size)**2) & \
              (((y*1.2)**2 + x**2) < (120/256*size)**2)
    obj[outline] = 0.0

    left_eye = ((y + 40/256*size)**2 + (x + 40/256*size)**2) < (20/256*size)**2
    obj[left_eye] = 0
    right_eye = (np.abs(y + 40/256*size) < 15/256*size) & \
                (np.abs(x - 40/256*size) < 30/256*size)
    obj[right_eye] = 0

    nose = (y + 20/256*size + x > 0) & (x < 0) & (y < 10/256*size)

    obj[nose] = (0.05j * x + 0.05j * y)[nose]

    mouth = (((y*1)**2 + x**2) > (50/256*size)**2) & \
            (((y*1)**2 + x**2) < (70/256*size)**2) & \
            (y > 20/256*size)

    obj[mouth] = 0

    tongue = (((y - 50/256*size)**2 + (x - 50/256*size)**2) < (20/256*size)**2) & \
             ((y**2 + x**2) > (70/256*size)**2)
    obj[tongue] = 0

    # This wave modulation introduces a strong signature in the diffraction pattern
    # that allows to confirm the correct scale and orientation.
    signature_wave = np.exp(1j*(3 * y + 7 * x) * 2*np.pi/size)

    obj += 0.3*signature_wave - 0.3

    obj = np.abs(obj)
    return obj


CoordMappingT = Callable[[CoordXY], PixelYX]


def get_forward_transformation_matrix(
        sim_params: Parameters4DSTEM, specimen_to_image: Optional[CoordMappingT] = None):
    '''
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

    It may be possible to derive this matrix from partial derivatives of the
    model. However, this is postponed for now since this matrix mixes input and
    output values with respect of the model, so one may have to work with a
    combination of forward and inverse derivatives.

    For the time being this method traces a number of sample rays and deduces the
    mapping matrix from these samples.
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
            model = Model4DSTEM.build(params=sim_params, scan_pos=scan_pos)
            ray = model.make_source_ray(source_dy=source_dy, source_dx=source_dx).ray
            res = model.trace(ray)
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
                res['detector'].sampling['detector_px'].y,
                res['detector'].sampling['detector_px'].x,
                1.
            )
            output_sample = (
                spec_px.y,
                spec_px.x,
                source_dy,
                source_dx,
                1.,
            )
            output_samples.append(output_sample)
            input_samples.append(input_sample)

    output_samples = np.array(output_samples)
    input_samples = np.array(input_samples)

    x, residuals, rank, s = np.linalg.lstsq(input_samples, output_samples)

    # FIXME include test also based on singular values
    # FIXME confirm correct operation with realistic TEM parameters
    assert len(residuals) == rank
    # Confirm that the solution is exact, in particular that
    # the model is linear
    assert rank == 5

    no_residuals = np.allclose(residuals, 0., rtol=1e-12, atol=1e-12)

    test_samples = np.empty_like(output_samples)
    for i in range(len(input_samples)):
        test_samples[i] = input_samples[i] @ x

    reproduced = np.allclose(
        output_samples,
        test_samples,
        rtol=1e-12,
        atol=1e-12
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


@numba.njit
def project_frame_forward(obj, source_semiconv, mat, scan_y, scan_x, out):
    limit = np.abs(np.tan(source_semiconv))**2
    for det_y in range(out.shape[0]):
        for det_x in range(out.shape[1]):
            # Manually unrolled matrix-vector product to allow skipping before
            # calculating all values and facilitate auto-vectorization of the
            # loop

            # _one = (
            #       scan_y * mat[0, 4] + scan_x * mat[1, 4]
            #       + det_y * mat[2, 4] + det_x * mat[3, 4] + mat[4, 4]
            # )
            # assert np.allclose(_one, 1)
            tilt_y = (
                scan_y * mat[0, 2] + scan_x * mat[1, 2]
                + det_y * mat[2, 2] + det_x * mat[3, 2] + mat[4, 2]
            )
            tilt_x = (
                scan_y * mat[0, 3] + scan_x * mat[1, 3]
                + det_y * mat[2, 3] + det_x * mat[3, 3] + mat[4, 3]
            )
            if np.abs(tilt_y)**2 + np.abs(tilt_x)**2 < limit:
                spec_y = (
                    scan_y * mat[0, 0] + scan_x * mat[1, 0]
                    + det_y * mat[2, 0] + det_x * mat[3, 0] + mat[4, 0]
                )
                spec_x = (
                    scan_y * mat[0, 1] + scan_x * mat[1, 1]
                    + det_y * mat[2, 1] + det_x * mat[3, 1] + mat[4, 1]
                )
                spec_y = int(np.round(spec_y))
                spec_x = int(np.round(spec_x))
                if spec_y >= 0 and spec_y < obj.shape[0] and spec_x >= 0 and spec_x < obj.shape[1]:
                    out[det_y, det_x] = obj[spec_y, spec_x]
            else:
                out[det_y, det_x] = 0.


def project(
        image, scan_shape, detector_shape,
        sim_params: Parameters4DSTEM,
        specimen_to_image: Optional[CoordMappingT] = None):
    result = np.zeros(tuple(scan_shape) + tuple(detector_shape), dtype=image.dtype)
    mat = get_forward_transformation_matrix(
        sim_params=sim_params, specimen_to_image=specimen_to_image
    )
    model = Model4DSTEM.build(params=sim_params, scan_pos=PixelYX(x=0., y=0.))
    for scan_y in range(result.shape[0]):
        for scan_x in range(result.shape[1]):
            project_frame_forward(
                obj=image,
                source_semiconv=model.source.semi_conv,
                mat=mat,
                scan_y=scan_y,
                scan_x=scan_x,
                out=result[scan_y, scan_x]
            )
    return result
