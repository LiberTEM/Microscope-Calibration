'''
Independent reference implementation of the ray tracing from detector to object
to allow simulating a dataset.

This can then be used to test the UDF that performs the inverse projection.
'''

import numpy as np
import numba

from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM, PixelYX


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


def get_forward_transformation_matrix(sim_params: Parameters4DSTEM):
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
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
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
            input_sample = (
                scan_pos.y,
                scan_pos.x,
                res['detector'].sampling['detector_px'].y,
                res['detector'].sampling['detector_px'].x,
                1.
            )
            output_sample = (
                res['specimen'].sampling['scan_px'].y,
                res['specimen'].sampling['scan_px'].x,
                source_dy,
                source_dx,
                1.,
            )
            output_samples.append(output_sample)
            input_samples.append(input_sample)

    output_samples = np.array(output_samples)
    input_samples = np.array(input_samples)

    x, residuals, rank, s = np.linalg.lstsq(np.array(input_samples), np.array(output_samples))

    # FIXME include test also based on singular values
    # FIXME confirm correct operation with realistic TEM parameters
    assert len(residuals) == rank
    # Confirm that the solution is exact, in particular that
    # the model is linear
    assert np.allclose(residuals, 0.)
    assert rank == 5

    for i in range(len(input_samples)):
        assert np.allclose(
            output_samples[i],
            input_samples[i] @ x,
            rtol=1e-6,
            atol=1e-6
        )
    return x


@numba.njit
def project_frame_forward(obj, source_semiconv, mat, scan_y, scan_x, out):
    limit = np.abs(source_semiconv)**2
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


def project(image, scan_shape, detector_shape, sim_params: Parameters4DSTEM):
    result = np.zeros(tuple(scan_shape) + tuple(detector_shape), dtype=image.dtype)
    mat = get_forward_transformation_matrix(sim_params=sim_params)
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
