'''
Independent reference implementation of the ray tracing from detector to object
to allow simulating a dataset.

This can then be used to test the UDF that performs the inverse projection.
'''

import numpy as np
import numba

from libertem.analysis import com as com_analysis

from microscope_calibration.common.model import Parameters4DSTEM


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


def get_transformation_matrix(sim_params: Parameters4DSTEM):
    '''
    Calculate a transformation matrix for :func:`detector_px_to_specimen_px`
    from the provided parameters.

    Internally this uses :func:`libertem.analysis.com.apply_correction` to
    transform unit vectors in order to determine the matrix.
    '''
    transformation_matrix = np.array(com_analysis.apply_correction(
        y_centers=np.array((1, 0)),
        x_centers=np.array((0, 1)),
        scan_rotation=sim_params['scan_rotation'],
        flip_y=sim_params['flip_y'],
    ))
    return transformation_matrix


@numba.njit(inline='always', cache=True)
def detector_px_to_specimen_px(
        y_px, x_px, cy, cx, detector_pixel_size, scan_pixel_size, camera_length,
        overfocus, transformation_matrix, fov_size_y, fov_size_x):
    '''
    Model Figure 2 of https://arxiv.org/abs/2403.08538

    The pixel coordinates refer to the center of a pixel: In :func:`_project`
    they are rounded to the nearest integer. This function is just transforming
    coordinates independent of actual scan and detector sizes. Rounding and
    bounds checks are performed in :func:`_project`.

    The specimen pixel coordinates calculated by this function are combined with
    the scan coordinates in :func:`_project` to model the scan.

    Parameters
    ----------

    y_px, x_px : float
        Detector pixel coordinates to project. They are relative to (cy, cx),
        where specifying pixel coordinates (0.0, 0.0) maps to physical
        coordinates (-cy * detector_pixel_size, -cx *detector_pixel_size), and
        pixel coordinates (cy, cx) map to physical coordinates (0.0. 0.0).
    cy, cx : float
        Detector center in detector pixel coordinates. This defines the position
        of the "straight through" beam on the detector.
    detector_pixel_size, scan_pixel_size : float
        Physical pixel sizes in m. This assumes a uniform scan and detector grid
        in x and y direction
    camera_length : float
        Virtual distance from specimen to detector in m
    overfocus : float
        Virtual distance from focus point to specimen in m. Underfocus is
        specified as a negative overfocus.
    transformation_matrix : np.ndarray[float]
        2x2 transformation matrix for detector coordinates. It acts around (cy,
        cx). This is used to specify rotation and handedness change consistent
        with other methods in LiberTEM. It can be calculated with
        :fun:`get_transformation_matrix`.
    fov_size_y, fov_size_x : float
        Size of the scan area (field of view) in scan pixels. The scan
        coordinate system is centered in the middle of this field of view,
        meaning that the "straight through" beam (y_px, x_px) == (cy, cx) is
        mapped to (fov_size_y/2, fov_size_x/2). Please note that the actual scan
        coordinates are not calculated in this function, but added as an offset
        in :func:`_project`. The field of view specified here is just used to calculate
        the center of the "straight through" beam in the middle of the scan.

    Returns
    -------
    specimen_px_y, specimen_px_x : float
        Beam position on the specimen in scan pixel coordinates.
    '''
    position_y, position_x = (y_px - cy) * detector_pixel_size, (x_px - cx) * detector_pixel_size
    position_y, position_x = transformation_matrix @ np.array((position_y, position_x))
    specimen_position_y = position_y / (overfocus + camera_length) * overfocus
    specimen_position_x = position_x / (overfocus + camera_length) * overfocus
    specimen_px_x = specimen_position_x / scan_pixel_size + fov_size_x / 2
    specimen_px_y = specimen_position_y / scan_pixel_size + fov_size_y / 2
    return specimen_px_y, specimen_px_x


@numba.njit(cache=True)
def _project(
        image, cy, cx, detector_pixel_size, scan_pixel_size, camera_length,
        overfocus, transformation_matrix, result_out):
    scan_shape = result_out.shape[:2]
    for det_y in range(result_out.shape[2]):
        for det_x in range(result_out.shape[3]):
            specimen_px_y, specimen_px_x = detector_px_to_specimen_px(
                y_px=det_y,
                x_px=det_x,
                cy=cy,
                cx=cx,
                detector_pixel_size=detector_pixel_size,
                scan_pixel_size=scan_pixel_size,
                camera_length=camera_length,
                overfocus=overfocus,
                transformation_matrix=transformation_matrix,
                fov_size_y=image.shape[0],
                fov_size_x=image.shape[1],
            )
            for scan_y in range(scan_shape[0]):
                for scan_x in range(scan_shape[1]):
                    offset_y = scan_y - scan_shape[0] // 2
                    offset_x = scan_x - scan_shape[1] // 2
                    image_px_y = int(np.round(specimen_px_y + offset_y))
                    image_px_x = int(np.round(specimen_px_x + offset_x))
                    if image_px_y < 0 or image_px_y >= image.shape[0]:
                        continue
                    if image_px_x < 0 or image_px_x >= image.shape[1]:
                        continue
                    result_out[scan_y, scan_x, det_y, det_x] = image[image_px_y, image_px_x]


def project(image, scan_shape, detector_shape, sim_params: Parameters4DSTEM):
    result = np.zeros(tuple(scan_shape) + tuple(detector_shape), dtype=image.dtype)
    _project(
        image=image,
        cy=sim_params['cy'],
        cx=sim_params['cx'],
        detector_pixel_size=sim_params['detector_pixel_size'],
        scan_pixel_size=sim_params['scan_pixel_size'],
        camera_length=sim_params['camera_length'],
        overfocus=sim_params['overfocus'],
        transformation_matrix=get_transformation_matrix(sim_params),
        result_out=result
    )
    return result
