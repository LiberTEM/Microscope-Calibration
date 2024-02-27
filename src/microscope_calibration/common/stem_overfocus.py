from typing import TypedDict, TYPE_CHECKING

import numpy as np
from temgymbasic.model import STEMModel
import numba

if TYPE_CHECKING:
    from libertem.common import Shape


class OverfocusParams(TypedDict):
    overfocus: float  # m
    scan_pixel_size: float  # m
    camera_length: float  # m
    detector_pixel_size: float  # m
    semiconv: float  # rad
    cy: float
    cx: float
    scan_rotation: float  # deg
    flip_y: bool


def make_model(params: OverfocusParams, dataset_shape: 'Shape') -> STEMModel:
    model = STEMModel()
    model.set_stem_params(
        overfocus=params['overfocus'],
        semiconv_angle=params['semiconv'],
        scan_step_yx=(params['scan_pixel_size'], params['scan_pixel_size']),
        scan_shape=dataset_shape.nav.to_tuple(),
        camera_length=params['camera_length'],
    )
    model.detector.pixel_size = params['detector_pixel_size']
    model.detector.shape = dataset_shape.sig.to_tuple()
    model.detector.flip_y = params['flip_y']
    model.detector.rotation = params['scan_rotation']
    model.detector.set_center_px((params['cy'], params['cx']))
    return model


def get_translation_matrix(model: STEMModel) -> np.ndarray:
    yxs = (
        (0, 0),
        (model.sample.scan_shape[0], model.sample.scan_shape[1]),
        (0, model.sample.scan_shape[1]),
        (model.sample.scan_shape[0], 0),
    )
    num_rays = 7

    a = []
    b = []

    for yx in yxs:
        for rays in model.scan_point_iter(num_rays=num_rays, yx=yx):
            if rays.location is model.sample:
                yyxx = np.stack(
                    model.sample.on_grid(rays, as_int=False),
                    axis=-1,
                )
                coordinates = np.tile(
                    np.asarray((*yx, 1)).reshape(-1, 3),
                    (rays.num, 1),
                )
                a.append(np.concatenate((yyxx, coordinates), axis=-1))
            elif rays.location is model.detector:
                yy, xx = model.detector.on_grid(rays, as_int=False)
                b.append(np.stack((yy, xx), axis=-1))

    res, *_ = np.linalg.lstsq(
        np.concatenate(a, axis=0),
        np.concatenate(b, axis=0),
        rcond=None,
    )
    return res


@numba.njit(cache=True, fastmath=True)
def project_frame(frame, scan_y, scan_x, translation_matrix, result_out):
    for t_y in range(result_out.shape[0]):
        for t_x in range(result_out.shape[1]):
            s_y = t_y * translation_matrix[0, 0]
            s_x = t_y * translation_matrix[0, 1]

            s_y += t_x * translation_matrix[1, 0]
            s_x += t_x * translation_matrix[1, 1]

            s_y += scan_y * translation_matrix[2, 0]
            s_x += scan_y * translation_matrix[2, 1]

            s_y += scan_x * translation_matrix[3, 0]
            s_x += scan_x * translation_matrix[3, 1]

            s_y += translation_matrix[4, 0]
            s_x += translation_matrix[4, 1]

            ss_y = int(np.round(s_y))
            ss_x = int(np.round(s_x))
            if ss_y >= 0 and ss_x >= 0 and ss_y < frame.shape[0] and ss_x < frame.shape[1]:
                result_out[t_y, t_x] += frame[ss_y, ss_x]
