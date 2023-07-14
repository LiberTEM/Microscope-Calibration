from typing import TypedDict

import numpy as np
from temgymbasic import components as comp
from temgymbasic.model import Model
from temgymbasic.functions import get_pixel_coords
import numba


class OverfocusParams(TypedDict):
    overfocus: float  # m
    scan_pixel_size: float  # m
    camera_length: float  # m
    detector_pixel_size: float  # m
    semiconv: float  # rad
    cy: float
    cx: float
    scan_rotation: float
    flip_y: bool


def make_model(params: OverfocusParams, dataset_shape):
    # We have to make it square
    sample = np.ones((dataset_shape[0], dataset_shape[0]))
    # Create a list of components to model a simplified 4DSTEM experiment
    components = [
        comp.DoubleDeflector(name='Scan Coils', z_up=0.3, z_low=0.25),
        comp.Lens(name='Lens', z=0.20),
        comp.Sample(
            name='Sample',
            sample=sample,
            z=params['camera_length'],
            width=sample.shape[0] * params['scan_pixel_size']
        ),
        comp.DoubleDeflector(
            name='Descan Coils',
            z_up=0.1,
            z_low=0.05,
            scan_rotation=0.
        )
    ]

    # Create the model Electron microscope. Initially we create a parallel
    # circular beam leaving the "gun"
    model = Model(
        components,
        beam_z=0.4,
        beam_type='paralell',
        num_rays=7,  # somehow the minimum
        experiment='4DSTEM',
        detector_pixels=dataset_shape[2],
        detector_size=dataset_shape[2] * params['detector_pixel_size']
    )

    model.scan_pixel_size = params['scan_pixel_size']
    model.set_obj_lens_f_from_overfocus(params['overfocus'])
    model.scan_pixels = dataset_shape[0]
    return model


def get_translation_matrix(params: OverfocusParams, model):
    a = []
    b = []
    model.scan_pixel_x = 0
    model.scan_pixel_y = 0
    for scan_y in (0, model.scan_pixels - 1):
        for scan_x in (0, model.scan_pixels - 1):
            model.scan_pixel_y = scan_y
            model.scan_pixel_x = scan_x
            model.update_scan_coil_ratio()
            model.step()
            sample_rays_x = model.r[model.sample_r_idx, 0, :]
            sample_rays_y = model.r[model.sample_r_idx, 2, :]
            detector_rays_x = model.r[-1, 0, :]
            detector_rays_y = model.r[-1, 2, :]
            sample_coords_x, sample_coords_y = get_pixel_coords(
                rays_x=sample_rays_x,
                rays_y=sample_rays_y,
                size=model.components[model.sample_idx].sample_size,
                pixels=model.components[model.sample_idx].sample_pixels,
            )
            detector_coords_x, detector_coords_y = get_pixel_coords(
                rays_x=detector_rays_x,
                rays_y=detector_rays_y,
                size=model.detector_size,
                pixels=model.detector_pixels,
                flip_y=params['flip_y'],
                scan_rotation=params['scan_rotation'],
            )
            for i in range(len(sample_coords_x)):
                a.append((
                    sample_coords_y[i],
                    sample_coords_x[i],
                    model.scan_pixels-model.scan_pixel_y,
                    model.scan_pixels-model.scan_pixel_x,
                    1
                ))
                b.append((detector_coords_y[i], detector_coords_x[i]))
    res = np.linalg.lstsq(a, b, rcond=None)
    return res[0]


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
            if s_y >= 0 and s_x >= 0 and s_y < frame.shape[0] and s_x < frame.shape[1]:
                result_out[int(t_y), int(t_x)] += frame[int(s_y), int(s_x)]
