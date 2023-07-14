from typing import TypedDict

import numpy as np
from libertem.udf.base import UDF
from libertem.masks import circular

from microscope_calibration.common.stem_overfocus import (
    get_translation_matrix, OverfocusParams, make_model, project_frame
)


class OverfocusProcessParams(TypedDict):
    pair_distance: float
    pair_radius: float


class OverfocusUDF(UDF):
    def __init__(
            self, overfocus_params: OverfocusParams,
            process_params: OverfocusProcessParams,
            point_y=None, point_x=None):
        super().__init__(
            overfocus_params=overfocus_params,
            process_params=process_params,
            point_y=point_y, point_x=point_x
        )

    def _get_fov(self):
        fov_size_y = int(self.meta.dataset_shape.nav[0])
        fov_size_x = int(self.meta.dataset_shape.nav[1])
        return fov_size_y, fov_size_x

    def get_task_data(self):
        overfocus_params = self.params.overfocus_params
        translation_matrix = get_translation_matrix(
            params=overfocus_params,
            model=make_model(overfocus_params, self.meta.dataset_shape)
        )
        pair_roi = np.zeros(self.meta.dataset_shape.nav)
        pair_roi_centeroffset = self.params.process_params['pair_distance'] / np.sqrt(8)
        nav_y, nav_x = self.meta.dataset_shape.nav
        pair_roi += circular(
            centerX=nav_x/2 - pair_roi_centeroffset,
            centerY=nav_y/2 - pair_roi_centeroffset,
            imageSizeX=pair_roi.shape[1],
            imageSizeY=pair_roi.shape[0],
            radius=self.params.process_params['pair_radius'],
        )
        if self.params.process_params['pair_distance'] >= 1:
            pair_roi -= circular(
                centerX=nav_x/2 + pair_roi_centeroffset,
                centerY=nav_y/2 + pair_roi_centeroffset,
                imageSizeX=pair_roi.shape[1],
                imageSizeY=pair_roi.shape[0],
                radius=self.params.process_params['pair_radius'],
            )
        return {
            'translation_matrix': translation_matrix,
            'pair_roi': pair_roi
        }

    def get_result_buffers(self):
        fov = self._get_fov()
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            'point': self.buffer(kind='nav', dtype=dtype, where='device'),
            'shifted_sum': self.buffer(kind='single', dtype=dtype, extra_shape=fov, where='device'),
            'shifted_pair': self.buffer(
                kind='single', dtype=dtype, extra_shape=fov, where='device'
            ),
            'sum': self.buffer(kind='single', dtype=dtype, extra_shape=fov, where='device'),
        }

    def process_frame(self, frame):
        scan_y, scan_x = self.meta.coordinates[0]
        center_y = self.meta.dataset_shape.nav[0] // 2
        center_x = self.meta.dataset_shape.nav[1] // 2
        if self.task_data.pair_roi[scan_y, scan_x] != 0:
            buf = np.zeros_like(self.results.shifted_sum)
            project_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                translation_matrix=self.task_data.translation_matrix,
                result_out=buf
            )
            self.results.shifted_sum += buf
            self.results.shifted_pair += self.task_data.pair_roi[scan_y, scan_x] * buf
        else:  # This saves allocation of buf and a copy
            project_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                translation_matrix=self.task_data.translation_matrix,
                result_out=self.results.shifted_sum
            )
        point_y = self.params.point_y
        if point_y is None:
            point_y = frame.shape[0]//2
        point_x = self.params.point_x
        if point_x is None:
            point_x = frame.shape[1]//2

        self.results.point[:] = frame[point_y, point_x]

        project_frame(
            frame=frame,
            scan_y=center_y,
            scan_x=center_x,
            translation_matrix=self.task_data.translation_matrix,
            result_out=self.results.sum
        )

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum
        dest.shifted_pair += src.shifted_pair
        dest.sum += src.sum
        dest.point[:] = src.point
