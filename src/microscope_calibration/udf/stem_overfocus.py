import numpy as np
from libertem.udf.base import UDF

from microscope_calibration.common.model import Parameters4DSTEM
from microscope_calibration.common.stem_overfocus import (
    get_backward_transformation_matrix,
    get_detector_correction_matrix,
    project_frame_backwards,
    correct_frame,
)


class OverfocusUDF(UDF):
    def __init__(
            self, overfocus_params: dict):
        super().__init__(
            overfocus_params=overfocus_params,
        )

    def _get_fov(self):
        fov_size_y = int(self.meta.dataset_shape.nav[0])
        fov_size_x = int(self.meta.dataset_shape.nav[1])
        return fov_size_y, fov_size_x

    def get_task_data(self):
        overfocus_params = self.params.overfocus_params['params']
        backprojection_matrix = get_backward_transformation_matrix(
            rec_params=overfocus_params,
        )
        correction_matrix = get_detector_correction_matrix(
            rec_params=overfocus_params,
        )
        select_roi = np.zeros(self.meta.dataset_shape.nav, dtype=bool)
        nav_y, nav_x = self.meta.dataset_shape.nav
        select_roi[nav_y//2, nav_x//2] = True
        return {
            'backprojection_matrix': backprojection_matrix,
            'correction_matrix': correction_matrix,
            'select_roi': select_roi
        }

    def get_result_buffers(self):
        fov = self._get_fov()
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            'point': self.buffer(kind='nav', dtype=dtype, where='device'),
            'shifted_sum': self.buffer(kind='single', dtype=dtype, extra_shape=fov, where='device'),
            'selected': self.buffer(
                kind='single', dtype=dtype, extra_shape=fov, where='device'
            ),
            'corrected_sum': self.buffer(kind='single', dtype=dtype, extra_shape=fov, where='device'),
        }

    def process_frame(self, frame):
        scan_y, scan_x = self.meta.coordinates[0]
        overfocus_params: Parameters4DSTEM = self.params.overfocus_params['params']
        if self.task_data.select_roi[scan_y, scan_x]:
            buf = np.zeros_like(self.results.shifted_sum)
            project_frame_backwards(
                frame=frame,
                source_semiconv=overfocus_params.semiconv,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.backprojection_matrix,
                image_out=buf
            )
            self.results.shifted_sum += buf
            self.results.selected += buf
        else:  # This saves allocation of buf and a copy
            project_frame_backwards(
                frame=frame,
                source_semiconv=overfocus_params.semiconv,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.backprojection_matrix,
                image_out=self.results.shifted_sum
            )
        center = overfocus_params.detector_center
        self.results.point[:] = frame[int(center.y), int(center.x)]

        correct_frame(
            frame=frame,
            scan_y=scan_y,
            scan_x=scan_x,
            mat=self.task_data.correction_matrix,
            detector_out=self.results.corrected_sum
        )

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum
        dest.selected += src.selected
        dest.corrected_sum += src.corrected_sum
        dest.point[:] = src.point
