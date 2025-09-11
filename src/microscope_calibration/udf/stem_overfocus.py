import numpy as np
from libertem.udf.base import UDF

from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM, PixelYX, CoordXY


class OverfocusUDF(UDF):
    def __init__(
            self, overfocus_params: Parameters4DSTEM):
        super().__init__(
            overfocus_params=overfocus_params,
        )

    def _get_fov(self):
        fov_size_y = int(self.meta.dataset_shape.nav[0])
        fov_size_x = int(self.meta.dataset_shape.nav[1])
        return fov_size_y, fov_size_x

    def get_task_data(self):
        overfocus_params = self.params.overfocus_params
        translation_matrix = get_translation_matrix(
            make_model(overfocus_params, self.meta.dataset_shape)
        )
        select_roi = np.zeros(self.meta.dataset_shape.nav, dtype=bool)
        nav_y, nav_x = self.meta.dataset_shape.nav
        select_roi[nav_y//2, nav_x//2] = True
        return {
            'translation_matrix': translation_matrix,
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
            'sum': self.buffer(kind='single', dtype=dtype, extra_shape=fov, where='device'),
        }

    def process_frame(self, frame):
        scan_y, scan_x = self.meta.coordinates[0]
        center_y = self.meta.dataset_shape.nav[0] // 2
        center_x = self.meta.dataset_shape.nav[1] // 2
        overfocus_params = self.params.overfocus_params
        if self.task_data.select_roi[scan_y, scan_x]:
            buf = np.zeros_like(self.results.shifted_sum)
            project_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                translation_matrix=self.task_data.translation_matrix,
                result_out=buf
            )
            self.results.shifted_sum += buf
            self.results.selected += buf
        else:  # This saves allocation of buf and a copy
            project_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                translation_matrix=self.task_data.translation_matrix,
                result_out=self.results.shifted_sum
            )

        self.results.point[:] = frame[int(overfocus_params['cy']), int(overfocus_params['cx'])]

        project_frame(
            frame=frame,
            scan_y=center_y,
            scan_x=center_x,
            translation_matrix=self.task_data.translation_matrix,
            result_out=self.results.sum
        )

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum
        dest.selected += src.selected
        dest.sum += src.sum
        dest.point[:] = src.point
