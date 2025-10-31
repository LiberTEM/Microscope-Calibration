import logging
from functools import lru_cache

import numpy as np

from libertem.common.math import prod, count_nonzero
from libertem.udf.base import UDF

from microscope_calibration.common.model import Parameters4DSTEM
from microscope_calibration.common.stem_overfocus import (
    get_backward_transformation_matrix,
    get_detector_correction_matrix,
    project_frame_backwards,
    correct_frame,
    corrected_det_y,
    corrected_det_x,
    FittingError,
)


log = logging.getLogger(__name__)


class BaseCorrectionUDF(UDF):
    def __init__(
        self,
        overfocus_params: dict,
        *args,
        back_mat=None,
        corr_mat=None,
        ref_params=None,
        **kwargs,
    ):
        """
        Params are wrapped in a dict since that allows in-place updates, see
        https://github.com/LiberTEM/LiberTEM/issues/1780
        """
        # Detect changes so that the mapping matrices are recalculated
        if overfocus_params["params"] != ref_params:
            back_mat = self._back_mat(rec_params=overfocus_params["params"])
            corr_mat = self._corr_mat(rec_params=overfocus_params["params"])
        return super().__init__(
            *args,
            overfocus_params=overfocus_params,
            back_mat=back_mat,
            corr_mat=corr_mat,
            ref_params=overfocus_params["params"],
            **kwargs,
        )

    @staticmethod
    @lru_cache
    def _back_mat(*args, **kwargs):
        try:
            return get_backward_transformation_matrix(*args, **kwargs)
        except FittingError:
            return None

    @staticmethod
    @lru_cache
    def _corr_mat(*args, **kwargs):
        try:
            return get_detector_correction_matrix(*args, **kwargs)
        except FittingError:
            return None

    def _get_fov(self):
        fov_size_y = int(self.meta.dataset_shape.nav[0])
        fov_size_x = int(self.meta.dataset_shape.nav[1])
        return fov_size_y, fov_size_x

    @property
    def has_correction(self):
        return self.params.corr_mat is not None

    @property
    def has_backprojection(self):
        return self.params.back_mat is not None


class OverfocusUDF(BaseCorrectionUDF):
    def get_result_buffers(self):
        fov = self._get_fov()
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            "backprojected_sum": self.buffer(
                kind="single", dtype=dtype, extra_shape=fov, where="device"
            ),
            "corrected_point": self.buffer(kind="nav", dtype=dtype, where="device"),
            "corrected_sum": self.buffer(kind="sig", dtype=dtype, where="device"),
        }

    def process_frame(self, frame):
        scan_y, scan_x = self.meta.coordinates[0]
        overfocus_params: Parameters4DSTEM = self.params.overfocus_params["params"]
        if self.has_backprojection:
            project_frame_backwards(
                frame=frame,
                source_semiconv=overfocus_params.semiconv,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.params.back_mat,
                image_out=self.results.backprojected_sum,
            )
        center = overfocus_params.detector_center
        if self.has_correction:
            det_y = corrected_det_y(
                det_corr_y=center.y,
                det_corr_x=center.x,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.params.corr_mat,
            )
            det_x = corrected_det_x(
                det_corr_y=center.y,
                det_corr_x=center.x,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.params.corr_mat,
            )
            det_y = int(np.round(det_y))
            det_x = int(np.round(det_x))
            if (
                det_y >= 0
                and det_y < frame.shape[0]
                and det_x >= 0
                and det_x < frame.shape[1]
            ):
                self.results.corrected_point[:] = frame[det_y, det_x]

            correct_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.params.corr_mat,
                detector_out=self.results.corrected_sum,
            )

    def merge(self, dest, src):
        dest.backprojected_sum += src.backprojected_sum
        dest.corrected_sum += src.corrected_sum
        dest.corrected_point[:] = src.corrected_point


# Copied and adapted from libertem.udf.raw.PickUDF
# to allow re-using the correction basics and
# not run into inheritance complications
class CorrectedPickUDF(BaseCorrectionUDF):
    def __init__(
        self,
        overfocus_params,
        back_mat=None,
        corr_mat=None,
        ref_params=None,
    ):
        super().__init__(
            overfocus_params,
            back_mat=back_mat,
            corr_mat=corr_mat,
            ref_params=ref_params,
        )

    def get_preferred_input_dtype(self):
        ""
        return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
        ""
        dtype = self.meta.input_dtype
        sigshape = tuple(self.meta.dataset_shape.sig)
        backprojection_fov = self._get_fov()
        if self.meta.roi is not None:
            navsize = count_nonzero(self.meta.roi)
        else:
            navsize = prod(self.meta.dataset_shape.nav)
        warn_limit = 2**28
        loaded_size = prod(sigshape) * navsize * np.dtype(dtype).itemsize
        if loaded_size > warn_limit:
            log.warning(
                "CorrectedPickUDF is loading %s bytes per buffer, exceeding warning limit %s. "
                "Consider using or implementing an UDF to process data on the worker "
                "nodes instead." % (loaded_size, warn_limit)
            )
        # We are using a "single" buffer since we mostly load single frames. A
        # "sig" buffer would work as well, but would require a transpose to
        # accomodate multiple frames in the last and not first dimension.
        # A "nav" buffer would allocate a NaN-filled buffer for the whole dataset.
        return {
            "corrected": self.buffer(
                kind="single", extra_shape=(navsize,) + sigshape, dtype=dtype
            ),
            "backprojected": self.buffer(
                kind="single", extra_shape=(navsize,) + backprojection_fov, dtype=dtype
            ),
        }

    def process_frame(self, frame):
        ""
        scan_y, scan_x = self.meta.coordinates[0]
        overfocus_params: Parameters4DSTEM = self.params.overfocus_params["params"]
        # We work in flattened nav space with ROI applied
        sl = self.meta.slice.get()
        if self.has_correction:
            correct_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.params.corr_mat,
                detector_out=self.results.corrected[sl],
            )
        if self.has_backprojection:
            project_frame_backwards(
                frame=frame,
                source_semiconv=overfocus_params.semiconv,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.params.back_mat,
                image_out=self.results.backprojected[sl],
            )

    def merge(self, dest, src):
        ""
        # We receive full-size buffers from each node that
        # contributes at least one frame and rely on the rest being filled
        # with zeros correctly.
        dest.corrected[:] += src.corrected
        dest.backprojected[:] += src.backprojected

    def merge_all(self, ordered_results):
        ""
        res = {}
        for attr in ("corrected", "backprojected"):
            chunks = [getattr(b, attr) for b in ordered_results.values()]
            # We receive full-size buffers from each node that
            # contributes at least one frame and rely on the rest being filled
            # with zeros correctly.
            ssum = np.stack(chunks, axis=0).sum(axis=0)
            res[attr] = ssum
        return res
