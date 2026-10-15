import logging
from functools import lru_cache

import numpy as np
from libertem.common.math import count_nonzero, prod
from libertem.udf.base import UDF

from microscope_calibration.common.model import Model4DSTEM
from microscope_calibration.common.stem_overfocus import (
    FittingError,
    correct_frame,
    corrected_det_x,
    corrected_det_y,
    get_backward_transformation_matrix,
    get_detector_correction_matrix,
    project_frame_backwards,
)

log = logging.getLogger(__name__)


@lru_cache
def _backward_matrix(model: Model4DSTEM) -> np.ndarray | None:
    try:
        return _read_only(get_backward_transformation_matrix(rec_model=model))
    except FittingError as e:
        log.warning("Skipping back-projection: %s", e)
        return None


@lru_cache
def _correction_matrix(model: Model4DSTEM) -> np.ndarray | None:
    try:
        return _read_only(get_detector_correction_matrix(rec_model=model))
    except FittingError as e:
        log.warning("Skipping detector correction: %s", e)
        return None


def _read_only(mat: np.ndarray) -> np.ndarray:
    # Cached and shared between calls
    mat.setflags(write=False)
    return mat


class BaseCorrectionUDF(UDF):
    """
    Base class for UDFs that use the transformation matrices of a model

    Parameters
    ----------
    overfocus_model
        :code:`{"model": model}` with a :class:`Model4DSTEM`. The model is
        wrapped in a dict so that it can be updated in place, see
        https://github.com/LiberTEM/LiberTEM/issues/1780. The matrices are
        calculated from the current model for each task.

    If a matrix can't be calculated, e.g. the back-projection in focus, the
    UDF logs a warning and skips the corresponding results.
    """

    def __init__(self, overfocus_model: dict, **kwargs):
        super().__init__(overfocus_model=overfocus_model, **kwargs)

    @property
    def _model(self) -> Model4DSTEM:
        # LiberTEM doesn't type the parameters of UDFs
        return self.params.overfocus_model["model"]  # pyright: ignore[reportIndexIssue, reportReturnType]

    def get_task_data(self):
        model = self._model.normalize_types()
        return {
            "back_mat": _backward_matrix(model),  # pyright: ignore[reportArgumentType]  # jax_dataclasses models are hashable
            "corr_mat": _correction_matrix(model),  # pyright: ignore[reportArgumentType]  # jax_dataclasses models are hashable
        }

    def _get_fov(self) -> tuple[int, int]:
        nav_shape = tuple(self.meta.dataset_shape.nav)
        if len(nav_shape) != 2:
            raise ValueError(f"Only 2D scans are supported, got navigation shape {nav_shape}.")
        return nav_shape

    @property
    def has_correction(self):
        return self.task_data.corr_mat is not None

    @property
    def has_backprojection(self):
        return self.task_data.back_mat is not None


class OverfocusUDF(BaseCorrectionUDF):
    def get_result_buffers(self):
        fov = self._get_fov()
        dtype = np.result_type(self.meta.input_dtype, np.float32)
        return {
            "backprojected_sum": self.buffer(kind="single", dtype=dtype, extra_shape=fov),
            "corrected_point": self.buffer(kind="nav", dtype=dtype),
            "corrected_sum": self.buffer(kind="sig", dtype=dtype),
        }

    def process_frame(self, frame):
        scan_y, scan_x = self.meta.coordinates[0]
        overfocus_model = self._model
        if self.has_backprojection:
            project_frame_backwards(
                frame=frame,
                source_semiconv=overfocus_model.semiconv,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.back_mat,
                image_out=self.results.backprojected_sum,
            )
        center = overfocus_model.detector_center
        if self.has_correction:
            det_y = corrected_det_y(
                det_corr_y=center.y,
                det_corr_x=center.x,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.corr_mat,
            )
            det_x = corrected_det_x(
                det_corr_y=center.y,
                det_corr_x=center.x,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.corr_mat,
            )
            det_y = int(np.round(det_y))
            det_x = int(np.round(det_x))
            if det_y >= 0 and det_y < frame.shape[0] and det_x >= 0 and det_x < frame.shape[1]:
                self.results.corrected_point[:] = frame[det_y, det_x]  # pyright: ignore[reportIndexIssue]  # LiberTEM doesn't type UDF buffers

            correct_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.corr_mat,
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
    def get_preferred_input_dtype(self):
        return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
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
                "nodes instead.",
                loaded_size,
                warn_limit,
            )
        # We are using a "single" buffer since we mostly load single frames. A
        # "sig" buffer would work as well, but would require a transpose to
        # accommodate multiple frames in the last and not first dimension.
        # A "nav" buffer would allocate a NaN-filled buffer for the whole dataset.
        return {
            "corrected": self.buffer(kind="single", extra_shape=(navsize, *sigshape), dtype=dtype),
            "backprojected": self.buffer(
                kind="single", extra_shape=(navsize, *backprojection_fov), dtype=dtype
            ),
        }

    def process_frame(self, frame):
        scan_y, scan_x = self.meta.coordinates[0]
        overfocus_model = self._model
        # We work in flattened nav space with ROI applied
        sl = self.meta.slice.get()  # pyright: ignore[reportOptionalMemberAccess]  # LiberTEM types the slice as optional
        if self.has_correction:
            correct_frame(
                frame=frame,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.corr_mat,
                detector_out=self.results.corrected[sl][0],  # pyright: ignore[reportIndexIssue]  # LiberTEM doesn't type UDF buffers
            )
        if self.has_backprojection:
            project_frame_backwards(
                frame=frame,
                source_semiconv=overfocus_model.semiconv,
                scan_y=scan_y,
                scan_x=scan_x,
                mat=self.task_data.back_mat,
                image_out=self.results.backprojected[sl][0],  # pyright: ignore[reportIndexIssue]  # LiberTEM doesn't type UDF buffers
            )

    def merge(self, dest, src):
        # We receive full-size buffers from each node that
        # contributes at least one frame and rely on the rest being filled
        # with zeros correctly.
        dest.corrected[:] += src.corrected
        dest.backprojected[:] += src.backprojected

    def merge_all(self, ordered_results):
        res = {}
        for attr in ("corrected", "backprojected"):
            chunks = [getattr(b, attr) for b in ordered_results.values()]
            # We receive full-size buffers from each node that
            # contributes at least one frame and rely on the rest being filled
            # with zeros correctly.
            res[attr] = sum(chunks[1:], start=chunks[0].copy())
        return res
