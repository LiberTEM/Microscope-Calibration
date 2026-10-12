import warnings

import jax_dataclasses as jdc
import numpy as np
import pytest
from numpy.testing import assert_allclose
from temgym_core import CoordXY, PixelYX

from microscope_calibration.common.model import (
    DescanError,
    Model4DSTEM,
    NonlinearModelError,
    RuntimeDerivationWarning,
    _runtime_descan_compensation,
)
from microscope_calibration.common.stem_overfocus import (
    _ray_transfer_matrix_numeric,
    get_backward_transformation_matrix,
    ray_transfer_matrix,
)

# Subclasses of Model4DSTEM can change the result of trace() without
# overriding trace() itself, which makes the generated code invalid for them.
# By default, they derive the functions at runtime instead.


@jdc.pytree_dataclass
class ShiftedDetectorModel(Model4DSTEM):
    def real_to_detector(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        res = super().real_to_detector(coords, _one)
        return PixelYX(y=res.y + 3.0 * _one, x=res.x - 2.0 * _one)


@jdc.pytree_dataclass
class CompatibleModel(Model4DSTEM):
    # Doesn't change trace(), so the generated code of Model4DSTEM is valid
    @classmethod
    def _generated_module(cls):
        return Model4DSTEM._generated_module()

    def scan_area(self):
        return self.scan_pixel_pitch**2


@jdc.pytree_dataclass
class NonlinearModel(Model4DSTEM):
    def real_to_detector(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        res = super().real_to_detector(coords, _one)
        return PixelYX(y=res.y**2, x=res.x)


class UndecoratedModel(Model4DSTEM):
    pass


def make_model(model_type=Model4DSTEM) -> Model4DSTEM:
    return model_type(
        overfocus=0.7,
        scan_pixel_pitch=0.3,
        scan_center=PixelYX(y=3.0, x=-2.0),
        scan_rotation=0.4,
        camera_length=1.3,
        detector_pixel_pitch=0.05,
        detector_center=PixelYX(y=11.0, x=7.0),
        semiconv=0.01,
        flip_factor=-1.0,
        descan_error=DescanError(*np.linspace(-0.3, 0.4, 12)),
        detector_rotation=0.9,
    )


@pytest.fixture(autouse=True)
def clear_caches():
    # The runtime derivation warns once per class and function
    _runtime_descan_compensation.cache_clear()
    _ray_transfer_matrix_numeric.cache_clear()


def central_ray_px(model: Model4DSTEM):
    res = []
    for scan_y, scan_x in ((0.0, 0.0), (5.0, -3.0), (-7.0, 2.0)):
        for cl in (0.5, 1.3):
            r = model.derive(camera_length=cl).trace(
                scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0.0, source_dx=0.0
            )
            res.append(tuple(r["specimen"].sampling["scan_px"]))
            res.append(tuple(r["detector"].sampling["detector_px"]))
    return np.array(res, dtype=np.float64)


ADJUSTMENTS = {
    "scan_rotation": 1.1,
    "scan_pixel_pitch": 0.7,
    "scan_center": PixelYX(y=-1.0, x=4.0),
    "detector_rotation": -0.3,
    "flip_factor": 1.0,
    "detector_center": PixelYX(y=2.0, x=-5.0),
    "detector_pixel_pitch": 0.11,
}


@pytest.mark.parametrize("param", sorted(ADJUSTMENTS))
def test_adjust_runtime_derivation(param):
    model = make_model(ShiftedDetectorModel)
    with pytest.warns(RuntimeDerivationWarning, match="ShiftedDetectorModel"):
        modified = getattr(model, f"adjust_{param}")(ADJUSTMENTS[param])
    assert type(modified) is ShiftedDetectorModel
    assert getattr(modified, param) == ADJUSTMENTS[param]
    assert_allclose(central_ray_px(modified), central_ray_px(model), atol=1e-9)


def test_derive_keeps_class():
    model = make_model(ShiftedDetectorModel)
    assert type(model.derive(overfocus=0.1)) is ShiftedDetectorModel
    assert type(model.normalize_types()) is ShiftedDetectorModel


def test_ray_transfer_runtime_derivation():
    model = make_model(ShiftedDetectorModel)
    with pytest.warns(RuntimeDerivationWarning, match="ray_transfer_matrix"):
        transfer = ray_transfer_matrix(model)
    # The generated code for Model4DSTEM doesn't know about the shift
    base = ray_transfer_matrix(make_model())
    assert_allclose(transfer[4, 2:4] - base[4, 2:4], (3.0, -2.0))
    get_backward_transformation_matrix(model)


def test_compatible_subclass_uses_generated_code():
    model = make_model(CompatibleModel)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeDerivationWarning)
        modified = model.adjust_scan_rotation(1.1)
        transfer = ray_transfer_matrix(model)
    assert type(modified) is CompatibleModel
    assert_allclose(transfer, ray_transfer_matrix(make_model()))


def test_base_class_uses_generated_code():
    model = make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeDerivationWarning)
        model.adjust_scan_rotation(1.1).adjust_camera_length(0.3)
        get_backward_transformation_matrix(model)


def test_nonlinear_subclass():
    model = make_model(NonlinearModel)
    with pytest.warns(RuntimeDerivationWarning):
        with pytest.raises(NonlinearModelError):
            model.adjust_scan_rotation(1.1)
    with pytest.warns(RuntimeDerivationWarning):
        with pytest.raises(NonlinearModelError):
            ray_transfer_matrix(model)


def test_undecorated_subclass():
    model = make_model(UndecoratedModel)
    with pytest.warns(RuntimeDerivationWarning):
        with pytest.raises(TypeError, match="pytree_dataclass"):
            model.adjust_scan_rotation(1.1)
