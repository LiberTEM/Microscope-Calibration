import numpy as np
import pytest
from numpy.testing import assert_allclose

from microscope_calibration.common.model import DescanError, Model4DSTEM
from microscope_calibration.util import jeol_dm4
from microscope_calibration.util.jeol_dm4 import (
    IMAGE,
    CalibratedJEOLDM4,
    acceleration_from_dm4,
    compare,
    derive_model_from_dm4,
    derive_model_relative_dm4,
    microscope_info,
)


def make_tags(**changes):
    # Subset of the tags of a 4D STEM DM4 file, keys relative to IMAGE
    tags = {
        # sig
        "ImageData.Dimensions.1": 64,
        "ImageData.Dimensions.2": 48,
        # nav
        "ImageData.Dimensions.3": 32,
        "ImageData.Dimensions.4": 16,
        "ImageData.Calibrations.Dimension.3.Scale": 2.0,
        "ImageData.Calibrations.Dimension.3.Units": "nm",
        "ImageData.Calibrations.Dimension.4.Scale": 2.0,
        "ImageData.Calibrations.Dimension.4.Units": "nm",
        "ImageTags.Acquisition.Frame.CCD.Pixel Size (um)": np.array([55.0, 55.0]),
        "ImageTags.Microscope Info.STEM Camera Length": np.float32(80.0),
        "ImageTags.Microscope Info.Voltage": 200000.0,
        "ImageTags.Microscope Info.Stage Position.Stage Z": 30.0,
        "ImageTags.DigiScan.Rotation": 10.0,
        "ImageTags.DigiScan.Rotation Offset": 3.0,
    }
    tags.update(changes)
    res = {f"{IMAGE}.{key}": value for key, value in tags.items()}
    # Thumbnail
    res[".ImageList.1.ImageTags.Microscope Info.Voltage"] = 1.0
    return res


def test_derive_model():
    model = derive_model_from_dm4(make_tags())
    assert model.scan_center == (16.0, 8.0)
    assert model.detector_center == (32.0, 24.0)
    assert_allclose(model.scan_pixel_pitch, 2e-9)
    assert_allclose(model.detector_pixel_pitch, 55e-6)
    assert_allclose(model.camera_length, 0.08)
    # Opposite sign of the DM4 tags
    assert_allclose(model.scan_rotation, np.deg2rad(-10.0))
    assert_allclose(model.detector_rotation, np.deg2rad(-3.0))


def test_derive_model_start():
    start = Model4DSTEM.default(dataset_shape=(1, 1, 1, 1)).derive(overfocus=0.1)
    model = derive_model_from_dm4(make_tags(), model=start)
    assert model.overfocus == 0.1
    assert model.scan_center == start.scan_center


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"ImageData.Calibrations.Dimension.4.Scale": 3.0}, "uniform scan step"),
        ({"ImageTags.Acquisition.Frame.CCD.Pixel Size (um)": np.array([55.0, 50.0])}, "uniform"),
        ({"ImageTags.Acquisition.Frame.CCD.Pixel Size (um)": np.ones(3)}, "two dimensions"),
    ],
)
def test_derive_model_invalid(changes, match):
    with pytest.raises(ValueError, match=match):
        derive_model_from_dm4(make_tags(**changes))


def test_acceleration():
    assert acceleration_from_dm4(make_tags()).to("kV").magnitude == 200


def test_derive_model_relative():
    model = Model4DSTEM.default(dataset_shape=(32, 16, 64, 48)).derive(
        overfocus=1e-3,
        scan_pixel_pitch=3e-9,
        detector_pixel_pitch=50e-6,
        camera_length=0.1,
        scan_rotation=0.2,
    )
    new_tags = make_tags(
        **{
            "ImageData.Calibrations.Dimension.3.Scale": 4.0,
            "ImageData.Calibrations.Dimension.4.Scale": 4.0,
            "ImageTags.Microscope Info.STEM Camera Length": 120.0,
            "ImageTags.DigiScan.Rotation": 15.0,
            "ImageTags.Microscope Info.Stage Position.Stage Z": 50.0,
        }
    )
    res = derive_model_relative_dm4(old_tags=make_tags(), new_tags=new_tags, model=model)
    assert_allclose(res.scan_pixel_pitch, 6e-9)
    assert_allclose(res.detector_pixel_pitch, 50e-6)
    assert_allclose(res.camera_length, 0.15)
    assert_allclose(res.scan_rotation, 0.2 + np.deg2rad(-5.0))
    # Stage Z up by 20 um reduces the overfocus
    assert_allclose(res.overfocus, 1e-3 - 20e-6)


def test_microscope_info():
    info = microscope_info(make_tags())
    assert info["Voltage"] == 200000.0
    assert info["STEM Camera Length"] == 80.0
    assert type(info["STEM Camera Length"]) is float
    # Only the 4D STEM data, not the thumbnail
    assert len(info) == 3


def test_compare():
    old_model = derive_model_from_dm4(make_tags())
    new_tags = make_tags(**{"ImageTags.Microscope Info.Voltage": 300000.0})
    new_model = old_model.derive(
        overfocus=old_model.overfocus + 1.0, descan_error=DescanError(offpxi=1.0)
    )
    res = compare(old_tags=make_tags(), old_model=old_model, new_tags=new_tags, new_model=new_model)
    assert res == {
        "dm4": {"Voltage": (200000.0, 300000.0)},
        "model": {"overfocus": (old_model.overfocus, new_model.overfocus)},
        "descan_error": {"offpxi": (0.0, 1.0)},
    }


def test_read_tags_closes_file(monkeypatch):
    events = []

    class FakeFileDM:
        def __init__(self, path):
            self.allTags = {"path": path}

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

    monkeypatch.setattr(jeol_dm4, "fileDM", FakeFileDM)
    assert jeol_dm4.read_tags("test.dm4") == {"path": "test.dm4"}
    assert events == ["enter", "exit"]


class FakeContext:
    def load(self, filetype, path):
        return ("dataset", path)


def test_save_load(monkeypatch, tmp_path):
    monkeypatch.setattr(jeol_dm4, "read_tags", lambda path: make_tags())
    ctx = FakeContext()
    calibrated = CalibratedJEOLDM4.new("data.dm4", ctx=ctx)
    assert calibrated.dataset == ("dataset", "data.dm4")
    model = calibrated.model.derive(overfocus=0.3, descan_error=DescanError(sxo_pyi=0.5))
    calibrated = calibrated.calibrated(model, ctx=ctx)
    calibrated.save(tmp_path / "calibration.json")
    loaded = CalibratedJEOLDM4.load(tmp_path / "calibration.json", ctx=ctx)
    assert loaded.path == "data.dm4"
    assert loaded.model == model.normalize_types()
    assert loaded.compare(calibrated) == {"dm4": {}, "model": {}, "descan_error": {}}
