import numpy as np
import pytest
from libertem.api import Context
from numpy.testing import assert_allclose

from microscope_calibration.common.model import Model4DSTEM, PixelYX
from microscope_calibration.ui import CalibratedDataset, CoordinateCorrectionLayout


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"nav_mode": "point", "sig_mode": "lin"},
        {"nav_mode": "sumsig", "sig_mode": "log", "twothetas": np.array((0.01, 0.02, 0.03))},
    ],
)
def test_smoke(kwargs):
    ctx = Context.make_with("inline")
    data = np.zeros((4, 5, 6, 7))
    ds = ctx.load("memory", data=data)
    calib_ds = CalibratedDataset(dataset=ds, model=Model4DSTEM.default(dataset_shape=ds.shape))

    _ = CoordinateCorrectionLayout(calibrated_dataset=calib_ds, ctx=ctx, **kwargs).layout


class MockEvent(PixelYX):
    pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nav_mode": "sumsig", "sig_mode": "log", "twothetas": np.array((0.01, 0.02, 0.03))},
    ],
)
def test_events(kwargs):
    ctx = Context.make_with("inline")
    data = np.zeros((4, 5, 6, 7))
    ds = ctx.load("memory", data=data)
    calib_ds = CalibratedDataset(dataset=ds, model=Model4DSTEM.default(dataset_shape=ds.shape))

    layout = CoordinateCorrectionLayout(calibrated_dataset=calib_ds, ctx=ctx, **kwargs)

    _ = layout.layout

    layout.cl_input.value = 1.2
    layout.semiconv_input.value = 2
    layout.scalebar_input.value = 23
    layout.detector_pitch_input.value = 60
    layout.scan_rotation_input.value = 17

    calibrated = layout.current_calibrated()
    assert calibrated.dataset == ds
    m = calibrated.model
    assert_allclose(m.camera_length, 1.2)
    assert_allclose(m.semiconv, 2 / 1000)
    assert_allclose(m.detector_pixel_pitch, 60e-6)
    assert_allclose(m.scan_rotation, np.deg2rad(17))

    layout.move_scan_to(MockEvent(y=1, x=2))
    layout.move_feature_to(MockEvent(y=2, x=3))
    layout.move_rings_to(MockEvent(y=3, x=4))
    layout.move_feature_select_to(MockEvent(y=4, x=5))

    layout.descan_add_row()
    layout.descan_add_row()
    layout.move_scan_to(MockEvent(y=0, x=1))
    layout.move_rings_to(MockEvent(y=3, x=4))
    layout.descan_add_row()
    layout.descan_move_to_row(0)
    layout.perform_descan_update()
    layout.center_correlation_regression()

    layout.descan_delete_row(0)
    layout.descan_drop()

    layout.coord_add_row()
    layout.move_feature_select_to(MockEvent(y=3, x=2))
    layout.coord_add_row()
    layout.move_scan_to(MockEvent(y=0, x=0))
    layout.coord_add_row()
    layout.perform_coord_update()
    layout.sharpen()
    layout.invert_focus()

    layout.coord_move_to_row(0)

    layout.coord_delete_row(0)
    layout.coord_drop()
