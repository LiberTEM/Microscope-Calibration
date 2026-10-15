import numpy as np
import pytest
from bokeh.document import Document
from libertem.api import Context
from numpy.testing import assert_allclose

from microscope_calibration.common.model import Model4DSTEM, PixelYX
from microscope_calibration.ui import CalibratedDataset, CoordinateCorrectionLayout


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"nav_mode": "point", "sig_mode": "lin"},
        {"nav_mode": "sumsig", "sig_mode": "log", "twothetas": np.array((0.01, 0.02, 0.03))},
    ),
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
    ({"nav_mode": "sumsig", "sig_mode": "log", "twothetas": np.array((0.01, 0.02, 0.03))},),
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
    # Returns the class of the calibrated dataset's model
    assert type(m) is Model4DSTEM
    assert_allclose(m.camera_length, 1.2)
    # The UI shows the angle, the model holds the slope
    assert_allclose(m.semiconv, np.tan(2 / 1000))
    assert_allclose(m.detector_pixel_pitch, 60e-6)
    assert_allclose(m.scan_rotation, np.deg2rad(17))

    layout.move_scan_to(MockEvent(y=1, x=2))
    layout.move_feature_to(MockEvent(y=2, x=3))
    layout.move_rings_to(MockEvent(y=3, x=4))
    layout.move_feature_select_to(MockEvent(y=4, x=5))

    layout.descan_add_row()
    # Replaces the previous row at the same scan position
    layout.descan_add_row()
    layout.move_scan_to(MockEvent(y=0, x=1))
    layout.move_rings_to(MockEvent(y=3, x=4))
    layout.descan_add_row()
    layout.move_scan_to(MockEvent(y=3, x=0))
    layout.descan_add_row()
    assert len(layout.descan_fixpoint_table.value) == 3
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


def test_in_focus_keeps_feature_selection(monkeypatch):
    ctx = Context.make_with("inline")
    ds = ctx.load("memory", data=np.zeros((4, 5, 6, 7)))
    model = Model4DSTEM.default(dataset_shape=ds.shape)
    assert model.overfocus == 0
    layout = CoordinateCorrectionLayout(
        calibrated_dataset=CalibratedDataset(dataset=ds, model=model), ctx=ctx
    )
    start = dict(layout.feature_select_model.data)

    def fail(**kwargs):
        raise AssertionError("Feature position on the detector is not defined in focus")

    monkeypatch.setattr(layout, "get_feature_on_detector", fail)
    layout.move_scan_to(MockEvent(y=1, x=2))
    layout.move_feature_to(MockEvent(y=2, x=3))
    layout.cl_input.value = 1.2
    assert layout.feature_select_model.data == start


@pytest.mark.parametrize("held", (False, True))
def test_update_with_force(held):
    ctx = Context.make_with("inline")
    ds = ctx.load("memory", data=np.zeros((4, 5, 6, 7)))
    model = Model4DSTEM.default(dataset_shape=ds.shape)
    layout = CoordinateCorrectionLayout(
        calibrated_dataset=CalibratedDataset(dataset=ds, model=model), ctx=ctx
    )
    calls = []
    layout._on_data_change(layout.feature_model, lambda attr, old, new: calls.append(dict(new)))
    doc = Document()
    doc.add_root(layout.feature_model)
    if held:
        # Like Panel in a notebook
        doc.hold("combine")
    update = layout.scan_pos_update(PixelYX(y=1.0, x=2.0))
    layout.update_with_force(layout.feature_model, update)
    # Callbacks run once, also during a hold
    assert calls == [dict(layout.feature_model.data)]
    assert layout.feature_model.data["scan_y"] == [1.0]
    if held:
        # The change is collected to be sent to the browser
        assert doc.callbacks._held_events


def test_status_messages():
    ctx = Context.make_with("inline")
    ds = ctx.load("memory", data=np.zeros((4, 5, 6, 7)))
    model = Model4DSTEM.default(dataset_shape=ds.shape).derive(overfocus=0.01)
    layout = CoordinateCorrectionLayout(
        calibrated_dataset=CalibratedDataset(dataset=ds, model=model), ctx=ctx
    )
    _ = layout.layout
    assert not layout.descan_status.visible
    # Empty table
    layout.apply_button.clicks += 1
    assert layout.descan_status.visible
    assert layout.descan_status.alert_type == "danger"
    assert "at least 3 points" in layout.descan_status.object
    for y, x in ((0, 0), (1, 3), (3, 1)):
        layout.move_scan_to(MockEvent(y=y, x=x))
        layout.descan_add_row()
    layout.apply_button.clicks += 1
    assert layout.descan_status.alert_type == "success"
    assert "from 3 points: RMS residual" in layout.descan_status.object
    assert layout.descan_status.visible
    # Hidden on the next change
    layout.move_scan_to(MockEvent(y=2, x=2))
    assert not layout.descan_status.visible
    layout.apply_button.clicks += 1
    assert layout.descan_status.visible
    layout.descan_drop()
    assert not layout.descan_status.visible
