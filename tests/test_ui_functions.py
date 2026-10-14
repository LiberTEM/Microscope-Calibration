import numpy as np
from numpy.testing import assert_allclose

from microscope_calibration.common.model import DescanError, Model4DSTEM, PixelYX
from microscope_calibration.common.stem_overfocus import get_center
from microscope_calibration.ui import (
    center_regression,
    correlation_pattern,
    expected_center_shifts,
    format_descan_error,
    format_model,
)


def test_center_regression():
    center = PixelYX(y=10.0, x=20.0)
    y, x = np.mgrid[:5, :7]
    field = np.stack(
        (center.y + 1.0 + 0.2 * y - 0.3 * x, center.x - 2.0 + 0.4 * y + 0.5 * x), axis=-1
    ).astype(np.float32)
    regression = center_regression(field, center)
    assert_allclose(regression, [[1.0, -2.0], [0.2, 0.4], [-0.3, 0.5]], atol=1e-5)
    # The input is not modified
    assert field[0, 0, 0] == np.float32(center.y + 1.0)


def test_expected_center_shifts():
    model = Model4DSTEM.default(dataset_shape=(4, 5, 32, 32)).derive(
        overfocus=1e-3,
        descan_error=DescanError(offpxi=1e-5, pxo_pxi=0.3, pyo_pyi=-0.2),
    )
    shifts = expected_center_shifts(model, nav_shape=(4, 5))
    assert shifts.shape == (4, 5, 2)
    for scan_y, scan_x in ((0, 0), (3, 1), (2, 4)):
        center = get_center(model, PixelYX(y=float(scan_y), x=float(scan_x)))
        assert_allclose(
            shifts[scan_y, scan_x],
            (center.y - model.detector_center.y, center.x - model.detector_center.x),
            atol=1e-9,
        )
    no_descan = model.derive(descan_error=DescanError())
    center = get_center(no_descan, PixelYX(y=0.0, x=0.0))
    assert_allclose(
        expected_center_shifts(no_descan, nav_shape=(4, 5)),
        np.broadcast_to(
            (center.y - model.detector_center.y, center.x - model.detector_center.x), (4, 5, 2)
        ),
        atol=1e-9,
    )


def test_correlation_pattern():
    pattern = correlation_pattern(ri=np.array([0.0, 20.0]), ro=np.array([5.0, 30.0]))
    assert pattern.radius == 5.0
    assert pattern.radius_outer == 10.0
    pattern = correlation_pattern(ri=np.array([0.0]), ro=np.array([5.0]))
    assert pattern.radius_outer == 6.0


def test_format_model():
    model = Model4DSTEM.default(dataset_shape=(8, 8, 8, 8)).derive(
        scan_rotation=np.pi / 2, descan_error=DescanError(offpxi=1.23456789)
    )
    res = format_model(model, ["scan_rotation", "camera_length", "scan_center"])
    assert res == {
        "scan_rotation": "90.0°",
        "camera_length": "1.0",
        "scan_center": str(model.scan_center),
    }
    assert format_descan_error(model.descan_error)["offpxi"] == "1.2346"
