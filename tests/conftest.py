import zlib

import numpy as np
import pytest

from microscope_calibration.common.model import DescanError, Model4DSTEM, PixelYX


@pytest.fixture(autouse=True)
def _seed_random(request):
    # Reproducible random test inputs with a different seed for each test
    np.random.seed(zlib.crc32(request.node.nodeid.encode()))


@pytest.fixture
def random_model() -> Model4DSTEM:
    camera_length = np.random.uniform(0.1, 2)
    if np.random.random() < 0.5:
        overfocus = np.random.uniform(0.1, 2)
    else:
        # Underfocus, not close to a crossover at the detector
        overfocus = -np.random.uniform(0.05, 0.9) * camera_length
    return Model4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=np.random.uniform(0.01, 2),
        scan_center=PixelYX(
            y=np.random.uniform(-10, 10),
            x=np.random.uniform(-10, 10),
        ),
        scan_rotation=np.random.uniform(-np.pi, np.pi),
        camera_length=camera_length,
        detector_pixel_pitch=np.random.uniform(0.01, 2),
        detector_center=PixelYX(
            y=np.random.uniform(-10, 10),
            x=np.random.uniform(-10, 10),
        ),
        semiconv=np.random.uniform(0.0001, np.pi / 2),
        flip_factor=np.random.choice([-1.0, 1.0]),
        descan_error=DescanError(*np.random.uniform(-1, 1, size=len(DescanError()))),
        detector_rotation=np.random.uniform(-np.pi, np.pi),
    )
