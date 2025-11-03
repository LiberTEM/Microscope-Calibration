import pytest

import numpy as np

from microscope_calibration.common.model import Parameters4DSTEM, PixelYX, DescanError


@pytest.fixture
def random_params() -> Parameters4DSTEM:
    return Parameters4DSTEM(
        overfocus=np.random.uniform(0.1, 2),
        scan_pixel_pitch=np.random.uniform(0.01, 2),
        scan_center=PixelYX(
            y=np.random.uniform(-10, 10),
            x=np.random.uniform(-10, 10),
        ),
        scan_rotation=np.random.uniform(-np.pi, np.pi),
        camera_length=np.random.uniform(0.1, 2),
        detector_pixel_pitch=np.random.uniform(0.01, 2),
        detector_center=PixelYX(
            y=np.random.uniform(-10, 10),
            x=np.random.uniform(-10, 10),
        ),
        semiconv=np.random.uniform(0.0001, np.pi/2),
        flip_factor=np.random.choice([-1., 1.]),
        descan_error=DescanError(
            *np.random.uniform(-1, 1, size=len(DescanError()))
        ),
        detector_rotation=np.random.uniform(-np.pi, np.pi),
    )
