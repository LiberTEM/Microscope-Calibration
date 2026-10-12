import numpy as np
import pytest
from numpy.testing import assert_allclose

from microscope_calibration.common.model import (
    CoordXY,
    DescanError,
    Model4DSTEM,
    PixelYX,
)
from microscope_calibration.common.stem_overfocus import (
    FittingError,
    get_backward_transformation_matrix,
)
from microscope_calibration.udf.stem_overfocus import BaseCorrectionUDF
from microscope_calibration.util.sympy import lambdify

# The transformation matrices are derived symbolically from
# Model4DSTEM.trace(). The function below is the previous implementation that
# fitted the matrix to sample rays. It serves as a reference to confirm that
# the symbolic derivation gives the same result.

# scan position y/x, source tilt y/x
TEST_PARAMETERS = np.array(
    (
        [0.0, 0.0, 0.0, 0.0],
        [100.0, 100.0, 0.0, 0.0],
        [-100.0, 100.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
    )
)


def _lstsq(get_sample):
    input_samples = []
    output_samples = []
    for test_param_raw in TEST_PARAMETERS:
        for factor in (1.0, 2.0):
            input_sample, output_sample = get_sample(*(test_param_raw * factor))
            input_samples.append(input_sample)
            output_samples.append(output_sample)
    x, residuals, rank, s = np.linalg.lstsq(
        np.array(input_samples, dtype=np.float64), np.array(output_samples, dtype=np.float64)
    )
    assert rank == 5
    assert_allclose(residuals, 0.0, atol=1e-11)
    return x


def reference_get_backward_transformation_matrix(rec_model: Model4DSTEM, specimen_to_image=None):
    @lambdify(modules=np)
    def get_sample(scan_pos_x, scan_pos_y, source_dy, source_dx):
        scan_pos = PixelYX(x=scan_pos_x, y=scan_pos_y)
        res = rec_model.trace(
            scan_pos=scan_pos,
            source_dy=source_dy,
            source_dx=source_dx,
        )

        if specimen_to_image is None:
            spec_px = res["specimen"].sampling["scan_px"]
        else:
            spec_px = specimen_to_image(CoordXY(x=res["specimen"].ray.x, y=res["specimen"].ray.y))
        input_sample = (scan_pos.y, scan_pos.x, spec_px.y, spec_px.x, 1.0)
        output_sample = (
            res["detector"].sampling["detector_px"].y,
            res["detector"].sampling["detector_px"].x,
            source_dy,
            source_dx,
            1.0,
        )
        return (input_sample, output_sample)

    return _lstsq(get_sample)


def affine_mapping(coords: CoordXY) -> PixelYX:
    # Arbitrary rotation, anisotropic scaling, shear and offset
    return PixelYX(
        y=3.1 * coords.y - 0.7 * coords.x + 17.0,
        x=0.4 * coords.y + 2.3 * coords.x - 5.0,
    )


def default_model(**kwargs) -> Model4DSTEM:
    return Model4DSTEM.default(dataset_shape=(32, 32, 32, 32)).derive(
        descan_error=DescanError(*np.linspace(-0.3, 0.4, 12)), **kwargs
    )


@pytest.mark.parametrize("specimen_to_image", (None, affine_mapping))
@pytest.mark.parametrize("repeat", range(5))
def test_backward_matrix_reference(random_model, specimen_to_image, repeat):
    mat = get_backward_transformation_matrix(
        rec_model=random_model, specimen_to_image=specimen_to_image
    )
    ref = reference_get_backward_transformation_matrix(
        rec_model=random_model, specimen_to_image=specimen_to_image
    )
    assert mat.dtype == np.float64
    assert mat.shape == (5, 5)
    assert_allclose(mat, ref, rtol=1e-9, atol=1e-9 * np.max(np.abs(ref)))


@pytest.mark.parametrize("overfocus", (0.0, 1e-12, -1e-12))
def test_backward_matrix_singular(overfocus):
    model = default_model(overfocus=overfocus)
    with pytest.raises(FittingError):
        get_backward_transformation_matrix(rec_model=model)


def test_backward_matrix_min_footprint():
    # With semiconv 1 mrad and 1 um scan pixel pitch, the footprint radius is
    # 1e-7 px at 0.1 nm overfocus, below the default threshold of 1e-6 px.
    model = default_model(overfocus=1e-10)
    with pytest.raises(FittingError, match="footprint"):
        get_backward_transformation_matrix(rec_model=model)
    # Methods with very small defocus can lower the threshold
    mat = get_backward_transformation_matrix(rec_model=model, min_footprint_px=1e-8)
    assert np.all(np.isfinite(mat))
    # Change of tilt per specimen pixel: scan pixel pitch / overfocus
    assert_allclose(np.abs(np.linalg.det(mat[2:4, 2:4])), (1e-6 / 1e-10) ** 2)


def test_backward_matrix_not_affine():
    def mapping(coords: CoordXY) -> PixelYX:
        return PixelYX(y=coords.y**2, x=coords.x)

    with pytest.raises(ValueError, match="not affine"):
        get_backward_transformation_matrix(rec_model=default_model(), specimen_to_image=mapping)


@pytest.mark.parametrize("overfocus", (0.0, 1e-12))
def test_udf_no_backprojection_in_focus(overfocus):
    # The UDF continues without backprojection for singular transformations.
    # Before deriving the matrix symbolically, near-singular cases raised a
    # RuntimeError instead.
    BaseCorrectionUDF._back_mat.cache_clear()
    assert BaseCorrectionUDF._back_mat(rec_model=default_model(overfocus=overfocus)) is None
