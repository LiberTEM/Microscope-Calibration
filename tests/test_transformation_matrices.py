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
    get_detector_correction_matrix,
    ray_transfer_matrix,
)
from microscope_calibration.udf.stem_overfocus import BaseCorrectionUDF
from microscope_calibration.util.stem_overfocus_sim import get_forward_transformation_matrix
from microscope_calibration.util.sympy import lambdify

# The transformation matrices are calculated from the ray transfer matrix,
# which is derived symbolically from Model4DSTEM.trace(). The functions below
# are the previous implementations that fitted the matrices to sample rays.
# They serve as a reference to confirm that the new calculation gives the
# same results.

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


def reference_get_detector_correction_matrix(rec_model: Model4DSTEM, ref_model: Model4DSTEM):
    @lambdify(modules=np)
    def get_sample(scan_pos_x, scan_pos_y, source_dy, source_dx):
        scan_pos = PixelYX(x=scan_pos_x, y=scan_pos_y)
        res = rec_model.trace(
            scan_pos=scan_pos,
            source_dy=source_dy,
            source_dx=source_dx,
        )

        ref_res = ref_model.trace(
            scan_pos=scan_pos,
            source_dy=source_dy,
            source_dx=source_dx,
        )

        input_sample = (
            scan_pos.y,
            scan_pos.x,
            ref_res["detector"].sampling["detector_px"].y,
            ref_res["detector"].sampling["detector_px"].x,
            1.0,
        )
        output_sample = (
            res["detector"].sampling["detector_px"].y,
            res["detector"].sampling["detector_px"].x,
            source_dy,
            source_dx,
            1.0,
        )
        return (input_sample, output_sample)

    return _lstsq(get_sample)


def reference_get_forward_transformation_matrix(sim_model: Model4DSTEM, specimen_to_image=None):
    @lambdify(modules=np)
    def get_sample(scan_pos_x, scan_pos_y, source_dy, source_dx):
        scan_pos = PixelYX(x=scan_pos_x, y=scan_pos_y)
        res = sim_model.trace(scan_pos=scan_pos, source_dy=source_dy, source_dx=source_dx)
        if specimen_to_image is None:
            spec_px = res["specimen"].sampling["scan_px"]
        else:
            spec_px = specimen_to_image(CoordXY(x=res["specimen"].ray.x, y=res["specimen"].ray.y))
        input_sample = (
            scan_pos.y,
            scan_pos.x,
            res["detector"].sampling["detector_px"].y,
            res["detector"].sampling["detector_px"].x,
            1.0,
        )
        output_sample = (
            spec_px.y,
            spec_px.x,
            source_dy,
            source_dx,
            1.0,
        )
        return (input_sample, output_sample)

    return _lstsq(get_sample)


def assert_matrix_close(mat, ref):
    assert mat.dtype == np.float64
    assert mat.shape == (5, 5)
    assert_allclose(mat, ref, rtol=1e-9, atol=1e-9 * np.max(np.abs(ref)))


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
    assert_matrix_close(mat, ref)


@pytest.mark.parametrize("specimen_to_image", (None, affine_mapping))
@pytest.mark.parametrize("repeat", range(5))
def test_forward_matrix_reference(random_model, specimen_to_image, repeat):
    mat = get_forward_transformation_matrix(
        sim_model=random_model, specimen_to_image=specimen_to_image
    )
    ref = reference_get_forward_transformation_matrix(
        sim_model=random_model, specimen_to_image=specimen_to_image
    )
    assert_matrix_close(mat, ref)


@pytest.mark.parametrize("repeat", range(5))
def test_detector_correction_matrix_reference(random_model, repeat):
    ref_model = Model4DSTEM(
        overfocus=random_model.overfocus,
        scan_pixel_pitch=np.random.uniform(0.01, 2),
        scan_center=PixelYX(y=np.random.uniform(-10, 10), x=np.random.uniform(-10, 10)),
        scan_rotation=np.random.uniform(-np.pi, np.pi),
        camera_length=np.random.uniform(0.1, 2),
        detector_pixel_pitch=np.random.uniform(0.01, 2),
        detector_center=PixelYX(y=np.random.uniform(-10, 10), x=np.random.uniform(-10, 10)),
        semiconv=random_model.semiconv,
        flip_factor=np.random.choice([-1.0, 1.0]),
        descan_error=DescanError(*np.random.uniform(-1, 1, size=12)),
        detector_rotation=np.random.uniform(-np.pi, np.pi),
    )
    mat = get_detector_correction_matrix(rec_model=random_model, ref_model=ref_model)
    ref = reference_get_detector_correction_matrix(rec_model=random_model, ref_model=ref_model)
    assert_matrix_close(mat, ref)


def test_detector_correction_matrix_default_reference(random_model):
    mat = get_detector_correction_matrix(rec_model=random_model)
    ref_model = random_model.derive(
        scan_rotation=0.0,
        flip_factor=1.0,
        descan_error=DescanError(),
        detector_rotation=random_model.scan_rotation,
    )
    ref = reference_get_detector_correction_matrix(rec_model=random_model, ref_model=ref_model)
    assert_matrix_close(mat, ref)


def test_ray_transfer_matrix(random_model):
    transfer = ray_transfer_matrix(random_model, specimen_to_image=affine_mapping)
    for scan_y, scan_x, tilt_y, tilt_x in TEST_PARAMETERS:
        res = random_model.trace(
            scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=tilt_y, source_dx=tilt_x
        )
        spec = affine_mapping(CoordXY(x=res["specimen"].ray.x, y=res["specimen"].ray.y))
        det = res["detector"].sampling["detector_px"]
        expected = np.array((spec.y, spec.x, det.y, det.x, 1.0), dtype=np.float64)
        assert_allclose(
            np.array((scan_y, scan_x, tilt_y, tilt_x, 1.0)) @ transfer,
            expected,
            rtol=1e-9,
            atol=1e-9 * np.max(np.abs(expected)),
        )


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


@pytest.mark.parametrize("delta", (0.0, 1e-12))
def test_detector_matrices_singular_crossover(delta):
    # The detector in the crossover at the source: the detector position
    # doesn't depend on the tilt
    model = default_model(overfocus=0.1)
    model = model.derive(camera_length=-model.overfocus + delta)
    with pytest.raises(FittingError):
        get_forward_transformation_matrix(sim_model=model)
    with pytest.raises(FittingError):
        get_detector_correction_matrix(rec_model=model)


def test_zero_pixel_pitch():
    model = default_model(overfocus=0.1, detector_pixel_pitch=0.0)
    with pytest.raises(FittingError):
        get_backward_transformation_matrix(rec_model=model)
