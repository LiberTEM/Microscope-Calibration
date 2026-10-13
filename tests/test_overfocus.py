import numpy as np
import pytest
from libertem.corrections.coordinates import rotate, scale
from numpy.testing import assert_allclose

from microscope_calibration.common.model import (
    DescanError,
    Model4DSTEM,
    PixelYX,
)
from microscope_calibration.common.stem_overfocus import (
    correct_frame,
    get_backward_transformation_matrix,
    get_center,
    get_detector_correction_matrix,
    get_diffraction_pixel_radius,
    get_primary_beam_radius,
    project_frame_backwards,
    ring_radii,
)
from microscope_calibration.util.stem_overfocus_sim import project

NumpyModel = Model4DSTEM.lambdify(np)


@pytest.mark.with_numba
def test_model_consistency_backproject():
    model = Model4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.234,
        camera_length=0.73,
        detector_pixel_pitch=0.0321,
        semiconv=0.023,
        scan_center=PixelYX(x=0.13, y=0.23),
        scan_rotation=0.752,
        flip_factor=-1.0,
        detector_center=PixelYX(x=23, y=42),
        detector_rotation=2.134,
        descan_error=DescanError(
            pxo_pxi=0.2,
            pxo_pyi=0.3,
            pyo_pxi=0.5,
            pyo_pyi=0.7,
            sxo_pxi=0.11,
            sxo_pyi=0.13,
            syo_pxi=0.17,
            syo_pyi=0.19,
            offpxi=0.23,
            offpyi=0.29,
            offsxi=0.31,
            offsyi=0.37,
        ),
    )
    mat = get_backward_transformation_matrix(rec_model=model)

    inp = np.array((2, 3, 5, 7, 1))
    out = inp @ mat
    scan_pos = PixelYX(
        y=inp[0],
        x=inp[1],
    )
    source_dy = out[2]
    source_dx = out[3]

    assert_allclose(out[4], 1)
    res = NumpyModel.from_model(model).trace(
        scan_pos=scan_pos, source_dx=source_dx, source_dy=source_dy
    )
    assert_allclose(out[0], res["detector"].sampling["detector_px"].y, rtol=1e-12, atol=1e-12)
    assert_allclose(out[1], res["detector"].sampling["detector_px"].x, rtol=1e-12, atol=1e-12)
    assert_allclose(inp[2], res["specimen"].sampling["scan_px"].y, rtol=1e-12, atol=1e-12)
    assert_allclose(inp[3], res["specimen"].sampling["scan_px"].x, rtol=1e-12, atol=1e-12)


@pytest.mark.with_numba
def test_model_consistency_correct():
    model = Model4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.234,
        camera_length=0.73,
        detector_pixel_pitch=0.0321,
        semiconv=0.023,
        scan_center=PixelYX(x=0.13, y=0.23),
        scan_rotation=0.752,
        flip_factor=-1.0,
        detector_center=PixelYX(x=23, y=42),
        detector_rotation=2.134,
        descan_error=DescanError(
            pxo_pxi=0.2,
            pxo_pyi=0.3,
            pyo_pxi=0.5,
            pyo_pyi=0.7,
            sxo_pxi=0.11,
            sxo_pyi=0.13,
            syo_pxi=0.17,
            syo_pyi=0.19,
            offpxi=0.23,
            offpyi=0.29,
            offsxi=0.31,
            offsyi=0.37,
        ),
    )
    ref_model = Model4DSTEM(
        overfocus=1.1523,
        scan_pixel_pitch=0.4234,
        camera_length=0.7453,
        detector_pixel_pitch=0.03421,
        semiconv=0.042,
        scan_center=PixelYX(x=0.4, y=0.345),
        scan_rotation=0.75,
        flip_factor=1.0,
        detector_center=PixelYX(x=2, y=4),
        detector_rotation=2.4134,
        descan_error=DescanError(
            pxo_pxi=0.234,
            pxo_pyi=0.3345,
            pyo_pxi=0.534,
            pyo_pyi=0.735,
            sxo_pxi=0.1134,
            sxo_pyi=0.134,
            syo_pxi=0.173,
            syo_pyi=0.194,
            offpxi=0.234,
            offpyi=0.293,
            offsxi=0.313,
            offsyi=0.373,
        ),
    )
    mat = get_detector_correction_matrix(rec_model=model, ref_model=ref_model)

    inp = np.array((2, 3, 5, 7, 1))
    out = inp @ mat
    scan_pos = PixelYX(
        y=inp[0],
        x=inp[1],
    )
    source_dy = out[2]
    source_dx = out[3]

    assert_allclose(out[4], 1)
    res = NumpyModel.from_model(model).trace(
        scan_pos=scan_pos, source_dx=source_dx, source_dy=source_dy
    )

    ref_res = NumpyModel.from_model(ref_model).trace(
        scan_pos=scan_pos, source_dx=source_dx, source_dy=source_dy
    )

    assert_allclose(inp[2], ref_res["detector"].sampling["detector_px"].y, rtol=1e-12, atol=1e-12)
    assert_allclose(inp[3], ref_res["detector"].sampling["detector_px"].x, rtol=1e-12, atol=1e-12)
    assert_allclose(out[0], res["detector"].sampling["detector_px"].y, rtol=1e-12, atol=1e-12)
    assert_allclose(out[1], res["detector"].sampling["detector_px"].x, rtol=1e-12, atol=1e-12)


@pytest.mark.with_numba
def test_backproject_identity():
    # 1:1 size mapping between detector and specimen
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.inf,
        scan_center=PixelYX(x=7.1, y=16.0),
        scan_rotation=0.0,
        flip_factor=1.0,
        detector_center=PixelYX(x=7.1, y=16.0),
        descan_error=DescanError(),
    )
    obj = np.random.random((32, 13))
    res = np.zeros_like(obj)
    mat = get_backward_transformation_matrix(rec_model=model)
    project_frame_backwards(
        frame=obj,
        source_semiconv=np.inf,
        mat=mat,
        scan_y=16,
        scan_x=7,
        image_out=res,
    )
    assert_allclose(obj, res)


def test_backproject_counterrotate():
    # 1:1 size mapping between detector and specimen
    # Rotating detector and scan rotates the whole reference frame
    # so that the result is identity
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.inf,
        scan_center=PixelYX(x=16, y=16.0),
        scan_rotation=np.pi / 2,
        flip_factor=1.0,
        detector_center=PixelYX(x=16, y=16.0),
        detector_rotation=np.pi / 2,
        descan_error=DescanError(),
    )
    obj = np.random.random((32, 32))
    res = np.zeros_like(obj)
    mat = get_backward_transformation_matrix(rec_model=model)
    project_frame_backwards(
        frame=obj,
        source_semiconv=np.inf,
        mat=mat,
        scan_y=16,
        scan_x=16,
        image_out=res,
    )
    assert_allclose(obj, res)


@pytest.mark.parametrize("rotate_scan", (False, True))
@pytest.mark.parametrize("rotate_detector", (False, True))
@pytest.mark.parametrize("fixed_reference", (False, True))
@pytest.mark.parametrize("flip_factor", (1.0, -1.0))
def test_backproject_rot90_flip(rotate_scan, rotate_detector, fixed_reference, flip_factor):
    # 1:1 size mapping between detector and specimen
    # rotating scan and detector in fixed reference frame and
    # scan reference frame.
    # Projecting into 4D STEM dataset and then back-projecting a
    # detector frame into the reference coordinate system restores the object,
    # i.e. rotation and flip are canceled out.
    if rotate_detector:
        detector_rotation = np.pi / 2
    else:
        detector_rotation = 0.0
    if rotate_scan:
        scan_rotation = np.pi / 2
    else:
        scan_rotation = 0.0

    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.inf,
        scan_center=PixelYX(x=16, y=16.0),
        scan_rotation=scan_rotation,
        flip_factor=flip_factor,
        detector_center=PixelYX(x=16, y=16.0),
        detector_rotation=detector_rotation,
        descan_error=DescanError(),
    )

    if fixed_reference:

        def map_coord(inp):
            cy = obj.shape[0] / 2
            cx = obj.shape[1] / 2
            inp_vec = np.array((inp.y, inp.x))
            y, x = scale(1) @ inp_vec
            return PixelYX(y=y + cy, x=x + cx)
    else:
        map_coord = None

    obj = np.random.random((32, 32))

    projected = project(
        image=obj,
        scan_shape=((32, 32)),
        detector_shape=((32, 32)),
        sim_model=model,
        specimen_to_image=map_coord,
    )
    mat = get_backward_transformation_matrix(
        rec_model=model,
        specimen_to_image=map_coord,
    )
    # We back-project several scan positions and confirm that
    # we are getting back the object in the chosen reference coordinate system,
    # minus clipping at the borders
    for pick_y in (15, 16, 17):
        for pick_x in (15, 16, 17):
            res = np.zeros_like(obj)
            project_frame_backwards(
                frame=projected[pick_y, pick_x],
                source_semiconv=np.inf,
                mat=mat,
                scan_y=pick_y,
                scan_x=pick_x,
                image_out=res,
            )

            assert_allclose(obj[2:-2, 2:-2], res[2:-2, 2:-2])


def test_backproject_scale_fixed():
    # scan coordinates are 2x detector coordinates relative to object,
    # but we project from and back-project into fixed 1:1 reference coordinates
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=2,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.inf,
        scan_center=PixelYX(x=16.0, y=16.0),
        scan_rotation=0.0,
        flip_factor=1.0,
        detector_center=PixelYX(x=32, y=32.0),
        detector_rotation=0.0,
        descan_error=DescanError(),
    )
    obj = np.random.random((64, 64))
    res = np.zeros((64, 64))

    def map_coord(inp):
        cy = obj.shape[0] / 2
        cx = obj.shape[1] / 2
        inp_vec = np.array((inp.y, inp.x))
        y, x = scale(1) @ inp_vec
        return PixelYX(y=y + cy, x=x + cx)

    projected = project(
        image=obj,
        scan_shape=((32, 32)),
        detector_shape=((64, 64)),
        sim_model=model,
        specimen_to_image=map_coord,
    )

    mat = get_backward_transformation_matrix(rec_model=model, specimen_to_image=map_coord)
    project_frame_backwards(
        frame=projected[16, 16],
        source_semiconv=np.inf,
        mat=mat,
        scan_y=16,
        scan_x=16,
        image_out=res,
    )

    assert_allclose(obj, res)


def test_backproject_scale_scanref():
    # scan coordinates are 2x detector coordinates,
    # and we project from and back-project into that coordinate system
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=2,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.inf,
        scan_center=PixelYX(x=16.0, y=16.0),
        scan_rotation=0.0,
        flip_factor=1.0,
        detector_center=PixelYX(x=32, y=32.0),
        detector_rotation=0.0,
        descan_error=DescanError(),
    )
    obj = np.random.random((64, 64))
    res = np.zeros((32, 32))

    projected = project(
        image=obj,
        scan_shape=((32, 32)),
        detector_shape=((64, 64)),
        sim_model=model,
        specimen_to_image=None,
    )

    mat = get_backward_transformation_matrix(rec_model=model, specimen_to_image=None)
    project_frame_backwards(
        frame=projected[16, 16],
        source_semiconv=np.inf,
        mat=mat,
        scan_y=16,
        scan_x=16,
        image_out=res,
    )
    # The back-projection result corresponds to the trace
    # of the central pixel, i.e. scan coordinates
    assert_allclose(projected[:, :, 32, 32], res)


SCAN_PIXEL_PITCH = 0.1
DETECTOR_PIXEL_PITCH = 0.2
CAMERA_LENGTH = 1.0


def correction_shapes(obj_half_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    # Not square to detect mix-ups of y and x
    scan_shape = (obj_half_size * 2, obj_half_size * 2 + 2)
    # Detector larger than object to avoid clipping at the borders
    detector_shape = (obj_half_size * 4, obj_half_size * 4 + 6)
    return scan_shape, detector_shape


def correction_model(obj_half_size: int, **changes) -> Model4DSTEM:
    """
    Model without aberrations for the shapes of :func:`correction_shapes`
    """
    overfocus = 1.0
    scan_shape, detector_shape = correction_shapes(obj_half_size)
    semiconv = (obj_half_size * DETECTOR_PIXEL_PITCH / 2 + 0.00314157) / (overfocus + CAMERA_LENGTH)
    return Model4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=SCAN_PIXEL_PITCH,
        camera_length=CAMERA_LENGTH,
        detector_pixel_pitch=DETECTOR_PIXEL_PITCH,
        semiconv=semiconv,
        scan_center=PixelYX(y=scan_shape[0] / 2, x=scan_shape[1] / 2),
        scan_rotation=0.0,
        flip_factor=1.0,
        detector_center=PixelYX(y=detector_shape[0] / 2, x=detector_shape[1] / 2),
        detector_rotation=0.0,
        descan_error=DescanError(),
    ).derive(**changes)


# Descan error designed to give whole pixel shifts
WHOLE_PIXEL_DESCAN_ERROR = DescanError(
    offpxi=DETECTOR_PIXEL_PITCH,
    offpyi=DETECTOR_PIXEL_PITCH * 2,
    offsxi=-1 * DETECTOR_PIXEL_PITCH / CAMERA_LENGTH,
    offsyi=-2 * DETECTOR_PIXEL_PITCH / CAMERA_LENGTH,
    pxo_pxi=2 * DETECTOR_PIXEL_PITCH / SCAN_PIXEL_PITCH,
    pyo_pyi=3 * DETECTOR_PIXEL_PITCH / SCAN_PIXEL_PITCH,
    sxo_pxi=-3 * DETECTOR_PIXEL_PITCH / SCAN_PIXEL_PITCH / CAMERA_LENGTH,
    syo_pyi=-4 * DETECTOR_PIXEL_PITCH / SCAN_PIXEL_PITCH / CAMERA_LENGTH,
)


def check_correction(obj_half_size, model, ref_model, sim_ref_model, specimen_to_image=None):
    """
    Correct a dataset simulated with :code:`model` according to
    :code:`ref_model` and compare with a dataset simulated with
    :code:`sim_ref_model`
    """
    scan_shape, detector_shape = correction_shapes(obj_half_size)
    mat = get_detector_correction_matrix(rec_model=model, ref_model=ref_model)
    obj = np.random.random(scan_shape)
    projected = project(
        image=obj,
        detector_shape=detector_shape,
        scan_shape=scan_shape,
        sim_model=model,
        specimen_to_image=specimen_to_image,
    )
    out = np.zeros_like(projected)
    for scan_y in range(out.shape[0]):
        for scan_x in range(out.shape[1]):
            correct_frame(
                frame=projected[scan_y, scan_x],
                mat=mat,
                scan_y=scan_y,
                scan_x=scan_x,
                detector_out=out[scan_y, scan_x],
            )
    projected_ref = project(
        image=obj,
        detector_shape=detector_shape,
        scan_shape=scan_shape,
        sim_model=sim_ref_model,
        # Detector image correction doesn't interfere with
        # how scan positions are mapped, so we have to use the same mapping here
        specimen_to_image=specimen_to_image,
    )
    # 100 % match between corrected frames and simulated reference
    assert_allclose(projected_ref, out)
    return obj, projected, out


@pytest.mark.parametrize("scan_rotation", (0.0, np.pi / 2))
@pytest.mark.parametrize("detector_rotation", (0.0, np.pi / 2))
@pytest.mark.parametrize("flip_factor", (1.0, -1.0))
@pytest.mark.parametrize("manual_reference", (False, True))
def test_correct(scan_rotation, detector_rotation, flip_factor, manual_reference):
    obj_half_size = 16
    model = correction_model(
        obj_half_size,
        scan_rotation=scan_rotation,
        flip_factor=flip_factor,
        detector_rotation=detector_rotation,
        descan_error=WHOLE_PIXEL_DESCAN_ERROR,
    )
    # Manual reference parameters to check that code path. Should be identical
    # to the default calculated by get_detector_correction_matrix(). Note that
    # this rotates the detector to follow the scan in order to cancel out the
    # scan rotation.
    ref_model = correction_model(obj_half_size, detector_rotation=scan_rotation)
    # Transform the data as if the rotations were 0, no flip, and the descan error was 0.
    obj, projected, out = check_correction(
        obj_half_size,
        model=model,
        ref_model=ref_model if manual_reference else None,
        sim_ref_model=correction_model(obj_half_size),
    )
    center = model.detector_center
    center = (int(center.y), int(center.x))
    # no descan error left: Trace of central pixel is the object
    assert_allclose(obj, out[:, :, *center])
    # Counter-test: Trace of central pixel of simulated dataset with descan
    # error doesn't match the object
    assert not np.allclose(obj, projected[:, :, *center])


@pytest.mark.parametrize("scan_rotation", (0.0, np.pi / 2))
@pytest.mark.parametrize("detector_rotation", (0.0, np.pi / 2))
def test_correct_flip(scan_rotation, detector_rotation):
    obj_half_size = 16
    model = correction_model(
        obj_half_size,
        scan_rotation=scan_rotation,
        detector_rotation=detector_rotation,
        descan_error=WHOLE_PIXEL_DESCAN_ERROR,
    )
    # Manual reference parameters that introduce flip_y and compensate the rotations
    ref_model = correction_model(obj_half_size, flip_factor=-1.0, detector_rotation=scan_rotation)
    # Transform the data as if the rotations were 0, flip_y, and the descan error was 0.
    check_correction(
        obj_half_size,
        model=model,
        ref_model=ref_model,
        sim_ref_model=correction_model(obj_half_size, flip_factor=-1.0),
    )


@pytest.mark.parametrize(
    "scan_rotation, detector_rotation, obj_half_size",
    [
        (0.0, 0.0, 16),
        (0.0, np.pi / 2, 16),
        (np.pi / 2, 0.0, 16),
        (np.pi / 2, np.pi / 2, 16),
        pytest.param(0.0, 0.0, 1, marks=pytest.mark.with_numba),
    ],
)
def test_correct_fixed_manualref(scan_rotation, detector_rotation, obj_half_size):
    scan_shape, _ = correction_shapes(obj_half_size)

    # Fixed mapping from physical to image for forward simulations
    def map_coord(inp):
        cy = scan_shape[0] / 2
        cx = scan_shape[1] / 2
        inp_vec = np.array((inp.y, inp.x))
        y, x = rotate(-np.pi / 2) @ scale(1 / SCAN_PIXEL_PITCH) @ inp_vec
        return PixelYX(y=y + cy + 2, x=x + cx - 3)

    model = correction_model(
        obj_half_size,
        scan_rotation=scan_rotation,
        detector_rotation=detector_rotation,
        descan_error=WHOLE_PIXEL_DESCAN_ERROR,
    )
    base = correction_model(obj_half_size)
    detector_center = PixelYX(y=base.detector_center.y + 2, x=base.detector_center.x - 1)
    # Manual reference parameters that introduce flip_y and compensate the rotations
    ref_model = base.derive(
        # No impact on correction
        scan_pixel_pitch=SCAN_PIXEL_PITCH * 42,
        detector_pixel_pitch=DETECTOR_PIXEL_PITCH * 2,
        # Has no impact since we don't remap the scan dimension,
        # only the projection after the specimen
        scan_rotation=np.pi / 23,
        flip_factor=-1.0,
        detector_center=detector_center,
    )
    # Simulated result with flip_y
    sim_ref_model = base.derive(
        detector_pixel_pitch=DETECTOR_PIXEL_PITCH * 2,
        # Has to match the input scan rotation since we don't remap the scan dimension
        scan_rotation=scan_rotation,
        flip_factor=-1.0,
        detector_center=detector_center,
    )
    # Detector image correction doesn't interfere with how scan positions are mapped
    check_correction(
        obj_half_size,
        model=model,
        ref_model=ref_model,
        sim_ref_model=sim_ref_model,
        specimen_to_image=map_coord,
    )


@pytest.mark.parametrize("overfocus", [0.01, -0.02])
def test_beam_geometry(overfocus):
    detector_pixel_pitch = 0.2
    camera_length = 1.234
    model = Model4DSTEM.default(dataset_shape=(32, 32, 64, 64)).derive(
        overfocus=overfocus,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=0.02,
        detector_rotation=0.3,
    )
    beam_radius = get_primary_beam_radius(model)
    # The beam diverges from the crossover at the source
    assert_allclose(
        beam_radius, 0.02 * abs(overfocus + camera_length) / detector_pixel_pitch, rtol=1e-12
    )
    twothetas = (0.05, 0.1)
    radii = [get_diffraction_pixel_radius(model, twotheta) for twotheta in twothetas]
    # Diffraction at the specimen
    assert_allclose(radii, np.tan(twothetas) * camera_length / detector_pixel_pitch, rtol=1e-12)
    ri, ro = ring_radii(model, twothetas)
    assert_allclose(ri, np.maximum(np.array(radii) - beam_radius, 0), rtol=1e-12)
    assert_allclose(ro, np.array(radii) + beam_radius, rtol=1e-12)


def test_get_center(random_model):
    scan_pos = PixelYX(y=3.0, x=-2.0)
    center = get_center(random_model, scan_pos)
    ref = random_model.trace(scan_pos=scan_pos, source_dx=0.0, source_dy=0.0)
    assert_allclose(center, [float(v) for v in ref["detector"].sampling["detector_px"]], rtol=1e-12)
