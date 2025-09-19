import pytest
from numpy.testing import assert_allclose

import jax.numpy as jnp
import numpy as np

from microscope_calibration.common.stem_overfocus import (
    get_backward_transformation_matrix, get_detector_correction_matrix,
    project_frame_backwards, correct_frame
)
from microscope_calibration.util.stem_overfocus_sim import project
from microscope_calibration.common.model import (
    Parameters4DSTEM, Model4DSTEM, PixelYX, DescanError,
    scale, rotate, trace
)


def test_model_consistency_backproject():
    params = Parameters4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.234,
        camera_length=0.73,
        detector_pixel_pitch=0.0321,
        semiconv=0.023,
        scan_center=PixelYX(x=0.13, y=0.23),
        scan_rotation=0.752,
        flip_y=True,
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
            offsyi=0.37
        )
    )
    mat = get_backward_transformation_matrix(rec_params=params)

    inp = np.array((2, 3, 5, 7, 1))
    out = inp @ mat
    scan_pos = PixelYX(
        y=inp[0],
        x=inp[1],
    )
    source_dy = out[2]
    source_dx = out[3]

    assert_allclose(out[4], 1)
    res = trace(
        params=params, scan_pos=scan_pos, source_dx=source_dx, source_dy=source_dy)
    assert_allclose(out[0], res['detector'].sampling['detector_px'].y, rtol=1e-12, atol=1e-12)
    assert_allclose(out[1], res['detector'].sampling['detector_px'].x, rtol=1e-12, atol=1e-12)
    assert_allclose(inp[2], res['specimen'].sampling['scan_px'].y, rtol=1e-12, atol=1e-12)
    assert_allclose(inp[3], res['specimen'].sampling['scan_px'].x, rtol=1e-12, atol=1e-12)


def test_model_consistency_correct():
    params = Parameters4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.234,
        camera_length=0.73,
        detector_pixel_pitch=0.0321,
        semiconv=0.023,
        scan_center=PixelYX(x=0.13, y=0.23),
        scan_rotation=0.752,
        flip_y=True,
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
            offsyi=0.37
        )
    )
    ref_params = Parameters4DSTEM(
        overfocus=1.1523,
        scan_pixel_pitch=0.4234,
        camera_length=0.7453,
        detector_pixel_pitch=0.03421,
        semiconv=0.042,
        scan_center=PixelYX(x=0.4, y=0.345),
        scan_rotation=0.75,
        flip_y=False,
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
            offsyi=0.373
        )
    )
    mat = get_detector_correction_matrix(rec_params=params, ref_params=ref_params)

    inp = np.array((2, 3, 5, 7, 1))
    out = inp @ mat
    scan_pos = PixelYX(
        y=inp[0],
        x=inp[1],
    )
    source_dy = out[2]
    source_dx = out[3]

    assert_allclose(out[4], 1)
    model = Model4DSTEM.build(params=params, scan_pos=scan_pos)
    ray = model.make_source_ray(source_dx=source_dx, source_dy=source_dy).ray
    res = model.trace(ray)

    ref_model = Model4DSTEM.build(params=ref_params, scan_pos=scan_pos)
    ref_ray = ref_model.make_source_ray(source_dx=source_dx, source_dy=source_dy).ray
    ref_res = ref_model.trace(ref_ray)

    assert_allclose(inp[2], ref_res['detector'].sampling['detector_px'].y, rtol=1e-12, atol=1e-12)
    assert_allclose(inp[3], ref_res['detector'].sampling['detector_px'].x, rtol=1e-12, atol=1e-12)
    assert_allclose(out[0], res['detector'].sampling['detector_px'].y, rtol=1e-12, atol=1e-12)
    assert_allclose(out[1], res['detector'].sampling['detector_px'].x, rtol=1e-12, atol=1e-12)


def test_backproject_identity():
    # 1:1 size mapping between detector and specimen
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.pi/2,
        scan_center=PixelYX(x=7.1, y=16.),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=7.1, y=16.),
        descan_error=DescanError()
    )
    obj = np.random.random((32, 13))
    res = np.zeros_like(obj)
    mat = get_backward_transformation_matrix(rec_params=params)
    project_frame_backwards(
        frame=obj,
        source_semiconv=np.pi/2,
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
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.pi/2,
        scan_center=PixelYX(x=16, y=16.),
        scan_rotation=np.pi/2,
        flip_y=False,
        detector_center=PixelYX(x=16, y=16.),
        detector_rotation=np.pi/2,
        descan_error=DescanError()
    )
    obj = np.random.random((32, 32))
    res = np.zeros_like(obj)
    mat = get_backward_transformation_matrix(rec_params=params)
    project_frame_backwards(
        frame=obj,
        source_semiconv=np.pi/2,
        mat=mat,
        scan_y=16,
        scan_x=16,
        image_out=res,
    )
    assert_allclose(obj, res)


@pytest.mark.parametrize(
    'rotate_scan', (False, True)
)
@pytest.mark.parametrize(
    'rotate_detector', (False, True)
)
@pytest.mark.parametrize(
    'fixed_reference', (False, True)
)
@pytest.mark.parametrize(
    'flip_y', (False, True)
)
def test_backproject_rot90_flip(rotate_scan, rotate_detector, fixed_reference, flip_y):
    # 1:1 size mapping between detector and specimen
    # rotating scan and detector in fixed reference frame and
    # scan reference frame.
    # Projecting into 4D STEM dataset and then back-projecting a
    # detector frame into the reference coordinate system restores the object,
    # i.e. rotation and flip are canceled out.
    if rotate_detector:
        detector_rotation = np.pi/2
    else:
        detector_rotation = 0.
    if rotate_scan:
        scan_rotation = np.pi/2
    else:
        scan_rotation = 0.

    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.pi/2,
        scan_center=PixelYX(x=16, y=16.),
        scan_rotation=scan_rotation,
        flip_y=flip_y,
        detector_center=PixelYX(x=16, y=16.),
        detector_rotation=detector_rotation,
        descan_error=DescanError()
    )

    if fixed_reference:
        def map_coord(inp):
            cy = obj.shape[0] / 2
            cx = obj.shape[1] / 2
            inp_vec = jnp.array((inp.y, inp.x))
            y, x = scale(1) @ inp_vec
            return PixelYX(y=y+cy, x=x+cx)
    else:
        map_coord = None

    obj = np.random.random((32, 32))

    projected = project(
        image=obj,
        scan_shape=((32, 32)),
        detector_shape=((32, 32)),
        sim_params=params,
        specimen_to_image=map_coord,
    )
    mat = get_backward_transformation_matrix(
        rec_params=params,
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
                source_semiconv=np.pi/2,
                mat=mat,
                scan_y=pick_y,
                scan_x=pick_x,
                image_out=res,
            )

            assert_allclose(obj[2:-2, 2:-2], res[2:-2, 2:-2])


def test_backproject_scale_fixed():
    # scan coordinates are 2x detector coordinates relative to object,
    # but we project from and back-project into fixed 1:1 reference coordinates
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=2,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.pi/2,
        scan_center=PixelYX(x=16., y=16.),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=32, y=32.),
        detector_rotation=0.,
        descan_error=DescanError()
    )
    obj = np.random.random((64, 64))
    res = np.zeros((64, 64))

    def map_coord(inp):
        cy = obj.shape[0] / 2
        cx = obj.shape[1] / 2
        inp_vec = jnp.array((inp.y, inp.x))
        y, x = scale(1) @ inp_vec
        return PixelYX(y=y+cy, x=x+cx)

    projected = project(
        image=obj,
        scan_shape=((32, 32)),
        detector_shape=((64, 64)),
        sim_params=params,
        specimen_to_image=map_coord,
    )

    mat = get_backward_transformation_matrix(
        rec_params=params,
        specimen_to_image=map_coord
    )
    project_frame_backwards(
        frame=projected[16, 16],
        source_semiconv=np.pi/2,
        mat=mat,
        scan_y=16,
        scan_x=16,
        image_out=res,
    )

    assert_allclose(obj, res)


def test_backproject_scale_scanref():
    # scan coordinates are 2x detector coordinates,
    # and we project from and back-project into that coordinate system
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=2,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.pi/2,
        scan_center=PixelYX(x=16., y=16.),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=32, y=32.),
        detector_rotation=0.,
        descan_error=DescanError()
    )
    obj = np.random.random((64, 64))
    res = np.zeros((32, 32))

    projected = project(
        image=obj,
        scan_shape=((32, 32)),
        detector_shape=((64, 64)),
        sim_params=params,
        specimen_to_image=None,
    )

    mat = get_backward_transformation_matrix(
        rec_params=params,
        specimen_to_image=None
    )
    project_frame_backwards(
        frame=projected[16, 16],
        source_semiconv=np.pi/2,
        mat=mat,
        scan_y=16,
        scan_x=16,
        image_out=res,
    )
    # The back-projection result corresponds to the trace
    # of the central pixel, i.e. scan coordinates
    assert_allclose(projected[:, :, 32, 32], res)


@pytest.mark.parametrize(
    'scan_rotation', (0., np.pi/2)
)
@pytest.mark.parametrize(
    'detector_rotation', (0., np.pi/2)
)
@pytest.mark.parametrize(
    'flip_y', (False, True)
)
@pytest.mark.parametrize(
    'manual_reference', (False, True)
)
def test_correct(scan_rotation, detector_rotation, flip_y, manual_reference):
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 1.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 16
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2 + 0.00314157, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_y=flip_y,
        # Simulate detector larger than object to avoid clipping at the borders
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=detector_rotation,
        # Descan error designed to give whole pixel shifts
        descan_error=DescanError(
            offpxi=detector_pixel_pitch,
            offpyi=detector_pixel_pitch * 2,
            offsxi=-1 * detector_pixel_pitch/camera_length,
            offsyi=-2 * detector_pixel_pitch/camera_length,
            pxo_pxi=2 * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=3 * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=-3 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=-4 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        )
    )
    # Manual reference parametes to check that code path
    # Should be identical to the default calculated by get_detector_correction_matrix()
    # Note that this rotates the detector to follow the scan in order to cancel out
    # the scan rotation.
    params_ref_manual = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=scan_rotation,
        descan_error=DescanError()
    )
    # Parameters for simulated result without aberrations
    params_ref_sim = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=0.,
        descan_error=DescanError()
    )
    # Obtain correction matrix for 4D STEM dataset, i.e. transform the data as
    # if the rotations were 0, no flip, and the descan error was 0.
    mat = get_detector_correction_matrix(
        rec_params=params,
        ref_params=params_ref_manual if manual_reference else None,
    )
    obj = np.random.random((obj_half_size * 2, obj_half_size * 2))
    projected = project(
        image=obj,
        detector_shape=(obj_half_size * 4, obj_half_size * 4),
        scan_shape=(obj_half_size * 2, obj_half_size * 2),
        sim_params=params,
    )
    # Calculate corrected 4D STEM dataset, i.e. as if the rotations were 0,
    # no flip, and the descan error was 0.
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
        detector_shape=(obj_half_size * 4, obj_half_size * 4),
        scan_shape=(obj_half_size * 2, obj_half_size * 2),
        sim_params=params_ref_sim,
    )
    # 100 % match between corrected frames and simulated reference without aberrations
    assert_allclose(projected_ref, out)
    # no descan error left: Trace of central pixel is the object
    assert_allclose(obj, out[:, :, obj_half_size * 2, obj_half_size * 2])
    # Counter-test: Trace of central pixel of simulate ddataset with descan
    # error doesn't match the object
    assert not np.allclose(obj, projected[:, :, obj_half_size * 2, obj_half_size * 2])


@pytest.mark.parametrize(
    'scan_rotation', (0., np.pi/2)
)
@pytest.mark.parametrize(
    'detector_rotation', (0., np.pi/2)
)
def test_correct_flip(scan_rotation, detector_rotation):
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 1.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 16
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2 + 0.00314157, propagation_distance)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_y=False,
        # Simulate detector larger than object to avoid clipping at the borders
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=detector_rotation,
        # Descan error designed to give whole pixel shifts
        descan_error=DescanError(
            offpxi=detector_pixel_pitch,
            offpyi=detector_pixel_pitch * 2,
            offsxi=-1 * detector_pixel_pitch/camera_length,
            offsyi=-2 * detector_pixel_pitch/camera_length,
            pxo_pxi=2 * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=3 * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=-3 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=-4 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        )
    )
    # Manual reference parametes that introduce flip_y
    # and compensate the rotations
    params_ref_manual = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=0.,
        flip_y=True,
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=scan_rotation,
        descan_error=DescanError()
    )
    # Parameters for simulated result with flip_y
    params_ref_sim = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=0.,
        flip_y=True,
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=0.,
        descan_error=DescanError()
    )
    # Obtain correction matrix for 4D STEM dataset that transforms the data as
    # if the rotations were 0, flip_y, and the descan error was 0.
    mat = get_detector_correction_matrix(
        rec_params=params,
        ref_params=params_ref_manual,
    )
    obj = np.random.random((obj_half_size * 2, obj_half_size * 2))
    projected = project(
        image=obj,
        detector_shape=(obj_half_size * 4, obj_half_size * 4),
        scan_shape=(obj_half_size * 2, obj_half_size * 2),
        sim_params=params,
    )
    # Calculate corrected 4D STEM dataset with the rotations were 0,
    # flip, and no descan error.
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
        detector_shape=(obj_half_size * 4, obj_half_size * 4),
        scan_shape=(obj_half_size * 2, obj_half_size * 2),
        sim_params=params_ref_sim,
    )
    # 100 % match between corrected frames and simulated reference without aberrations
    assert_allclose(projected_ref, out)


@pytest.mark.parametrize(
    'scan_rotation', (0., np.pi/2)
)
@pytest.mark.parametrize(
    'detector_rotation', (0., np.pi/2)
)
def test_correct_fixed_manualref(scan_rotation, detector_rotation):
    scan_pixel_pitch = 0.1
    detector_pixel_pitch = 0.2
    overfocus = 1.
    camera_length = 1.
    propagation_distance = overfocus + camera_length
    obj_half_size = 16
    angle = np.arctan2(obj_half_size*detector_pixel_pitch/2 + 0.00314157, propagation_distance)

    # Fixed mapping from physical to image for forward simulations
    def map_coord(inp):
        cy = obj.shape[0] / 2
        cx = obj.shape[1] / 2
        inp_vec = jnp.array((inp.y, inp.x))
        y, x = rotate(-np.pi/2) @ scale(1/scan_pixel_pitch) @ inp_vec
        return PixelYX(y=y+cy + 2, x=x+cx - 3)

    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        scan_rotation=scan_rotation,
        flip_y=False,
        # Simulate detector larger than object to avoid clipping at the borders
        detector_center=PixelYX(x=obj_half_size * 2, y=obj_half_size * 2),
        detector_rotation=detector_rotation,
        # Descan error designed to give whole pixel shifts
        descan_error=DescanError(
            offpxi=detector_pixel_pitch,
            offpyi=detector_pixel_pitch * 2,
            offsxi=-1 * detector_pixel_pitch/camera_length,
            offsyi=-2 * detector_pixel_pitch/camera_length,
            pxo_pxi=2 * detector_pixel_pitch/scan_pixel_pitch,
            pyo_pyi=3 * detector_pixel_pitch/scan_pixel_pitch,
            sxo_pxi=-3 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
            syo_pyi=-4 * detector_pixel_pitch/scan_pixel_pitch/camera_length,
        )
    )
    # Manual reference parametes that introduce flip_y
    # and compensate the rotations
    params_ref_manual = Parameters4DSTEM(
        overfocus=overfocus,
        # No impact on correction
        scan_pixel_pitch=scan_pixel_pitch * 42,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch * 2,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        # Has no impact since we don't remap the scan dimension,
        # only the projection after the specimen
        scan_rotation=np.pi/23,
        flip_y=True,
        detector_center=PixelYX(x=obj_half_size * 2 - 1, y=obj_half_size * 2 + 2),
        detector_rotation=0.,
        descan_error=DescanError()
    )
    # Parameters for simulated result with flip_y
    params_ref_sim = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=scan_pixel_pitch,
        camera_length=camera_length,
        detector_pixel_pitch=detector_pixel_pitch * 2,
        semiconv=angle,
        scan_center=PixelYX(x=obj_half_size, y=obj_half_size),
        # Has to match the input scan rotation since we don't
        # remap the scan dimension
        scan_rotation=scan_rotation,
        flip_y=True,
        detector_center=PixelYX(x=obj_half_size * 2 - 1, y=obj_half_size * 2 + 2),
        detector_rotation=0.,
        descan_error=DescanError()
    )
    # Obtain correction matrix for 4D STEM dataset that transforms the data as
    # if the rotations were 0, flip_y, and the descan error was 0.
    mat = get_detector_correction_matrix(
        rec_params=params,
        ref_params=params_ref_manual,
    )
    obj = np.random.random((obj_half_size * 2, obj_half_size * 2))
    projected = project(
        image=obj,
        detector_shape=(obj_half_size * 4, obj_half_size * 4),
        scan_shape=(obj_half_size * 2, obj_half_size * 2),
        sim_params=params,
        # Detector image correction doesn't interfere with
        # how scan positions are mapped
        specimen_to_image=map_coord,
    )
    # Calculate corrected 4D STEM dataset with the rotations were 0,
    # flip, and no descan error.
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
        detector_shape=(obj_half_size * 4, obj_half_size * 4),
        scan_shape=(obj_half_size * 2, obj_half_size * 2),
        sim_params=params_ref_sim,
        # Detector image correction doesn't interfere with
        # how scan positions are mapped, so we have to use the same mapping here
        specimen_to_image=map_coord,
    )
    # 100 % match between corrected frames and simulated reference without aberrations
    assert_allclose(projected_ref, out)
