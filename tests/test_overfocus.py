import pytest
from numpy.testing import assert_allclose

import jax.numpy as jnp
import numpy as np
from skimage.measure import blur_effect

from microscope_calibration.common.stem_overfocus import (
    get_backward_transformation_matrix, get_detector_correction_matrix,
    project_frame_backwards, correct_frame
)
from microscope_calibration.util.stem_overfocus_sim import project
from microscope_calibration.udf.stem_overfocus import OverfocusUDF
from microscope_calibration.common.model import (
    Parameters4DSTEM, Model4DSTEM, PixelYX, DescanError, Result4DSTEM, ResultSection,
    identity, scale, rotate
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
    model = Model4DSTEM.build(params=params, scan_pos=scan_pos)
    ray = model.make_source_ray(source_dx=source_dx, source_dy=source_dy).ray
    res = model.trace(ray)
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
        scan_y=7,
        scan_x=16,
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


def test_project():
    size = 16
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_size=0.5,
        camera_length=1,
        detector_pixel_size=1,
        semiconv=0.004,
        cy=size/2,
        cx=size/2,
        scan_rotation=0,
        flip_y=False
    )
    obj = smiley(size)
    projected = project(
        image=obj,
        scan_shape=(size, size),
        detector_shape=(size, size),
        sim_params=params,
    )
    assert_allclose(obj, projected[size//2, size//2])
    assert_allclose(obj, projected[:, :, size//2, size//2])


def test_project_zerocl():
    # Camera length is zero, 1:1 match of scan and detector
    size = 16
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_size=1,
        camera_length=0,
        detector_pixel_size=1,
        semiconv=0.004,
        cy=size/2,
        cx=size/2,
        scan_rotation=0,
        flip_y=False
    )
    obj = smiley(size)
    projected = project(
        image=obj,
        scan_shape=(size, size),
        detector_shape=(size, size),
        sim_params=params,
    )
    assert_allclose(obj, projected[size//2, size//2])
    assert_allclose(obj, projected[:, :, size//2, size//2])


def test_project_scale():
    # With overfocus == 1 and cl == 1, the image is 2x magnified on the
    # detector. With same pixel size and twice the number of pixels, every
    # second detector pixel maps to a single scan pixel
    size = 16
    detector_size = 2 * size
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_size=1,
        camera_length=1,
        detector_pixel_size=1,
        semiconv=0.004,
        cy=detector_size/2,
        cx=detector_size/2,
        scan_rotation=0,
        flip_y=False
    )
    obj = smiley(size)
    projected = project(
        image=obj,
        scan_shape=(size, size),
        detector_shape=(detector_size, detector_size),
        sim_params=params,
    )
    # Center of the scan, every second detector pixel
    assert_allclose(obj, projected[size//2, size//2, ::2, ::2])
    # Scan area, trace of center of detector
    assert_allclose(obj, projected[:, :, detector_size//2, detector_size//2])


def test_project_2():
    size = 16
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_size=0.5,
        camera_length=1,
        detector_pixel_size=1,
        semiconv=0.004,
        cy=size/2 + 3,
        cx=size/2 - 7,
        scan_rotation=0,
        flip_y=False
    )
    obj = smiley(size)
    projected = project(
        image=obj,
        scan_shape=(size, size),
        detector_shape=(size, size),
        sim_params=params,
    )
    assert_allclose(obj, projected[size//2 + 3, size//2 - 7])
    assert_allclose(obj, projected[:, :, size//2 + 3, size//2 - 7])


def test_project_3():
    size = 16
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_size=0.5,
        camera_length=1,
        detector_pixel_size=0.5,
        semiconv=0.004,
        cy=size/2,
        cx=size/2,
        scan_rotation=0,
        flip_y=False
    )
    obj = smiley(size)
    projected = project(
        image=obj,
        scan_shape=(size, size),
        detector_shape=(size, size),
        sim_params=params,
    )
    assert_allclose(
        obj[size//4:size//4*3, size//4:size//4*3],
        projected[size//2, size//2, ::2, ::2]
    )
    assert_allclose(obj, projected[:, :, size//2, size//2])


def test_project_rotate():
    size = 16
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_size=0.5,
        camera_length=1,
        detector_pixel_size=1,
        semiconv=0.004,
        cy=size/2,
        cx=size/2,
        scan_rotation=180,
        flip_y=False
    )
    obj = smiley(size)
    projected = project(
        image=obj,
        scan_shape=(size, size),
        detector_shape=(size, size),
        sim_params=params,
    )
    # Rotated around "pixel corner", so shifted by 1
    assert_allclose(obj, projected[size//2 - 1, size//2 - 1, ::-1, ::-1])
    assert_allclose(obj, projected[:, :, size//2, size//2])


def test_project_odd():
    det_y = 29
    det_x = 31
    scan_y = 17
    scan_x = 13
    obj_y = 19
    obj_x = 23
    size = 32
    params = Parameters4DSTEM(
        overfocus=0.01,
        scan_pixel_size=0.01,
        camera_length=1.,
        detector_pixel_size=1,
        semiconv=0.004,
        cy=det_y/2,
        cx=det_x/2,
        scan_rotation=0.,
        flip_y=False
    )
    obj = smiley(size)[:obj_y, :obj_x]
    projected = project(
        image=obj,
        scan_shape=(scan_y, scan_x),
        detector_shape=(det_y, det_x),
        sim_params=params,
    )
    dy = (obj_y-scan_y)//2
    dx = (obj_x - scan_x)//2
    assert_allclose(obj[dy:scan_y+dy, dx:scan_x+dx], projected[:, :, det_y//2, det_x//2])
    dy = (det_y - obj_y) // 2
    dx = (det_x - obj_x) // 2
    assert_allclose(obj, projected[scan_y//2, scan_x//2, dy:obj_y+dy, dx:obj_x+dx])


def get_ref_translation_matrix(params: Parameters4DSTEM, nav_shape):
    a = []
    b = []

    for det_y in (0, 1):
        for det_x in (0, 1):
            spec_y, spec_x = detector_px_to_specimen_px(
                y_px=float(det_y),
                x_px=float(det_x),
                fov_size_y=float(nav_shape[0]),
                fov_size_x=float(nav_shape[1]),
                transformation_matrix=get_transformation_matrix(params),
                cy=params['cy'],
                cx=params['cx'],
                detector_pixel_size=float(params['detector_pixel_size']),
                scan_pixel_size=float(params['scan_pixel_size']),
                camera_length=float(params['camera_length']),
                overfocus=float(params['overfocus']),
            )
            # Code lifted from util.stem_overfocus_sim._project
            for scan_y in (0, 1):
                for scan_x in (0, 1):
                    offset_y = scan_y - nav_shape[0] / 2
                    offset_x = scan_x - nav_shape[1] / 2
                    image_px_y = spec_y + offset_y
                    image_px_x = spec_x + offset_x
                    a.append((
                        image_px_y,
                        image_px_x,
                        scan_y,
                        scan_x,
                        1
                    ))
                    b.append((det_y, det_x))
    res = np.linalg.lstsq(a, b, rcond=None)
    return res[0]


class RefOverfocusUDF(OverfocusUDF):
    def get_task_data(self):
        overfocus_params = self.params.overfocus_params
        translation_matrix = get_ref_translation_matrix(
            params=overfocus_params,
            nav_shape=self._get_fov()
        )
        select_roi = np.zeros(self.meta.dataset_shape.nav, dtype=bool)
        nav_y, nav_x = self.meta.dataset_shape.nav
        select_roi[nav_y//2, nav_x//2] = True
        return {
            'translation_matrix': translation_matrix,
            'select_roi': select_roi
        }


@pytest.mark.parametrize(
    # make sure the test is sensitive enough
    'fail', [False, True]
)
def test_translation_ref(fail):
    fail_factor = 1.001 if fail else 1

    nav_shape = (8, 8)
    sig_shape = (8, 8)

    params = Parameters4DSTEM(
        overfocus=0.0001,
        scan_pixel_size=0.00000001,
        camera_length=1,
        detector_pixel_size=0.0001,
        semiconv=0.01,
        cy=3,
        cx=3,
        scan_rotation=33.3,
        flip_y=True,
    )
    fail_params = params.copy()
    fail_params['overfocus'] /= fail_factor
    fail_params['scan_pixel_size'] *= fail_factor
    fail_params['camera_length'] *= fail_factor
    fail_params['detector_pixel_size'] /= fail_factor
    fail_params['cy'] *= fail_factor
    fail_params['cx'] /= fail_factor
    fail_params['scan_rotation'] *= fail_factor

    ref_translation_matrix = get_ref_translation_matrix(
        params=fail_params,
        nav_shape=nav_shape,
    )

    model = make_model(params, Shape(nav_shape + sig_shape, sig_dims=2))
    translation_matrix = get_translation_matrix(model)
    if fail:
        with pytest.raises(AssertionError):
            assert translation_matrix == pytest.approx(ref_translation_matrix, rel=0.001)
    else:
        assert translation_matrix == pytest.approx(ref_translation_matrix, rel=0.001)


def test_udf_ref():
    params = Parameters4DSTEM(
        overfocus=0.0001,
        scan_pixel_size=0.00000001,
        camera_length=1,
        detector_pixel_size=0.0001,
        semiconv=0.001,
        cy=3.,
        cx=3.,
        scan_rotation=0,
        flip_y=False
    )
    obj = np.zeros((8, 8))
    obj[3, 3] = 1
    sim = project(obj, scan_shape=(8, 8), detector_shape=(8, 8), sim_params=params)
    assert sim[3, 3, 3, 3] == 1

    ctx = Context.make_with('inline')
    ds = ctx.load('memory', data=sim)

    ref_udf = RefOverfocusUDF(params)
    res_udf = OverfocusUDF(params)

    res = ctx.run_udf(dataset=ds, udf=(ref_udf, res_udf))
    assert_allclose(res[0]['shifted_sum'].data.astype(bool), obj.astype(bool))
    assert_allclose(res[1]['shifted_sum'].data.astype(bool), obj.astype(bool))


def test_optimize():
    params = Parameters4DSTEM(
        overfocus=0.0001,
        scan_pixel_size=0.00000001,
        camera_length=1,
        detector_pixel_size=0.0001,
        semiconv=np.pi,
        cy=3.,
        cx=3.,
        scan_rotation=0,
        flip_y=False
    )
    obj = np.zeros((8, 8))
    obj[3, 3] = 1
    sim = project(obj, scan_shape=(8, 8), detector_shape=(8, 8), sim_params=params)
    ctx = Context.make_with('inline')
    ds = ctx.load('memory', data=sim)
    ref_udf = RefOverfocusUDF(params)
    make_new_params, loss = make_overfocus_loss_function(
        params=params,
        ctx=ctx,
        dataset=ds,
        overfocus_udf=ref_udf,
    )
    res = optimize(loss=loss)
    res_params = make_new_params(res.x)
    assert_allclose(res_params['scan_rotation'], params['scan_rotation'], atol=0.1)
    assert_allclose(res_params['overfocus'], params['overfocus'], rtol=0.1)

    valdict = {'val': False}

    def callback(args, new_params, udf_results, current_loss):
        if valdict['val']:
            pass
        else:
            valdict['val'] = True
            assert_allclose(args, [0, 0])
            assert params == new_params
            assert_allclose(udf_results[0]['shifted_sum'].data.astype(bool), obj.astype(bool))

    make_new_params, loss = make_overfocus_loss_function(
        params=params,
        ctx=ctx,
        dataset=ds,
        overfocus_udf=ref_udf,
        callback=callback,
        blur_function=blur_effect,
        extra_udfs=(OverfocusUDF(params), ),
        plots=(),
    )
    res = optimize(
        loss=loss, minimizer_kwargs={'method': 'SLSQP'},
        bounds=[(-10, 10), (-10, 10)],
    )
    res_params = make_new_params(res.x)
    assert_allclose(res_params['scan_rotation'], params['scan_rotation'], atol=0.1)
    assert_allclose(res_params['overfocus'], params['overfocus'], rtol=0.1)
