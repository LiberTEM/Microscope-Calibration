from numpy.testing import assert_allclose

import numpy as np

from microscope_calibration.util.stem_overfocus_sim import (
    get_forward_transformation_matrix, project_frame_forward,
    project
)
from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM, PixelYX, DescanError


def test_project_frame_forward():
    for repeat in range(10):
        scan_y = np.random.random()
        scan_x = np.random.random()
        semiconv = np.random.random()

        def ref_project(obj, source_semiconv, mat, scan_y, scan_x, out):
            for det_y in range(out.shape[0]):
                for det_x in range(out.shape[1]):
                    inp = np.array((scan_y, scan_x, det_y, det_x, 1.))
                    spec_y, spec_x, tilt_y, tilt_x, _one = inp @ mat
                    if np.linalg.norm((tilt_y, tilt_x)) < source_semiconv:
                        spec_y = int(np.round(spec_y))
                        spec_x = int(np.round(spec_x))
                        if (
                                spec_y >= 0 and spec_y < obj.shape[0]
                                and spec_x >= 0 and spec_x < obj.shape[1]):
                            out[det_y, det_x] = obj[spec_y, spec_x]
                        else:
                            out[det_y, det_x] = 0.

        mat = np.random.random((5, 5))
        obj = np.random.random((13, 17))
        out = np.empty((19, 23))
        out_ref = out.copy()

        project_frame_forward(
            obj=obj,
            source_semiconv=semiconv,
            mat=mat,
            scan_y=scan_y,
            scan_x=scan_x,
            out=out
        )
        ref_project(
            obj=obj,
            source_semiconv=semiconv,
            mat=mat,
            scan_y=scan_y,
            scan_x=scan_x,
            out=out_ref
        )
        assert_allclose(out, out_ref)


def test_model_consistency():
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
    mat = get_forward_transformation_matrix(sim_params=params)

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
    assert_allclose(inp[2], res['detector'].sampling['detector_px'].y, rtol=1e-6, atol=1e-6)
    assert_allclose(inp[3], res['detector'].sampling['detector_px'].x, rtol=1e-6, atol=1e-6)
    assert_allclose(out[0], res['specimen'].sampling['scan_px'].y, rtol=1e-6, atol=1e-6)
    assert_allclose(out[1], res['specimen'].sampling['scan_px'].x, rtol=1e-6, atol=1e-6)


def test_project_identity():
    # 1:1 size mapping between detector and specimen
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=2,
        semiconv=np.inf,
        scan_center=PixelYX(x=6.9, y=16.),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=7.1, y=16.),
        descan_error=DescanError()
    )
    obj = np.random.random((32, 13))
    res = project(
        image=obj,
        detector_shape=(32, 13),
        scan_shape=(32, 13),
        sim_params=params,
    )
    assert_allclose(obj, res[16, 7])


def test_project_scale():
    # 1:2 upscaling on the detector
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        camera_length=1,
        detector_pixel_pitch=1,
        semiconv=np.inf,
        scan_center=PixelYX(x=16, y=16.),
        scan_rotation=0.,
        flip_y=False,
        detector_center=PixelYX(x=32, y=32.),
        descan_error=DescanError()
    )
    obj = np.random.random((32, 32))
    res = project(
        image=obj,
        detector_shape=(64, 64),
        scan_shape=(32, 32),
        sim_params=params,
    )
    assert_allclose(obj, res[16, 16, ::2, ::2])
