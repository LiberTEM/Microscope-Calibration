import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.measure import blur_effect

from microscope_calibration.common.stem_overfocus import (
    get_backward_transformation_matrix, get_detector_correction_matrix,
    project_frame_backwards, correct_frame
)
from microscope_calibration.udf.stem_overfocus import OverfocusUDF
from microscope_calibration.common.model import (
    Parameters4DSTEM, Model4DSTEM, PixelYX, DescanError, Result4DSTEM, ResultSection,
    identity, scale, rotate, flip_y
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


def test_project_identity():
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
