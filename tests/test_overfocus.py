import numpy as np
import pytest
from numpy.testing import assert_allclose

from microscope_calibration.util.stem_overfocus_sim import (
    get_transformation_matrix, detector_px_to_specimen_px, project, smiley
)
from microscope_calibration.common.stem_overfocus import (
    OverfocusParams, make_model, get_translation_matrix
)
from microscope_calibration.udf.stem_overfocus import OverfocusUDF
from libertem.api import Context
from libertem.common import Shape


@pytest.mark.parametrize(
    'params', [
        ({'scan_rotation':   0, 'flip_y': False}, ((1, 0), (0, 1))),
        ({'scan_rotation': 180, 'flip_y': False}, ((-1, 0), (0, -1))),
        ({'scan_rotation':  90, 'flip_y': True}, ((0, 1), (1, 0))),
        ({'scan_rotation':  0, 'flip_y': True}, ((-1, 0), (0, 1))),
        (
            {'scan_rotation':  45, 'flip_y': False},
            ((1/np.sqrt(2), 1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)))
        ),
    ]
)
def test_get_transformation_matrix(params):
    inp, ref = params
    sim_params = OverfocusParams(
        overfocus=1,
        scan_pixel_size=1,
        camera_length=1,
        detector_pixel_size=2,
        semiconv=0.004,
        cy=8,
        cx=8,
        scan_rotation=0,
        flip_y=False
    )
    sim_params.update(inp)
    res = get_transformation_matrix(sim_params)
    assert_allclose(res, ref, atol=1e-8)
    for vec in res:
        assert_allclose(np.linalg.norm(vec), 1)


@pytest.mark.parametrize(
    # params are relative to default parameters in function below
    'params', [
        (
            {},
            (0, 0)
        ),
        (
            {
                'y_px': 3.,
                'x_px': -7.,
            },
            (3, -7)
        ),
        (
            {
                'y_px': 3.,
                'x_px': -7.,
                'transformation_matrix': np.array(((-1., 0.), (0., -1.))),
            },
            (-3, 7)
        ),
        (
            {
                'detector_pixel_size': 2,
                'y_px': 3.,
                'x_px': -7.,
            },
            (6, -14)
        ),
        (
            {
                'scan_pixel_size': 0.5,
                'camera_length': 2,
                'y_px': 3.,
                'x_px': -7.,
            },
            (3, -7)
        ),
        (
            {
                'overfocus': 0.1,
                'scan_pixel_size': 0.1,
                'y_px': 3.,
                'x_px': -7.,
            },
            (3, -7)
        ),
        (
            {
                'cy': 1,
                'cx': 5,
                'y_px': 3.,
                'x_px': -7.,
            },
            (2, -12)
        ),
        (
            {
                'y_px': 3.,
                'x_px': -7.,
                'fov_size_y': 4,
                'fov_size_x': 6,
            },
            (5, -4)
        ),
        (
            {
                'overfocus': 1,
                'scan_pixel_size': 0.5,
                'camera_length': 1,
                'detector_pixel_size': 1,
                'cy': 17,
                'cx': 19,
                'y_px': 3.,
                'x_px': -7.,
                'fov_size_y': 4,
                'fov_size_x': 10,
                'transformation_matrix': np.array(((-1., 0.), (0., -1.))),
            },
            # y: (-3 + 17) * 2 + 2
            # x: (7 + 19) * 2 + 5
            (30, 57)
        ),
        (
            {
                'overfocus': 0.1,
                'scan_pixel_size': 0.1,
                'y_px': 3.,
                'x_px': -7.,
                'transformation_matrix': np.array(((-1., 0.), (0., 1.))),
            },
            (-3, -7)
        ),
        (
            {
                'overfocus': 0.1,
                'scan_pixel_size': 0.1,
                'y_px': 3.,
                'x_px': -7.,
                'fov_size_y': 4,
                'fov_size_x': 10,
                'transformation_matrix': np.array(((-1., 0.), (0., 1.))),
            },
            (-1, -2)
        ),
    ]
)
def test_detector_specimen_px(params):
    inp, ref = params
    func_params = {
        'overfocus': 1,
        'scan_pixel_size': 1,
        'camera_length': 1,
        'detector_pixel_size': 1,
        'cy': 0,
        'cx': 0,
        'y_px': 0.,
        'x_px': 0.,
        'fov_size_y': 0,
        'fov_size_x': 0,
        'transformation_matrix': np.array(((1., 0.), (0., 1.))),
    }
    func_params.update(inp)
    res = detector_px_to_specimen_px(**func_params)
    assert_allclose(res, ref, atol=1e-8)


def test_project():
    size = 16
    params = OverfocusParams(
        overfocus=0.1,
        scan_pixel_size=0.1,
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


def test_project_2():
    size = 16
    params = OverfocusParams(
        overfocus=0.1,
        scan_pixel_size=0.1,
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
    params = OverfocusParams(
        overfocus=0.1,
        scan_pixel_size=0.1,
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
    params = OverfocusParams(
        overfocus=0.1,
        scan_pixel_size=0.1,
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
    params = OverfocusParams(
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


def get_ref_translation_matrix(params: OverfocusParams, nav_shape):
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


def test_translation_ref():
    nav_shape = (8, 8)
    sig_shape = (8, 8)

    params = OverfocusParams(
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
    ref_translation_matrix = get_ref_translation_matrix(
        params=params,
        nav_shape=nav_shape,
    )

    model = make_model(params, Shape(nav_shape + sig_shape, sig_dims=2))
    translation_matrix = get_translation_matrix(model)
    assert translation_matrix == pytest.approx(ref_translation_matrix, rel=0.001)


def test_udf_ref():
    params = OverfocusParams(
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
