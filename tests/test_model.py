import pytest
from numpy.testing import assert_allclose

import jax_dataclasses as jdc
from libertem.udf.com import guess_corrections
import numpy as np

from temgym_core.ray import Ray
from temgym_core.components import DescanError, Component
from temgym_core.source import Source
from temgym_core import PixelsYX

from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM


def test_params():
    params = Parameters4DSTEM(
        overfocus=0.7,
        scan_pixel_pitch=0.005,
        scan_cy=17,
        scan_cx=13,
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_cy=11,
        detector_cx=19,
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=13, x=7))
    assert model.params == params


def test_trace_smoke():
    params = Parameters4DSTEM(
        overfocus=0.7,
        scan_pixel_pitch=0.005,
        scan_cy=17,
        scan_cx=13,
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_cy=11,
        detector_cx=19,
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    model.trace(ray=ray)


def test_trace_focused_smoke():
    params = Parameters4DSTEM(
        overfocus=0.,
        scan_pixel_pitch=0.005,
        scan_cy=17,
        scan_cx=13,
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_cy=11,
        detector_cx=19,
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    model.trace(ray=ray)


def test_trace_noproject_smoke():
    params = Parameters4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.005,
        scan_cy=17,
        scan_cx=13,
        scan_rotation=1.234,
        camera_length=0.,
        detector_pixel_pitch=0.0247,
        detector_cy=11,
        detector_cx=19,
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    model.trace(ray=ray)


def test_trace_underfocused_smoke():
    params = Parameters4DSTEM(
        overfocus=-0.23,
        scan_pixel_pitch=0.005,
        scan_cy=17,
        scan_cx=13,
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_cy=11,
        detector_cx=19,
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    model.trace(ray=ray)


# Beam straight along the optical axis, no scan deflection, scan and detector
# coordinate system identical with physical coordinates.
def test_straight():
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_cy=0.,
        scan_cx=0.,
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_cy=0.,
        detector_cx=0.,
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=0., x=0.))
    ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
    res = model.trace(ray=ray)

    for key, sect in res.items():
        if isinstance(sect.component, Component) or isinstance(sect.component, Source):
            assert sect.component.z == sect.ray.z
            assert sect.component.z == sect.ray.pathlength
        assert sect.ray.x == 0.
        assert sect.ray.y == 0.
    assert res['detector'].ray.z == params.overfocus + params.camera_length
    assert res['source'].ray.z == 0.
    assert res['specimen'].sampling['scan_px'] == PixelsYX(x=0., y=0.)
    assert res['detector'].sampling['detector_px'] == PixelsYX(x=0., y=0.)


# Scan deflection test: beam is shifted
@pytest.mark.parametrize(
    'dy', (-0.2, 0., 0.34)
)
@pytest.mark.parametrize(
    'dx', (-0.7, 0., 0.42)
)
@pytest.mark.parametrize(
    'scan_y', (-17, 0., 23.4)
)
@pytest.mark.parametrize(
    'scan_x', (-23, 0., 29)
)
def test_scan(dy, dx, scan_y, scan_x):
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_cy=0.,
        scan_cx=0.,
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_cy=0.,
        detector_cx=0.,
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model_straight = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=0., x=0.))
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=scan_y, x=scan_x))

    ray_straight = model_straight.make_source_ray(source_dx=dx, source_dy=dy).ray
    ray = model.make_source_ray(source_dx=dx, source_dy=dy).ray
    assert ray == ray_straight

    res_straight = model_straight.trace(ray=ray)
    res = model.trace(ray=ray)

    for key in res.keys():
        sect = res[key]
        sect_straight = res_straight[key]
        assert sect.ray.z == sect_straight.ray.z
        assert sect.ray.pathlength == sect_straight.ray.pathlength
        if isinstance(sect.component, Component) or isinstance(sect.component, Source):
            assert sect.component.z == sect.ray.z
            assert sect.component.z == sect.ray.pathlength
        # Beam is deflected
        if key in ('scanner', 'specimen'):
            assert sect.ray.x - sect_straight.ray.x == scan_x
            assert sect.ray.y - sect_straight.ray.y == scan_y
        # Beam is not deflected
        else:
            assert_allclose(sect.ray.x, sect_straight.ray.x)
            assert_allclose(sect.ray.y, sect_straight.ray.y)
            # Ray propagates straight
            assert_allclose(sect.ray.x, sect.ray.z * dx)
            assert_allclose(sect.ray.y, sect.ray.z * dy)
    assert_allclose(res['detector'].ray.z, params.overfocus + params.camera_length)
    assert_allclose(res['source'].ray.z, 0.)
    # Correct scan deflection
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelsYX(
            x=scan_x + res_straight['specimen'].sampling['scan_px'].x,
            y=scan_y + res_straight['specimen'].sampling['scan_px'].y
        )
    )
    # check physical coords equals pixel coords
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelsYX(
            x=res['specimen'].ray.x,
            y=res['specimen'].ray.y,
        )
    )
    assert_allclose(res['detector'].sampling['detector_px'], PixelsYX(
        x=dx*(params.overfocus + params.camera_length),
        y=dy*(params.overfocus + params.camera_length)
    ))
    # check physical coords equals pixel coords
    assert_allclose(
        res['detector'].sampling['detector_px'],
        PixelsYX(
            x=res['detector'].ray.x,
            y=res['detector'].ray.y,
        )
    )


# detector coordinate systems
@pytest.mark.parametrize(
    'detector_cy', (-0.11, 0., 29)
)
@pytest.mark.parametrize(
    'detector_cx', (-0.32, 0., 43)
)
@pytest.mark.parametrize(
    'detector_pixel_pitch', (0.09, 1., 1.53)
)
@pytest.mark.parametrize(
    'flip_y', (True, False)
)
def test_detector_coordinate_shift_scale_flip(
        detector_cy, detector_cx, detector_pixel_pitch, flip_y):
    scan_cy = -0.7
    scan_cx = 23.
    scan_pixel_pitch = 1.34
    dy = -0.2
    dx = 0.42
    scan_y = -17
    scan_x = 29
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=scan_pixel_pitch,
        scan_cy=scan_cy,
        scan_cx=scan_cx,
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=detector_pixel_pitch,
        detector_cy=detector_cy,
        detector_cx=detector_cx,
        semiconv=0.023,
        flip_y=flip_y,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=scan_y, x=scan_x))

    ray = model.make_source_ray(source_dx=dx, source_dy=dy).ray

    res = model.trace(ray=ray)
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelsYX(
            x=res['specimen'].ray.x/scan_pixel_pitch + scan_cx,
            y=res['specimen'].ray.y/scan_pixel_pitch + scan_cy,
        ),
        rtol=1e-6, atol=1e-6
    )
    flip_factor = -1. if flip_y else 1.
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['detector'].sampling['detector_px'],
        PixelsYX(
            x=res['detector'].ray.x/detector_pixel_pitch + detector_cx,
            y=flip_factor*(res['detector'].ray.y/detector_pixel_pitch + flip_factor*detector_cy),
        ),
        rtol=1e-6, atol=1e-6
    )


# scan coordinate systems
@pytest.mark.parametrize(
    'scan_cy', (-0.7, 0., 21)
)
@pytest.mark.parametrize(
    'scan_cx', (-0.22, 0., 23)
)
@pytest.mark.parametrize(
    'scan_pixel_pitch', (0.07, 1., 1.34)
)
def test_scan_coordinate_shift_scale(scan_cy, scan_cx, scan_pixel_pitch):
    detector_cy = -11.
    detector_cx = 43.
    detector_pixel_pitch = 0.09
    flip_y = True
    dy = -0.2
    dx = 0.42
    scan_y = -17
    scan_x = 29
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=scan_pixel_pitch,
        scan_cy=scan_cy,
        scan_cx=scan_cx,
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=detector_pixel_pitch,
        detector_cy=detector_cy,
        detector_cx=detector_cx,
        semiconv=0.023,
        flip_y=flip_y,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelsYX(y=scan_y, x=scan_x))

    ray = model.make_source_ray(source_dx=dx, source_dy=dy).ray

    res = model.trace(ray=ray)
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelsYX(
            x=res['specimen'].ray.x/scan_pixel_pitch + scan_cx,
            y=res['specimen'].ray.y/scan_pixel_pitch + scan_cy,
        ),
        rtol=1e-6, atol=1e-6
    )
    flip_factor = -1. if flip_y else 1.
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['detector'].sampling['detector_px'],
        PixelsYX(
            x=res['detector'].ray.x/detector_pixel_pitch + detector_cx,
            y=flip_factor*res['detector'].ray.y/detector_pixel_pitch + detector_cy,
        ),
        rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    'scan_rotation', (-1.345, 0, 0.987)
)
@pytest.mark.parametrize(
    'flip_y', (False, True)
)
@pytest.mark.parametrize(
    'detector_cy', (-13, 0., 7)
)
@pytest.mark.parametrize(
    'detector_cx', (-11, 0., 5)
)
def test_com_validation(scan_rotation, flip_y, detector_cy, detector_cx):
    @jdc.pytree_dataclass
    class PointChargeComponent(Component):
        z: float

        def __call__(self, ray: Ray) -> Ray:
            distance = np.linalg.norm(np.array((ray.y, ray.x)))
            if distance > 1e-6:
                # field strength is 1/distance**2,
                # additionally normalize displacement to unit vector
                dx = -ray.x / distance**3 * 1e-2
                dy = -ray.y / distance**3 * 1e-2
                return ray.derive(dx=ray.dx+dx, dy=ray.dy+dy)
            else:
                return ray

    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_cy=0.,
        scan_cx=0.,
        scan_rotation=scan_rotation,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_cy=detector_cy,
        detector_cx=detector_cx,
        semiconv=0.023,
        flip_y=flip_y,
        descan_error=DescanError()
    )

    y_deflections = np.linspace(start=-1, stop=1, num=3)
    x_deflections = np.linspace(start=-1, stop=1, num=3)
    com = np.empty((len(y_deflections), len(x_deflections), 2))
    for y, scan_y in enumerate(y_deflections):
        for x, scan_x in enumerate(x_deflections):
            model = Model4DSTEM.build(
                params=params,
                scan_pos=PixelsYX(x=float(scan_x), y=float(scan_y)),
                specimen=PointChargeComponent(z=params.overfocus)
            )
            ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
            res = model.trace(ray)
            com[y, x, 0] = res['detector'].sampling['detector_px'].y
            com[y, x, 1] = res['detector'].sampling['detector_px'].x

    guess_result = guess_corrections(
        y_centers=com[..., 0],
        x_centers=com[..., 1],
    )
    assert_allclose(guess_result.scan_rotation / 180 * np.pi, scan_rotation, atol=5e-2, rtol=5e-2)
    assert guess_result.flip_y == flip_y
    assert_allclose(guess_result.cy, detector_cy, atol=5e-3, rtol=1e-3)
    assert_allclose(guess_result.cx, detector_cx, atol=5e-3, rtol=1e-3)
