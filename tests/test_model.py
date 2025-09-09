import pytest
from numpy.testing import assert_allclose

import jax_dataclasses as jdc
from libertem.udf.com import guess_corrections, apply_correction
import numpy as np

from temgym_core.ray import Ray
from temgym_core.components import DescanError, Component
from temgym_core.propagator import Propagator
from temgym_core.source import Source
from temgym_core import PixelYX

from microscope_calibration.common.model import Parameters4DSTEM, Model4DSTEM


def test_params():
    params = Parameters4DSTEM(
        overfocus=0.7,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError(offpxi=.345, pxo_pxi=948)
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=13, x=7))
    assert model.params == params


def test_trace_smoke():
    params = Parameters4DSTEM(
        overfocus=0.7,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    res = model.trace(ray=ray)
    keys = (
        'source', 'overfocus', 'scanner', 'specimen', 'descanner',
        'camera_length', 'detector'
    )
    for key in keys:
        assert key in res
        sect = res[key]
        assert isinstance(sect.ray, Ray)
    components = ('scanner', 'specimen', 'descanner', 'detector')
    propagators = ('camera_length', 'camera_length')
    for key in components:
        sect = res[key]
        assert isinstance(sect.component, Component)
    for key in propagators:
        sect = res[key]
        assert isinstance(sect.component, Propagator)
    assert isinstance(res['source'].component, Source)
    assert isinstance(res['specimen'].sampling['scan_px'], PixelYX)
    assert isinstance(res['detector'].sampling['detector_px'], PixelYX)


def test_trace_focused():
    params = Parameters4DSTEM(
        overfocus=0.,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=13, x=7))
    ray1 = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    res1 = model.trace(ray=ray1)
    ray2 = model.make_source_ray(source_dx=0., source_dy=0.).ray
    res2 = model.trace(ray=ray2)
    assert_allclose(res1['specimen'].ray.x, res2['specimen'].ray.x)
    assert_allclose(res1['specimen'].ray.y, res2['specimen'].ray.y)
    assert_allclose(res1['specimen'].sampling['scan_px'].x, 7)
    assert_allclose(res1['specimen'].sampling['scan_px'].y, 13)


def test_trace_noproject():
    params = Parameters4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=0.,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    model.trace(ray=ray)


def test_trace_underfocused_smoke():
    params = Parameters4DSTEM(
        overfocus=-0.23,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=13, x=7))
    ray = model.make_source_ray(source_dx=0.034, source_dy=0.042).ray
    model.trace(ray=ray)


# Beam straight along the optical axis, no scan deflection, scan and detector
# coordinate system identical with physical coordinates.
def test_straight():
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=0., x=0.))
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
    assert res['specimen'].sampling['scan_px'] == PixelYX(x=0., y=0.)
    assert res['detector'].sampling['detector_px'] == PixelYX(x=0., y=0.)


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
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model_straight = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=0., x=0.))
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=scan_y, x=scan_x))

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
        PixelYX(
            x=scan_x + res_straight['specimen'].sampling['scan_px'].x,
            y=scan_y + res_straight['specimen'].sampling['scan_px'].y
        )
    )
    # Check that central ray goes through scan position
    if dx == 0. and dy == 0.:
        assert_allclose(
            res['specimen'].sampling['scan_px'],
            PixelYX(
                x=scan_x,
                y=scan_y,
            ),
            rtol=1e-6, atol=1e-6
        )
    # check physical coords equals pixel coords
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelYX(
            x=res['specimen'].ray.x,
            y=res['specimen'].ray.y,
        )
    )
    assert_allclose(res['detector'].sampling['detector_px'], PixelYX(
        x=dx*(params.overfocus + params.camera_length),
        y=dy*(params.overfocus + params.camera_length)
    ))
    # check physical coords equals pixel coords
    assert_allclose(
        res['detector'].sampling['detector_px'],
        PixelYX(
            x=res['detector'].ray.x,
            y=res['detector'].ray.y,
        )
    )


# detector coordinate systems
@pytest.mark.parametrize(
    'detector_cycx', ((-0.11, 43.), (0., 0.))
)
@pytest.mark.parametrize(
    'detector_pixel_pitch', (0.09, 1., 1.53)
)
@pytest.mark.parametrize(
    'flip_y', (True, False)
)
@pytest.mark.parametrize(
    'dydx', ((0., 0.), (-0.2, 0.42))
)
def test_detector_coordinate_shift_scale_flip(
        detector_cycx, detector_pixel_pitch, flip_y, dydx):
    detector_cy, detector_cx = detector_cycx
    scan_cy = -0.7
    scan_cx = 23.
    scan_pixel_pitch = 1.34
    dy, dx = dydx
    scan_y = -17
    scan_x = 29
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=scan_pixel_pitch,
        scan_center=PixelYX(y=scan_cy, x=scan_cx),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=detector_pixel_pitch,
        detector_center=PixelYX(y=detector_cy, x=detector_cx),
        semiconv=0.023,
        flip_y=flip_y,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=scan_y, x=scan_x))

    ray = model.make_source_ray(source_dx=dx, source_dy=dy).ray

    res = model.trace(ray=ray)
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelYX(
            x=res['specimen'].ray.x/scan_pixel_pitch + scan_cx,
            y=res['specimen'].ray.y/scan_pixel_pitch + scan_cy,
        ),
        rtol=1e-6, atol=1e-6
    )
    flip_factor = -1. if flip_y else 1.
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['detector'].sampling['detector_px'],
        PixelYX(
            x=res['detector'].ray.x/detector_pixel_pitch + detector_cx,
            y=flip_factor*(res['detector'].ray.y/detector_pixel_pitch + flip_factor*detector_cy),
        ),
        rtol=1e-6, atol=1e-6
    )
    if dy == 0.:
        assert_allclose(
            res['detector'].sampling['detector_px'].y,
            detector_cy
        )
    if dx == 0.:
        assert_allclose(
            res['detector'].sampling['detector_px'].x,
            detector_cx
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
        scan_center=PixelYX(y=scan_cy, x=scan_cx),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=detector_pixel_pitch,
        detector_center=PixelYX(y=detector_cy, x=detector_cx),
        semiconv=0.023,
        flip_y=flip_y,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(params=params, scan_pos=PixelYX(y=scan_y, x=scan_x))

    ray = model.make_source_ray(source_dx=dx, source_dy=dy).ray

    res = model.trace(ray=ray)
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['specimen'].sampling['scan_px'],
        PixelYX(
            x=res['specimen'].ray.x/scan_pixel_pitch + scan_cx,
            y=res['specimen'].ray.y/scan_pixel_pitch + scan_cy,
        ),
        rtol=1e-6, atol=1e-6
    )
    flip_factor = -1. if flip_y else 1.
    # check physical coords vs pixel coords scale and shift
    assert_allclose(
        res['detector'].sampling['detector_px'],
        PixelYX(
            x=res['detector'].ray.x/detector_pixel_pitch + detector_cx,
            y=flip_factor*res['detector'].ray.y/detector_pixel_pitch + detector_cy,
        ),
        rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    # work in exact degree values since guess_corrections() can only
    # find these exactly. Otherwise we have larger residuals
    'scan_rotation', (73/180*np.pi, 0, 23/180*np.pi)
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
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=scan_rotation,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=detector_cy, x=detector_cx),
        semiconv=0.023,
        flip_y=flip_y,
        descan_error=DescanError()
    )

    y_deflections = np.linspace(start=-1, stop=1, num=3)
    x_deflections = np.linspace(start=-1, stop=1, num=3)
    com_y = np.empty((len(y_deflections), len(x_deflections)))
    com_x = np.empty((len(y_deflections), len(x_deflections)))
    for y, scan_y in enumerate(y_deflections):
        for x, scan_x in enumerate(x_deflections):
            model = Model4DSTEM.build(
                params=params,
                scan_pos=PixelYX(x=float(scan_x), y=float(scan_y)),
                specimen=PointChargeComponent(z=params.overfocus)
            )
            ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
            res = model.trace(ray)
            # Validate that the ray is deflected towards the center
            # by the point charge component
            phys_y = res['detector'].ray.y
            phys_x = res['detector'].ray.x
            pass_y = res['specimen'].ray.y
            pass_x = res['specimen'].ray.x
            if phys_y != 0 or phys_x != 0:
                assert_allclose(
                    # The displacement in the detector plane in corrected pixel
                    # coordinates is pointing in the opposite direction of the
                    # displacement from the center when passing through the
                    # specimen plane, i.e. the beam is deflected towards the
                    # center
                    np.array((phys_y, phys_x))/np.linalg.norm((phys_y, phys_x)),
                    -np.array((pass_y, pass_x))/np.linalg.norm((pass_y, pass_x))
                )
            com_y[y, x] = res['detector'].sampling['detector_px'].y
            com_x[y, x] = res['detector'].sampling['detector_px'].x

    guess_result = guess_corrections(y_centers=com_y, x_centers=com_x)
    corrected_y, corrected_x = apply_correction(
        y_centers=com_y-detector_cy, x_centers=com_x-detector_cx,
        scan_rotation=guess_result.scan_rotation,
        flip_y=guess_result.flip_y,
    )
    # Make sure the correction actually corrected
    for y, scan_y in enumerate(y_deflections):
        for x, scan_x in enumerate(x_deflections):
            if corrected_y[y, x] != 0 or corrected_x[y, x] != 0:
                assert_allclose(
                    # The corrected displacement in corrected pixel coordinates
                    # in the detector plane is pointing in the opposite
                    # direction of the displacement from the center in scan
                    # coordinates
                    np.array((scan_y, scan_x))/np.linalg.norm((scan_y, scan_x)),
                    -np.array((
                        corrected_y[y, x], corrected_x[y, x]
                    ))/np.linalg.norm((
                        corrected_y[y, x], corrected_x[y, x]
                    )),
                    atol=1e-4, rtol=1e-4
                )

    # See https://github.com/LiberTEM/LiberTEM/issues/1775
    # Rotation direction is opposite
    assert_allclose(-guess_result.scan_rotation / 180 * np.pi, scan_rotation, atol=1e-4, rtol=1e-4)
    assert guess_result.flip_y == flip_y
    assert_allclose(guess_result.cy, detector_cy, atol=1e-2, rtol=1e-2)
    assert_allclose(guess_result.cx, detector_cx, atol=1e-2, rtol=1e-2)


def test_rotation_direction_0():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away, and therefore positive scan rotation
    # rotates the scan points to the right.
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=0., x=1.)
    )
    ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
    res = model.trace(ray)
    assert_allclose(res['specimen'].sampling['scan_px'].x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].sampling['scan_px'].y, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.y, 0., atol=1e-6, rtol=1e-6)

    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=1., x=0.)
    )
    ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
    res = model.trace(ray)
    assert_allclose(res['specimen'].sampling['scan_px'].x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].sampling['scan_px'].y, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.y, 1., atol=1e-6, rtol=1e-6)


def test_rotation_direction_90():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away, and therefore positive scan rotation
    # rotates the scan points to the right in physical coordinates
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=np.pi/2,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=0., x=1.)
    )
    ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
    res = model.trace(ray)

    assert_allclose(res['specimen'].sampling['scan_px'].x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].sampling['scan_px'].y, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.y, 1., atol=1e-6, rtol=1e-6)

    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=1., x=0.)
    )
    ray = model.make_source_ray(source_dx=0., source_dy=0.).ray
    res = model.trace(ray)
    assert_allclose(res['specimen'].sampling['scan_px'].x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].sampling['scan_px'].y, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.x, -1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['specimen'].ray.y, 0., atol=1e-6, rtol=1e-6)


def test_detector_px():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away.
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=0., x=0.)
    )
    ray = model.make_source_ray(source_dx=0.5, source_dy=0.).ray
    res = model.trace(ray)
    assert_allclose(res['detector'].sampling['detector_px'].x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].sampling['detector_px'].y, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.y, 0., atol=1e-6, rtol=1e-6)

    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=0., x=0.)
    )
    ray = model.make_source_ray(source_dx=0., source_dy=0.5).ray
    res = model.trace(ray)
    assert_allclose(res['detector'].sampling['detector_px'].x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].sampling['detector_px'].y, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.y, 1., atol=1e-6, rtol=1e-6)


def test_detector_px_flipy():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away.
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=True,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=0., x=0.)
    )
    ray = model.make_source_ray(source_dx=0.5, source_dy=0.).ray
    res = model.trace(ray)
    assert_allclose(res['detector'].sampling['detector_px'].x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].sampling['detector_px'].y, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.x, 1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.y, 0., atol=1e-6, rtol=1e-6)

    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=0., x=0.)
    )
    ray = model.make_source_ray(source_dx=0., source_dy=0.5).ray
    res = model.trace(ray)
    assert_allclose(res['detector'].sampling['detector_px'].x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].sampling['detector_px'].y, -1., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.x, 0., atol=1e-6, rtol=1e-6)
    assert_allclose(res['detector'].ray.y, 1., atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    'scan', (PixelYX(y=0., x=0.), PixelYX(y=-3., x=5.), )
)
@pytest.mark.parametrize(
    'overfocus', (-2., 0., 0.1)
)
@pytest.mark.parametrize(
    'camera_length', (-4., 0., 1.2)
)
@pytest.mark.parametrize(
    'dydx', ((-4., 13.), (0., 0.))
)
def test_geometry(scan, overfocus, camera_length, dydx):
    dy, dx = dydx
    params = Parameters4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=camera_length,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=scan
    )
    ray = model.make_source_ray(source_dx=dx, source_dy=dy).ray
    res = model.trace(ray)
    # No descan error means rays not bent
    for key, sect in res.items():
        assert sect.ray.dy == dy
        assert sect.ray.dx == dx
        if scan.x == 0. or key not in ('scanner', 'specimen'):
            assert_allclose(sect.ray.x, dx*sect.ray.z)
        if scan.y == 0. or key not in ('scanner', 'specimen'):
            assert_allclose(sect.ray.y, dy*sect.ray.z)
    assert res['source'].ray.z == 0
    for key in ('overfocus', 'scanner', 'specimen', 'descanner'):
        assert_allclose(res[key].ray.z, overfocus)
    for key in ('camera_length', 'detector'):
        assert_allclose(res[key].ray.z, overfocus+camera_length)


def test_descan_offset():
    params_ref = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model_ref = Model4DSTEM.build(
        params=params_ref,
        scan_pos=PixelYX(y=23., x=-13.)
    )
    ray_ref = model_ref.make_source_ray(source_dx=0.5, source_dy=-0.1).ray
    res_ref = model_ref.trace(ray_ref)

    offpxi = 0.11
    offpyi = 0.13
    offsxi = 0.17
    offsyi = 0.19
    params = Parameters4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError(
            offpxi=offpxi,
            offpyi=offpyi,
            offsxi=offsxi,
            offsyi=offsyi
        )
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=PixelYX(y=23., x=-13.)
    )
    ray = model.make_source_ray(source_dx=0.5, source_dy=-0.1).ray
    res = model.trace(ray)

    for key in ('source', 'overfocus', 'scanner', 'specimen'):
        sect_ref = res_ref[key]
        sect = res[key]
        for attr in ('y', 'x', 'dy', 'dx', 'z'):
            assert_allclose(
                getattr(sect.ray, attr),
                getattr(sect_ref.ray, attr),
            )
    sect_ref = res_ref['descanner']
    sect = res['descanner']
    assert_allclose(
        sect.ray.x,
        sect_ref.ray.x + offpxi
    )
    assert_allclose(
        sect.ray.y,
        sect_ref.ray.y + offpyi
    )
    assert_allclose(
        sect.ray.dx,
        sect_ref.ray.dx + offsxi
    )
    assert_allclose(
        sect.ray.dy,
        sect_ref.ray.dy + offsyi
    )
    assert_allclose(
        sect.ray.z,
        sect_ref.ray.z
    )
    # Straight propagation
    for key in ('camera_length', 'detector'):
        start = res['descanner']
        stop = res[key]
        assert_allclose(
            stop.ray.x,
            start.ray.x + start.ray.dx*(stop.ray.z - start.ray.z)
        )
        assert_allclose(
            stop.ray.y,
            start.ray.y + start.ray.dy*(stop.ray.z - start.ray.z)
        )
        assert_allclose(
            stop.ray.dx,
            start.ray.dx
        )
        assert_allclose(
            stop.ray.dy,
            start.ray.dy
        )


@pytest.mark.parametrize(
    'scan', (PixelYX(y=0., x=0.), PixelYX(y=-3., x=5.), )
)
def test_descan_position(scan):
    params_ref = Parameters4DSTEM(
        overfocus=1.,
        scan_pixel_pitch=1.,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1.,
        detector_pixel_pitch=1.,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model_ref = Model4DSTEM.build(
        params=params_ref,
        scan_pos=scan
    )
    ray_ref = model_ref.make_source_ray(source_dx=0.5, source_dy=-0.1).ray
    res_ref = model_ref.trace(ray_ref)

    pxo_pxi = 0.11
    pxo_pyi = 0.13
    pyo_pxi = 0.17
    pyo_pyi = 0.19
    params = Parameters4DSTEM(
        overfocus=1.,
        scan_pixel_pitch=1.,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1.,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError(
            pxo_pxi=pxo_pxi,
            pxo_pyi=pxo_pyi,
            pyo_pxi=pyo_pxi,
            pyo_pyi=pyo_pyi
        )
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=scan
    )
    ray = model.make_source_ray(source_dx=0.5, source_dy=-0.1).ray
    res = model.trace(ray)

    # no descan error contribution from p*o_p*i parameters
    # if beam is not deflected by scanner
    if scan.x == 0 and scan.y == 0:
        keys = (
            'source', 'overfocus', 'scanner', 'specimen',
            'descanner', 'camera_length', 'detector'
        )
    else:
        keys = ('source', 'overfocus', 'scanner', 'specimen')
    for key in keys:
        sect_ref = res_ref[key]
        sect = res[key]
        for attr in ('y', 'x', 'dy', 'dx', 'z'):
            assert_allclose(
                getattr(sect.ray, attr),
                getattr(sect_ref.ray, attr),
            )
    sect_ref = res_ref['descanner']
    sect = res['descanner']
    assert_allclose(
        sect.ray.x,
        sect_ref.ray.x + pxo_pxi * scan.x + pxo_pyi * scan.y
    )
    assert_allclose(
        sect.ray.y,
        sect_ref.ray.y + pyo_pxi * scan.x + pyo_pyi * scan.y
    )
    assert_allclose(
        sect.ray.dx,
        sect_ref.ray.dx
    )
    assert_allclose(
        sect.ray.dy,
        sect_ref.ray.dy
    )
    assert_allclose(
        sect.ray.z,
        sect_ref.ray.z
    )
    # Straight propagation
    for key in ('camera_length', 'detector'):
        start = res['descanner']
        stop = res[key]
        assert_allclose(
            stop.ray.x,
            start.ray.x + start.ray.dx*(stop.ray.z - start.ray.z)
        )
        assert_allclose(
            stop.ray.y,
            start.ray.y + start.ray.dy*(stop.ray.z - start.ray.z)
        )
        assert_allclose(
            stop.ray.dx,
            start.ray.dx
        )
        assert_allclose(
            stop.ray.dy,
            start.ray.dy
        )


@pytest.mark.parametrize(
    'scan', (PixelYX(y=0., x=0.), PixelYX(y=-3., x=5.), )
)
def test_descan_slope(scan):
    params_ref = Parameters4DSTEM(
        overfocus=1.,
        scan_pixel_pitch=1.,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1.,
        detector_pixel_pitch=1.,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError()
    )
    model_ref = Model4DSTEM.build(
        params=params_ref,
        scan_pos=scan
    )
    ray_ref = model_ref.make_source_ray(source_dx=0.5, source_dy=-0.1).ray
    res_ref = model_ref.trace(ray_ref)

    sxo_pxi = 0.11
    sxo_pyi = 0.13
    syo_pxi = 0.17
    syo_pyi = 0.19
    params = Parameters4DSTEM(
        overfocus=1.,
        scan_pixel_pitch=1.,
        scan_center=PixelYX(y=0., x=0.),
        scan_rotation=0.,
        camera_length=1.,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0., x=0.),
        semiconv=0.023,
        flip_y=False,
        descan_error=DescanError(
            sxo_pxi=sxo_pxi,
            sxo_pyi=sxo_pyi,
            syo_pxi=syo_pxi,
            syo_pyi=syo_pyi
        )
    )
    model = Model4DSTEM.build(
        params=params,
        scan_pos=scan
    )
    ray = model.make_source_ray(source_dx=0.5, source_dy=-0.1).ray
    res = model.trace(ray)

    # no descan error contribution from s*o_p*i parameters
    # if beam is not deflected by scanner
    if scan.x == 0 and scan.y == 0:
        keys = (
            'source', 'overfocus', 'scanner', 'specimen',
            'descanner', 'camera_length', 'detector'
        )
    else:
        keys = ('source', 'overfocus', 'scanner', 'specimen')
    for key in keys:
        sect_ref = res_ref[key]
        sect = res[key]
        for attr in ('y', 'x', 'dy', 'dx', 'z'):
            assert_allclose(
                getattr(sect.ray, attr),
                getattr(sect_ref.ray, attr),
            )
    sect_ref = res_ref['descanner']
    sect = res['descanner']
    assert_allclose(
        sect.ray.dx,
        sect_ref.ray.dx + sxo_pxi * scan.x + sxo_pyi * scan.y
    )
    assert_allclose(
        sect.ray.dy,
        sect_ref.ray.dy + syo_pxi * scan.x + syo_pyi * scan.y
    )
    assert_allclose(
        sect.ray.x,
        sect_ref.ray.x
    )
    assert_allclose(
        sect.ray.y,
        sect_ref.ray.y
    )
    assert_allclose(
        sect.ray.z,
        sect_ref.ray.z
    )
    # Straight propagation
    for key in ('camera_length', 'detector'):
        start = res['descanner']
        stop = res[key]
        assert_allclose(
            stop.ray.x,
            start.ray.x + start.ray.dx*(stop.ray.z - start.ray.z)
        )
        assert_allclose(
            stop.ray.y,
            start.ray.y + start.ray.dy*(stop.ray.z - start.ray.z)
        )
        assert_allclose(
            stop.ray.dx,
            start.ray.dx
        )
        assert_allclose(
            stop.ray.dy,
            start.ray.dy
        )
