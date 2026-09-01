import pytest
from numpy.testing import assert_allclose

import jax; jax.config.update("jax_enable_x64", True)  # noqa fmt: skip

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import sympy as sym
from libertem.udf.com import apply_correction, guess_corrections
from temgym_core import PixelYX
from temgym_core.components import Component, DescanError, NamedTuple
from temgym_core.propagator import Propagator
from temgym_core.ray import Ray
from temgym_core.source import Source

from microscope_calibration.common.model import Model4DSTEM
from microscope_calibration.util.sympy import lambdify


def norm(y, x):
    return sym.sqrt(sym.Abs(y) ** 2 + sym.Abs(x) ** 2)


def test_trace_smoke():
    model = Model4DSTEM(
        overfocus=0.7,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=13, x=7), source_dx=0.034, source_dy=0.042)
    keys = (
        "source",
        "overfocus",
        "scanner",
        "specimen",
        "descanner",
        "camera_length",
        "detector",
    )
    for key in keys:
        assert key in res
        sect = res[key]
        assert isinstance(sect.ray, Ray)
    components = ("scanner", "specimen", "descanner", "detector")
    propagators = ("camera_length", "camera_length")
    for key in components:
        sect = res[key]
        assert isinstance(sect.component, Component)
    for key in propagators:
        sect = res[key]
        assert isinstance(sect.component, Propagator)
    assert isinstance(res["source"].component, Source)
    assert isinstance(res["specimen"].sampling["scan_px"], PixelYX)
    assert isinstance(res["detector"].sampling["detector_px"], PixelYX)


def test_trace_focused():
    model = Model4DSTEM(
        overfocus=0.0,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(),
    )
    res1 = model.trace(scan_pos=PixelYX(y=13, x=7), source_dx=0.034, source_dy=0.042)
    res2 = model.trace(scan_pos=PixelYX(y=13, x=7), source_dx=0.0, source_dy=0.0)
    assert sym.sympify(res1["specimen"].ray.x).equals(res2["specimen"].ray.x)
    # assert_allclose(res1['specimen'].ray.x, res2['specimen'].ray.x)
    assert sym.sympify(res1["specimen"].ray.y).equals(res2["specimen"].ray.y)
    assert sym.sympify(res1["specimen"].sampling["scan_px"].x).equals(7)
    assert sym.sympify(res1["specimen"].sampling["scan_px"].y).equals(13)


def test_trace_noproject():
    model = Model4DSTEM(
        overfocus=0.123,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=0.0,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(),
    )
    model.trace(scan_pos=PixelYX(y=13, x=7), source_dx=0.034, source_dy=0.042)


def test_trace_underfocused_smoke():
    model = Model4DSTEM(
        overfocus=-0.23,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(),
    )
    model.trace(scan_pos=PixelYX(y=13, x=7), source_dx=0.034, source_dy=0.042)


# Beam straight along the optical axis, no scan deflection, scan and detector
# coordinate system identical with physical coordinates.
def test_straight():
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.0, source_dy=0.0)

    for key, sect in res.items():
        if isinstance(sect.component, Component) or isinstance(sect.component, Source):
            assert sect.component.z == sect.ray.z
            assert sect.component.z == sect.ray.pathlength
        assert sym.sympify(sect.ray.x).equals(0.0)
        assert sym.sympify(sect.ray.y).equals(0.0)
    assert res["detector"].ray.z == model.overfocus + model.camera_length
    assert res["source"].ray.z == 0.0
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(0.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(0.0)


# Scan deflection test: beam is shifted
# Also test that comparisons with symbols are
# equal when they should be, and not equal when they shouldn't.
@pytest.mark.parametrize("dy", (-0.2, 0.0, 0.34))
@pytest.mark.parametrize("dx", (-0.7, 0.0, 0.42))
@pytest.mark.parametrize("scan_y", (sym.Symbol("scan_y"),))
@pytest.mark.parametrize("scan_x", (sym.Symbol("scan_x"),))
def test_scan(dy, dx, scan_y, scan_x):
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res_straight = model.trace(
        scan_pos=PixelYX(y=0.0, x=0.0), source_dx=dx, source_dy=dy
    )
    res = model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=dx, source_dy=dy)

    for key in res.keys():
        sect = res[key]
        sect_straight = res_straight[key]
        assert sect.ray.z == sect_straight.ray.z
        assert sect.ray.pathlength == sect_straight.ray.pathlength
        if isinstance(sect.component, Component) or isinstance(sect.component, Source):
            assert sect.component.z == sect.ray.z
            assert sect.component.z == sect.ray.pathlength
        # Beam is deflected
        if key in ("scanner", "specimen"):
            assert sym.sympify(sect.ray.x - sect_straight.ray.x).equals(scan_x)
            assert sym.sympify(sect.ray.y - sect_straight.ray.y).equals(scan_y)
        # Beam is not deflected
        else:
            assert sym.sympify(sect.ray.x).equals(sect_straight.ray.x)
            assert sym.sympify(sect.ray.y).equals(sect_straight.ray.y)
            # Some counter checks to make sure deviations actually let
            # the values to be detected as different
            with pytest.raises(AssertionError):
                assert sym.sympify(sect.ray.x).equals(sect_straight.ray.x + 1)
            # Test that we can clearly distinguish tests that should fail and
            # tests that should succeed when a mixture of symbols (scan_y, scan_x) and
            # numerical values (dx, dy) is used. If the ray has propagated in z and there is
            # a different deflection in x and y,
            # then the x and y values should not be equal anymore.
            if sect.ray.z != 0 and dx != dy:
                with pytest.raises(AssertionError):
                    assert sym.sympify(sect.ray.x).equals(sect_straight.ray.y)
            else:
                assert sym.sympify(sect.ray.x).equals(sect_straight.ray.y)
            # Ray propagates straight
            assert sym.sympify(sect.ray.x).equals(sect.ray.z * dx)
            assert sym.sympify(sect.ray.y).equals(sect.ray.z * dy)
    assert sym.sympify(res["detector"].ray.z).equals(
        model.overfocus + model.camera_length
    )
    assert sym.sympify(res["source"].ray.z).equals(0.0)
    # Correct scan deflection
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(
        scan_x + res_straight["specimen"].sampling["scan_px"].x
    )
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(
        scan_y + res_straight["specimen"].sampling["scan_px"].y
    )
    # Check that central ray goes through scan position
    if dx == 0.0 and dy == 0.0:
        assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(scan_x)
        assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(scan_y)
    # check physical coords equals pixel coords
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(
        res["specimen"].ray.x
    )
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(
        res["specimen"].ray.y
    )
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(
        dx * (model.overfocus + model.camera_length)
    )
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(
        dy * (model.overfocus + model.camera_length)
    )
    # check physical coords equals pixel coords
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(
        res["detector"].ray.x
    )
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(
        res["detector"].ray.y
    )


# detector coordinate systems
@pytest.mark.parametrize("detector_cycx", ((-0.11, 43.0), (0.0, 0.0)))
@pytest.mark.parametrize("detector_pixel_pitch", (0.09, 1.0, 1.53))
@pytest.mark.parametrize("flip_factor", (-1.0, 1.0))
@pytest.mark.parametrize("dydx", ((0.0, 0.0), (-0.2, 0.42)))
def test_detector_coordinate_shift_scale_flip(
    detector_cycx, detector_pixel_pitch, flip_factor, dydx
):
    detector_cy, detector_cx = detector_cycx
    scan_cy = -0.7
    scan_cx = 23.0
    scan_pixel_pitch = 1.34
    dy, dx = dydx
    scan_y = -17
    scan_x = 29
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=scan_pixel_pitch,
        scan_center=PixelYX(y=scan_cy, x=scan_cx),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=detector_pixel_pitch,
        detector_center=PixelYX(y=detector_cy, x=detector_cx),
        semiconv=0.023,
        flip_factor=flip_factor,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=dx, source_dy=dy)
    # check physical coords vs pixel coords scale and shift
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(
        res["specimen"].ray.x / scan_pixel_pitch + scan_cx
    )
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(
        res["specimen"].ray.y / scan_pixel_pitch + scan_cy
    )
    # check physical coords vs pixel coords scale and shift
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(
        res["detector"].ray.x / detector_pixel_pitch + detector_cx
    )
    assert_allclose(
        float(sym.sympify(res["detector"].sampling["detector_px"].y).evalf()),
        float(
            sym.sympify(
                flip_factor
                * (
                    res["detector"].ray.y / detector_pixel_pitch
                    + flip_factor * detector_cy
                )
            ).evalf()
        ),
    )
    if dy == 0.0:
        assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(
            detector_cy
        )
    if dx == 0.0:
        assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(
            detector_cx
        )


# scan coordinate systems
@pytest.mark.parametrize("scan_cy", (-0.7, 0.0, 21))
@pytest.mark.parametrize("scan_cx", (-0.22, 0.0, 23))
@pytest.mark.parametrize("scan_pixel_pitch", (0.07, 1.0, 1.34))
def test_scan_coordinate_shift_scale(scan_cy, scan_cx, scan_pixel_pitch):
    detector_cy = -11.0
    detector_cx = 43.0
    detector_pixel_pitch = 0.09
    flip_factor = -1.0
    dy = -0.2
    dx = 0.42
    scan_y = -17
    scan_x = 29
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=scan_pixel_pitch,
        scan_center=PixelYX(y=scan_cy, x=scan_cx),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=detector_pixel_pitch,
        detector_center=PixelYX(y=detector_cy, x=detector_cx),
        semiconv=0.023,
        flip_factor=flip_factor,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=scan_y, x=scan_x), source_dx=dx, source_dy=dy)
    # check physical coords vs pixel coords scale and shift
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(
        res["specimen"].ray.x / scan_pixel_pitch + scan_cx
    )
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(
        res["specimen"].ray.y / scan_pixel_pitch + scan_cy
    )
    flip_factor = -1.0 if flip_factor else 1.0
    # check physical coords vs pixel coords scale and shift
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(
        res["detector"].ray.x / detector_pixel_pitch + detector_cx
    )
    assert_allclose(
        float(sym.sympify(res["detector"].sampling["detector_px"].y).evalf()),
        float(
            sym.sympify(
                flip_factor
                * (
                    res["detector"].ray.y / detector_pixel_pitch
                    + flip_factor * detector_cy
                )
            ).evalf()
        ),
    )


@pytest.mark.parametrize(
    # work in exact degree values since guess_corrections() can only
    # find these exactly. Otherwise we have larger residuals
    "scan_rotation",
    (73 / 180 * np.pi, 0, 23 / 180 * np.pi),
)
@pytest.mark.parametrize("flip_factor", (1.0, -1.0))
@pytest.mark.parametrize("detector_cy", (-13, 0.0, 7))
@pytest.mark.parametrize("detector_cx", (-11, 0.0, 5))
def test_com_validation(scan_rotation, flip_factor, detector_cy, detector_cx):
    @jdc.pytree_dataclass
    class PointChargeComponent(Component):
        z: float

        def __call__(self, ray: Ray) -> Ray:
            distance = sym.sqrt(sym.Abs(ray.x) ** 2 + sym.Abs(ray.y) ** 2)
            # field strength is 1/distance**2,
            # additionally normalize displacement to unit vector
            normfield = sym.Piecewise((1 / distance**3, distance > 1e-6), (0, True))
            dx = -ray.x * normfield * 1e-2
            dy = -ray.y * normfield * 1e-2
            return ray.derive(dx=ray.dx + dx, dy=ray.dy + dy)

    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=scan_rotation,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=detector_cy, x=detector_cx),
        semiconv=0.023,
        flip_factor=flip_factor,
        descan_error=DescanError(),
    )

    class ReturnT(NamedTuple):
        phys_y: float
        phys_x: float
        pass_y: float
        pass_x: float
        detector_scan_px_y: float
        detector_scan_px_x: float

    @lambdify(modules=jax.numpy)
    def calculate(scan_y, scan_x):
        res = model.trace(
            scan_pos=PixelYX(x=scan_x, y=scan_y),
            source_dy=0.0,
            source_dx=0.0,
            specimen=PointChargeComponent(z=model.overfocus),
        )
        return ReturnT(
            phys_y=res["detector"].ray.y,
            phys_x=res["detector"].ray.x,
            pass_y=res["specimen"].ray.y,
            pass_x=res["specimen"].ray.x,
            detector_scan_px_y=res["detector"].sampling["detector_px"].y,
            detector_scan_px_x=res["detector"].sampling["detector_px"].x,
        )

    y_deflections = np.linspace(start=-1, stop=1, num=3)
    x_deflections = np.linspace(start=-1, stop=1, num=3)
    com_y = np.empty((len(y_deflections), len(x_deflections)))
    com_x = np.empty((len(y_deflections), len(x_deflections)))
    for y, scan_y in enumerate(y_deflections):
        for x, scan_x in enumerate(x_deflections):
            res = calculate(scan_y=scan_y, scan_x=scan_x)
            # Validate that the ray is deflected towards the center
            # by the point charge component
            phys_y = res.phys_y
            phys_x = res.phys_x
            pass_y = res.pass_y
            pass_x = res.pass_x
            if phys_y != 0 or phys_x != 0:
                assert_allclose(
                    # The displacement in the detector plane in corrected pixel
                    # coordinates is pointing in the opposite direction of the
                    # displacement from the center when passing through the
                    # specimen plane, i.e. the beam is deflected towards the
                    # center
                    np.array((phys_y, phys_x)) / np.linalg.norm((phys_y, phys_x)),
                    -np.array((pass_y, pass_x)) / np.linalg.norm((pass_y, pass_x)),
                )
            print(res.detector_scan_px_y)
            com_y[y, x] = res.detector_scan_px_y
            com_x[y, x] = res.detector_scan_px_x

    guess_result = guess_corrections(y_centers=com_y, x_centers=com_x)
    corrected_y, corrected_x = apply_correction(
        y_centers=com_y - detector_cy,
        x_centers=com_x - detector_cx,
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
                    np.array((scan_y, scan_x)) / np.linalg.norm((scan_y, scan_x)),
                    -np.array((corrected_y[y, x], corrected_x[y, x]))
                    / np.linalg.norm((corrected_y[y, x], corrected_x[y, x])),
                    atol=1e-12,
                    rtol=1e-12,
                )

    assert_allclose(
        -guess_result.scan_rotation / 180 * np.pi, scan_rotation, atol=1e-12, rtol=1e-4
    )
    if flip_factor == 1.0:
        flip_y = False
    elif flip_factor == -1.0:
        flip_y = True
    else:
        raise ValueError(0)
    assert guess_result.flip_y == flip_y
    assert_allclose(guess_result.cy, detector_cy, atol=1e-2, rtol=1e-2)
    assert_allclose(guess_result.cx, detector_cx, atol=1e-2, rtol=1e-2)
    assert_allclose(guess_result.cy, detector_cy, atol=1e-2, rtol=1e-2)
    assert_allclose(guess_result.cx, detector_cx, atol=1e-2, rtol=1e-2)


def test_rotation_direction_0():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away, and therefore positive scan rotation
    # rotates the scan points to the right.
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=1.0), source_dx=0.0, source_dy=0.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(1.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(0.0)
    assert sym.sympify(res["specimen"].ray.x).equals(1.0)
    assert sym.sympify(res["specimen"].ray.y).equals(0.0)

    res = model.trace(scan_pos=PixelYX(y=1.0, x=0.0), source_dx=0.0, source_dy=0.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(0.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(1.0)
    assert sym.sympify(res["specimen"].ray.x).equals(0.0)
    assert sym.sympify(res["specimen"].ray.y).equals(1.0)


@pytest.mark.parametrize("flip_factor", (1.0, -1.0))
def test_rotation_direction_90(flip_factor):
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away, and therefore positive scan rotation
    # rotates the scan points to the right in physical coordinates
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=sym.pi / 2,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=flip_factor,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=1.0), source_dx=0.0, source_dy=0.0)

    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(1.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(0.0)
    assert sym.sympify(res["specimen"].ray.x).equals(0.0)
    assert sym.sympify(res["specimen"].ray.y).equals(1.0)

    res = model.trace(scan_pos=PixelYX(y=1.0, x=0.0), source_dx=0.0, source_dy=0.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].x).equals(0.0)
    assert sym.sympify(res["specimen"].sampling["scan_px"].y).equals(1.0)
    assert sym.sympify(res["specimen"].ray.x).equals(-1.0)
    assert sym.sympify(res["specimen"].ray.y).equals(0.0)


def test_detector_px():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away.
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.5, source_dy=0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(1.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(0.0)
    assert sym.sympify(res["detector"].ray.x).equals(1.0)
    assert sym.sympify(res["detector"].ray.y).equals(0.0)

    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.0, source_dy=0.5)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(1.0)
    assert sym.sympify(res["detector"].ray.x).equals(0.0)
    assert sym.sympify(res["detector"].ray.y).equals(1.0)


def test_detector_px_flipy():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away.
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        detector_rotation=0.0,
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dy=0.0, source_dx=0.5)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(1.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(0.0)
    assert sym.sympify(res["detector"].ray.x).equals(1.0)
    assert sym.sympify(res["detector"].ray.y).equals(0.0)

    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dy=0.5, source_dx=0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(-1.0)
    assert sym.sympify(res["detector"].ray.x).equals(0.0)
    assert sym.sympify(res["detector"].ray.y).equals(1.0)


def test_detector_px_rotate():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away.
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        detector_rotation=sym.pi / 2,
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.5, source_dy=0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(-1.0)
    assert sym.sympify(res["detector"].ray.x).equals(1.0)
    assert sym.sympify(res["detector"].ray.y).equals(0.0)

    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.0, source_dy=0.5)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(1.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(0.0)
    assert sym.sympify(res["detector"].ray.x).equals(0.0)
    assert sym.sympify(res["detector"].ray.y).equals(1.0)


def test_detector_px_rotate_flipy():
    # Check conformance with
    # https://libertem.github.io/LiberTEM/concepts.html#coordinate-system: y
    # points down, x to the right, z away.
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        detector_rotation=sym.pi / 2,
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.5, source_dy=0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(0.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(1.0)
    assert sym.sympify(res["detector"].ray.x).equals(1.0)
    assert sym.sympify(res["detector"].ray.y).equals(0.0)

    res = model.trace(scan_pos=PixelYX(y=0.0, x=0.0), source_dx=0.0, source_dy=0.5)
    assert sym.sympify(res["detector"].sampling["detector_px"].x).equals(1.0)
    assert sym.sympify(res["detector"].sampling["detector_px"].y).equals(0.0)
    assert sym.sympify(res["detector"].ray.x).equals(0.0)
    assert sym.sympify(res["detector"].ray.y).equals(1.0)


@pytest.mark.parametrize(
    "scan",
    (
        PixelYX(y=0.0, x=0.0),
        PixelYX(y=-3.0, x=5.0),
    ),
)
@pytest.mark.parametrize("overfocus", (-2.0, 0.0, 0.1))
@pytest.mark.parametrize("camera_length", (-4.0, 0.0, 1.2))
@pytest.mark.parametrize("dydx", ((sym.Symbol("dy"), sym.Symbol("dx")),))
def test_geometry(scan, overfocus, camera_length, dydx):
    dy, dx = dydx
    model = Model4DSTEM(
        overfocus=overfocus,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=camera_length,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res = model.trace(scan_pos=scan, source_dx=dx, source_dy=dy)
    # No descan error means rays not bent
    for key, sect in res.items():
        assert sym.sympify(sect.ray.dy).equals(dy)
        assert sym.sympify(sect.ray.dx).equals(dx)
        if scan.x == 0.0 or key not in ("scanner", "specimen"):
            assert sym.sympify(sect.ray.x).equals(dx * sect.ray.z)
        if scan.y == 0.0 or key not in ("scanner", "specimen"):
            assert sym.sympify(sect.ray.y).equals(dy * sect.ray.z)
    assert res["source"].ray.z == 0
    for key in ("overfocus", "scanner", "specimen", "descanner"):
        assert sym.sympify(res[key].ray.z).equals(overfocus)
    for key in ("camera_length", "detector"):
        assert sym.sympify(res[key].ray.z).equals(overfocus + camera_length)


def test_descan_offset():
    model_ref = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res_ref = model_ref.trace(
        scan_pos=PixelYX(y=23.0, x=-13.0), source_dx=0.5, source_dy=-0.1
    )

    offpxi = sym.Symbol("offpxi")
    offpyi = sym.Symbol("offpyi")
    offsxi = sym.Symbol("offsxi")
    offsyi = sym.Symbol("offsyi")
    model = Model4DSTEM(
        overfocus=1,
        scan_pixel_pitch=1,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(
            offpxi=offpxi, offpyi=offpyi, offsxi=offsxi, offsyi=offsyi
        ),
    )
    res = model.trace(scan_pos=PixelYX(y=23.0, x=-13.0), source_dx=0.5, source_dy=-0.1)

    for key in ("source", "overfocus", "scanner", "specimen"):
        sect_ref = res_ref[key]
        sect = res[key]
        for attr in ("y", "x", "dy", "dx", "z"):
            assert sym.sympify(getattr(sect.ray, attr)).equals(
                getattr(sect_ref.ray, attr)
            )
    sect_ref = res_ref["descanner"]
    sect = res["descanner"]
    assert sym.sympify(sect.ray.x).equals(sect_ref.ray.x + offpxi)
    assert sym.sympify(sect.ray.y).equals(sect_ref.ray.y + offpyi)
    assert sym.sympify(sect.ray.dx).equals(sect_ref.ray.dx + offsxi)
    assert sym.sympify(sect.ray.dy).equals(sect_ref.ray.dy + offsyi)
    assert sym.sympify(sect.ray.z).equals(sect_ref.ray.z)
    # Straight propagation
    for key in ("camera_length", "detector"):
        start = res["descanner"]
        stop = res[key]
        assert sym.sympify(stop.ray.x).equals(
            start.ray.x + start.ray.dx * (stop.ray.z - start.ray.z)
        )
        assert sym.sympify(stop.ray.y).equals(
            start.ray.y + start.ray.dy * (stop.ray.z - start.ray.z)
        )
        assert sym.sympify(stop.ray.dx).equals(start.ray.dx)
        assert sym.sympify(stop.ray.dy).equals(start.ray.dy)


@pytest.mark.parametrize(
    "scan",
    (
        PixelYX(y=0.0, x=0.0),
        PixelYX(y=-3.0, x=5.0),
    ),
)
def test_descan_position(scan):
    model_ref = Model4DSTEM(
        overfocus=1.0,
        scan_pixel_pitch=1.0,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1.0,
        detector_pixel_pitch=1.0,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res_ref = model_ref.trace(scan_pos=scan, source_dx=0.5, source_dy=-0.1)

    pxo_pxi = sym.Symbol("pxo_pxi")
    pxo_pyi = sym.Symbol("pxo_pyi")
    pyo_pxi = sym.Symbol("pyo_pxi")
    pyo_pyi = sym.Symbol("pyo_pyi")
    model = Model4DSTEM(
        overfocus=1.0,
        scan_pixel_pitch=1.0,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1.0,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(
            pxo_pxi=pxo_pxi, pxo_pyi=pxo_pyi, pyo_pxi=pyo_pxi, pyo_pyi=pyo_pyi
        ),
    )
    res = model.trace(scan_pos=scan, source_dx=0.5, source_dy=-0.1)

    # no descan error contribution from p*o_p*i parameters
    # if beam is not deflected by scanner
    if scan.x == 0 and scan.y == 0:
        keys = (
            "source",
            "overfocus",
            "scanner",
            "specimen",
            "descanner",
            "camera_length",
            "detector",
        )
    else:
        keys = ("source", "overfocus", "scanner", "specimen")
    for key in keys:
        sect_ref = res_ref[key]
        sect = res[key]
        for attr in ("y", "x", "dy", "dx", "z"):
            assert sym.sympify(getattr(sect.ray, attr)).equals(
                getattr(sect_ref.ray, attr)
            )
    sect_ref = res_ref["descanner"]
    sect = res["descanner"]
    assert sym.sympify(sect.ray.x).equals(
        sect_ref.ray.x + pxo_pxi * scan.x + pxo_pyi * scan.y
    )
    assert sym.sympify(sect.ray.y).equals(
        sect_ref.ray.y + pyo_pxi * scan.x + pyo_pyi * scan.y
    )
    assert sym.sympify(sect.ray.dx).equals(sect_ref.ray.dx)
    assert sym.sympify(sect.ray.dy).equals(sect_ref.ray.dy)
    assert sym.sympify(sect.ray.z).equals(sect_ref.ray.z)
    # Straight propagation
    for key in ("camera_length", "detector"):
        start = res["descanner"]
        stop = res[key]
        assert sym.sympify(stop.ray.x).equals(
            start.ray.x + start.ray.dx * (stop.ray.z - start.ray.z)
        )
        assert sym.sympify(stop.ray.y).equals(
            start.ray.y + start.ray.dy * (stop.ray.z - start.ray.z)
        )
        assert sym.sympify(stop.ray.dx).equals(start.ray.dx)
        assert sym.sympify(stop.ray.dy).equals(start.ray.dy)


@pytest.mark.parametrize(
    "scan",
    (
        PixelYX(y=0.0, x=0.0),
        PixelYX(y=-3.0, x=5.0),
    ),
)
def test_descan_slope(scan):
    model_ref = Model4DSTEM(
        overfocus=1.0,
        scan_pixel_pitch=1.0,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1.0,
        detector_pixel_pitch=1.0,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(),
    )
    res_ref = model_ref.trace(scan_pos=scan, source_dx=0.5, source_dy=-0.1)

    sxo_pxi = sym.Symbol("sxo_pxi")
    sxo_pyi = sym.Symbol("sxo_pyi")
    syo_pxi = sym.Symbol("syo_pxi")
    syo_pyi = sym.Symbol("syo_pyi")
    model = Model4DSTEM(
        overfocus=1.0,
        scan_pixel_pitch=1.0,
        scan_center=PixelYX(y=0.0, x=0.0),
        scan_rotation=0.0,
        camera_length=1.0,
        detector_pixel_pitch=1,
        detector_center=PixelYX(y=0.0, x=0.0),
        semiconv=0.023,
        flip_factor=1.0,
        descan_error=DescanError(
            sxo_pxi=sxo_pxi, sxo_pyi=sxo_pyi, syo_pxi=syo_pxi, syo_pyi=syo_pyi
        ),
    )
    res = model.trace(scan_pos=scan, source_dx=0.5, source_dy=-0.1)

    # no descan error contribution from s*o_p*i parameters
    # if beam is not deflected by scanner
    if scan.x == 0 and scan.y == 0:
        keys = (
            "source",
            "overfocus",
            "scanner",
            "specimen",
            "descanner",
            "camera_length",
            "detector",
        )
    else:
        keys = ("source", "overfocus", "scanner", "specimen")
    for key in keys:
        sect_ref = res_ref[key]
        sect = res[key]
        for attr in ("y", "x", "dy", "dx", "z"):
            assert sym.sympify(getattr(sect.ray, attr)).equals(
                getattr(sect_ref.ray, attr)
            )
    sect_ref = res_ref["descanner"]
    sect = res["descanner"]
    assert sym.sympify(sect.ray.dx).equals(
        sect_ref.ray.dx + sxo_pxi * scan.x + sxo_pyi * scan.y
    )
    assert sym.sympify(sect.ray.dy).equals(
        sect_ref.ray.dy + syo_pxi * scan.x + syo_pyi * scan.y
    )
    assert sym.sympify(sect.ray.x).equals(sect_ref.ray.x)
    assert sym.sympify(sect.ray.y).equals(sect_ref.ray.y)
    assert sym.sympify(sect.ray.z).equals(sect_ref.ray.z)
    # Straight propagation
    for key in ("camera_length", "detector"):
        start = res["descanner"]
        stop = res[key]
        assert sym.sympify(stop.ray.x).equals(
            start.ray.x + start.ray.dx * (stop.ray.z - start.ray.z)
        )
        assert sym.sympify(stop.ray.y).equals(
            start.ray.y + start.ray.dy * (stop.ray.z - start.ray.z)
        )
        assert sym.sympify(stop.ray.dx).equals(start.ray.dx)
        assert sym.sympify(stop.ray.dy).equals(start.ray.dy)


def test_jax_smoke():
    model = Model4DSTEM(
        overfocus=0.7,
        scan_pixel_pitch=0.005,
        scan_center=PixelYX(y=17, x=13),
        scan_rotation=1.234,
        camera_length=2.3,
        detector_pixel_pitch=0.0247,
        detector_center=PixelYX(y=11, x=19),
        semiconv=0.023,
        flip_factor=-1.0,
        descan_error=DescanError(offpxi=0.345, pxo_pxi=948),
    )

    class InputArrT(NamedTuple):
        scan_y: float
        scan_x: float
        tilt_y: float
        tilt_x: float
        one: float = 1.0

    inp = InputArrT(
        scan_y=sym.Symbol("scan_y"),
        scan_x=sym.Symbol("scan_x"),
        tilt_y=sym.Symbol("tilt_y"),
        tilt_x=sym.Symbol("tilt_x"),
        one=1.0,
    )

    @jax.jit
    @lambdify(modules=jnp, kwargs={"arr": inp})
    def test_func(arr: InputArrT):
        scan_y, scan_x, tilt_y, tilt_x, _one = arr
        scan_pos = PixelYX(x=scan_x, y=scan_y)
        res = model.trace(
            scan_pos=scan_pos, source_dy=tilt_y, source_dx=tilt_x, _one=_one
        )
        return (
            res["specimen"].sampling["scan_px"].y,
            res["specimen"].sampling["scan_px"].x,
            res["detector"].sampling["detector_px"].y,
            res["detector"].sampling["detector_px"].x,
            res["detector"].ray._one,
        )

    sample = InputArrT(0.0, 0.0, 0.0, 0.0, 1.0)
    test_func(sample)
    jax.jacobian(test_func)(sample)


def measure_descan_deviation(model, target_model):
    distances = []

    @lambdify(modules=np)
    def distance(scan_y, scan_x, cl):
        ref_model = model.derive(camera_length=cl)
        ref = ref_model.trace(
            scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0.0, source_dx=0.0
        )
        opt_model = target_model.derive(
            camera_length=cl,
        )
        opt = opt_model.trace(
            scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0.0, source_dx=0.0
        )
        return (
            (
                opt["detector"].sampling["detector_px"].y
                - ref["detector"].sampling["detector_px"].y
            ),
            (
                opt["detector"].sampling["detector_px"].x
                - ref["detector"].sampling["detector_px"].x
            ),
        )

    for scan_y in (0, 1):
        for scan_x in (0, 1):
            for cl in (0, 1):
                distances.append(distance(scan_y=scan_y, scan_x=scan_x, cl=cl))
    return np.linalg.norm(np.array(distances))


def test_adjust_scan_rotation(random_model: Model4DSTEM):
    scan_rotation = np.random.uniform(-sym.pi, sym.pi)
    modified = random_model.adjust_scan_rotation(
        scan_rotation=scan_rotation,
    )
    print(random_model, scan_rotation, modified)
    assert_allclose(
        0,
        measure_descan_deviation(
            random_model,
            modified,
        ),
        atol=1e-12,
    )
    assert modified.scan_rotation == scan_rotation


def test_adjust_scan_pixel_pitch(random_model):
    scan_pixel_pitch = np.random.uniform(0.0001, 2)
    modified = random_model.adjust_scan_pixel_pitch(
        scan_pixel_pitch=scan_pixel_pitch,
    )
    print(random_model, scan_pixel_pitch, modified)
    assert_allclose(
        0,
        measure_descan_deviation(
            random_model,
            modified,
        ),
        atol=1e-12,
    )
    assert modified.scan_pixel_pitch == scan_pixel_pitch


def test_adjust_scan_center(random_model):
    scan_center = PixelYX(
        y=np.random.uniform(-10, 10),
        x=np.random.uniform(-10, 10),
    )
    modified = random_model.adjust_scan_center(
        scan_center=scan_center,
    )
    print(random_model, scan_center, modified)
    assert_allclose(
        0,
        measure_descan_deviation(
            random_model,
            modified,
        ),
        atol=1e-12,
    )
    assert modified.scan_center == scan_center


def test_adjust_detector_rotation(random_model):
    detector_rotation = np.random.uniform(-sym.pi, sym.pi)
    modified = random_model.adjust_detector_rotation(
        detector_rotation=detector_rotation,
    )
    print(random_model, detector_rotation, modified)
    assert_allclose(
        0,
        measure_descan_deviation(
            random_model,
            modified,
        ),
        atol=1e-12,
    )
    assert modified.detector_rotation == detector_rotation


def test_adjust_flip_y(random_model):
    for flip_factor in (-1.0, 1.0):
        modified = random_model.adjust_flip_factor(
            flip_factor=flip_factor,
        )
        print(random_model, flip_factor, modified)
        assert_allclose(
            0,
            measure_descan_deviation(
                random_model,
                modified,
            ),
            atol=1e-12,
        )
        assert modified.flip_factor == flip_factor


def test_adjust_detector_center(random_model):
    detector_center = PixelYX(
        y=np.random.uniform(-10, 10),
        x=np.random.uniform(-10, 10),
    )
    modified = random_model.adjust_detector_center(
        detector_center=detector_center,
    )
    print(random_model, detector_center, modified)
    assert_allclose(
        0,
        measure_descan_deviation(
            random_model,
            modified,
        ),
        atol=1e-12,
    )
    assert modified.detector_center == detector_center


def test_adjust_detector_pixel_pitch(random_model):
    detector_pixel_pitch = np.random.uniform(0.0001, 2)
    modified = random_model.adjust_detector_pixel_pitch(
        detector_pixel_pitch=detector_pixel_pitch,
    )
    print(random_model, detector_pixel_pitch, modified)
    assert_allclose(
        0,
        measure_descan_deviation(
            random_model,
            modified,
        ),
        atol=1e-12,
    )
    assert modified.detector_pixel_pitch == detector_pixel_pitch


def test_adjust_camera_length(random_model):
    camera_length = np.random.uniform(0.0001, 2)
    modified = random_model.adjust_camera_length(camera_length)
    ratio = modified.camera_length / random_model.camera_length
    print(random_model, camera_length, modified)

    distances = []

    @lambdify(modules=np)
    def distance(scan_y, scan_x, cl):
        ref_model = random_model.derive(camera_length=cl)
        ref = ref_model.trace(
            scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0.0, source_dx=0.0
        )
        # Scale by `ratio`
        opt_model = modified.derive(
            camera_length=cl * ratio,
        )
        opt = opt_model.trace(
            scan_pos=PixelYX(y=scan_y, x=scan_x), source_dy=0.0, source_dx=0.0
        )
        return (
            opt["detector"].sampling["detector_px"].y
            - ref["detector"].sampling["detector_px"].y,
            opt["detector"].sampling["detector_px"].x
            - ref["detector"].sampling["detector_px"].x,
        )

    # We check that the model produces the same pixel offsets
    # at the camera length scaled by `ratio`
    for scan_y in (0, 1):
        for scan_x in (0, 1):
            for cl in (0, 1):
                distances.append(distance(scan_y=scan_y, scan_x=scan_x, cl=cl))
    assert_allclose(0, np.linalg.norm(distances), atol=1e-12)
    assert modified.camera_length == camera_length
