from typing import Optional, NamedTuple, Union
from collections import OrderedDict

import jax; jax.config.update("jax_enable_x64", True)  # noqa
import jax_dataclasses as jdc
import jax.numpy as jnp
from jax.errors import TracerBoolConversionError

from temgym_core.ray import Ray
from temgym_core import PixelYX, CoordXY
from temgym_core.components import Component, Plane, Descanner, Scanner, DescanError
from temgym_core.run import run_iter
from temgym_core.source import Source, PointSource
from temgym_core.propagator import Propagator, FreeSpaceParaxial


# Jax-compatible versions of libertem.corrections.coordinates functions
def scale(factor):
    return jnp.eye(2) * factor


def rotate(radians):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    radians = jnp.astype(radians, jnp.float64)
    return jnp.array(
        [(jnp.cos(radians), jnp.sin(radians)), (-jnp.sin(radians), jnp.cos(radians))]
    )


# The flip_factor is introduced to make it differentiable
def flip_y(flip_factor: float = -1.0):
    return jnp.array([(flip_factor, 0), (0, 1)], dtype=jnp.float64)


def identity():
    return jnp.eye(2, dtype=jnp.float64)


def scale_rotate_flip_y(mat: jnp.ndarray):
    """
    Deconstruct a matrix generated with scale() @ rotate() @ flip_y()
    into the individual parameters
    """
    scale_y = jnp.linalg.norm(mat[:, 0])
    scale_x = jnp.linalg.norm(mat[:, 1])
    if not jnp.allclose(scale_y, scale_x):
        raise ValueError(f"y scale {scale_y} and x scale {scale_x} are different.")

    scan_rot_flip = mat / scale_y
    # 2D cross product
    flip_factor = (
        scan_rot_flip[0, 0] * scan_rot_flip[1, 1]
        - scan_rot_flip[0, 1] * scan_rot_flip[1, 0]
    )
    # Make sure no scale or shear left
    if not jnp.allclose(jnp.abs(flip_factor), 1.0):
        raise ValueError(
            f"Contains shear: flip factor (2D cross product) is {flip_factor}."
        )
    flip_y = bool(flip_factor < 0)
    # undo flip_y
    rot = scan_rot_flip.copy()
    rot = rot.at[:, 0].set(rot[:, 0] * flip_factor)

    angle1 = jnp.arctan2(-rot[1, 0], rot[0, 0])
    angle2 = jnp.arctan2(rot[0, 1], rot[1, 1])

    # So far not reached in tests since inconsistencies are caught as shear before
    if not jnp.allclose(
        jnp.array((jnp.sin(angle1), jnp.cos(angle1))),
        jnp.array((jnp.sin(angle2), jnp.cos(angle2))),
    ):
        raise ValueError(
            f"Rotation angle 1 {angle1} and rotation angle 2 {angle2} are inconsistent."
        )

    return (scale_y, angle1, flip_y)


# TODO use LiberTEM-schema later
@jdc.pytree_dataclass
class Parameters4DSTEM:
    overfocus: float  # m
    scan_pixel_pitch: float  # m
    scan_center: PixelYX
    scan_rotation: float  # rad
    camera_length: float  # m
    detector_pixel_pitch: float  # m
    detector_center: PixelYX
    semiconv: float  # rad
    flip_y: bool
    descan_error: DescanError = DescanError()
    detector_rotation: float = 0.0  # rad

    def derive(
        self,
        overfocus: float | None = None,  # m
        scan_pixel_pitch: float | None = None,  # m
        scan_center: PixelYX | None = None,
        scan_rotation: float | None = None,  # rad
        camera_length: float | None = None,  # m
        detector_pixel_pitch: float | None = None,  # m
        detector_center: PixelYX | None = None,
        detector_rotation: float | None = None,  # rad
        semiconv: float | None = None,  # rad
        flip_y: bool | None = None,
        descan_error: DescanError | None = None,
    ) -> "Parameters4DSTEM":
        return Parameters4DSTEM(
            overfocus=overfocus if overfocus is not None else self.overfocus,
            scan_pixel_pitch=(
                scan_pixel_pitch
                if scan_pixel_pitch is not None
                else self.scan_pixel_pitch
            ),
            scan_center=scan_center if scan_center is not None else self.scan_center,
            scan_rotation=scan_rotation
            if scan_rotation is not None
            else self.scan_rotation,
            camera_length=camera_length
            if camera_length is not None
            else self.camera_length,
            detector_pixel_pitch=(
                detector_pixel_pitch
                if detector_pixel_pitch is not None
                else self.detector_pixel_pitch
            ),
            detector_center=(
                detector_center if detector_center is not None else self.detector_center
            ),
            detector_rotation=(
                detector_rotation
                if detector_rotation is not None
                else self.detector_rotation
            ),
            semiconv=semiconv if semiconv is not None else self.semiconv,
            flip_y=flip_y if flip_y is not None else self.flip_y,
            descan_error=descan_error
            if descan_error is not None
            else self.descan_error,
        )

    def normalize_types(self):
        return self.derive(
            overfocus=float(self.overfocus),
            scan_pixel_pitch=float(self.scan_pixel_pitch),
            scan_center=PixelYX(
                y=float(self.scan_center.y),
                x=float(self.scan_center.x),
            ),
            scan_rotation=float(self.scan_rotation),
            camera_length=float(self.camera_length),
            detector_pixel_pitch=float(self.detector_pixel_pitch),
            detector_center=PixelYX(
                y=float(self.detector_center.y),
                x=float(self.detector_center.x),
            ),
            detector_rotation=float(self.detector_rotation),
            semiconv=float(self.semiconv),
            flip_y=bool(self.flip_y),
            descan_error=DescanError(
                pxo_pyi=float(self.descan_error.pxo_pyi),
                pyo_pyi=float(self.descan_error.pyo_pyi),
                pxo_pxi=float(self.descan_error.pxo_pxi),
                pyo_pxi=float(self.descan_error.pyo_pxi),
                sxo_pyi=float(self.descan_error.sxo_pyi),
                syo_pyi=float(self.descan_error.syo_pyi),
                sxo_pxi=float(self.descan_error.sxo_pxi),
                syo_pxi=float(self.descan_error.syo_pxi),
                offpxi=float(self.descan_error.offpxi),
                offpyi=float(self.descan_error.offpyi),
                offsxi=float(self.descan_error.offsxi),
                offsyi=float(self.descan_error.offsyi),
            ),
        )

    def adjust_scan_rotation(self, scan_rotation: float) -> "Parameters4DSTEM":
        """
        Adjust the scan rotation while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        de = self.descan_error
        angle = scan_rotation - self.scan_rotation
        # Rotate the input direction
        pxo_pyi, pxo_pxi = rotate(angle) @ jnp.array((de.pxo_pyi, de.pxo_pxi))
        pyo_pyi, pyo_pxi = rotate(angle) @ jnp.array((de.pyo_pyi, de.pyo_pxi))
        sxo_pyi, sxo_pxi = rotate(angle) @ jnp.array((de.sxo_pyi, de.sxo_pxi))
        syo_pyi, syo_pxi = rotate(angle) @ jnp.array((de.syo_pyi, de.syo_pxi))
        new_de = DescanError(
            pxo_pyi=pxo_pyi,
            pyo_pyi=pyo_pyi,
            pxo_pxi=pxo_pxi,
            pyo_pxi=pyo_pxi,
            sxo_pyi=sxo_pyi,
            syo_pyi=syo_pyi,
            sxo_pxi=sxo_pxi,
            syo_pxi=syo_pxi,
            offpxi=de.offpxi,
            offpyi=de.offpyi,
            offsxi=de.offsxi,
            offsyi=de.offsyi,
        )
        return self.derive(
            scan_rotation=scan_rotation,
            descan_error=new_de,
        )

    def adjust_scan_pixel_pitch(self, scan_pixel_pitch: float) -> "Parameters4DSTEM":
        """
        Adjust the scan pixel pitch while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        de = self.descan_error
        ratio = self.scan_pixel_pitch / scan_pixel_pitch

        new_de = DescanError(
            pxo_pyi=de.pxo_pyi * ratio,
            pyo_pyi=de.pyo_pyi * ratio,
            pxo_pxi=de.pxo_pxi * ratio,
            pyo_pxi=de.pyo_pxi * ratio,
            sxo_pyi=de.sxo_pyi * ratio,
            syo_pyi=de.syo_pyi * ratio,
            sxo_pxi=de.sxo_pxi * ratio,
            syo_pxi=de.syo_pxi * ratio,
            offpxi=de.offpxi,
            offpyi=de.offpyi,
            offsxi=de.offsxi,
            offsyi=de.offsyi,
        )
        return self.derive(
            scan_pixel_pitch=scan_pixel_pitch,
            descan_error=new_de,
        )

    def adjust_scan_center(self, scan_center: PixelYX) -> "Parameters4DSTEM":
        # Compensate effect of different scan centers with
        # constant offsets of the descanner. We simply measure how much these offsets should be
        # by comparing rays along the optical axis
        res1 = trace(self, scan_pos=self.scan_center, source_dx=0.0, source_dy=0.0)
        res2 = trace(self, scan_pos=scan_center, source_dx=0.0, source_dy=0.0)

        de = self.descan_error
        offpxi = de.offpxi + res2["descanner"].ray.x - res1["descanner"].ray.x
        offpyi = de.offpyi + res2["descanner"].ray.y - res1["descanner"].ray.y
        offsxi = de.offsxi + res2["descanner"].ray.dx - res1["descanner"].ray.dx
        offsyi = de.offsyi + res2["descanner"].ray.dy - res1["descanner"].ray.dy

        new_de = DescanError(
            pxo_pyi=de.pxo_pyi,
            pyo_pyi=de.pyo_pyi,
            pxo_pxi=de.pxo_pxi,
            pyo_pxi=de.pyo_pxi,
            sxo_pyi=de.sxo_pyi,
            syo_pyi=de.syo_pyi,
            sxo_pxi=de.sxo_pxi,
            syo_pxi=de.syo_pxi,
            offpxi=offpxi,
            offpyi=offpyi,
            offsxi=offsxi,
            offsyi=offsyi,
        )
        return self.derive(
            scan_center=scan_center,
            descan_error=new_de,
        )

    def adjust_detector_rotation(self, detector_rotation: float) -> "Parameters4DSTEM":
        de = self.descan_error
        angle = detector_rotation - self.detector_rotation
        # rotate the output direction
        pyo_pyi, pxo_pyi = rotate(angle) @ jnp.array((de.pyo_pyi, de.pxo_pyi))
        pyo_pxi, pxo_pxi = rotate(angle) @ jnp.array((de.pyo_pxi, de.pxo_pxi))
        syo_pyi, sxo_pyi = rotate(angle) @ jnp.array((de.syo_pyi, de.sxo_pyi))
        syo_pxi, sxo_pxi = rotate(angle) @ jnp.array((de.syo_pxi, de.sxo_pxi))
        offpyi, offpxi = rotate(angle) @ jnp.array((de.offpyi, de.offpxi))
        offsyi, offsxi = rotate(angle) @ jnp.array((de.offsyi, de.offsxi))
        new_de = DescanError(
            pxo_pyi=pxo_pyi,
            pyo_pyi=pyo_pyi,
            pxo_pxi=pxo_pxi,
            pyo_pxi=pyo_pxi,
            sxo_pyi=sxo_pyi,
            syo_pyi=syo_pyi,
            sxo_pxi=sxo_pxi,
            syo_pxi=syo_pxi,
            offpxi=offpxi,
            offpyi=offpyi,
            offsxi=offsxi,
            offsyi=offsyi,
        )
        return self.derive(
            detector_rotation=detector_rotation,
            descan_error=new_de,
        )

    def adjust_flip_y(self, flip_y: bool) -> "Parameters4DSTEM":
        # Some import gymnastic to keep the naming clean
        from .model import flip_y as fl

        de = self.descan_error
        angle = self.detector_rotation
        if flip_y != self.flip_y:
            # Rotate into detector directions, flip, then rotate back
            trans = rotate(angle) @ fl() @ rotate(-angle)
            # transform the output direction
            pyo_pyi, pxo_pyi = trans @ jnp.array((de.pyo_pyi, de.pxo_pyi))
            pyo_pxi, pxo_pxi = trans @ jnp.array((de.pyo_pxi, de.pxo_pxi))
            syo_pyi, sxo_pyi = trans @ jnp.array((de.syo_pyi, de.sxo_pyi))
            syo_pxi, sxo_pxi = trans @ jnp.array((de.syo_pxi, de.sxo_pxi))
            offpyi, offpxi = trans @ jnp.array((de.offpyi, de.offpxi))
            offsyi, offsxi = trans @ jnp.array((de.offsyi, de.offsxi))
            new_de = DescanError(
                pxo_pyi=pxo_pyi,
                pyo_pyi=pyo_pyi,
                pxo_pxi=pxo_pxi,
                pyo_pxi=pyo_pxi,
                sxo_pyi=sxo_pyi,
                syo_pyi=syo_pyi,
                sxo_pxi=sxo_pxi,
                syo_pxi=syo_pxi,
                offpxi=offpxi,
                offpyi=offpyi,
                offsxi=offsxi,
                offsyi=offsyi,
            )
            return self.derive(
                flip_y=not self.flip_y,
                descan_error=new_de,
            )
        else:
            return self

    def adjust_detector_center(self, detector_center: PixelYX) -> "Parameters4DSTEM":
        cls = Model4DSTEM
        de = self.descan_error
        zero = PixelYX(0, 0)
        model1 = cls.build(params=self, scan_pos=zero)
        model2 = cls.build(
            params=self.derive(
                detector_center=detector_center,
            ),
            scan_pos=zero,
        )
        physical_1 = model1.detector_to_real(zero)
        physical_2 = model2.detector_to_real(zero)
        offpyi = de.offpyi + physical_2.y - physical_1.y
        offpxi = de.offpxi + physical_2.x - physical_1.x
        new_de = DescanError(
            pxo_pyi=de.pxo_pyi,
            pyo_pyi=de.pyo_pyi,
            pxo_pxi=de.pxo_pxi,
            pyo_pxi=de.pyo_pxi,
            sxo_pyi=de.sxo_pyi,
            syo_pyi=de.syo_pyi,
            sxo_pxi=de.sxo_pxi,
            syo_pxi=de.syo_pxi,
            offpxi=offpxi,
            offpyi=offpyi,
            offsxi=de.offsxi,
            offsyi=de.offsyi,
        )
        return self.derive(
            detector_center=detector_center,
            descan_error=new_de,
        )

    def adjust_detector_pixel_pitch(
        self, detector_pixel_pitch: float
    ) -> "Parameters4DSTEM":
        de = self.descan_error
        ratio = detector_pixel_pitch / self.detector_pixel_pitch

        new_de = DescanError(
            pxo_pyi=de.pxo_pyi * ratio,
            pyo_pyi=de.pyo_pyi * ratio,
            pxo_pxi=de.pxo_pxi * ratio,
            pyo_pxi=de.pyo_pxi * ratio,
            sxo_pyi=de.sxo_pyi * ratio,
            syo_pyi=de.syo_pyi * ratio,
            sxo_pxi=de.sxo_pxi * ratio,
            syo_pxi=de.syo_pxi * ratio,
            offpxi=de.offpxi * ratio,
            offpyi=de.offpyi * ratio,
            offsxi=de.offsxi * ratio,
            offsyi=de.offsyi * ratio,
        )
        return self.derive(
            detector_pixel_pitch=detector_pixel_pitch,
            descan_error=new_de,
        )

    def adjust_camera_length(self, camera_length: float) -> "Parameters4DSTEM":
        de = self.descan_error
        ratio = self.camera_length / camera_length

        new_de = DescanError(
            pxo_pyi=de.pxo_pyi,
            pyo_pyi=de.pyo_pyi,
            pxo_pxi=de.pxo_pxi,
            pyo_pxi=de.pyo_pxi,
            sxo_pyi=de.sxo_pyi * ratio,
            syo_pyi=de.syo_pyi * ratio,
            sxo_pxi=de.sxo_pxi * ratio,
            syo_pxi=de.syo_pxi * ratio,
            offpxi=de.offpxi,
            offpyi=de.offpyi,
            offsxi=de.offsxi * ratio,
            offsyi=de.offsyi * ratio,
        )
        return self.derive(
            camera_length=camera_length,
            descan_error=new_de,
        )


# "Layer" of a beam passing through a model
class ResultSection(NamedTuple):
    component: Union[Component, Source, Propagator]
    ray: Ray
    sampling: Optional[dict] = None


# Layer stack, result of tracing a ray through a model
Result4DSTEM = OrderedDict[str, ResultSection]


@jdc.pytree_dataclass
class Model4DSTEM:
    source: PointSource
    scanner: Scanner
    specimen: Plane
    descanner: Descanner
    detector: Plane

    _scan_to_real: jnp.ndarray  # 2x2 matrix from libertem.corrections.coordinates
    _real_to_scan: jnp.ndarray  # 2x2 matrix from libertem.corrections.coordinates
    _detector_to_real: jnp.ndarray  # 2x2 matrix from libertem.corrections.coordinates
    _real_to_detector: jnp.ndarray  # 2x2 matrix from libertem.corrections.coordinates

    scan_center: PixelYX
    detector_center: PixelYX

    @property
    def overfocus(self) -> float:
        return self.specimen.z - self.source.z

    @property
    def camera_length(self) -> float:
        return self.detector.z - self.specimen.z

    def scan_to_real(self, pixels: PixelYX, _one: float = 1.0) -> CoordXY:
        (y, x) = self._scan_to_real @ jnp.array(
            (pixels.y - self.scan_center.y * _one, pixels.x - self.scan_center.x * _one)
        )
        return CoordXY(y=y, x=x)

    def real_to_scan(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        (y, x) = self._real_to_scan @ jnp.array((coords.y, coords.x))
        return PixelYX(y=y + self.scan_center.y * _one, x=x + self.scan_center.x * _one)

    def detector_to_real(self, pixels: PixelYX, _one: float = 1.0) -> CoordXY:
        (y, x) = self._detector_to_real @ jnp.array(
            (
                pixels.y - self.detector_center.y * _one,
                pixels.x - self.detector_center.x * _one,
            )
        )
        return CoordXY(y=y, x=x)

    def real_to_detector(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        (y, x) = self._real_to_detector @ jnp.array((coords.y, coords.x))
        return PixelYX(
            y=y + self.detector_center.y * _one, x=x + self.detector_center.x * _one
        )

    @classmethod
    def build(
        cls,
        params: Parameters4DSTEM,
        scan_pos: PixelYX,
        specimen: Optional[Component] = None,
    ) -> "Model4DSTEM":
        scan_to_real = rotate(params.scan_rotation) @ scale(params.scan_pixel_pitch)
        real_to_scan = scale(1 / params.scan_pixel_pitch) @ rotate(
            -params.scan_rotation
        )
        scan_y, scan_x = scan_to_real @ jnp.array(
            (
                scan_pos.y - params.scan_center.y,
                scan_pos.x - params.scan_center.x,
            )
        )
        do_flip = flip_y((-1) ** params.flip_y)
        detector_to_real = (
            scale(params.detector_pixel_pitch)
            @ rotate(params.detector_rotation)
            @ do_flip
        )
        real_to_detector = (
            do_flip
            @ rotate(-params.detector_rotation)
            @ scale(1 / params.detector_pixel_pitch)
        )
        if specimen is None:
            specimen = Plane(z=params.overfocus)
        else:
            try:
                # FIXME better solution later?
                assert jnp.allclose(specimen.z, params.overfocus)
            except TracerBoolConversionError:
                pass
        return cls(
            source=PointSource(z=0, semi_conv=params.semiconv),
            _scan_to_real=scan_to_real,
            _real_to_scan=real_to_scan,
            _detector_to_real=detector_to_real,
            _real_to_detector=real_to_detector,
            scanner=Scanner(z=params.overfocus, scan_pos_x=scan_x, scan_pos_y=scan_y),
            specimen=specimen,
            descanner=Descanner(
                z=params.overfocus,
                scan_pos_x=scan_x,
                scan_pos_y=scan_y,
                descan_error=params.descan_error,
            ),
            detector=Plane(z=params.overfocus + params.camera_length),
            scan_center=params.scan_center,
            detector_center=params.detector_center,
        )

    @property
    def params(self) -> Parameters4DSTEM:
        scan_scale, scan_rotation, scan_flip = scale_rotate_flip_y(self._scan_to_real)
        assert scan_flip is False
        detector_scale, detector_rotation, detector_flip = scale_rotate_flip_y(
            self._detector_to_real
        )
        assert jnp.allclose(detector_rotation, 0.0)
        return Parameters4DSTEM(
            overfocus=self.specimen.z - self.source.z,
            scan_pixel_pitch=scan_scale,
            scan_center=self.scan_center,
            scan_rotation=scan_rotation,
            camera_length=self.detector.z - self.specimen.z,
            detector_pixel_pitch=detector_scale,
            detector_center=self.detector_center,
            semiconv=self.source.semi_conv,
            flip_y=detector_flip,
            descan_error=self.descanner.descan_error,
        )

    @property
    def scan_pos(self):
        y = self.scanner.scan_pos_y
        x = self.scanner.scan_pos_x

        assert self.scanner.scan_tilt_y == 0.0
        assert self.scanner.scan_tilt_x == 0.0

        assert self.scanner.scan_pos_y == self.descanner.scan_pos_y
        assert self.scanner.scan_pos_x == self.descanner.scan_pos_x
        assert self.descanner.scan_tilt_y == 0.0
        assert self.descanner.scan_tilt_x == 0.0

        return self.real_to_scan(CoordXY(x=x, y=y))

    def make_source_ray(
        self, source_dx: float, source_dy: float, _one: float = 1.0
    ) -> ResultSection:
        ray = Ray(
            x=self.source.offset_xy.x,
            y=self.source.offset_xy.y,
            dx=source_dx,
            dy=source_dy,
            z=self.source.z,
            pathlength=0.0,
            _one=_one,
        )
        return ResultSection(component=self.source, ray=ray)

    @property
    def components(self):
        return (self.source, self.scanner, self.specimen, self.descanner, self.detector)

    def trace(self, ray: Ray) -> Result4DSTEM:
        result = OrderedDict()

        # run_iter() currently inserts a propagation if two subsequent
        # components have a non-zero distance, but skips for equal z. We
        # therefore check meticulously that we are actually getting the
        # components and rays we expect. Furthermore, we make sure that our
        # result ALWAYS has the same schema independent of parameters by
        # inserting gaps of zero length manually.
        run_result = list(run_iter(ray=ray, components=self.components))

        # skip the first propagation, which should be zero distance
        comp, r = run_result.pop(0)
        try:
            assert isinstance(comp, Propagator)
            assert comp.distance == 0.0
            assert r == ray
        except TracerBoolConversionError:
            pass

        comp, r = run_result.pop(0)
        try:
            assert comp == self.source
            assert r == ray
        except TracerBoolConversionError:
            pass
        result["source"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert isinstance(comp.propagator, FreeSpaceParaxial)
        assert isinstance(r, Ray)
        result["overfocus"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        try:
            assert comp == self.scanner
            assert isinstance(r, Ray)
        except TracerBoolConversionError:
            pass
        result["scanner"] = ResultSection(component=comp, ray=r)

        # Skip zero distance propagation between scanner and specimen
        comp, r = run_result.pop(0)
        try:
            assert isinstance(comp, Propagator)
            assert comp.distance == 0.0
            assert isinstance(r, Ray)
            assert r == result["scanner"].ray
        except TracerBoolConversionError:
            pass

        comp, r = run_result.pop(0)
        try:
            assert comp == self.specimen
            assert isinstance(r, Ray)
        except TracerBoolConversionError:
            pass
        scan_px = self.real_to_scan(CoordXY(x=r.x, y=r.y), _one=ray._one)
        result["specimen"] = ResultSection(
            component=comp,
            ray=r,
            sampling={"scan_px": scan_px},
        )

        # Skip zero distance propagation between specimen and descanner
        comp, r = run_result.pop(0)
        try:
            assert isinstance(comp, Propagator)
            assert comp.distance == 0.0
            assert r == result["specimen"].ray
        except TracerBoolConversionError:
            pass
        comp, r = run_result.pop(0)
        try:
            assert comp == self.descanner
            assert isinstance(r, Ray)
        except TracerBoolConversionError:
            pass
        result["descanner"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert isinstance(comp.propagator, FreeSpaceParaxial)
        assert isinstance(r, Ray)
        result["camera_length"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        try:
            assert comp == self.detector
            assert isinstance(r, Ray)
        except TracerBoolConversionError:
            pass
        detector_px = self.real_to_detector(CoordXY(x=r.x, y=r.y), _one=ray._one)
        result["detector"] = ResultSection(
            component=comp,
            ray=r,
            sampling={"detector_px": detector_px},
        )

        assert len(run_result) == 0
        return result


def trace(
    params: Parameters4DSTEM,
    scan_pos: PixelYX,
    source_dx: float,
    source_dy: float,
    specimen: Component | None = None,
    _one: float = 1.0,
) -> Result4DSTEM:
    model = Model4DSTEM.build(params, scan_pos=scan_pos, specimen=specimen)
    ray = model.make_source_ray(source_dy=source_dy, source_dx=source_dx, _one=_one).ray
    return model.trace(ray)
