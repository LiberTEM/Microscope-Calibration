from typing import Optional, NamedTuple, Union
from collections import OrderedDict

import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax_dataclasses as jdc
import jax.numpy as jnp

from temgym_core.ray import Ray
from temgym_core import PixelYX, CoordXY
from temgym_core.components import (
    Component, Plane, Descanner, Scanner, DescanError
)
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
    return jnp.array([
        (jnp.cos(radians), jnp.sin(radians)),
        (-jnp.sin(radians), jnp.cos(radians))
    ])


def flip_y():
    return jnp.array([
        (-1, 0),
        (0, 1)
    ], dtype=jnp.float64)


def identity():
    return jnp.eye(2, dtype=jnp.float64)


def scale_rotate_flip_y(mat: jnp.ndarray):
    '''
    Deconstruct a matrix generated with scale() @ rotate() @ flip_y()
    into the individual parameters
    '''
    scale_y = jnp.linalg.norm(mat[:, 0])
    scale_x = jnp.linalg.norm(mat[:, 1])
    if not jnp.allclose(scale_y, scale_x):
        raise ValueError(f'y scale {scale_y} and x scale {scale_x} are different.')

    scan_rot_flip = mat / scale_y
    # 2D cross product
    flip_factor = (
        scan_rot_flip[0, 0] * scan_rot_flip[1, 1]
        - scan_rot_flip[0, 1] * scan_rot_flip[1, 0]
    )
    # Make sure no scale or shear left
    if not jnp.allclose(jnp.abs(flip_factor), 1.):
        raise ValueError(
            f'Contains shear: flip factor (2D cross product) is {flip_factor}.'
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
        jnp.array((jnp.sin(angle2), jnp.cos(angle2)))
    ):
        raise ValueError(
            f'Rotation angle 1 {angle1} and rotation angle 2 {angle2} are inconsistent.'
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
    detector_rotation: float = 0.  # rad

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
    ) -> 'Parameters4DSTEM':
        return Parameters4DSTEM(
            overfocus=overfocus if overfocus is not None else self.overfocus,
            scan_pixel_pitch=(
                scan_pixel_pitch if scan_pixel_pitch is not None
                else self.scan_pixel_pitch
            ),
            scan_center=scan_center if scan_center is not None else self.scan_center,
            scan_rotation=scan_rotation if scan_rotation is not None else self.scan_rotation,
            camera_length=camera_length if camera_length is not None else self.camera_length,
            detector_pixel_pitch=(
                detector_pixel_pitch if detector_pixel_pitch is not None
                else self.detector_pixel_pitch
            ),
            detector_center=(
                detector_center if detector_center is not None
                else self.detector_center
            ),
            detector_rotation=(
                detector_rotation if detector_rotation is not None
                else self.detector_rotation
            ),
            semiconv=semiconv if semiconv is not None else self.semiconv,
            flip_y=flip_y if flip_y is not None else self.flip_y,
            descan_error=descan_error if descan_error is not None else self.descan_error,
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

    def scan_to_real(self, pixels: PixelYX, _one: float = 1.) -> CoordXY:
        (y, x) = self._scan_to_real @ jnp.array((
            pixels.y - self.scan_center.y*_one, pixels.x - self.scan_center.x*_one
        ))
        return CoordXY(y=y, x=x)

    def real_to_scan(self, coords: CoordXY, _one: float = 1.) -> PixelYX:
        (y, x) = self._real_to_scan @ jnp.array((coords.y, coords.x))
        return PixelYX(y=y + self.scan_center.y*_one, x=x + self.scan_center.x*_one)

    def detector_to_real(self, pixels: PixelYX, _one: float = 1.) -> CoordXY:
        (y, x) = self._detector_to_real @ jnp.array((
            pixels.y - self.detector_center.y*_one, pixels.x - self.detector_center.x*_one))
        return CoordXY(y=y, x=x)

    def real_to_detector(self, coords: CoordXY, _one: float = 1.) -> PixelYX:
        (y, x) = self._real_to_detector @ jnp.array((coords.y, coords.x))
        return PixelYX(y=y + self.detector_center.y*_one, x=x + self.detector_center.x*_one)

    @classmethod
    def build(
            cls, params: Parameters4DSTEM, scan_pos: PixelYX,
            specimen: Optional[Component] = None) -> 'Model4DSTEM':
        scan_to_real = rotate(params.scan_rotation)\
            @ scale(params.scan_pixel_pitch)
        real_to_scan = scale(1/params.scan_pixel_pitch) @ rotate(-params.scan_rotation)
        scan_y, scan_x = scan_to_real @ jnp.array((
            scan_pos.y - params.scan_center.y,
            scan_pos.x - params.scan_center.x,
        ))
        do_flip = flip_y() if params.flip_y else identity()
        detector_to_real = scale(params.detector_pixel_pitch) @ \
            rotate(params.detector_rotation) @ do_flip
        real_to_detector = do_flip @ rotate(-params.detector_rotation) @ \
            scale(1/params.detector_pixel_pitch)
        if specimen is None:
            specimen = Plane(z=params.overfocus)
        else:
            # FIXME better solution later?
            assert jnp.allclose(specimen.z, params.overfocus)

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
                descan_error=params.descan_error
            ),
            detector=Plane(z=params.overfocus + params.camera_length),
            scan_center=params.scan_center,
            detector_center=params.detector_center
        )

    @property
    def params(self) -> Parameters4DSTEM:
        scan_scale, scan_rotation, scan_flip = scale_rotate_flip_y(self._scan_to_real)
        assert scan_flip is False
        detector_scale, detector_rotation, detector_flip = scale_rotate_flip_y(
            self._detector_to_real
        )
        assert jnp.allclose(detector_rotation, 0.)
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

    def make_source_ray(
            self,
            source_dx: float, source_dy: float,
            _one: float = 1.) -> ResultSection:
        ray = Ray(
            x=self.source.offset_xy.x,
            y=self.source.offset_xy.y,
            dx=source_dx,
            dy=source_dy,
            z=self.source.z,
            pathlength=0.,
            _one=_one,
        )
        return ResultSection(component=self.source, ray=ray)

    def trace(self, ray: Ray) -> Result4DSTEM:
        result = OrderedDict()

        components = (self.source, self.scanner, self.specimen, self.descanner, self.detector)
        # run_iter() currently inserts a propagation if two subsequent
        # components have a non-zero distance, but skips for equal z. We
        # therefore check meticulously that we are actually getting the
        # components and rays we expect. Furthermore, we make sure that our
        # result ALWAYS has the same schema independent of parameters by
        # inserting gaps of zero length manually.
        run_result = list(run_iter(ray=ray, components=components))

        # skip the first propagation, which should be zero distance
        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert comp.distance == 0.
        assert r == ray

        comp, r = run_result.pop(0)
        assert comp == self.source
        assert r == ray
        result['source'] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert isinstance(comp.propagator, FreeSpaceParaxial)
        assert isinstance(r, Ray)
        result['overfocus'] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert comp == self.scanner
        assert isinstance(r, Ray)
        result['scanner'] = ResultSection(component=comp, ray=r)

        # Skip zero distance propagation between scanner and specimen
        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert comp.distance == 0.
        assert isinstance(r, Ray)
        assert r == result['scanner'].ray

        comp, r = run_result.pop(0)
        assert comp == self.specimen
        assert isinstance(r, Ray)
        scan_px = self.real_to_scan(CoordXY(x=r.x, y=r.y), _one=ray._one)
        result['specimen'] = ResultSection(
            component=comp,
            ray=r,
            sampling={'scan_px': scan_px},
        )

        # Skip zero distance propagation between specimen and descanner
        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert comp.distance == 0.
        assert r == result['specimen'].ray

        comp, r = run_result.pop(0)
        assert comp == self.descanner
        assert isinstance(r, Ray)
        result['descanner'] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert isinstance(comp.propagator, FreeSpaceParaxial)
        assert isinstance(r, Ray)
        result['camera_length'] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert comp == self.detector
        assert isinstance(r, Ray)
        detector_px = self.real_to_detector(CoordXY(x=r.x, y=r.y), _one=ray._one)
        result['detector'] = ResultSection(
            component=comp,
            ray=r,
            sampling={'detector_px': detector_px},
        )

        assert len(run_result) == 0
        return result
