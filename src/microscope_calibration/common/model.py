from typing import Optional, NamedTuple, Union
from collections import OrderedDict

import jax_dataclasses as jdc
import jax.numpy as jnp

from libertem.corrections import coordinates as ltcoords

from temgym_core.ray import Ray
from temgym_core import PixelYX, CoordXY
from temgym_core.components import (
    Component, Plane, Descanner, Scanner, DescanError
)
from temgym_core.run import run_iter
from temgym_core.source import Source, PointSource
from temgym_core.propagator import Propagator, FreeSpaceParaxial

# FIXME workaround until code is released in final location
try:
    from libertem.corrections.coordinates import scale_rotate_flip_y
except ImportError:
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
        rot[:, 0] *= flip_factor

        angle1 = jnp.arctan2(-rot[1, 0], rot[0, 0])
        angle2 = jnp.arctan2(rot[0, 1], rot[1, 1])

        # So far not reached in tests since inconsistencies are caught as shear before
        if not jnp.allclose((jnp.sin(angle1), jnp.cos(angle1)), (jnp.sin(angle2), jnp.cos(angle2))):
            raise ValueError(
                f'Rotation angle 1 {angle1} and rotation angle 2 {angle2} are inconsistent.'
            )

        return (scale_y, angle1, flip_y)


# TODO use LiberTEM-schema later
@jdc.pytree_dataclass
class Parameters4DSTEM:
    overfocus: float  # m
    scan_pixel_pitch: float  # m
    scan_cy: float  # px
    scan_cx: float  # px
    scan_rotation: float  # rad
    camera_length: float  # m
    detector_pixel_pitch: float  # m
    detector_cy: float  # px
    detector_cx: float  # px
    semiconv: float  # rad
    flip_y: bool
    descan_error: DescanError = DescanError()


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

    scan_cy: float
    scan_cx: float
    detector_cy: float
    detector_cx: float

    @property
    def overfocus(self) -> float:
        return self.specimen.z - self.source.z

    @property
    def camera_length(self) -> float:
        return self.detector.z - self.specimen.z

    def scan_to_real(self, pixels: PixelYX, _one: float = 1.) -> CoordXY:
        (y, x) = self._scan_to_real @ jnp.array((
            pixels.y - self.scan_cy*_one, pixels.x - self.scan_cx*_one
        ))
        return CoordXY(y=y, x=x)

    def real_to_scan(self, coords: CoordXY, _one: float = 1.) -> PixelYX:
        (y, x) = self._real_to_scan @ jnp.array((coords.y, coords.x))
        return PixelYX(y=y + self.scan_cy*_one, x=x + self.scan_cx*_one)

    def detector_to_real(self, pixels: PixelYX, _one: float = 1.) -> CoordXY:
        (y, x) = self._detector_to_real @ jnp.array((
            pixels.y - self.detector_cy*_one, pixels.x - self.detector_cx*_one))
        return CoordXY(y=y, x=x)

    def real_to_detector(self, coords: CoordXY, _one: float = 1.) -> PixelYX:
        (y, x) = self._real_to_detector @ jnp.array((coords.y, coords.x))
        return PixelYX(y=y + self.detector_cy*_one, x=x + self.detector_cx*_one)

    @classmethod
    def build(
            cls, params: Parameters4DSTEM, scan_pos: PixelYX,
            specimen: Optional[Component] = None) -> 'Model4DSTEM':
        scan_to_real = ltcoords.rotate(params.scan_rotation)\
            @ ltcoords.scale(params.scan_pixel_pitch)
        real_to_scan = jnp.linalg.inv(scan_to_real)
        scan_y, scan_x = scan_to_real @ (
            scan_pos.y - params.scan_cy,
            scan_pos.x - params.scan_cx,
        )
        do_flip = ltcoords.flip_y() if params.flip_y else ltcoords.identity()
        detector_to_real = ltcoords.scale(params.detector_pixel_pitch) @ do_flip
        real_to_detector = jnp.linalg.inv(detector_to_real)
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
            scan_cy=params.scan_cy,
            scan_cx=params.scan_cx,
            detector_cy=params.detector_cy,
            detector_cx=params.detector_cx,
        )

    @property
    def params(self) -> Parameters4DSTEM:
        scan_scale, scan_rotation, scan_flip = ltcoords.scale_rotate_flip_y(self._scan_to_real)
        assert scan_flip is False
        detector_scale, detector_rotation, detector_flip = ltcoords.scale_rotate_flip_y(
            self._detector_to_real
        )
        assert jnp.allclose(detector_rotation, 0.)
        return Parameters4DSTEM(
            overfocus=self.specimen.z - self.source.z,
            scan_pixel_pitch=scan_scale,
            scan_cy=self.scan_cy,
            scan_cx=self.scan_cx,
            scan_rotation=scan_rotation,
            camera_length=self.detector.z - self.specimen.z,
            detector_pixel_pitch=detector_scale,
            detector_cy=self.detector_cy,
            detector_cx=self.detector_cx,
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
