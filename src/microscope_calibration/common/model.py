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


# Simple transformation function similar to propagation by components
def scale(c: CoordXY | PixelYX, factor: float) -> CoordXY:
    return CoordXY(
        x=c.x * factor,
        y=c.y * factor,
    )


def rotate(c: CoordXY | PixelYX, radians: float) -> CoordXY:
    cosr = jnp.cos(radians)
    sinr = jnp.sin(radians)
    return CoordXY(
        x=c.x * cosr - c.y * sinr,
        y=c.y * cosr + c.x * sinr,
    )


# The flip_factor is introduced to make it differentiable
def flip_y(c: CoordXY | PixelYX, flip_factor: float = -1.0) -> CoordXY:
    return CoordXY(
        x=c.x,
        y=c.y * flip_factor,
    )


def shift(c: CoordXY | PixelYX, s: CoordXY | PixelYX, _one: float = 1.0) -> CoordXY:
    return CoordXY(
        y=c.y + s.y * _one,
        x=c.x + s.x * _one,
    )


# "Layer" of a beam passing through a model
class ResultSection(NamedTuple):
    component: Union[Component, Source, Propagator]
    ray: Ray
    sampling: Optional[dict] = None


# Layer stack, result of tracing a ray through a model
Result4DSTEM = OrderedDict[str, ResultSection]


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
    flip_factor: float  # 1.: no flip; -1.: flip
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
        flip_factor: float | None = None,
        descan_error: DescanError | None = None,
    ) -> "Parameters4DSTEM":
        if flip_factor is not None:
            assert flip_y is None
        if flip_y is not None:
            flip_factor = -1. if flip_y else 1.

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
            flip_factor=flip_factor if flip_factor is not None else self.flip_factor,
            descan_error=descan_error
            if descan_error is not None
            else self.descan_error,
        )

    def normalize_types(self) -> 'Parameters4DSTEM':
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
            flip_factor=float(self.flip_factor),
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

    def scan_to_real(self, pixels: PixelYX, _one: float = 1.0) -> CoordXY:
        return rotate(
            c=scale(
                c=shift(
                    c=pixels,
                    s=scale(
                        c=self.scan_center,
                        factor=-1,
                    ),
                    _one=_one,
                ),
                factor=self.scan_pixel_pitch,
            ),
            radians=self.scan_rotation,
        )

    def real_to_scan(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        res = shift(
            c=scale(
                c=rotate(
                    c=coords,
                    radians=-self.scan_rotation,
                ),
                factor=1/self.scan_pixel_pitch,
            ),
            s=self.scan_center,
            _one=_one,
        )
        return PixelYX(
            y=res.y,
            x=res.x,
        )

    def detector_to_real(self, pixels: PixelYX, _one: float = 1.0) -> CoordXY:
        return rotate(
            c=scale(
                c=flip_y(
                    c=shift(
                        c=pixels,
                        s=scale(
                            c=self.detector_center,
                            factor=-1,
                        ),
                        _one=_one,
                    ),
                    flip_factor=self.flip_factor,
                ),
                factor=self.detector_pixel_pitch,
            ),
            radians=self.detector_rotation,
        )

    def real_to_detector(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        res = shift(
            c=flip_y(
                c=scale(
                    c=rotate(
                        c=coords,
                        radians=-self.detector_rotation,
                    ),
                    factor=1/self.detector_pixel_pitch,
                ),
                flip_factor=self.flip_factor,
            ),
            s=self.detector_center,
            _one=_one,
        )
        return PixelYX(
            y=res.y,
            x=res.x,
        )

    def _components(
        self,
        scan_pos: PixelYX,
        specimen: Component | None = None,
        _one: float = 1.
    ) -> OrderedDict[str, Component]:
        if specimen is None:
            specimen = Plane(z=self.overfocus)
        else:
            try:
                # FIXME better solution later?
                assert jnp.allclose(specimen.z, self.overfocus)
            except TracerBoolConversionError:
                pass
        scan_pos_phys = self.scan_to_real(scan_pos, _one=_one)
        res = OrderedDict()
        res['source'] = PointSource(z=0, semi_conv=self.semiconv)
        res['scanner'] = Scanner(
            z=self.overfocus,
            scan_pos_x=scan_pos_phys.x, scan_pos_y=scan_pos_phys.y
        )
        res['specimen'] = specimen
        res['descanner'] = Descanner(
                z=self.overfocus,
                scan_pos_x=scan_pos_phys.x,
                scan_pos_y=scan_pos_phys.y,
                descan_error=self.descan_error,
            )
        res['detector'] = Plane(z=self.overfocus + self.camera_length)
        return res

    def trace(
        self,
        scan_pos: PixelYX,
        source_dx: float,
        source_dy: float,
        specimen: Component | None = None,
        _one: float = 1.0,
    ) -> Result4DSTEM:
        components = self._components(
            scan_pos=scan_pos,
            specimen=specimen,
        )
        source = components['source']
        ray = Ray(
            x=source.offset_xy.x,
            y=source.offset_xy.y,
            dx=source_dx,
            dy=source_dy,
            z=source.z,
            pathlength=0.0,
            _one=_one,
        )
        result = OrderedDict()

        # run_iter() currently inserts a propagation if two subsequent
        # components have a non-zero distance, but skips for equal z. We
        # therefore check meticulously that we are actually getting the
        # components and rays we expect. Furthermore, we make sure that our
        # result ALWAYS has the same schema independent of parameters by
        # inserting gaps of zero length manually.
        run_result = list(run_iter(ray=ray, components=components.values()))

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
            assert comp == components['source']
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
            assert comp == components['scanner']
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
            assert comp == components['specimen']
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
            assert comp == components['descanner']
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
            assert comp == components['detector']
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

    def adjust_scan_rotation(self, scan_rotation: float) -> "Parameters4DSTEM":
        """
        Adjust the scan rotation while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        de = self.descan_error
        angle = scan_rotation - self.scan_rotation
        # Rotate the input direction

        def trans(c: CoordXY):
            return rotate(c=c, radians=angle)

        pxo_pyx = trans(CoordXY(y=de.pxo_pyi, x=de.pxo_pxi))
        pyo_pyx = trans(CoordXY(y=de.pyo_pyi, x=de.pyo_pxi))
        sxo_pyx = trans(CoordXY(y=de.sxo_pyi, x=de.sxo_pxi))
        syo_pyx = trans(CoordXY(y=de.syo_pyi, x=de.syo_pxi))
        new_de = DescanError(
            pxo_pyi=pxo_pyx.y,
            pyo_pyi=pyo_pyx.y,
            pxo_pxi=pxo_pyx.x,
            pyo_pxi=pyo_pyx.x,
            sxo_pyi=sxo_pyx.y,
            syo_pyi=syo_pyx.y,
            sxo_pxi=sxo_pyx.x,
            syo_pxi=syo_pyx.x,
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
        res1 = self.trace(scan_pos=self.scan_center, source_dx=0.0, source_dy=0.0)
        res2 = self.trace(scan_pos=scan_center, source_dx=0.0, source_dy=0.0)

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

        def trans(c: CoordXY):
            return rotate(c=c, radians=angle)

        pxy_pyi = trans(CoordXY(y=de.pyo_pyi, x=de.pxo_pyi))
        pxy_pxi = trans(CoordXY(y=de.pyo_pxi, x=de.pxo_pxi))
        sxy_pyi = trans(CoordXY(y=de.syo_pyi, x=de.sxo_pyi))
        sxy_pxi = trans(CoordXY(y=de.syo_pxi, x=de.sxo_pxi))
        offp = trans(CoordXY(y=de.offpyi, x=de.offpxi))
        offs = trans(CoordXY(y=de.offsyi, x=de.offsxi))
        new_de = DescanError(
            pxo_pyi=pxy_pyi.x,
            pyo_pyi=pxy_pyi.y,
            pxo_pxi=pxy_pxi.x,
            pyo_pxi=pxy_pxi.y,
            sxo_pyi=sxy_pyi.x,
            syo_pyi=sxy_pyi.y,
            sxo_pxi=sxy_pxi.x,
            syo_pxi=sxy_pxi.y,
            offpxi=offp.x,
            offpyi=offp.y,
            offsxi=offs.x,
            offsyi=offs.y,
        )

        return self.derive(
            detector_rotation=detector_rotation,
            descan_error=new_de,
        )

    def adjust_flip_factor(self, flip_factor: float) -> "Parameters4DSTEM":
        # Some import gymnastic to keep the naming clean
        from .model import flip_y

        de = self.descan_error
        angle = self.detector_rotation

        if flip_factor != self.flip_factor:
            # Rotate into detector directions, flip, then rotate back
            def trans(c: CoordXY):
                return rotate(
                    flip_y(
                        c=rotate(
                            c=c,
                            radians=-angle
                        ),
                        flip_factor=flip_factor/self.flip_factor
                    ),
                    radians=angle
                )
            # transform the output direction
            pxy_pyi = trans(CoordXY(y=de.pyo_pyi, x=de.pxo_pyi))
            pxy_pxi = trans(CoordXY(y=de.pyo_pxi, x=de.pxo_pxi))
            sxy_pyi = trans(CoordXY(y=de.syo_pyi, x=de.sxo_pyi))
            sxy_pxi = trans(CoordXY(y=de.syo_pxi, x=de.sxo_pxi))
            offp = trans(CoordXY(y=de.offpyi, x=de.offpxi))
            offs = trans(CoordXY(y=de.offsyi, x=de.offsxi))
            new_de = DescanError(
                pxo_pyi=pxy_pyi.x,
                pyo_pyi=pxy_pyi.y,
                pxo_pxi=pxy_pxi.x,
                pyo_pxi=pxy_pxi.y,
                sxo_pyi=sxy_pyi.x,
                syo_pyi=sxy_pyi.y,
                sxo_pxi=sxy_pxi.x,
                syo_pxi=sxy_pxi.y,
                offpxi=offp.x,
                offpyi=offp.y,
                offsxi=offs.x,
                offsyi=offs.y,
            )
            return self.derive(
                flip_factor=flip_factor,
                descan_error=new_de,
            )
        else:
            return self

    def adjust_detector_center(self, detector_center: PixelYX) -> "Parameters4DSTEM":
        de = self.descan_error
        zero = PixelYX(0, 0)
        other = self.derive(
            detector_center=detector_center,
        )

        physical_1 = self.detector_to_real(zero)
        physical_2 = other.detector_to_real(zero)
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
