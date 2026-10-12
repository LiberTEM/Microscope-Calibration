from typing import NamedTuple, Self
from collections import OrderedDict
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
# ruff: disable[E402]
import jax_dataclasses as jdc
import sympy as sym
from jax.errors import TracerBoolConversionError
from temgym_core import CoordXY, PixelYX
from temgym_core.components import Component, DescanError, Descanner, Plane, Scanner
from temgym_core.propagator import FreeSpaceParaxial, Propagator
from temgym_core.ray import Ray
from temgym_core.run import run_iter
from temgym_core.source import PointSource, Source

from microscope_calibration.util.sympy import lambdify, symbol_maker
# ruff: enable[E402]


def equals(ray1: Ray, ray2: Ray) -> bool:
    for key in ray1.__dict__.keys():
        if not sym.sympify(ray1.__dict__[key]).equals(ray2.__dict__[key]):
            return False
    return True


# Simple transformation function similar to propagation by components
def scale(c: CoordXY | PixelYX, factor: float) -> CoordXY:
    return CoordXY(
        x=c.x * factor,
        y=c.y * factor,
    )


def rotate(c: CoordXY | PixelYX, radians: float) -> CoordXY:
    cosr = sym.cos(radians)
    sinr = sym.sin(radians)
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
    component: Component | Source | Propagator
    ray: Ray
    sampling: dict | None = None


# Layer stack, result of tracing a ray through a model
Result4DSTEM = OrderedDict[str, ResultSection]


# TODO use LiberTEM-schema later
@jdc.pytree_dataclass
class Model4DSTEM:
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

    @classmethod
    def default(cls, dataset_shape: tuple[int, int, int, int] = (0, 0, 0, 0)):
        """
        Model with somewhat sensible default values for a given dataset shape

        * Set beam center to scan and detector center
        * In focus
        * Camera length of 1 m
        * 1 um scan step
        * 50 um detector pixel pitch
        * No descan error, rotation, or flip
        """
        assert len(dataset_shape) == 4
        nav_shape = dataset_shape[:2]
        sig_shape = dataset_shape[2:4]

        scan_center = PixelYX(
            y=nav_shape[0] / 2,
            x=nav_shape[1] / 2,
        )
        detector_center = PixelYX(
            y=sig_shape[0] / 2,
            x=sig_shape[1] / 2,
        )
        return cls(
            overfocus=0.0,
            scan_pixel_pitch=1e-6,
            scan_center=scan_center,
            scan_rotation=0.0,
            camera_length=1.0,
            detector_pixel_pitch=50e-6,
            detector_center=detector_center,
            semiconv=1e-3,  # radian
            flip_factor=1.0,
        )

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
    ) -> Model4DSTEM:
        if flip_factor is not None:
            assert flip_y is None
        if flip_y is not None:
            flip_factor = -1.0 if flip_y else 1.0

        return Model4DSTEM(
            overfocus=overfocus if overfocus is not None else self.overfocus,
            scan_pixel_pitch=(
                scan_pixel_pitch if scan_pixel_pitch is not None else self.scan_pixel_pitch
            ),
            scan_center=scan_center if scan_center is not None else self.scan_center,
            scan_rotation=scan_rotation if scan_rotation is not None else self.scan_rotation,
            camera_length=camera_length if camera_length is not None else self.camera_length,
            detector_pixel_pitch=(
                detector_pixel_pitch
                if detector_pixel_pitch is not None
                else self.detector_pixel_pitch
            ),
            detector_center=(
                detector_center if detector_center is not None else self.detector_center
            ),
            detector_rotation=(
                detector_rotation if detector_rotation is not None else self.detector_rotation
            ),
            semiconv=semiconv if semiconv is not None else self.semiconv,
            flip_factor=flip_factor if flip_factor is not None else self.flip_factor,
            descan_error=descan_error if descan_error is not None else self.descan_error,
        )

    def normalize_types(self) -> Model4DSTEM:
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
                factor=1 / self.scan_pixel_pitch,
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
                    factor=1 / self.detector_pixel_pitch,
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
        self, scan_pos: PixelYX, specimen: Component | None = None, _one: float = 1.0
    ) -> OrderedDict[str, Component]:
        if specimen is None:
            specimen = Plane(z=self.overfocus)
        else:
            try:
                # FIXME better solution later?
                assert sym.simplify(specimen.z).equals(sym.simplify(self.overfocus))
            except TracerBoolConversionError:
                pass
        scan_pos_phys = self.scan_to_real(scan_pos, _one=_one)
        res = OrderedDict()
        res["source"] = PointSource(z=0, semi_conv=self.semiconv)
        res["scanner"] = Scanner(
            z=self.overfocus, scan_pos_x=scan_pos_phys.x, scan_pos_y=scan_pos_phys.y
        )
        res["specimen"] = specimen
        res["descanner"] = Descanner(
            z=self.overfocus,
            scan_pos_x=scan_pos_phys.x,
            scan_pos_y=scan_pos_phys.y,
            descan_error=self.descan_error,
        )
        res["detector"] = Plane(z=self.overfocus + self.camera_length)
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
        source = components["source"]
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
            assert sym.simplify(comp.distance).equals(0.0)
            assert equals(r, ray)
        except TracerBoolConversionError:
            pass

        comp, r = run_result.pop(0)
        try:
            assert comp == components["source"]
            assert equals(r, ray)
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
            assert comp == components["scanner"]
            assert isinstance(r, Ray)
        except TracerBoolConversionError:
            pass
        result["scanner"] = ResultSection(component=comp, ray=r)

        # Skip zero distance propagation between scanner and specimen
        comp, r = run_result.pop(0)
        try:
            assert isinstance(comp, Propagator)
            assert sym.simplify(comp.distance).equals(0.0)
            assert isinstance(r, Ray)
            assert equals(r, result["scanner"].ray)
        except TracerBoolConversionError:
            pass

        comp, r = run_result.pop(0)
        try:
            assert comp == components["specimen"]
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
            assert sym.simplify(comp.distance).equals(0.0)
            assert equals(r, result["specimen"].ray)
        except TracerBoolConversionError:
            pass
        comp, r = run_result.pop(0)
        try:
            assert comp == components["descanner"]
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
            assert comp == components["detector"]
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

    def _adjust_compensating_descan_error(self, param: str, value) -> Model4DSTEM:
        """
        Set :code:`param` to :code:`value` and compensate the effect on the
        central ray in pixel coordinates with the descan error.

        The compensation is derived symbolically from :meth:`trace`, see
        :func:`_derive_descan_compensation` and :data:`GENERATED_FUNCTIONS`.
        """
        # Imported here so that the generator can run even if
        # _generated.py is missing or outdated
        from microscope_calibration.common import _generated

        compensate = getattr(_generated, f"descan_compensation_{param}")
        return self.derive(
            **{param: value},
            descan_error=compensate(model=self, value=value),
        )

    def adjust_scan_rotation(self, scan_rotation: float) -> Model4DSTEM:
        """
        Adjust the scan rotation while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("scan_rotation", scan_rotation)

    def adjust_scan_pixel_pitch(self, scan_pixel_pitch: float) -> Model4DSTEM:
        """
        Adjust the scan pixel pitch while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("scan_pixel_pitch", scan_pixel_pitch)

    def adjust_scan_center(self, scan_center: PixelYX) -> Model4DSTEM:
        """
        Adjust the scan center while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("scan_center", scan_center)

    def adjust_detector_rotation(self, detector_rotation: float) -> Model4DSTEM:
        """
        Adjust the detector rotation while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("detector_rotation", detector_rotation)

    def adjust_flip_factor(self, flip_factor: float) -> Model4DSTEM:
        """
        Adjust the flip factor while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        Returns this model unchanged if the flip factor is the same.
        """
        if flip_factor != self.flip_factor:
            return self._adjust_compensating_descan_error("flip_factor", flip_factor)
        else:
            return self

    def adjust_detector_center(self, detector_center: PixelYX) -> Model4DSTEM:
        """
        Adjust the detector center while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("detector_center", detector_center)

    def adjust_detector_pixel_pitch(self, detector_pixel_pitch: float) -> Model4DSTEM:
        """
        Adjust the detector pixel pitch while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("detector_pixel_pitch", detector_pixel_pitch)

    def adjust_camera_length(self, camera_length: float) -> Model4DSTEM:
        """
        Adjust the camera length while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.

        The pixel coordinates are the same as before for all camera lengths
        scaled by the ratio of new and old camera length.
        """
        return self._adjust_compensating_descan_error("camera_length", camera_length)

    def invert_focus(self) -> Self:
        return (
            self.derive(overfocus=-self.overfocus)
            .adjust_detector_rotation(detector_rotation=self.detector_rotation + sym.pi)
            .adjust_camera_length(camera_length=self.camera_length + 2 * self.overfocus)
        )


def _derive_descan_compensation(param: str):
    """
    Derive how the descan error has to change so that a change of model
    parameter :code:`param` leaves the central ray unchanged in pixel coordinates.

    The derivation only uses :meth:`Model4DSTEM.trace`, which is the single
    source of truth for the model, and makes no further assumptions about the
    mathematical relations between parameters:

    The central ray (zero source tilt) is traced symbolically through a model
    with the old parameters and through a model with the new value of
    :code:`param` and an unknown new descan error. The difference of the
    resulting specimen and detector pixel coordinates is a polynomial in scan
    position and camera length. It must vanish for all values of these
    variables, i.e. all its coefficients must be zero, which yields enough
    equations to solve for all components of the new descan error.

    For :code:`param == "camera_length"`, the camera length itself can't be a
    free variable. Instead, the pixel coordinates have to be the same for all
    camera lengths scaled by the same factor, i.e. for old and new camera length
    multiplied with a free variable.

    Non-zero source tilts can't be included since the descan error only depends
    on the scan position, i.e. it can't compensate tilt-dependent changes.

    This approach was developed by Margarita Usoltceva
    (https://github.com/Margus2004) in
    https://github.com/Margus2004/Microscope-Calibration/tree/sympy

    Returns
    -------
    args, output
        Symbolic arguments :code:`{"model": Model4DSTEM, "value": ...}` and the
        new descan error as a function of them, see
        :func:`microscope_calibration.util.codegen.render_function`
    """
    recurse_for = (PixelYX, DescanError)
    old = symbol_maker(Model4DSTEM, postfix="old", recurse_for=recurse_for)
    value = symbol_maker(
        Model4DSTEM.__annotations__[param], postfix=f"{param}_new", recurse_for=recurse_for
    )
    new_de = symbol_maker(DescanError, postfix="new")
    new = old.derive(**{param: value}, descan_error=new_de)

    scan_y, scan_x = sym.symbols("scan_y scan_x")
    scan_pos = PixelYX(y=scan_y, x=scan_x)
    if param == "camera_length":
        cl_scale = sym.Symbol("cl_scale")
        free_variable = cl_scale
        old_traced = old.derive(camera_length=cl_scale * old.camera_length)
        new_traced = new.derive(camera_length=cl_scale * new.camera_length)
    else:
        free_variable = old.camera_length
        old_traced = old
        new_traced = new
    res_old = old_traced.trace(scan_pos=scan_pos, source_dx=0.0, source_dy=0.0)
    res_new = new_traced.trace(scan_pos=scan_pos, source_dx=0.0, source_dy=0.0)

    unknowns = set(new_de)
    equations = []
    for plane, key in (("specimen", "scan_px"), ("detector", "detector_px")):
        px_old = res_old[plane].sampling[key]
        px_new = res_new[plane].sampling[key]
        for diff in (px_old.y - px_new.y, px_old.x - px_new.x):
            poly = sym.Poly(sym.expand(diff), scan_y, scan_x, free_variable)
            for coeff in poly.coeffs():
                if coeff.free_symbols & unknowns:
                    equations.append(coeff)
                # Coefficients without unknowns must vanish identically, e.g.
                # sin**2 + cos**2 - 1. Otherwise the change of param can't be
                # compensated by the descan error.
                elif not coeff.equals(0):
                    raise ValueError(
                        f"Change of {param} can't be compensated by the descan error: {coeff} != 0"
                    )

    solutions = sym.solve(equations, list(new_de), dict=True)
    if len(solutions) != 1 or set(solutions[0].keys()) != set(new_de):
        raise ValueError(
            f"Could not derive a unique descan error compensation for {param}: {solutions}"
        )
    (solution,) = solutions
    solved_de = DescanError(*(sym.trigsimp(solution[s]) for s in new_de))
    return {"model": old, "value": value}, solved_de


# Functions that are derived symbolically from Model4DSTEM.trace() and rendered
# into _generated.py to avoid the runtime cost of the derivation. Regenerate with
# `python -m microscope_calibration.common.generate` after changing the model.
# tests/test_codegen.py verifies that _generated.py is up to date.
GENERATED_FUNCTIONS = {
    f"descan_compensation_{param}": partial(_derive_descan_compensation, param)
    for param in (
        "scan_pixel_pitch",
        "scan_rotation",
        "scan_center",
        "detector_rotation",
        "flip_factor",
        "detector_center",
        "detector_pixel_pitch",
        "camera_length",
    )
}


def lambdify_trace_for(modules):
    @lambdify(modules=modules, recurse_for=(PixelYX, Model4DSTEM, DescanError))
    def trace(model: Model4DSTEM, scan_pos: PixelYX, source_dy, source_dx):
        return model.trace(scan_pos=scan_pos, source_dy=source_dy, source_dx=source_dx)

    return trace
