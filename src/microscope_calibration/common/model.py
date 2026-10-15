import dataclasses
import warnings
from typing import Any, NamedTuple, Self, TypedDict, cast
from functools import cache, partial
from types import ModuleType

import jax

jax.config.update("jax_enable_x64", True)
# ruff: disable[E402]
import jax_dataclasses as jdc
import sympy as sym
from sympy.solvers.solveset import NonlinearError
from temgym_core import CoordXY, CoordsXY, PixelYX
from temgym_core.components import Component, DescanError, Descanner, Plane, Scanner
from temgym_core.propagator import FreeSpaceParaxial, Propagator
from temgym_core.ray import Ray
from temgym_core.run import run_iter
from temgym_core.source import PointSource, Source

from microscope_calibration.util.sympy import (
    lambdify_class,
    lambdify_method,
    lambdify_tree,
    symbol_maker,
)
# ruff: enable[E402]


def equals(ray1: Ray, ray2: Ray) -> bool:
    return all(sym.sympify(ray1.__dict__[key]).equals(ray2.__dict__[key]) for key in ray1.__dict__)


def _to_float(name: str, value) -> float:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} is a string: {value!r}")
    try:
        return float(value)
    except TypeError as e:
        raise TypeError(f"Can't convert {name} = {value!r} to float: {e}") from e


def _evaluate_sympy(name: str, value):
    """
    Convert SymPy numbers in :code:`value` or the leaves of a NamedTuple to :code:`float`
    """
    fields = getattr(value, "_fields", None)
    if isinstance(value, tuple) and fields is not None:
        return type(value)(
            **{key: _evaluate_sympy(f"{name}.{key}", getattr(value, key)) for key in fields}
        )
    return _to_float(name, value) if isinstance(value, sym.Basic) else value


def _to_float_tree(name: str, value):
    """
    Convert :code:`value` or the leaves of a NamedTuple like :class:`PixelYX`
    to :code:`float`, see :meth:`Model4DSTEM.normalize_types`
    """
    fields = getattr(value, "_fields", None)
    if isinstance(value, tuple) and fields is not None:
        return type(value)(
            **{key: _to_float_tree(f"{name}.{key}", getattr(value, key)) for key in fields}
        )
    return _to_float(name, value)


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


class NonlinearModelError(RuntimeError):
    """
    The :meth:`~Model4DSTEM.trace` of a model is not linear
    """


class RuntimeDerivationWarning(UserWarning):
    """
    A function is derived symbolically at runtime since no generated code is
    available for a model class, see :class:`Model4DSTEM`
    """


class ScanSampling(TypedDict):
    """
    Position of a ray on the specimen in scan pixel coordinates
    """

    scan_px: PixelYX


class DetectorSampling(TypedDict):
    """
    Position of a ray on the detector in detector pixel coordinates
    """

    detector_px: PixelYX


class ResultSection[S](NamedTuple):
    """
    "Layer" of a beam passing through a model, with :code:`sampling` of type
    :code:`S`, e.g. :class:`DetectorSampling`, or :code:`None`
    """

    component: Component | Source | Propagator
    ray: Ray
    # For sections without sampling, i.e. S = None
    sampling: S = None  # pyright: ignore[reportAssignmentType]


class Result4DSTEM(TypedDict):
    """
    Result of :meth:`Model4DSTEM.trace`, sections in the order of the beam path
    """

    source: ResultSection[None]
    overfocus: ResultSection[None]
    scanner: ResultSection[None]
    specimen: ResultSection[ScanSampling]
    descanner: ResultSection[None]
    camera_length: ResultSection[None]
    detector: ResultSection[DetectorSampling]


# TODO use LiberTEM-schema later
@jdc.pytree_dataclass
class Model4DSTEM:
    """
    Geometric calibration of a 4D STEM experiment

    :meth:`trace` is the single source of truth for the model. Functions that
    depend on it, such as the descan error compensation of the
    :code:`adjust_*()` methods and the
    :func:`~microscope_calibration.common.stem_overfocus.ray_transfer_matrix`,
    are derived symbolically from it with SymPy. Since that takes a few
    seconds, the result is generated in advance into
    :mod:`microscope_calibration.common._generated`, see
    :mod:`microscope_calibration.common.generate`.

    Subclassing
    -----------
    The generated code is only valid for :meth:`Model4DSTEM.trace` and the
    methods it uses, e.g. :meth:`real_to_detector`. A subclass may change the
    result of :meth:`trace`, so by default it doesn't use the generated code.
    Instead, the functions are derived from the subclass's :meth:`trace` on
    first use, once per subclass and function, with a
    :class:`RuntimeDerivationWarning`. A non-linear :meth:`trace` raises
    :class:`NonlinearModelError` in that case.

    The module with generated code that is valid for a class is returned by
    the classmethod :meth:`_generated_module`. A subclass can override it:

    * If the subclass doesn't change the result of :meth:`trace`, e.g. if it
      only adds convenience methods, return the module for
      :class:`Model4DSTEM`, i.e. :code:`super()._generated_module()` from
      :class:`Model4DSTEM`. Make sure that this stays true when the subclass
      changes.
    * To avoid the runtime derivation for a subclass that changes
      :meth:`trace`, generate code for it and return that module. The
      derivations accept the model class as a parameter, e.g.
      :code:`_derive_descan_compensation(param, model_type=MySubclass)` and
      :code:`stem_overfocus._derive_ray_transfer_matrix(MySubclass)`. Register
      them under the same names as in
      :data:`microscope_calibration.common.generate.GENERATED_FUNCTIONS`,
      render them with :func:`microscope_calibration.util.codegen.render_module`
      into a module of your own, and add a test like
      :code:`tests/test_codegen.py` that checks that it is up to date.

    Subclasses have to be decorated with
    :func:`jax_dataclasses.pytree_dataclass` like :class:`Model4DSTEM` so that
    JAX and :func:`~microscope_calibration.util.sympy.lambdify` see their
    parameters. They can't add fields that affect :meth:`trace`: the symbolic
    derivation only creates symbols for the fields of :class:`Model4DSTEM`.

    Units
    -----
    Lengths are in m, rotations in rad. :code:`semiconv` is the maximum
    slope of rays at the source relative to the optical axis, i.e.
    :code:`tan()` of the convergence semi-angle, since :meth:`trace` works with
    slopes.
    """

    overfocus: float  # m
    scan_pixel_pitch: float  # m
    scan_center: PixelYX
    scan_rotation: float  # rad
    camera_length: float  # m
    detector_pixel_pitch: float  # m
    detector_center: PixelYX
    # Slope at the source, i.e. tan() of the convergence semi-angle
    semiconv: float
    flip_factor: float  # 1.: no flip; -1.: flip
    descan_error: DescanError = DescanError()
    detector_rotation: float = 0.0  # rad

    @classmethod
    def _generated_module(cls) -> ModuleType | None:
        """
        Module with generated code that is valid for this class, or
        :code:`None` to derive functions at runtime, see :class:`Model4DSTEM`
        """
        if cls._symbolic_class() is Model4DSTEM:
            # Imported here so that the generator can run even if
            # _generated.py is missing or outdated
            from microscope_calibration.common import _generated

            return _generated
        return None

    @classmethod
    def lambdify(cls, modules) -> type[Self]:
        """
        Subclass with methods compiled for a numeric backend

        The methods that are marked with
        :func:`~microscope_calibration.util.sympy.lambdify_method`, e.g.
        :meth:`trace` and the :code:`adjust_*()` methods, are compiled from the
        SymPy-compatible methods of this class with
        :func:`~microscope_calibration.util.sympy.lambdify_tree` on first use.
        The class is created once per class and :code:`modules`.

        Use :meth:`from_model` to convert an instance. Classmethods like
        :meth:`default` create instances of the subclass directly. Instances
        keep the values as given, e.g. Python floats. Code compiled with
        :func:`equinox.filter_jit`, such as optimistix, treats these as static
        and compiles again for each new value. Convert them to arrays with
        :code:`jax.tree.map(jax.numpy.asarray, model)` before passing them.

        Parameters
        ----------
        modules
            :code:`modules` argument of :func:`sympy.lambdify`, e.g.
            :code:`numpy` or :code:`jax.numpy`

        Examples
        --------
        >>> import numpy as np
        >>> NumpyModel = Model4DSTEM.lambdify(np)
        >>> model = NumpyModel.default(dataset_shape=(32, 32, 64, 64))
        >>> float(model.adjust_scan_rotation(0.1).scan_rotation)
        0.1
        """
        # A subclass of the symbolic class, i.e. of this class unless this
        # class is a lambdified class itself
        return cast(
            type[Self],
            lambdify_class(
                cls._symbolic_class(),
                modules=modules,
                make_symbolic=symbolic_model,
                recurse_for=(PixelYX, CoordXY, CoordsXY, DescanError),
            ),
        )

    @classmethod
    def _symbolic_class(cls) -> type[Model4DSTEM]:
        """
        Class with SymPy-compatible methods, i.e. the base class of a class
        created with :meth:`lambdify`
        """
        return getattr(cls, "_lambdify_base", cls)

    @classmethod
    def from_model(cls, model: Model4DSTEM) -> Self:
        """
        Instance of this class with the parameters of :code:`model`

        This converts between the SymPy-compatible class and classes created
        with :meth:`lambdify`. For the latter, SymPy numbers are converted to
        :code:`float`.

        Raises
        ------
        TypeError
            If a parameter can't be converted to :code:`float` for a class
            created with :meth:`lambdify`, e.g. a SymPy expression with free
            symbols
        """
        values = {
            f.name: getattr(model, f.name)
            for f in dataclasses.fields(cls)
            if f.init and hasattr(model, f.name)
        }
        if cls._symbolic_class() is not cls:
            values = {name: _evaluate_sympy(name, value) for name, value in values.items()}
        return cls(**values)

    @classmethod
    def default(cls, dataset_shape: tuple[int, ...] = (0, 0, 0, 0)) -> Self:
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
            semiconv=1e-3,  # slope
            flip_factor=1.0,
        )

    def derive(
        self,
        *,
        overfocus: float | None = None,  # m
        scan_pixel_pitch: float | None = None,  # m
        scan_center: PixelYX | None = None,
        scan_rotation: float | None = None,  # rad
        camera_length: float | None = None,  # m
        detector_pixel_pitch: float | None = None,  # m
        detector_center: PixelYX | None = None,
        detector_rotation: float | None = None,  # rad
        semiconv: float | None = None,  # slope
        flip_y: bool | None = None,
        flip_factor: float | None = None,
        descan_error: DescanError | None = None,
    ) -> Self:
        """
        Copy of this model with the given parameters changed

        Unlike the :code:`adjust_*()` methods, this doesn't compensate the
        effect of the change with the descan error. :code:`flip_y` is an
        alternative to :code:`flip_factor`.
        """
        changes = {
            name: value
            for name, value in locals().items()
            if name not in ("self", "flip_y") and value is not None
        }
        if flip_y is not None:
            if flip_factor is not None:
                raise ValueError("Specify either flip_y or flip_factor, not both.")
            changes["flip_factor"] = -1.0 if flip_y else 1.0
        # Keeps the class and any other fields of subclasses
        return dataclasses.replace(self, **changes)

    def normalize_types(self) -> Self:
        """
        Copy of this model with all parameters converted to :code:`float`,
        e.g. from SymPy numbers or JAX and NumPy scalars

        Fields that subclasses add are kept as they are.

        Raises
        ------
        TypeError
            If a parameter can't be converted to :code:`float`, e.g. a SymPy
            expression with free symbols, a complex number, an array or a
            string. NaN and infinity are kept.
        """
        return dataclasses.replace(
            self,
            **{
                field.name: _to_float_tree(field.name, getattr(self, field.name))
                for field in dataclasses.fields(Model4DSTEM)
            },
        )

    @lambdify_method(static=("_one",))
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

    @lambdify_method(static=("_one",))
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

    @lambdify_method(static=("_one",))
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

    @lambdify_method(static=("_one",))
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
    ) -> dict[str, Component]:
        if specimen is None:
            specimen = Plane(z=self.overfocus)
        else:
            # The specimen is in the overfocus plane by definition
            # Components are dataclasses
            specimen = dataclasses.replace(cast(Any, specimen), z=self.overfocus)
        scan_pos_phys = self.scan_to_real(scan_pos, _one=_one)
        res = {}
        res["source"] = PointSource(z=0.0, semi_conv=self.semiconv)
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

    # _one is a constant in numeric calculations, which simplifies the expressions
    @lambdify_method(static=("_one",))
    def trace(
        self,
        scan_pos: PixelYX,
        source_dx: float,
        source_dy: float,
        specimen: Component | None = None,
        _one: float = 1.0,
    ) -> Result4DSTEM:
        """
        Trace a ray through the model

        Parameters
        ----------
        scan_pos
            Scan position in pixels
        source_dx, source_dy
            Tilt of the ray at the source in radians
        specimen
            Component that acts as the specimen, e.g. a
            :class:`~temgym_core.components.Scanner` to deflect the ray like a
            diffracting specimen. It is placed at the overfocus, i.e. its
            :code:`z` is replaced. By default, a
            :class:`~temgym_core.components.Plane` that doesn't change the ray.
        _one
            Homogeneous coordinate, see :class:`temgym_core.ray.Ray`

        Returns
        -------
        result
            Component and ray for each section of the model, in order
        """
        components = self._components(
            scan_pos=scan_pos,
            specimen=specimen,
            _one=_one,
        )
        source = components["source"]
        assert isinstance(source, PointSource)
        ray = Ray(
            x=source.offset_xy.x,
            y=source.offset_xy.y,
            dx=source_dx,
            dy=source_dy,
            z=source.z,
            pathlength=0.0,
            _one=_one,
        )
        result = {}

        # run_iter() currently inserts a propagation if two subsequent
        # components have a non-zero distance, but skips for equal z. We
        # therefore check meticulously that we are actually getting the
        # components and rays we expect. Furthermore, we make sure that our
        # result ALWAYS has the same schema independent of parameters by
        # inserting gaps of zero length manually.
        run_result = list(run_iter(ray=ray, components=list(components.values())))

        # skip the first propagation, which should be zero distance
        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert sym.simplify(comp.distance).equals(0.0)
        assert equals(r, ray)

        comp, r = run_result.pop(0)
        assert comp == components["source"]
        assert equals(r, ray)
        result["source"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert isinstance(comp.propagator, FreeSpaceParaxial)
        assert isinstance(r, Ray)
        result["overfocus"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert comp == components["scanner"]
        assert isinstance(r, Ray)
        result["scanner"] = ResultSection(component=comp, ray=r)

        # Skip zero distance propagation between scanner and specimen
        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert sym.simplify(comp.distance).equals(0.0)
        assert isinstance(r, Ray)
        assert equals(r, result["scanner"].ray)

        comp, r = run_result.pop(0)
        assert comp == components["specimen"]
        assert isinstance(r, Ray)
        scan_px = self.real_to_scan(CoordXY(x=r.x, y=r.y), _one=ray._one)
        result["specimen"] = ResultSection(
            component=comp,
            ray=r,
            sampling={"scan_px": scan_px},
        )

        # Skip zero distance propagation between specimen and descanner
        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert sym.simplify(comp.distance).equals(0.0)
        assert equals(r, result["specimen"].ray)
        comp, r = run_result.pop(0)
        assert comp == components["descanner"]
        assert isinstance(r, Ray)
        result["descanner"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert isinstance(comp, Propagator)
        assert isinstance(comp.propagator, FreeSpaceParaxial)
        assert isinstance(r, Ray)
        result["camera_length"] = ResultSection(component=comp, ray=r)

        comp, r = run_result.pop(0)
        assert comp == components["detector"]
        assert isinstance(r, Ray)
        detector_px = self.real_to_detector(CoordXY(x=r.x, y=r.y), _one=ray._one)
        result["detector"] = ResultSection(
            component=comp,
            ray=r,
            sampling={"detector_px": detector_px},
        )

        assert len(run_result) == 0
        return Result4DSTEM(**result)

    def _adjust_compensating_descan_error(self, param: str, value) -> Self:
        """
        Set :code:`param` to :code:`value` and compensate the effect on the
        central ray in pixel coordinates with the descan error.

        The compensation is derived symbolically from :meth:`trace`, see
        :func:`_derive_descan_compensation` and :data:`GENERATED_FUNCTIONS`.
        """
        name = f"descan_compensation_{param}"
        module = type(self)._generated_module()
        if module is None:
            compensate = _runtime_descan_compensation(type(self), param)
            descan_error = compensate({"model": self, "value": value})
        else:
            descan_error = getattr(module, name)(model=self, value=value)
        return self.derive(**{param: value}, descan_error=descan_error)

    @lambdify_method
    def adjust_scan_rotation(self, scan_rotation: float) -> Self:
        """
        Adjust the scan rotation while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("scan_rotation", scan_rotation)

    @lambdify_method
    def adjust_scan_pixel_pitch(self, scan_pixel_pitch: float) -> Self:
        """
        Adjust the scan pixel pitch while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("scan_pixel_pitch", scan_pixel_pitch)

    @lambdify_method
    def adjust_scan_center(self, scan_center: PixelYX) -> Self:
        """
        Adjust the scan center while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("scan_center", scan_center)

    @lambdify_method
    def adjust_detector_rotation(self, detector_rotation: float) -> Self:
        """
        Adjust the detector rotation while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("detector_rotation", detector_rotation)

    @lambdify_method
    def adjust_flip_factor(self, flip_factor: float) -> Self:
        """
        Adjust the flip factor while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("flip_factor", flip_factor)

    @lambdify_method
    def adjust_detector_center(self, detector_center: PixelYX) -> Self:
        """
        Adjust the detector center while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("detector_center", detector_center)

    @lambdify_method
    def adjust_detector_pixel_pitch(self, detector_pixel_pitch: float) -> Self:
        """
        Adjust the detector pixel pitch while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.
        """
        return self._adjust_compensating_descan_error("detector_pixel_pitch", detector_pixel_pitch)

    @lambdify_method
    def adjust_camera_length(self, camera_length: float) -> Self:
        """
        Adjust the camera length while keeping the effective descan error
        compensation the same.

        This allows first compensating descan error and then adjusting other parameters.

        The pixel coordinates are the same as before for all camera lengths
        scaled by the ratio of new and old camera length.
        """
        return self._adjust_compensating_descan_error("camera_length", camera_length)

    @lambdify_method
    def invert_focus(self) -> Self:
        """
        Model with inverted sign of the overfocus that projects the specimen
        to the same detector pixels

        Over- and underfocus can't be distinguished from the shadow image
        alone. This allows switching between the two interpretations.
        """
        return (
            self.derive(overfocus=-self.overfocus)
            .adjust_detector_rotation(detector_rotation=self.detector_rotation + sym.pi)
            .adjust_camera_length(camera_length=self.camera_length + 2 * self.overfocus)
        )


def symbolic_model(model_type: type[Model4DSTEM] = Model4DSTEM, postfix: str | None = None):
    """
    Instance of :code:`model_type` with SymPy symbols for all parameters
    """
    model_type = model_type._symbolic_class()
    model = symbol_maker(Model4DSTEM, postfix=postfix, recurse_for=(PixelYX, DescanError))
    if model_type is not Model4DSTEM:
        # symbol_maker() uses __annotations__, which subclasses don't inherit
        model = model_type(**{f.name: getattr(model, f.name) for f in dataclasses.fields(model)})
    return model


def check_pytree(model: Model4DSTEM):
    """
    Make sure that the class of :code:`model` is registered as a pytree

    Subclasses of :class:`Model4DSTEM` have to be decorated with
    :func:`jax_dataclasses.pytree_dataclass` as well. Otherwise, JAX and
    :func:`~microscope_calibration.util.sympy.lambdify` treat them as a single
    leaf instead of seeing their parameters.
    """
    if jax.tree.leaves(model) == [model]:
        raise TypeError(
            f"{type(model).__qualname__} is not registered as a pytree. Decorate subclasses "
            "of Model4DSTEM with @jax_dataclasses.pytree_dataclass."
        )


def warn_runtime_derivation(name: str, model_type: type):
    warnings.warn(
        f"No generated code for {name} of {model_type.__qualname__}, deriving it from "
        f"{model_type.__qualname__}.trace() at runtime. See the docstring of Model4DSTEM.",
        RuntimeDerivationWarning,
        stacklevel=3,
    )


@cache
def _runtime_descan_compensation(model_type: type[Model4DSTEM], param: str):
    """
    Descan error compensation for a model class without generated code,
    derived from its :meth:`~Model4DSTEM.trace` at runtime
    """
    warn_runtime_derivation(f"descan_compensation_{param}", model_type)
    args, output = _derive_descan_compensation(param, model_type=model_type)
    check_pytree(args["model"])
    return lambdify_tree(args, output, modules=sym)


def _derive_descan_compensation(param: str, model_type: type[Model4DSTEM] = Model4DSTEM):
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
    old = symbolic_model(model_type, postfix="old")
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
            try:
                poly = sym.Poly(sym.expand(diff), scan_y, scan_x, free_variable)
            except sym.PolynomialError as e:
                raise NonlinearModelError(
                    f"{model_type.__qualname__}.trace() is not linear: {diff}"
                ) from e
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

    # sym.solve() would try to solve a nonlinear system, which can take forever
    try:
        sym.linear_eq_to_matrix(equations, list(new_de))
    except NonlinearError as e:
        raise NonlinearModelError(f"{model_type.__qualname__}.trace() is not linear: {e}") from e
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
