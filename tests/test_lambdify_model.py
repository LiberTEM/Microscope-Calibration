import warnings

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import pytest
import sympy as sym
from numpy.testing import assert_allclose
from temgym_core import CoordXY, PixelYX

import microscope_calibration.util.sympy as mc_sympy
from microscope_calibration.common.model import Model4DSTEM, RuntimeDerivationWarning
from microscope_calibration.common.stem_overfocus import ray_transfer_matrix

BACKENDS = {"numpy": np, "jax": jnp}

ADJUST_VALUES = {
    "scan_rotation": 0.3,
    "scan_pixel_pitch": 0.7,
    "scan_center": PixelYX(y=1.0, x=-2.0),
    "detector_rotation": -0.4,
    "flip_factor": -1.0,
    "detector_center": PixelYX(y=-3.0, x=4.0),
    "detector_pixel_pitch": 0.6,
    "camera_length": 1.3,
}

TRACE_KWARGS = {"scan_pos": PixelYX(y=1.0, x=2.0), "source_dy": 0.001, "source_dx": 0.002}


def assert_numeric_equal(res, ref):
    assert jax.tree.structure(res) == jax.tree.structure(ref)
    for leaf, ref_leaf in zip(jax.tree.leaves(res), jax.tree.leaves(ref), strict=True):
        assert not isinstance(leaf, sym.Basic), leaf
        if isinstance(ref_leaf, (float, int, sym.Basic)):
            assert_allclose(float(leaf), float(ref_leaf), rtol=1e-12, atol=1e-12)
        else:
            assert type(leaf) is type(ref_leaf)


def numeric_tree(tree, cls):
    # Reference result of the symbolic class, with the type of the numeric class
    return jax.tree.map(
        lambda x: cls.from_model(x) if type(x) is Model4DSTEM else x,
        tree,
        is_leaf=lambda x: type(x) is Model4DSTEM,
    )


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_lambdify_class(backend):
    cls = Model4DSTEM.lambdify(BACKENDS[backend])
    assert issubclass(cls, Model4DSTEM)
    assert cls is not Model4DSTEM
    assert cls is Model4DSTEM.lambdify(BACKENDS[backend])
    # Lambdifies the symbolic base class again
    assert cls.lambdify(BACKENDS[backend]) is cls
    assert cls._symbolic_class() is Model4DSTEM
    assert type(cls.default(dataset_shape=(8, 8, 8, 8))) is cls


def test_lambdify_modules_list():
    modules = [{"cos": np.cos}, "numpy"]
    assert Model4DSTEM.lambdify(modules) is Model4DSTEM.lambdify(list(modules))
    assert Model4DSTEM.lambdify(modules) is not Model4DSTEM.lambdify(np)


def test_from_model(random_model):
    cls = Model4DSTEM.lambdify(np)
    model = cls.from_model(random_model)
    assert type(model) is cls
    assert jax.tree.leaves(model) == jax.tree.leaves(random_model)
    back = Model4DSTEM.from_model(model)
    assert type(back) is Model4DSTEM
    assert back == random_model


def test_from_model_evaluates_sympy(random_model):
    cls = Model4DSTEM.lambdify(np)
    symbolic = random_model.derive(scan_rotation=sym.pi / 4)
    model = cls.from_model(symbolic)
    assert type(model.scan_rotation) is float
    assert_allclose(model.scan_rotation, np.pi / 4)
    with pytest.raises(TypeError, match="scan_rotation"):
        cls.from_model(random_model.derive(scan_rotation=sym.Symbol("x")))


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("param", sorted(ADJUST_VALUES))
def test_adjust(random_model, backend, param):
    cls = Model4DSTEM.lambdify(BACKENDS[backend])
    res = getattr(cls.from_model(random_model), f"adjust_{param}")(ADJUST_VALUES[param])
    ref = getattr(random_model, f"adjust_{param}")(ADJUST_VALUES[param])
    assert type(res) is cls
    assert_numeric_equal(res, cls.from_model(ref))


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("invert_focus", {}),
        ("trace", TRACE_KWARGS),
        ("scan_to_real", {"pixels": PixelYX(y=3.0, x=-4.0)}),
        ("real_to_scan", {"coords": CoordXY(x=0.3, y=-0.4)}),
        ("detector_to_real", {"pixels": PixelYX(y=3.0, x=-4.0)}),
        ("real_to_detector", {"coords": CoordXY(x=0.3, y=-0.4)}),
    ],
)
def test_methods(random_model, backend, method, kwargs):
    cls = Model4DSTEM.lambdify(BACKENDS[backend])
    res = getattr(cls.from_model(random_model), method)(**kwargs)
    ref = getattr(random_model, method)(**kwargs)
    assert_numeric_equal(res, numeric_tree(ref, cls))


def test_chaining(random_model):
    cls = Model4DSTEM.lambdify(np)
    res = (
        cls.from_model(random_model)
        .derive(overfocus=0.3)
        .adjust_scan_rotation(0.2)
        .invert_focus()
        .trace(**TRACE_KWARGS)
    )
    ref = random_model.derive(overfocus=0.3).adjust_scan_rotation(0.2).invert_focus()
    assert_numeric_equal(res, ref.normalize_types().trace(**TRACE_KWARGS))


def test_compiled_once(monkeypatch):
    calls = []
    original = mc_sympy.lambdify_tree

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mc_sympy, "lambdify_tree", counting)
    # A list as modules gives a class that no other test uses
    cls = Model4DSTEM.lambdify([{}, "numpy"])
    model = cls.default(dataset_shape=(8, 8, 8, 8)).derive(overfocus=0.1)
    for value in (0.1, 0.2, 0.3):
        model.adjust_scan_rotation(value)
    assert len(calls) == 1
    for _ in range(3):
        model.trace(**TRACE_KWARGS)
    assert len(calls) == 2


@jdc.pytree_dataclass
class ShiftedDetectorModel(Model4DSTEM):
    def real_to_detector(self, coords: CoordXY, _one: float = 1.0) -> PixelYX:
        res = super().real_to_detector(coords, _one)
        return PixelYX(y=res.y + 3.0 * _one, x=res.x - 2.0 * _one)


def test_subclass(random_model):
    model = ShiftedDetectorModel.from_model(random_model)
    cls = ShiftedDetectorModel.lambdify(np)
    assert issubclass(cls, ShiftedDetectorModel)
    assert cls is not Model4DSTEM.lambdify(np)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeDerivationWarning)
        res = cls.from_model(model).adjust_detector_rotation(0.3).trace(**TRACE_KWARGS)
        ref = model.adjust_detector_rotation(0.3).trace(**TRACE_KWARGS)
    assert_numeric_equal(res, ref)


def test_ray_transfer_matrix(random_model):
    cls = Model4DSTEM.lambdify(np)
    with warnings.catch_warnings():
        # Uses the generated code of Model4DSTEM
        warnings.simplefilter("error", RuntimeDerivationWarning)
        res = ray_transfer_matrix(cls.from_model(random_model))
    assert_allclose(res, ray_transfer_matrix(random_model), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("param", ["scan_rotation", "camera_length"])
def test_jax_grad(random_model, param):
    # Gradients use a dedicated function with the inputs and outputs of
    # interest, see examples/model.ipynb
    model = Model4DSTEM.lambdify(jnp).from_model(random_model)

    def f(value):
        return sum(getattr(model, f"adjust_{param}")(value).descan_error)

    value = ADJUST_VALUES[param]
    eps = 1e-6
    expected = (f(value + eps) - f(value - eps)) / (2 * eps)
    assert_allclose(jax.grad(f)(value), expected, rtol=1e-5, atol=1e-8)


def test_static_argument():
    cls = Model4DSTEM.lambdify(np)
    model = cls.default(dataset_shape=(8, 8, 8, 8)).derive(overfocus=0.1)
    ref = Model4DSTEM.from_model(model)
    for _one in (1.0, 2.0):
        res = model.scan_to_real(PixelYX(y=1.0, x=2.0), _one=_one)
        assert_allclose(res, [float(v) for v in ref.scan_to_real(PixelYX(y=1.0, x=2.0), _one=_one)])
