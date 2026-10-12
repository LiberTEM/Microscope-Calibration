import inspect

import jax
import numpy as np
import pytest
import sympy as sym
from numpy.testing import assert_allclose
from temgym_core import PixelYX

from microscope_calibration.common import _generated
from microscope_calibration.common.generate import GENERATED_FUNCTIONS
from microscope_calibration.util.codegen import leaves, render_module


@pytest.mark.parametrize("name", sorted(GENERATED_FUNCTIONS))
def test_generated_up_to_date(name):
    # Compare symbolically instead of textually so that the test doesn't fail
    # for equivalent code, e.g. from a different SymPy version or ordering of terms.
    args, expected = GENERATED_FUNCTIONS[name]()
    msg = f"_generated.{name} is outdated, run `python -m microscope_calibration.common.generate`"
    assert hasattr(_generated, name), msg
    result = getattr(_generated, name)(**args)
    assert type(result) is type(expected), msg
    for res, exp in zip(leaves(result), leaves(expected), strict=True):
        assert sym.simplify(sym.sympify(res) - exp) == 0, msg


def test_no_stale_generated_functions():
    generated = {
        name
        for name, obj in inspect.getmembers(_generated, inspect.isfunction)
        if obj.__module__ == _generated.__name__
    }
    assert generated == set(GENERATED_FUNCTIONS), (
        "_generated.py is outdated, run `python -m microscope_calibration.common.generate`"
    )


def test_render_module():
    p = PixelYX(y=sym.Symbol("py"), x=sym.Symbol("px"))
    angle = sym.Symbol("angle")
    output = PixelYX(
        y=p.y * sym.cos(angle) + p.x * sym.sin(angle) + sym.pi,
        x=p.x / 2,
    )
    source = render_module(
        {"f": ({"p": p, "angle": angle}, output)},
        header="Test header",
    )
    assert source.startswith("# Test header\n")
    namespace = {}
    exec(source, namespace)
    f = namespace["f"]

    res = f(p=PixelYX(y=1.0, x=2.0), angle=0.3)
    assert isinstance(res, PixelYX)
    assert_allclose(res, (np.cos(0.3) + 2.0 * np.sin(0.3) + np.pi, 1.0))
    # Functions like cos() keep the backend of their argument
    assert not any(isinstance(v, sym.Basic) for v in res)
    # Plain arithmetic keeps plain types
    assert type(res.x) is float

    grad = jax.grad(lambda angle: f(p=PixelYX(y=1.0, x=2.0), angle=angle).y)(0.3)
    assert_allclose(grad, -np.sin(0.3) + 2.0 * np.cos(0.3))

    res = f(p=p, angle=angle)
    for r, o in zip(res, output, strict=True):
        # Constants like pi are printed as floats, like lambdify(modules=sympy)
        assert abs(sym.N(sym.simplify(r - o))) < 1e-14


def test_render_module_unknown_symbol():
    a, b = sym.symbols("a b")
    with pytest.raises(ValueError, match="not in args"):
        render_module({"f": ({"a": a}, a * b)}, header="")


def test_render_module_unknown_function():
    a = sym.Symbol("a")
    with pytest.raises(ValueError, match="not available"):
        render_module({"f": ({"a": a}, sym.tan(a))}, header="")
