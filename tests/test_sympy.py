import jax
import numpy as np
import pytest
import sympy as sym
from numpy.testing import assert_allclose

from microscope_calibration.util.sympy import lambdify, lambdify_tree


@pytest.mark.parametrize("sample", ["args", "kwargs"])
def test_lambdify_sample_input(sample):
    a = sym.Symbol("a")
    sample_input = {"args": (a, 2.0)} if sample == "args" else {"kwargs": {"a": a, "b": 2.0}}

    @lambdify(modules=np, **sample_input)
    def f(a: float, b: float):
        return a * b

    assert f(3.0, 2.0) == 6.0
    # The sample value for b is a constant
    with pytest.raises(ValueError, match="Constant value"):
        f(3.0, 4.0)


def test_lambdify_tree():
    a, b = sym.symbols("a b")
    f = lambdify_tree(
        inp={"x": a, "pair": (b, a), "const": 2.0, "pi": sym.pi},
        outp={"sum": a + b, "fixed": "label", "number": sym.Integer(3)},
        modules=np,
    )
    res = f({"x": 1.0, "pair": (2.0, 1.0), "const": 2.0, "pi": np.pi})
    assert res == {"sum": 3.0, "fixed": "label", "number": 3}


@pytest.mark.parametrize(
    "inp, match",
    [
        ({"x": 1.0, "pair": (2.0, 3.0), "const": 2.0}, "duplicate symbol"),
        ({"x": 1.0, "pair": (2.0, 1.0), "const": 4.0}, "Constant value"),
        ({"x": 1.0, "pair": (2.0, 1.0)}, "Tree definition"),
    ],
)
def test_lambdify_tree_input_mismatch(inp, match):
    a, b = sym.symbols("a b")
    f = lambdify_tree(inp={"x": a, "pair": (b, a), "const": 2.0}, outp=a + b, modules=np)
    with pytest.raises(ValueError, match=match):
        f(inp)


def test_lambdify_tree_array_constant():
    a = sym.Symbol("a")
    f = lambdify_tree(inp=(a, np.array([1.0, 2.0])), outp=a, modules=np)
    assert f((3.0, np.array([1.0, 2.0]))) == 3.0
    with pytest.raises(ValueError, match="Constant value"):
        f((3.0, np.array([1.0, 5.0])))


def test_lambdify_tree_expression_input():
    a = sym.Symbol("a")
    with pytest.raises(ValueError, match="not a symbol"):
        lambdify_tree(inp=(2 * a,), outp=a, modules=np)


def test_lambdify_tree_jit():
    a, b = sym.symbols("a b")
    f = lambdify_tree(inp=(a, b, a), outp=sym.cos(a) * b, modules="jax")
    assert_allclose(jax.jit(f)((0.5, 2.0, 0.5)), np.cos(0.5) * 2.0)
    assert_allclose(jax.grad(lambda x: f((x, 2.0, x)))(0.5), -np.sin(0.5) * 2.0)
