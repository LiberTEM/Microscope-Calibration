import numpy as np
import pytest
import sympy as sym

from microscope_calibration.util.sympy import lambdify


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
