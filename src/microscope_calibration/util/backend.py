"""
Math functions that work with SymPy, JAX and plain numbers

The functions dispatch on the type of their argument so that code using them,
e.g. :meth:`~microscope_calibration.common.model.Model4DSTEM.trace` and the
generated code in :mod:`microscope_calibration.common._generated`, keeps the
type of its inputs: SymPy expressions stay symbolic, JAX arrays and tracers
stay differentiable and traceable, and other numbers are calculated with NumPy.
"""

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
# ruff: disable[E402]
import jax.numpy as jnp
import numpy as np
import sympy as sym
# ruff: enable[E402]

__all__ = ["NAMESPACE", "cos", "sin"]


def _dispatch(name: str) -> Callable:
    sym_f = getattr(sym, name)
    jnp_f = getattr(jnp, name)
    np_f = getattr(np, name)

    def f(x):
        if isinstance(x, sym.Basic):
            return sym_f(x)
        # Includes tracers
        elif isinstance(x, jax.Array):
            return jnp_f(x)
        else:
            return np_f(x)

    f.__name__ = f.__qualname__ = name
    f.__doc__ = f"{name}() for SymPy expressions, JAX arrays and other numbers"
    return f


cos = _dispatch("cos")
sin = _dispatch("sin")

# Namespace for :func:`sympy.lambdify`, e.g. :code:`modules=[NAMESPACE, "sympy"]`
NAMESPACE = {name: globals()[name] for name in ("cos", "sin")}
