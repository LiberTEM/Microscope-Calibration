import dataclasses
import functools
import inspect
import types
import typing
from collections.abc import Callable, Container
from functools import cache
from numbers import Number
from typing import Any, TypeVar

import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
# ruff: disable[E402]
import jax_dataclasses as jdc
import numpy as np
import sympy as sym
from numpy.typing import NDArray
import wrapt
from frozendict import frozendict
from jax.errors import TracerBoolConversionError
# ruff: enable[E402]

T = TypeVar("T")


def isnumerictype(T):
    # Arrays with numeric dtype, e.g. NDArray[np.floating], since lambdified
    # functions work element-wise
    origin = typing.get_origin(T)
    # NDArray is a type alias in recent NumPy versions
    if origin is NDArray:
        (scalar_type,) = typing.get_args(T)
        return _is_numpy_number(scalar_type)
    if origin is np.ndarray:
        (_, dtype) = typing.get_args(T)
        (scalar_type,) = typing.get_args(dtype)
        return _is_numpy_number(scalar_type)
    for supercls in (Number, sym.Number):
        # try ... except because some reasonable Ts like int | float cause a TypeError here
        try:
            if issubclass(T, supercls):
                return True
        except TypeError:
            pass
    # This can also handle unions like int | float via isinstance()
    for number in (0, 0.0, 0.0 + 0.0j):
        # try ... except because weird types can cause a TypeError here,
        # but we can safely continue since we didn't positively recognize it as numeric
        try:
            if isinstance(number, T):
                return True
        except TypeError:
            pass
    return False


def _is_numpy_number(T) -> bool:
    if typing.get_origin(T) in (typing.Union, types.UnionType):
        return all(_is_numpy_number(t) for t in typing.get_args(T))
    return isinstance(T, type) and issubclass(T, np.number)


def is_sympy(expr: Any) -> bool:
    """
    Check if the given instance is a sympy expression.

    Some complex structure which contains a sympy expression as an element returns False.
    One can use a recursive function to check each element of the structure (e.g., typing.Iterable),
    However, because of a possibiblity to have a long sequence of elements, it does not seem to be
    effective enough.

    Any expressions that make sence mathematically are of class sympy.core.expr.Expr, which e.g.
    includes sympy.core.symbol.Symbol, sympy.core.numbers.Number, sympy.core.add.Add.
    However, we use sympy.core.basic.Basic which includes sympy.core.expr.Expr, in case we need
    to test also any logic operations, booleans, relations, etc.:
    isinstance(x>0, sympy.Expr) returns False, but isinstance(x>0, sympy.Basic) returns True.

    Some objects like matrices of class sympy.matrices.matrixbase.MatrixBase or sympy arrays
    of class sympy.tensor.array.ImmutableDenseNDimArray are not instances of sympy.Basic.


    Parameters
    ----------
    expr
        anything to be checked if it is a sympy instance
    """
    # FIXME confirm that this are all the base classes
    return isinstance(expr, (sym.Basic, sym.MatrixBase, sym.NDimArray))


def symbol_maker(params_cls: type[T], postfix: str | None = None, recurse_for: Container = ()) -> T:
    """
    Declare sympy symbols for each attribute of a given parameter class.

    Use __annotations__ to access the class attributes and their types.
    Make a dictionary of sympy symbols assigned to each class attribute of
    a primitive type. The symbol names are declared as names of the attributes
    with an appropriate postfix. After that construct a class instance with
    the symbolic variables as parameters.

    Parameters
    ----------
    params_cls: class
        class which needs symbolic variables to be assigned to its parameters
    postfix: str
        postfix for each symbol name to specify the parameters (e.g. 'new' or 'old')
    recurse_for: tuple
        list of class attributes of non-primitive types to receive symbols for each component

    Returns
    -------
    class instance
        instance of the class with symbols as parameters
    """

    def symbol_maker_inner(
        params_cls: type[T], postfix: str | None, recurse_for: Container, index: int
    ) -> tuple[int, T]:
        if hasattr(params_cls, "__annotations__"):
            symbols_dict = {}
            for attr in params_cls.__annotations__:
                cls = params_cls.__annotations__[attr]
                if cls in recurse_for:
                    (index, symbols_dict[attr]) = symbol_maker_inner(
                        cls, postfix, recurse_for, index
                    )
                elif isnumerictype(cls):
                    sym_name = attr if postfix is None else f"{attr}_{postfix}"
                    sym_name = f"{sym_name}_{index}"
                    symbols_dict[attr] = sym.Symbol(sym_name)
                    index += 1
                else:
                    raise TypeError(f"Can't generate symbol for type {cls}.")
            return (index, params_cls(**symbols_dict))
        elif isnumerictype(params_cls):
            attr = params_cls.__name__
            sym_name = attr if postfix is None else f"{attr}_{postfix}"
            sym_name = f"{sym_name}_{index}"
            index += 1
            return (index, sym.Symbol(sym_name))
        else:
            raise TypeError(f"Can't generate symbol for type {params_cls}.")

    (_, res) = symbol_maker_inner(
        params_cls=params_cls, postfix=postfix, recurse_for=recurse_for, index=0
    )
    return res


SymbolJaxTree = TypeVar("SymbolJaxTree")


def lambdify_tree(inp: SymbolJaxTree, outp: SymbolJaxTree, **kwargs):
    """
    Numeric function that calculates the tree :code:`outp` from the tree :code:`inp`

    The SymPy symbols in :code:`inp` are the arguments; other leaves of
    :code:`inp` are constants that must match in each call. SymPy expressions in
    :code:`outp` are evaluated with :func:`sympy.lambdify`, using
    :code:`kwargs`; other leaves are returned as they are.
    """
    inp_leaves, inp_treedef = jax.tree.flatten(inp)
    outp_leaves, outp_treedef = jax.tree.flatten(outp)

    inp_indices = []
    inp_symbols = []
    # index -> index of the first occurrence of the same symbol
    inp_dups = {}
    # index -> expected value
    inp_constants = {}

    for i, leaf in enumerate(inp_leaves):
        if isinstance(leaf, sym.Symbol):
            if leaf in inp_symbols:
                inp_dups[i] = inp_indices[inp_symbols.index(leaf)]
            else:
                inp_indices.append(i)
                inp_symbols.append(leaf)
        elif is_sympy(leaf) and not isinstance(leaf, (sym.NumberSymbol, sym.Number)):
            raise ValueError(
                f"SymPy leaf {leaf} found that is not a symbol or a constant number. "
                "Only symbols and constants are allowed in the input definition."
            )
        else:
            # Compare with plain numbers in calls, which is much faster
            inp_constants[i] = complex(leaf) if isinstance(leaf, sym.Basic) else leaf

    outp_indices = [i for i, leaf in enumerate(outp_leaves) if isinstance(leaf, sym.Basic)]
    inner_f = sym.lambdify(inp_symbols, [outp_leaves[i] for i in outp_indices], **kwargs)

    def check(ii, ii_leaves):
        mismatches = [
            (i, ref) for i, ref in inp_dups.items() if not _equal(ii_leaves[i], ii_leaves[ref])
        ]
        if mismatches:
            i, ref = mismatches[0]
            paths = _paths(ii)
            raise ValueError(
                f"Input value {ii_leaves[i]} with path {paths[i]} was a duplicate symbol in "
                "original input but is now not matching the input value "
                f"{ii_leaves[ref]} at {paths[ref]}"
            )
        for i, expected in inp_constants.items():
            if not _equal(ii_leaves[i], expected):
                raise ValueError(
                    f"Constant value {ii_leaves[i]} doesn't match reference input "
                    f"{inp_leaves[i]} for {_paths(ii)[i]}."
                )

    def outer(ii):
        ii_leaves, ii_treedef = jax.tree.flatten(ii)
        if ii_treedef != inp_treedef:
            raise ValueError(
                f"Tree definition of input {ii_treedef} does not match expected "
                f"tree definition {inp_treedef}."
            )
        try:
            check(ii, ii_leaves)
        # Error checking is incompatible with jax.jit
        # but can be skipped without affecting the result
        except TracerBoolConversionError:
            pass

        oo_inner = inner_f(*(ii_leaves[i] for i in inp_indices))
        outp = list(outp_leaves)
        for index, val in zip(outp_indices, oo_inner, strict=True):
            outp[index] = val
        return jax.tree.unflatten(outp_treedef, outp)

    return outer


def _equal(a, b) -> bool:
    try:
        return bool(a == b)
    # Arrays with more than one element
    except ValueError:
        return bool(np.array_equal(a, b))


def _paths(tree) -> list[str]:
    return [jax.tree_util.keystr(path) for path, _ in jax.tree.leaves_with_path(tree)]


@cache
def _signature(f) -> inspect.Signature:
    return inspect.signature(f)


def normalized_args(wrapped, args, kwargs):
    bound = _signature(wrapped).bind(*args, **kwargs)
    bound.apply_defaults()
    all_args = bound.arguments
    return all_args


@cache
def _outer_lambdify(wrapped, modules, sample_args, sample_kwargs, sym_kwargs, recurse_for):
    sig = inspect.signature(wrapped)
    spec = inspect.getfullargspec(wrapped)
    partial_bound = sig.bind_partial(*sample_args, **sample_kwargs)
    # partial_bound.apply_defaults()
    generated_args = {}
    for arg in spec.args:
        if arg not in partial_bound.arguments:
            cls = spec.annotations.get(arg, float)
            generated_args[arg] = symbol_maker(cls, postfix=arg, recurse_for=recurse_for)
    partial_bound.arguments.update(generated_args)
    normalized = normalized_args(wrapped, args=(), kwargs=partial_bound.arguments)
    f = lambdify_tree(normalized, wrapped(**normalized), modules=modules, **sym_kwargs)
    return f


def lambdify(recurse_for=(), modules=sym, args=None, kwargs=None, **sym_kwargs):
    # Rename to keep outer API succinct
    sample_args = () if args is None else args
    sample_kwargs = {} if kwargs is None else kwargs

    # wrapped -> numeric function, avoids hashing the arguments of
    # _outer_lambdify() in each call
    compiled = {}

    @wrapt.decorator
    def lambdified(wrapped, instance, args, kwargs):
        f = compiled.get(wrapped)
        if f is None:
            f = compiled[wrapped] = _outer_lambdify(
                wrapped=wrapped,
                modules=modules,
                sample_args=tuple(sample_args),
                sample_kwargs=frozendict(sample_kwargs),
                sym_kwargs=frozendict(sym_kwargs),
                recurse_for=recurse_for,
            )
        return f(normalized_args(wrapped, args, kwargs))

    return lambdified


def lambdify_method(method: Callable) -> Callable:
    """
    Mark a method to be compiled in classes created by :func:`lambdify_class`

    The method must be compatible with SymPy. Arguments whose type annotation
    is numeric or in :code:`recurse_for` of :func:`lambdify_class` become
    symbols, see :func:`symbol_maker`. All other arguments are static: They are
    passed on as they are, and the method is compiled separately for each of
    their values, which must be hashable.
    """
    method.__lambdify_method__ = True
    return method


class _HashableArg:
    """
    Wrapper to use arguments like :code:`modules=[{"cos": f}, "numpy"]` as cache keys
    """

    def __init__(self, value):
        self.value = value
        self.key = _freeze(value)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, _HashableArg) and self.key == other.key

    def __repr__(self):
        return getattr(self.value, "__name__", repr(self.value))


def _freeze(value):
    if isinstance(value, dict):
        return frozendict({k: _freeze(v) for k, v in value.items()})
    elif isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def lambdify_class(
    base: type[T],
    modules,
    make_symbolic: Callable[[type[T]], T],
    recurse_for: tuple = (),
) -> type[T]:
    """
    Subclass of the dataclass :code:`base` with numeric versions of its methods

    The methods marked with :func:`lambdify_method` are compiled with
    :func:`lambdify_tree` from the methods of :code:`base`, using
    :code:`modules`, on first use. Arguments and results of type :code:`base`
    are converted from and to the subclass, so that calls can be chained.

    The subclass is created once for each combination of arguments. Its
    attribute :code:`_lambdify_base` is :code:`base`.

    Parameters
    ----------
    base
        Dataclass registered as a pytree, with SymPy-compatible methods
    modules
        :code:`modules` argument of :func:`sympy.lambdify`
    make_symbolic
        Function that returns an instance of :code:`base` with SymPy symbols
        as parameters
    recurse_for
        Argument types to create symbols for, see :func:`symbol_maker`
    """
    return _lambdify_class(base, _HashableArg(modules), make_symbolic, tuple(recurse_for))


@cache
def _lambdify_class(base, modules: _HashableArg, make_symbolic, recurse_for):
    cls = type(
        f"{base.__name__}Lambdified",
        (base,),
        {
            "__module__": base.__module__,
            "__qualname__": f"{base.__qualname__}.lambdify({modules!r})",
            "__doc__": f"{base.__name__} with methods compiled for {modules!r}",
            "_lambdify_base": base,
            "_lambdify_modules": modules.value,
        },
    )
    for name in dir(base):
        if getattr(getattr(base, name), "__lambdify_method__", False):
            setattr(
                cls,
                name,
                _numeric_method(cls, base, name, modules.value, make_symbolic, recurse_for),
            )
    return jdc.pytree_dataclass(cls)


def _numeric_method(cls, base, name, modules, make_symbolic, recurse_for):
    method = getattr(base, name)
    signature = inspect.signature(method)
    hints = typing.get_type_hints(method)
    numeric_args = [
        arg
        for arg in list(signature.parameters)[1:]
        if hints.get(arg, float) in recurse_for or isnumerictype(hints.get(arg, float))
    ]
    # static arguments -> compiled function
    compiled = {}

    def compile_method(static: tuple):
        symbolic = {"self": make_symbolic(base)}
        for arg in numeric_args:
            symbolic[arg] = symbol_maker(
                hints.get(arg, float), postfix=arg, recurse_for=recurse_for
            )
        output = method(symbolic["self"], **{k: symbolic[k] for k in numeric_args}, **dict(static))
        return lambdify_tree(symbolic, output, modules=modules)

    def is_base(x) -> bool:
        return type(x) is base

    @functools.wraps(method)
    def numeric_method(self, *args, **kwargs):
        arguments = signature.bind(self, *args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        static = tuple((k, v) for k, v in list(values.items())[1:] if k not in numeric_args)
        f = compiled.get(static)
        if f is None:
            f = compiled[static] = compile_method(static)
        inputs = {"self": _convert(self, base), **{k: values[k] for k in numeric_args}}
        return jax.tree.map(
            lambda x: _convert(x, cls) if is_base(x) else x, f(inputs), is_leaf=is_base
        )

    return numeric_method


def _convert(obj, target: type[T]) -> T:
    """
    Instance of the dataclass :code:`target` with the field values of :code:`obj`
    """
    return target(**{f.name: getattr(obj, f.name) for f in dataclasses.fields(target) if f.init})
