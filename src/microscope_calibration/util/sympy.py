from copy import deepcopy
from typing import TypeVar, Any
from collections.abc import Container
from numbers import Number
import inspect
from frozendict import frozendict
from functools import cache

import wrapt
import sympy as sym
import jax
from jax.errors import TracerBoolConversionError

T = TypeVar("T")


def isnumerictype(T):
    for supercls in (Number, sym.Number):
        # try ... except because some reasonable Ts like int | float cause a TypeError here
        try:
            if issubclass(T, supercls):
                return True
        except TypeError:
            pass
    # This can also handle unions like int | float via isinstance()
    for number in (0, 0., 0.+0.j):
        # try ... except because weird types can cause a TypeError here,
        # but we can safely continue since we didn't positively recognize it as numeric
        try:
            if isinstance(number, T):
                return True
        except TypeError:
            pass
    return False


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


def symbol_maker(
        params_cls: type[T], postfix: str | None = None,
        recurse_for: Container = tuple()) -> T:
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
    def symbol_maker_inner(params_cls: type[T], postfix: str | None,
            recurse_for: Container, index: int) -> tuple[int, T]:
        if hasattr(params_cls, '__annotations__'):
            symbols_dict = {}
            for attr in params_cls.__annotations__.keys():
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
        params_cls=params_cls,
        postfix=postfix,
        recurse_for=recurse_for,
        index=0
    )
    return res


SymbolJaxTree = TypeVar("SymbolJaxTree")


def lambdify_tree(inp: SymbolJaxTree, outp: SymbolJaxTree, **kwargs):
    inp_leaves, inp_treedef = jax.tree.flatten(inp)
    outp_leaves, outp_treedef = jax.tree.flatten(outp)

    inp_indices = []
    inp_symbols = []
    inp_dups = {}

    for i, leave in enumerate(inp_leaves):
        if isinstance(leave, sym.Symbol):
            if leave in inp_symbols:
                inp_dups[i] = inp_symbols.index(leave)
            else:
                inp_indices.append(i)
                inp_symbols.append(leave)
        elif is_sympy(leave) and not isinstance(leave, (sym.NumberSymbol, sym.Number)):
            raise ValueError(
                f"SymPy leave {leave} found that is not a symbol or a constant number. "
                "Only symbols and constants are allowed in the input definition."
            )

    outp_indices = []
    outp_exprs = []

    for i, leave in enumerate(outp_leaves):
        if isinstance(leave, sym.Basic):
            outp_indices.append(i)
            outp_exprs.append(leave)

    inp_indices_set = set(inp_indices)
    inner_f = sym.lambdify(inp_symbols, outp_exprs, **kwargs)

    def outer(ii):
        ii_leaves, ii_treedef = jax.tree.flatten_with_path(ii)
        if ii_treedef != inp_treedef:
            raise ValueError(
                f'Tree definition of input {ii_treedef} does not match expected '
                f'tree definition {inp_treedef}.'
            )
        try:
            for i, (path, leave) in enumerate(ii_leaves):
                if i not in inp_indices_set:
                    if i in inp_dups:
                        orig_i = inp_dups[i]
                        orig_path, orig_leave = ii_leaves[orig_i]
                        if orig_leave != leave:
                            raise ValueError(
                                f"Input value {leave} with path {path} was a duplicate symbol in "
                                "original input but is now not matching the input value "
                                f"{orig_leave} at {orig_path}"
                            )
                    elif leave != inp_leaves[i]:
                        raise ValueError(
                            f"Constant value {leave} doesn't match reference input "
                            f"{inp_leaves[i]} for {path}.")
        # Error checking is incompatible with jax.jit
        # but can be skipped without affecting the result
        except TracerBoolConversionError:
            pass

        ii_vals = [ii_leaves[i][1] for i in inp_indices]
        oo_inner = inner_f(*ii_vals)
        outp = deepcopy(outp_leaves)
        for i, val in enumerate(oo_inner):
            index = outp_indices[i]
            outp[index] = val
        return jax.tree.unflatten(outp_treedef, outp)

    return outer


def normalized_args(wrapped, args, kwargs):
    sig = inspect.signature(wrapped)
    bound = sig.bind(*args, **kwargs)
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
    normalized = normalized_args(wrapped, args=tuple(), kwargs=partial_bound.arguments)
    f = lambdify_tree(normalized, wrapped(**normalized), modules=modules, **sym_kwargs)
    return f


def lambdify(recurse_for=tuple(), modules=sym, args=None, kwargs=None, **sym_kwargs):
    # Rename to keep outer API succinct
    sample_args = tuple() if args is None else args
    sample_kwargs = {} if kwargs is None else kwargs

    @wrapt.decorator
    def lambdified(wrapped, instance, args, kwargs):
        n_args = normalized_args(wrapped, args, kwargs)
        f = _outer_lambdify(
            wrapped=wrapped,
            modules=modules,
            sample_args=frozendict(sample_args),
            sample_kwargs=frozendict(sample_kwargs),
            sym_kwargs=frozendict(sym_kwargs),
            recurse_for=recurse_for,
        )
        return f(n_args)

    return lambdified
