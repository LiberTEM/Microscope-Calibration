"""
Render symbolic functions as Python source code

This allows to derive functions symbolically with SymPy at development time
and ship the result as plain Python code, avoiding the cost of the symbolic
derivation at runtime.
"""

import ast
import builtins
import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import sympy as sym
from sympy.printing.pycode import SymPyPrinter


def symbol_paths(tree: Any, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], sym.Symbol]]:
    """
    Yield the attribute path of each symbol in a tree of NamedTuples and dataclasses
    """
    if isinstance(tree, sym.Symbol):
        yield (path, tree)
    elif hasattr(tree, "_fields"):  # NamedTuple
        for name in tree._fields:
            yield from symbol_paths(getattr(tree, name), path + (name,))
    elif dataclasses.is_dataclass(tree):
        for field in dataclasses.fields(tree):
            yield from symbol_paths(getattr(tree, field.name), path + (field.name,))


def _render_tree(tree: Any, printer: SymPyPrinter, types: set[type], indent: str) -> str:
    if isinstance(tree, sym.Basic):
        return printer.doprint(tree)
    elif hasattr(tree, "_fields") or dataclasses.is_dataclass(tree):
        types.add(type(tree))
        names = (
            tree._fields
            if hasattr(tree, "_fields")
            else [field.name for field in dataclasses.fields(tree)]
        )
        inner = indent + "    "
        items = [
            f"{inner}{name}={_render_tree(getattr(tree, name), printer, types, inner)},\n"
            for name in names
        ]
        return f"{type(tree).__name__}(\n{''.join(items)}{indent})"
    elif isinstance(tree, tuple):
        inner = indent + "    "
        items = [f"{inner}{_render_tree(item, printer, types, inner)},\n" for item in tree]
        return f"(\n{''.join(items)}{indent})"
    else:
        return repr(tree)


def render_function(name: str, args: dict[str, Any], output: Any) -> tuple[str, set[type]]:
    """
    Render a function that calculates :code:`output` from :code:`args`

    Parameters
    ----------
    name
        Function name
    args
        Mapping of argument names to trees of NamedTuples and dataclasses
        with SymPy symbols as leaves. The symbols are replaced with the
        corresponding attributes of the arguments.
    output
        Tree of NamedTuples, dataclasses and tuples with SymPy expressions of
        the symbols in :code:`args` or numbers as leaves.

    Returns
    -------
    source, types
        Source code of the function and the NamedTuple/dataclass types it
        constructs
    """
    replacements = {}
    assignments = []
    for arg_name, tree in args.items():
        for path, symbol in symbol_paths(tree, (arg_name,)):
            local_name = "_".join(path)
            if symbol in replacements:
                raise ValueError(f"Duplicate symbol {symbol} in {path}")
            replacements[symbol] = sym.Symbol(local_name)
            if len(path) > 1:
                assignments.append(f"    {local_name} = {'.'.join(path)}\n")

    output = map_leaves(output, lambda leaf: _xreplace(leaf, replacements))
    used = {
        s.name for leaf in leaves(output) if isinstance(leaf, sym.Basic) for s in leaf.free_symbols
    }
    unknown = used - {s.name for s in replacements.values()}
    if unknown:
        raise ValueError(f"Output depends on symbols {unknown} that are not in args.")

    # Common subexpression elimination keeps the code short and fast
    if any(s.name.startswith(_CSE_PREFIX) for s in replacements.values()):
        raise ValueError(f"Argument names must not start with {_CSE_PREFIX}.")
    expressions = [leaf for leaf in leaves(output) if isinstance(leaf, sym.Basic)]
    intermediates, reduced = sym.cse(expressions, symbols=sym.numbered_symbols(_CSE_PREFIX))
    reduced_iter = iter(reduced)
    output = map_leaves(
        output, lambda leaf: next(reduced_iter) if isinstance(leaf, sym.Basic) else leaf
    )

    printer = SymPyPrinter({"fully_qualified_modules": False})
    types = set()
    body = _render_tree(output, printer, types, indent="    ")
    # Only assign the attributes that are used to keep the code concise
    lines = [line for line in assignments if line.split(" = ")[0].strip() in used]
    lines += [f"    {symbol} = {printer.doprint(expr)}\n" for symbol, expr in intermediates]
    source = f"def {name}({', '.join(args)}):\n" + "".join(lines) + f"    return {body}\n"
    return source, types


_CSE_PREFIX = "_cse"


def _xreplace(leaf: Any, replacements: dict) -> Any:
    return leaf.xreplace(replacements) if isinstance(leaf, sym.Basic) else leaf


def leaves(tree: Any) -> Iterator[sym.Basic]:
    """
    Yield the leaves, e.g. SymPy expressions or numbers, of a tree of
    NamedTuples, dataclasses and tuples
    """
    if hasattr(tree, "_fields"):
        for name in tree._fields:
            yield from leaves(getattr(tree, name))
    elif dataclasses.is_dataclass(tree):
        for field in dataclasses.fields(tree):
            yield from leaves(getattr(tree, field.name))
    elif isinstance(tree, tuple):
        for item in tree:
            yield from leaves(item)
    else:
        yield tree


def map_leaves(tree: Any, f: Callable[[Any], Any]) -> Any:
    """
    Apply :code:`f` to the leaves of a tree of NamedTuples, dataclasses and
    tuples, in the same order as :func:`leaves`
    """
    if hasattr(tree, "_fields"):
        return type(tree)(**{n: map_leaves(getattr(tree, n), f) for n in tree._fields})
    elif dataclasses.is_dataclass(tree):
        return dataclasses.replace(
            tree,
            **{
                fld.name: map_leaves(getattr(tree, fld.name), f) for fld in dataclasses.fields(tree)
            },
        )
    elif isinstance(tree, tuple):
        return tuple(map_leaves(item, f) for item in tree)
    else:
        return f(tree)


def _free_names(source: str) -> set[str]:
    """
    Names that are read in :code:`source` but neither assigned nor function
    arguments nor builtins
    """
    tree = ast.parse(source)
    loaded = set()
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.FunctionDef):
            bound.add(node.name)
    return loaded - bound - set(dir(builtins))


def render_module(functions: dict[str, tuple[dict[str, Any], Any]], header: str) -> str:
    """
    Render a module with the given functions

    Names that the printed SymPy expressions use, e.g. :code:`cos`, are imported
    from :code:`sympy`, so that the result works with SymPy and plain Python
    number types, like :func:`sympy.lambdify` with :code:`modules=sympy`.

    Parameters
    ----------
    functions
        Mapping of function names to :code:`(args, output)`, see :func:`render_function`
    header
        Comment at the top of the module
    """
    sources = []
    types = set()
    for name, (args, output) in sorted(functions.items()):
        source, function_types = render_function(name, args, output)
        sources.append(source)
        types |= function_types
    body = "\n\n".join(sources)

    imports = {}
    for t in types:
        imports.setdefault(t.__module__, set()).add(t.__name__)
    type_names = {t.__name__ for t in types}
    sympy_names = _free_names(body) - type_names
    for n in sympy_names:
        if not hasattr(sym, n):
            raise ValueError(f"Generated code uses name {n} that is not available in sympy.")
    if sympy_names:
        imports.setdefault("sympy", set()).update(sympy_names)

    import_lines = "".join(
        f"from {module} import {', '.join(sorted(names))}\n"
        for module, names in sorted(imports.items())
    )
    comment = "".join(f"# {line}\n".replace("# \n", "#\n") for line in header.splitlines())
    return f"{comment}\n{import_lines}\n\n{body}"
