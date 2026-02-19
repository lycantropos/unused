from __future__ import annotations

import ast
import builtins
import functools
import operator
from collections.abc import Callable, Mapping
from typing import Any, Final

from .lookup import lookup_namespace_by_expression_node
from .namespace import Namespace
from .object_path import BUILTINS_MODULE_PATH, LocalObjectPath, ModulePath


@functools.singledispatch
def evaluate_node(
    node: ast.expr,
    namespace: Namespace,  # noqa: ARG001
    /,
    *parent_namespaces: Namespace,  # noqa: ARG001
) -> Any:
    raise TypeError(type(node))


@evaluate_node.register(ast.Attribute)
def _(
    node: ast.Attribute, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    return getattr(
        evaluate_node(node.value, namespace, *parent_namespaces), node.attr
    )


_ALLOWED_CALLABLES: Final[
    Mapping[tuple[ModulePath, LocalObjectPath], Callable[..., Any]]
] = {
    (
        BUILTINS_MODULE_PATH,
        LocalObjectPath.from_object_name(callable_.__qualname__),
    ): callable_
    for callable_ in (
        builtins.getattr,
        builtins.hasattr,
        builtins.isinstance,
        builtins.issubclass,
        builtins.len,
        builtins.iter,
    )
}


@evaluate_node.register(ast.Call)
def _(
    node: ast.Call, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    try:
        callable_ = evaluate_node(node.func, namespace, *parent_namespaces)
    except EVALUATION_EXCEPTIONS:
        callable_namespace = lookup_namespace_by_expression_node(
            node.func, namespace, *parent_namespaces
        )
        if callable_namespace is None:
            raise
        callable_ = _ALLOWED_CALLABLES.get(
            (callable_namespace.module_path, callable_namespace.local_path)
        )
        if callable_ is None:
            raise
    args: list[Any] = []
    for positional_argument_node in node.args:
        if isinstance(positional_argument_node, ast.Starred):
            args.extend(
                evaluate_node(
                    positional_argument_node.value,
                    namespace,
                    *parent_namespaces,
                )
            )
        else:
            args.append(
                evaluate_node(
                    positional_argument_node, namespace, *parent_namespaces
                )
            )
    kwargs: dict[str, Any] = {}
    for keyword_argument_node in node.keywords:
        if (parameter_name := keyword_argument_node.arg) is not None:
            kwargs[parameter_name] = evaluate_node(
                keyword_argument_node.value, namespace, *parent_namespaces
            )
        else:
            kwargs.update(
                evaluate_node(
                    keyword_argument_node.value, namespace, *parent_namespaces
                )
            )
    return callable_(*args, **kwargs)


_binary_operators_by_operator_type: Mapping[
    type[ast.operator], Callable[[Any, Any], Any]
] = {
    ast.Add: operator.add,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.LShift: operator.lshift,
    ast.MatMult: operator.matmul,
    ast.Mod: operator.mod,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.RShift: operator.rshift,
    ast.Sub: operator.sub,
}


@evaluate_node.register(ast.BinOp)
def _(
    node: ast.BinOp, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    return _binary_operators_by_operator_type[type(node.op)](
        evaluate_node(node.left, namespace, *parent_namespaces),
        evaluate_node(node.right, namespace, *parent_namespaces),
    )


@evaluate_node.register(ast.BoolOp)
def _(
    node: ast.BoolOp, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    if isinstance(node.op, ast.And):
        try:
            return next(
                candidate
                for value_node in node.values[:-1]
                if not (
                    candidate := evaluate_node(
                        value_node, namespace, *parent_namespaces
                    )
                )
            )
        except StopIteration:
            return evaluate_node(
                node.values[-1], namespace, *parent_namespaces
            )
    assert isinstance(node.op, ast.Or), ast.unparse(node)
    try:
        return next(
            candidate
            for value_node in node.values[:-1]
            if (
                candidate := evaluate_node(
                    value_node, namespace, *parent_namespaces
                )
            )
        )
    except StopIteration:
        return evaluate_node(node.values[-1], namespace, *parent_namespaces)


_binary_comparison_operators_by_operator_node_type: Mapping[
    type[ast.cmpop], Callable[[Any, Any], bool]
] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


@evaluate_node.register(ast.Compare)
def _(
    node: ast.Compare, namespace: Namespace, /, *parent_namespaces: Namespace
) -> bool:
    value = evaluate_node(node.left, namespace, *parent_namespaces)
    for operator_node, next_value in zip(
        node.ops,
        (
            evaluate_node(operand_node, namespace, *parent_namespaces)
            for operand_node in node.comparators
        ),
        strict=True,
    ):
        if not _binary_comparison_operators_by_operator_node_type[
            type(operator_node)
        ](value, next_value):
            return False
        value = next_value
    return True


@evaluate_node.register(ast.Constant)
def _(
    node: ast.Constant,
    namespace: Namespace,  # noqa: ARG001
    /,
    *parent_namespaces: Namespace,  # noqa: ARG001
) -> Any:
    return node.value


@evaluate_node.register(ast.Dict)
def _(
    node: ast.Dict, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    result: dict[Any, Any] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        value = evaluate_node(value_node, namespace, *parent_namespaces)
        if key_node is None:
            result.update(**value)
        else:
            key = evaluate_node(key_node, namespace, *parent_namespaces)
            result[key] = value
    return result


@evaluate_node.register(ast.List)
def _(
    node: ast.List, namespace: Namespace, /, *parent_namespaces: Namespace
) -> list[Any]:
    result = []
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            result.extend(
                evaluate_node(
                    element_node.value, namespace, *parent_namespaces
                )
            )
        else:
            result.append(
                evaluate_node(element_node, namespace, *parent_namespaces)
            )
    return result


@evaluate_node.register(ast.Name)
def _(
    node: ast.Name, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    name = node.id
    try:
        return namespace.get_object_by_name(name)
    except KeyError:
        for parent_namespace in parent_namespaces:
            try:
                return parent_namespace.get_object_by_name(name)
            except KeyError:
                continue
        raise NameError(name) from None


@evaluate_node.register(ast.Set)
def _(
    node: ast.Set, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    result = set()
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            result.update(
                evaluate_node(
                    element_node.value, namespace, *parent_namespaces
                )
            )
        else:
            result.add(
                evaluate_node(element_node, namespace, *parent_namespaces)
            )
    return result


@evaluate_node.register(ast.Subscript)
def _(
    node: ast.Subscript, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    return evaluate_node(node.value, namespace, *parent_namespaces)[
        evaluate_node(node.slice, namespace, *parent_namespaces)
    ]


@evaluate_node.register(ast.Slice)
def _(
    node: ast.Slice, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    start = (
        evaluate_node(start_node, namespace, *parent_namespaces)
        if (start_node := node.lower) is not None
        else None
    )
    stop = (
        evaluate_node(stop_node, namespace, *parent_namespaces)
        if (stop_node := node.upper) is not None
        else None
    )
    step = (
        evaluate_node(step_node, namespace, *parent_namespaces)
        if (step_node := node.step) is not None
        else None
    )
    return slice(start, stop, step)


@evaluate_node.register(ast.Tuple)
def _(
    node: ast.Tuple, namespace: Namespace, /, *parent_namespaces: Namespace
) -> tuple[Any, ...]:
    result = []
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            result.extend(
                evaluate_node(
                    element_node.value, namespace, *parent_namespaces
                )
            )
        else:
            result.append(
                evaluate_node(element_node, namespace, *parent_namespaces)
            )
    return tuple(result)


_unary_operators_by_operator_type: Mapping[
    type[ast.unaryop], Callable[[Any], Any]
] = {
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@evaluate_node.register(ast.UnaryOp)
def _(
    node: ast.UnaryOp, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Any:
    return _unary_operators_by_operator_type[type(node.op)](
        evaluate_node(node.operand, namespace, *parent_namespaces)
    )


EVALUATION_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
    AttributeError,
    IndexError,
    KeyError,
    NameError,
    TypeError,
)
