from __future__ import annotations

import ast
import builtins
import functools
import operator
from collections.abc import Callable, Mapping
from typing import Any, Final

from .context import Context
from .lookup import lookup_object_by_expression_node
from .object_ import Scope
from .object_path import BUILTINS_MODULE_PATH, LocalObjectPath, ModulePath

EVALUATION_EXCEPTIONS: Final = (
    AttributeError,
    IndexError,
    KeyError,
    NameError,
    TypeError,
)


@functools.singledispatch
def evaluate_expression_node(
    node: ast.expr,
    scope: Scope,  # noqa: ARG001
    /,
    *parent_scopes: Scope,  # noqa: ARG001
    context: Context,  # noqa: ARG001
) -> Any:
    raise TypeError(type(node))


@evaluate_expression_node.register(ast.Attribute)
def _(
    node: ast.Attribute,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Any:
    return getattr(
        evaluate_expression_node(
            node.value, scope, *parent_scopes, context=context
        ),
        node.attr,
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
    )
}


@evaluate_expression_node.register(ast.Call)
def _(
    node: ast.Call, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    try:
        callable_ = evaluate_expression_node(
            node.func, scope, *parent_scopes, context=context
        )
    except EVALUATION_EXCEPTIONS:
        callable_object = lookup_object_by_expression_node(
            node.func, scope, *parent_scopes, context=context
        )
        if callable_object is None:
            raise
        callable_ = _ALLOWED_CALLABLES.get(
            (callable_object.module_path, callable_object.local_path)
        )
        if callable_ is None:
            raise
    args: list[Any] = []
    for positional_argument_node in node.args:
        if isinstance(positional_argument_node, ast.Starred):
            args.extend(
                evaluate_expression_node(
                    positional_argument_node.value,
                    scope,
                    *parent_scopes,
                    context=context,
                )
            )
        else:
            args.append(
                evaluate_expression_node(
                    positional_argument_node,
                    scope,
                    *parent_scopes,
                    context=context,
                )
            )
    kwargs: dict[str, Any] = {}
    for keyword_argument_node in node.keywords:
        if (parameter_name := keyword_argument_node.arg) is not None:
            kwargs[parameter_name] = evaluate_expression_node(
                keyword_argument_node.value,
                scope,
                *parent_scopes,
                context=context,
            )
        else:
            kwargs.update(
                evaluate_expression_node(
                    keyword_argument_node.value,
                    scope,
                    *parent_scopes,
                    context=context,
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


@evaluate_expression_node.register(ast.BinOp)
def _(
    node: ast.BinOp, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    return _binary_operators_by_operator_type[type(node.op)](
        evaluate_expression_node(
            node.left, scope, *parent_scopes, context=context
        ),
        evaluate_expression_node(
            node.right, scope, *parent_scopes, context=context
        ),
    )


@evaluate_expression_node.register(ast.BoolOp)
def _(
    node: ast.BoolOp, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    if isinstance(node.op, ast.And):
        try:
            return next(
                candidate
                for value_node in node.values[:-1]
                if not (
                    candidate := evaluate_expression_node(
                        value_node, scope, *parent_scopes, context=context
                    )
                )
            )
        except StopIteration:
            return evaluate_expression_node(
                node.values[-1], scope, *parent_scopes, context=context
            )
    assert isinstance(node.op, ast.Or), ast.unparse(node)
    try:
        return next(
            candidate
            for value_node in node.values[:-1]
            if (
                candidate := evaluate_expression_node(
                    value_node, scope, *parent_scopes, context=context
                )
            )
        )
    except StopIteration:
        return evaluate_expression_node(
            node.values[-1], scope, *parent_scopes, context=context
        )


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


@evaluate_expression_node.register(ast.Compare)
def _(
    node: ast.Compare, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> bool:
    value = evaluate_expression_node(
        node.left, scope, *parent_scopes, context=context
    )
    for operator_node, next_value in zip(
        node.ops,
        (
            evaluate_expression_node(
                operand_node, scope, *parent_scopes, context=context
            )
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


@evaluate_expression_node.register(ast.Constant)
def _(
    node: ast.Constant,
    scope: Scope,  # noqa: ARG001
    /,
    *parent_scopes: Scope,  # noqa: ARG001
    context: Context,  # noqa: ARG001
) -> Any:
    return node.value


@evaluate_expression_node.register(ast.Dict)
def _(
    node: ast.Dict, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    result: dict[Any, Any] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        value = evaluate_expression_node(
            value_node, scope, *parent_scopes, context=context
        )
        if key_node is None:
            result.update(**value)
        else:
            key = evaluate_expression_node(
                key_node, scope, *parent_scopes, context=context
            )
            result[key] = value
    return result


@evaluate_expression_node.register(ast.List)
def _(
    node: ast.List, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> list[Any]:
    result = []
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            result.extend(
                evaluate_expression_node(
                    element_node.value, scope, *parent_scopes, context=context
                )
            )
        else:
            result.append(
                evaluate_expression_node(
                    element_node, scope, *parent_scopes, context=context
                )
            )
    return result


@evaluate_expression_node.register(ast.Name)
def _(
    node: ast.Name,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,  # noqa: ARG001
) -> Any:
    name = node.id
    try:
        return scope.get_value(name)
    except KeyError:
        for parent_scope in parent_scopes:
            try:
                return parent_scope.get_value(name)
            except KeyError:
                continue
        raise NameError(name) from None


@evaluate_expression_node.register(ast.Set)
def _(
    node: ast.Set, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    result = set()
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            result.update(
                evaluate_expression_node(
                    element_node.value, scope, *parent_scopes, context=context
                )
            )
        else:
            result.add(
                evaluate_expression_node(
                    element_node, scope, *parent_scopes, context=context
                )
            )
    return result


@evaluate_expression_node.register(ast.Subscript)
def _(
    node: ast.Subscript,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Any:
    return evaluate_expression_node(
        node.value, scope, *parent_scopes, context=context
    )[
        evaluate_expression_node(
            node.slice, scope, *parent_scopes, context=context
        )
    ]


@evaluate_expression_node.register(ast.Slice)
def _(
    node: ast.Slice, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    start = (
        evaluate_expression_node(
            start_node, scope, *parent_scopes, context=context
        )
        if (start_node := node.lower) is not None
        else None
    )
    stop = (
        evaluate_expression_node(
            stop_node, scope, *parent_scopes, context=context
        )
        if (stop_node := node.upper) is not None
        else None
    )
    step = (
        evaluate_expression_node(
            step_node, scope, *parent_scopes, context=context
        )
        if (step_node := node.step) is not None
        else None
    )
    return slice(start, stop, step)


@evaluate_expression_node.register(ast.Tuple)
def _(
    node: ast.Tuple, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> tuple[Any, ...]:
    result = []
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            result.extend(
                evaluate_expression_node(
                    element_node.value, scope, *parent_scopes, context=context
                )
            )
        else:
            result.append(
                evaluate_expression_node(
                    element_node, scope, *parent_scopes, context=context
                )
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


@evaluate_expression_node.register(ast.UnaryOp)
def _(
    node: ast.UnaryOp, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    return _unary_operators_by_operator_type[type(node.op)](
        evaluate_expression_node(
            node.operand, scope, *parent_scopes, context=context
        )
    )
