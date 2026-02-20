from __future__ import annotations

import ast
import functools

from .context import Context, FunctionCallContext
from .module_namespaces import MODULE_NAMESPACES
from .namespace import Namespace, ObjectKind
from .object_path import (
    BUILTINS_GLOBALS_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    DICT_FIELD_NAME,
    ModulePath,
    SYS_MODULES_LOCAL_OBJECT_PATH,
    SYS_MODULE_PATH,
)


@functools.singledispatch
def lookup_namespace_by_expression_node(
    _node: ast.expr,
    _namespace: Namespace,
    /,
    *_parent_namespaces: Namespace,
    context: Context,  # noqa: ARG001
) -> Namespace | None:
    return None


@lookup_namespace_by_expression_node.register(ast.Attribute)
def _(
    node: ast.Attribute,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    context: Context,
) -> Namespace | None:
    assert isinstance(node.ctx, ast.Load), ast.unparse(node)
    object_namespace = lookup_namespace_by_expression_node(
        node.value, namespace, *parent_namespaces, context=context
    )
    if object_namespace is None:
        return None
    attribute_name = node.attr
    try:
        return object_namespace.get_namespace_by_name(attribute_name)
    except KeyError:
        raise AttributeError(attribute_name) from None


@lookup_namespace_by_expression_node.register(ast.Call)
def _(
    node: ast.Call,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    context: Context,
) -> Namespace | None:
    callable_namespace = lookup_namespace_by_expression_node(
        node.func, namespace, *parent_namespaces, context=context
    )
    if callable_namespace is None:
        return None
    if (
        callable_namespace.module_path == BUILTINS_MODULE_PATH
        and callable_namespace.local_path == BUILTINS_GLOBALS_LOCAL_OBJECT_PATH
    ):
        return MODULE_NAMESPACES[namespace.module_path].get_namespace_by_name(
            DICT_FIELD_NAME
        )
    if callable_namespace.kind is ObjectKind.CLASS:
        return Namespace(
            ObjectKind.INSTANCE,
            callable_namespace.module_path,
            callable_namespace.local_path,
            callable_namespace,
        )
    if callable_namespace.kind is ObjectKind.METACLASS:
        return Namespace(
            ObjectKind.CLASS,
            callable_namespace.module_path,
            callable_namespace.local_path,
            callable_namespace,
        )
    if callable_namespace.kind is ObjectKind.ROUTINE:
        return Namespace(
            ObjectKind.ROUTINE_CALL,
            callable_namespace.module_path,
            callable_namespace.local_path,
        )
    return None


@lookup_namespace_by_expression_node.register(ast.Name)
def _(
    node: ast.Name,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    context: Context,  # noqa: ARG001
) -> Namespace | None:
    assert isinstance(node.ctx, ast.Load)
    return lookup_namespace_by_object_name(
        node.id, namespace, *parent_namespaces
    )


@lookup_namespace_by_expression_node.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    context: Context,
) -> Namespace | None:
    return lookup_namespace_by_expression_node(
        node.value, namespace, *parent_namespaces, context=context
    )


@lookup_namespace_by_expression_node.register(ast.Subscript)
def _(
    node: ast.Subscript,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    context: Context,
) -> Namespace | None:
    from .evaluation import EVALUATION_EXCEPTIONS, evaluate_expression_node

    value_namespace = lookup_namespace_by_expression_node(
        node.value, namespace, *parent_namespaces, context=context
    )
    if value_namespace is None:
        return None
    if (
        value_namespace.kind is ObjectKind.INSTANCE
        and value_namespace.module_path == SYS_MODULE_PATH
        and value_namespace.local_path == SYS_MODULES_LOCAL_OBJECT_PATH
    ):
        try:
            module_name = evaluate_expression_node(
                node.slice, namespace, *parent_namespaces, context=context
            )
        except EVALUATION_EXCEPTIONS:
            if isinstance(context, FunctionCallContext):
                # assume that caller namespace is affected
                return MODULE_NAMESPACES[context.caller_module_path]
        else:
            assert isinstance(module_name, str), module_name
            return MODULE_NAMESPACES[ModulePath.from_module_name(module_name)]
    return None


def lookup_namespace_by_object_name(
    name: str, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace:
    try:
        return namespace.strict_get_namespace_by_name(name)
    except KeyError:
        for parent_namespace in parent_namespaces:
            try:
                return parent_namespace.strict_get_namespace_by_name(name)
            except KeyError:
                continue
    try:
        return namespace.get_namespace_by_name(name)
    except KeyError:
        for parent_namespace in parent_namespaces:
            try:
                return parent_namespace.get_namespace_by_name(name)
            except KeyError:
                continue
        raise NameError(name) from None
