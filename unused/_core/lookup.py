from __future__ import annotations

import ast
import functools

from .namespace import Namespace, ObjectKind


@functools.singledispatch
def lookup_namespace_by_expression_node(
    _node: ast.expr, _namespace: Namespace, /, *_parent_namespaces: Namespace
) -> Namespace | None:
    return None


@lookup_namespace_by_expression_node.register(ast.Attribute)
def _(
    node: ast.Attribute, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    assert isinstance(node.ctx, ast.Load), ast.unparse(node)
    object_namespace = lookup_namespace_by_expression_node(
        node.value, namespace, *parent_namespaces
    )
    if object_namespace is None:
        return None
    try:
        return object_namespace.get_namespace_by_name(node.attr)
    except KeyError:
        raise AttributeError(node.attr) from None


@lookup_namespace_by_expression_node.register(ast.Call)
def _(
    node: ast.Call, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    callable_namespace = lookup_namespace_by_expression_node(
        node.func, namespace, *parent_namespaces
    )
    if callable_namespace is None:
        return None
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
    node: ast.Name, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    assert isinstance(node.ctx, ast.Load)
    return lookup_namespace_by_object_name(
        node.id, namespace, *parent_namespaces
    )


@lookup_namespace_by_expression_node.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    return lookup_namespace_by_expression_node(
        node.value, namespace, *parent_namespaces
    )


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
