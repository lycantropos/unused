from __future__ import annotations

import ast
import functools

from .context import Context, FunctionCallContext
from .modules import MODULES
from .object_ import Object, ObjectKind, PlainObject, Scope
from .object_path import (
    BUILTINS_GLOBALS_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    DICT_FIELD_NAME,
    ModulePath,
    SYS_MODULES_LOCAL_OBJECT_PATH,
    SYS_MODULE_PATH,
)


@functools.singledispatch
def lookup_object_by_expression_node(
    _node: ast.expr,
    _scope: Scope,
    /,
    *_parent_scopes: Scope,
    context: Context,  # noqa: ARG001
) -> Object | None:
    return None


@lookup_object_by_expression_node.register(ast.Attribute)
def _(
    node: ast.Attribute,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object | None:
    assert isinstance(node.ctx, ast.Load), ast.unparse(node)
    value_object = lookup_object_by_expression_node(
        node.value, scope, *parent_scopes, context=context
    )
    if value_object is None:
        return None
    attribute_name = node.attr
    try:
        return value_object.get_attribute(attribute_name)
    except KeyError:
        raise AttributeError(attribute_name) from None


@lookup_object_by_expression_node.register(ast.Call)
def _(
    node: ast.Call, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object | None:
    callable_object = lookup_object_by_expression_node(
        node.func, scope, *parent_scopes, context=context
    )
    if callable_object is None:
        return None
    if (
        callable_object.module_path == BUILTINS_MODULE_PATH
        and callable_object.local_path == BUILTINS_GLOBALS_LOCAL_OBJECT_PATH
    ):
        return MODULES[scope.module_path].get_attribute(DICT_FIELD_NAME)
    if callable_object.kind is ObjectKind.CLASS:
        return PlainObject(
            ObjectKind.INSTANCE,
            callable_object.module_path,
            callable_object.local_path,
            callable_object,
        )
    if callable_object.kind is ObjectKind.METACLASS:
        return PlainObject(
            ObjectKind.CLASS,
            callable_object.module_path,
            callable_object.local_path,
            callable_object,
        )
    if callable_object.kind is ObjectKind.ROUTINE:
        return PlainObject(
            ObjectKind.ROUTINE_CALL,
            callable_object.module_path,
            callable_object.local_path,
        )
    return None


@lookup_object_by_expression_node.register(ast.Name)
def _(
    node: ast.Name,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,  # noqa: ARG001
) -> Object | None:
    assert isinstance(node.ctx, ast.Load)
    return lookup_object_by_name(node.id, scope, *parent_scopes)


@lookup_object_by_expression_node.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object | None:
    return lookup_object_by_expression_node(
        node.value, scope, *parent_scopes, context=context
    )


@lookup_object_by_expression_node.register(ast.Subscript)
def _(
    node: ast.Subscript,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object | None:
    from .evaluation import EVALUATION_EXCEPTIONS, evaluate_expression_node

    value_object = lookup_object_by_expression_node(
        node.value, scope, *parent_scopes, context=context
    )
    if value_object is None:
        return None
    if (
        value_object.kind is ObjectKind.INSTANCE
        and value_object.module_path == SYS_MODULE_PATH
        and value_object.local_path == SYS_MODULES_LOCAL_OBJECT_PATH
    ):
        try:
            module_name = evaluate_expression_node(
                node.slice, scope, *parent_scopes, context=context
            )
        except EVALUATION_EXCEPTIONS:
            if isinstance(context, FunctionCallContext):
                # assume that caller module is affected
                return MODULES[context.caller_module_path]
        else:
            assert isinstance(module_name, str), module_name
            return MODULES[ModulePath.from_module_name(module_name)]
    return None


def lookup_object_by_name(
    name: str, scope: Scope, /, *parent_scopes: Scope
) -> Object:
    try:
        return scope.strict_get_object(name)
    except KeyError:
        for parent_scope in parent_scopes:
            try:
                return parent_scope.strict_get_object(name)
            except KeyError:
                continue
    try:
        return scope.get_object(name)
    except KeyError:
        for parent_scope in parent_scopes:
            try:
                return parent_scope.get_object(name)
            except KeyError:
                continue
        raise NameError(name) from None
