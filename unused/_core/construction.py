from __future__ import annotations

import ast
import builtins
import functools

from .context import Context
from .enums import ObjectKind, ScopeKind
from .evaluation import (
    EVALUATION_EXCEPTIONS,
    evaluate_expression_node,
    function_node_to_keyword_only_defaults,
    function_node_to_positional_defaults,
    value_to_object,
)
from .lookup import lookup_object_by_expression_node, lookup_object_by_name
from .missing import MISSING
from .modules import BUILTINS_MODULE, MODULES, TYPES_MODULE
from .object_ import Call, Class, Instance, Object, Routine, UnknownObject
from .object_path import (
    BUILTINS_DICT_LOCAL_OBJECT_PATH,
    BUILTINS_GLOBALS_LOCAL_OBJECT_PATH,
    BUILTINS_LIST_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    BUILTINS_OBJECT_LOCAL_OBJECT_PATH,
    BUILTINS_SET_LOCAL_OBJECT_PATH,
    BUILTINS_TUPLE_LOCAL_OBJECT_PATH,
    BUILTINS_TYPE_LOCAL_OBJECT_PATH,
    COLLECTIONS_MODULE_PATH,
    COLLECTIONS_NAMEDTUPLE_LOCAL_OBJECT_PATH,
    DICT_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH,
)
from .scope import Scope
from .utils import ensure_type


@functools.singledispatch
def construct_object_from_expression_node(
    node: ast.expr,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    try:
        value = evaluate_expression_node(
            node, scope, *parent_scopes, context=context
        ).value
    except EVALUATION_EXCEPTIONS:
        value = MISSING
    return value_to_object(
        value, module_path=module_path, local_path=local_path
    )


@construct_object_from_expression_node.register(ast.Attribute)
def _(
    node: ast.Attribute,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    if (
        value_object := lookup_object_by_expression_node(
            node.value, scope, *parent_scopes, context=context
        )
    ) is not None:
        attribute_name = node.attr
        try:
            return value_object.get_attribute(attribute_name)
        except KeyError:
            raise AttributeError(attribute_name) from None
    return UnknownObject(module_path, local_path, value=MISSING)


@construct_object_from_expression_node.register(ast.Call)
def _(
    node: ast.Call,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    callable_object = lookup_object_by_expression_node(
        node.func, scope, *parent_scopes, context=context
    )
    if callable_object is None:
        return UnknownObject(module_path, local_path, value=MISSING)
    if callable_object.module_path == BUILTINS_MODULE_PATH and (
        callable_object.local_path == BUILTINS_TYPE_LOCAL_OBJECT_PATH
    ):
        assert callable_object.kind is ObjectKind.METACLASS, callable_object
        first_argument_object = construct_object_from_expression_node(
            node.args[0],
            scope,
            *parent_scopes,
            context=context,
            local_path=local_path.join('__args_0__'),
            module_path=module_path,
        )
        return (
            Class(
                Scope(ScopeKind.METACLASS, module_path, local_path),
                ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_TYPE_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                metacls=MISSING,
            )
            if (
                len(node.args) == 1
                and first_argument_object is not None
                and first_argument_object.kind is ObjectKind.CLASS
            )
            else Class(
                Scope(
                    ScopeKind.UNKNOWN_CLASS
                    if (
                        first_argument_object is None
                        or (
                            first_argument_object.kind
                            in (ObjectKind.UNKNOWN_CLASS, ObjectKind.UNKNOWN)
                        )
                    )
                    else ScopeKind.CLASS,
                    module_path,
                    local_path,
                ),
                ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_OBJECT_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                metacls=MISSING,
            )
        )
    if callable_object.kind is ObjectKind.CLASS:
        return Instance(
            module_path, local_path, cls=callable_object, value=MISSING
        )
    if callable_object.kind is ObjectKind.METACLASS:
        return Class(
            Scope(ScopeKind.CLASS, module_path, local_path),
            metacls=callable_object,
        )
    if (
        callable_object.module_path == BUILTINS_MODULE_PATH
        and callable_object.local_path == BUILTINS_GLOBALS_LOCAL_OBJECT_PATH
    ):
        try:
            assert callable_object.kind is ObjectKind.ROUTINE, callable_object
        except AssertionError:
            raise
        return MODULES[scope.module_path].get_attribute(DICT_FIELD_NAME)
    if (
        callable_object.kind is ObjectKind.ROUTINE
        and callable_object.module_path == BUILTINS_MODULE_PATH
        and (
            callable_object.local_path
            == LocalObjectPath.from_object_name(builtins.vars.__qualname__)
        )
    ):
        (argument_node,) = node.args
        argument_object = lookup_object_by_expression_node(
            argument_node, scope, *parent_scopes, context=context
        )
        assert argument_object is not None
        return argument_object.get_attribute(DICT_FIELD_NAME)
    if callable_object.module_path == COLLECTIONS_MODULE_PATH and (
        callable_object.local_path == COLLECTIONS_NAMEDTUPLE_LOCAL_OBJECT_PATH
    ):
        _, namedtuple_field_name_node = node.args
        try:
            named_tuple_field_names = evaluate_expression_node(
                namedtuple_field_name_node,
                scope,
                *parent_scopes,
                context=context,
            ).value
        except EVALUATION_EXCEPTIONS:
            return UnknownObject(module_path, local_path, value=MISSING)
        if isinstance(named_tuple_field_names, str):
            named_tuple_field_names = named_tuple_field_names.replace(
                ',', ' '
            ).split()
        assert isinstance(named_tuple_field_names, tuple | list), ast.unparse(
            node
        )
        named_tuple_object = Class(
            Scope(ScopeKind.CLASS, module_path, local_path),
            ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_TUPLE_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_OBJECT_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            metacls=MISSING,
        )
        for field_name in named_tuple_field_names:
            assert isinstance(field_name, str), field_name
            named_tuple_object.set_attribute(
                field_name,
                UnknownObject(
                    named_tuple_object.module_path,
                    named_tuple_object.local_path.join(field_name),
                    value=MISSING,
                ),
            )
        return named_tuple_object
    return Call(
        module_path,
        local_path,
        callable_object,
        [
            (
                isinstance(argument_node, ast.Starred),
                construct_object_from_expression_node(
                    (
                        argument_node.value
                        if isinstance(argument_node, ast.Starred)
                        else argument_node
                    ),
                    scope,
                    *parent_scopes,
                    context=context,
                    local_path=local_path.join(f'__args_{argument_index}__'),
                    module_path=module_path,
                ),
            )
            for argument_index, argument_node in enumerate(node.args)
        ],
        [
            (
                argument_node.arg,
                construct_object_from_expression_node(
                    argument_node.value,
                    scope,
                    *parent_scopes,
                    context=context,
                    local_path=local_path.join(f'__args_{argument_index}__'),
                    module_path=module_path,
                ),
            )
            for argument_index, argument_node in enumerate(node.keywords)
        ],
    )


@construct_object_from_expression_node.register(ast.Dict)
@construct_object_from_expression_node.register(ast.DictComp)
def _(
    node: ast.Dict | ast.DictComp,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    try:
        value = evaluate_expression_node(
            node, scope, *parent_scopes, context=context
        ).value
    except EVALUATION_EXCEPTIONS:
        value = MISSING
    return Instance(
        module_path,
        local_path,
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_DICT_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )


@construct_object_from_expression_node.register(ast.Lambda)
def _(
    node: ast.Lambda,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    return Routine(
        module_path,
        local_path,
        ast_node=node,
        cls=ensure_type(
            TYPES_MODULE.get_nested_attribute(
                TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        keyword_only_defaults=function_node_to_keyword_only_defaults(
            node.args, scope, *parent_scopes, context=context
        ),
        positional_defaults=function_node_to_positional_defaults(
            node.args, scope, *parent_scopes, context=context
        ),
    )


@construct_object_from_expression_node.register(ast.List)
@construct_object_from_expression_node.register(ast.ListComp)
def _(
    node: ast.List | ast.ListComp,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    try:
        value = evaluate_expression_node(
            node, scope, *parent_scopes, context=context
        ).value
    except EVALUATION_EXCEPTIONS:
        value = MISSING
    return Instance(
        module_path,
        local_path,
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_LIST_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )


@construct_object_from_expression_node.register(ast.Name)
def _(
    node: ast.Name,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,  # noqa: ARG001
    local_path: LocalObjectPath,  # noqa: ARG001
    module_path: ModulePath,  # noqa: ARG001
) -> Object:
    return lookup_object_by_name(node.id, scope, *parent_scopes)


@construct_object_from_expression_node.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    return (
        result
        if (
            result := lookup_object_by_expression_node(
                node.value, scope, *parent_scopes, context=context
            )
        )
        is not None
        else UnknownObject(module_path, local_path, value=MISSING)
    )


@construct_object_from_expression_node.register(ast.Set)
@construct_object_from_expression_node.register(ast.SetComp)
def _(
    node: ast.Set | ast.SetComp,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    try:
        value = evaluate_expression_node(
            node, scope, *parent_scopes, context=context
        ).value
    except EVALUATION_EXCEPTIONS:
        value = MISSING
    return Instance(
        module_path,
        local_path,
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_SET_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )


@construct_object_from_expression_node.register(ast.Tuple)
def _(
    node: ast.Tuple,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Object:
    try:
        value = evaluate_expression_node(
            node, scope, *parent_scopes, context=context
        ).value
    except EVALUATION_EXCEPTIONS:
        value = MISSING
    return Instance(
        module_path,
        local_path,
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_TUPLE_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )
