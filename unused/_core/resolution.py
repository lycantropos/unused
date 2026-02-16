import ast
import functools

from .evaluation import EVALUATION_EXCEPTIONS, evaluate_node
from .module_namespaces import BUILTINS_MODULE_NAMESPACE
from .namespace import Namespace, ObjectKind
from .object_path import (
    BUILTINS_MODULE_PATH,
    COLLECTIONS_MODULE_PATH,
    GLOBALS_LOCAL_OBJECT_PATH,
    LocalObjectPath,
    ModulePath,
    NAMED_TUPLE_LOCAL_OBJECT_PATH,
)


@functools.singledispatch
def lookup_node(
    _node: ast.expr, _namespace: Namespace, /, *_parent_namespaces: Namespace
) -> Namespace | None:
    return None


@lookup_node.register(ast.Attribute)
def _(
    node: ast.Attribute, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    object_namespace = lookup_node(node.value, namespace, *parent_namespaces)
    return (
        None
        if object_namespace is None
        else object_namespace.get_namespace_by_name(node.attr)
    )


@lookup_node.register(ast.Call)
def _(
    node: ast.Call, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    callable_namespace = lookup_node(node.func, namespace, *parent_namespaces)
    if callable_namespace is None:
        return None
    if callable_namespace.kind is ObjectKind.CLASS:
        return Namespace(
            ObjectKind.INSTANCE,
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


@lookup_node.register(ast.Name)
def _(
    node: ast.Name, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    return _resolve_object_name(node.id, namespace, *parent_namespaces)


@lookup_node.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace | None:
    return lookup_node(node.value, namespace, *parent_namespaces)


@functools.singledispatch
def resolve_node(
    _node: ast.expr,
    _namespace: Namespace,
    /,
    *_parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    return Namespace(ObjectKind.UNKNOWN, module_path, local_path)


@resolve_node.register(ast.Attribute)
def _(
    node: ast.Attribute,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    if (
        object_namespace := lookup_node(
            node.value, namespace, *parent_namespaces
        )
    ) is not None:
        attribute_name = node.attr
        try:
            return object_namespace.get_namespace_by_name(attribute_name)
        except KeyError:
            raise AttributeError(attribute_name) from None
    return Namespace(ObjectKind.UNKNOWN, module_path, local_path)


@resolve_node.register(ast.Call)
def _(
    node: ast.Call,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    callable_namespace = lookup_node(node.func, namespace, *parent_namespaces)
    if callable_namespace is None:
        return Namespace(ObjectKind.UNKNOWN, module_path, local_path)
    if callable_namespace.kind is ObjectKind.CLASS:
        return Namespace(
            ObjectKind.INSTANCE,
            callable_namespace.module_path,
            callable_namespace.local_path,
            callable_namespace,
        )
    if (
        callable_namespace.module_path == BUILTINS_MODULE_PATH
        and callable_namespace.local_path == GLOBALS_LOCAL_OBJECT_PATH
    ):
        return Namespace(
            ObjectKind.ROUTINE_CALL,
            callable_namespace.module_path,
            callable_namespace.local_path,
        )
    if (callable_namespace.module_path == COLLECTIONS_MODULE_PATH) and (
        callable_namespace.local_path == NAMED_TUPLE_LOCAL_OBJECT_PATH
    ):
        _, namedtuple_field_name_node = node.args
        try:
            named_tuple_field_names = evaluate_node(
                namedtuple_field_name_node, namespace, *parent_namespaces
            )
        except EVALUATION_EXCEPTIONS:
            return Namespace(ObjectKind.UNKNOWN, module_path, local_path)
        if isinstance(named_tuple_field_names, str):
            named_tuple_field_names = named_tuple_field_names.replace(
                ',', ' '
            ).split()
        assert isinstance(named_tuple_field_names, tuple | list), ast.unparse(
            node
        )
        named_tuple_namespace = Namespace(
            ObjectKind.CLASS,
            module_path,
            local_path,
            BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                LocalObjectPath.from_object_name(tuple.__qualname__)
            ),
            BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                LocalObjectPath.from_object_name(object.__qualname__)
            ),
        )
        for field_name in named_tuple_field_names:
            named_tuple_namespace.set_namespace_by_name(
                field_name,
                Namespace(
                    ObjectKind.UNKNOWN,
                    named_tuple_namespace.module_path,
                    named_tuple_namespace.local_path.join(field_name),
                ),
            )
        return named_tuple_namespace
    return Namespace(ObjectKind.UNKNOWN, module_path, local_path)


@resolve_node.register(ast.Dict)
@resolve_node.register(ast.DictComp)
def _(
    _node: ast.Dict | ast.DictComp,
    _namespace: Namespace,
    /,
    *_parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    return Namespace(
        ObjectKind.INSTANCE,
        module_path,
        local_path,
        BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
            LocalObjectPath.from_object_name(dict.__qualname__)
        ),
    )


@resolve_node.register(ast.List)
@resolve_node.register(ast.ListComp)
def _(
    _node: ast.List | ast.ListComp,
    _namespace: Namespace,
    /,
    *_parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    return Namespace(
        ObjectKind.INSTANCE,
        module_path,
        local_path,
        BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
            LocalObjectPath.from_object_name(list.__qualname__)
        ),
    )


@resolve_node.register(ast.Name)
def _(
    node: ast.Name,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    local_path: LocalObjectPath,  # noqa: ARG001
    module_path: ModulePath,  # noqa: ARG001
) -> Namespace:
    return _resolve_object_name(node.id, namespace, *parent_namespaces)


@resolve_node.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    return (
        result
        if (result := lookup_node(node.value, namespace, *parent_namespaces))
        is not None
        else Namespace(ObjectKind.UNKNOWN, module_path, local_path)
    )


@resolve_node.register(ast.Set)
@resolve_node.register(ast.SetComp)
def _(
    _node: ast.Set | ast.SetComp,
    _namespace: Namespace,
    /,
    *_parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    return Namespace(
        ObjectKind.INSTANCE,
        module_path,
        local_path,
        BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
            LocalObjectPath.from_object_name(set.__qualname__)
        ),
    )


@resolve_node.register(ast.Tuple)
def _(
    _node: ast.Tuple,
    _namespace: Namespace,
    /,
    *_parent_namespaces: Namespace,
    local_path: LocalObjectPath,
    module_path: ModulePath,
) -> Namespace:
    return Namespace(
        ObjectKind.INSTANCE,
        module_path,
        local_path,
        BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
            LocalObjectPath.from_object_name(tuple.__qualname__)
        ),
    )


def _resolve_object_name(
    name: str, namespace: Namespace, /, *parent_namespaces: Namespace
) -> Namespace:
    try:
        return namespace.get_namespace_by_name(name)
    except KeyError:
        for parent_namespace in parent_namespaces:
            try:
                return parent_namespace.get_namespace_by_name(name)
            except KeyError:
                continue
        raise NameError(name) from None
