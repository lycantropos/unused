from __future__ import annotations

import ast
import builtins
import ctypes
import graphlib
import inspect
import sys
import types
from collections import deque
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from functools import partial, singledispatch
from pathlib import Path
from typing import Any, Final, TypeAlias, TypeVar

from typing_extensions import TypeIs

from .enums import ObjectKind, ScopeKind
from .missing import MISSING
from .object_ import (
    CALLABLE_OBJECT_CLASSES,
    Class,
    Method,
    Module,
    MutableObject,
    Object,
    PlainObject,
    Routine,
    UnknownObject,
)
from .object_path import (
    BUILTINS_MODULE_PATH,
    BUILTINS_OBJECT_LOCAL_OBJECT_PATH,
    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    TYPES_METHOD_TYPE_LOCAL_OBJECT_PATH,
    TYPES_MODULE_PATH,
)
from .safety import is_safe
from .scope import Scope
from .utils import AnyFunctionDefinitionAstNode, ensure_type

ObjectPath: TypeAlias = tuple[ModulePath, LocalObjectPath]


class DefinitionAstNodeParser(ast.NodeVisitor):
    def __init__(
        self,
        definition_nodes: MutableMapping[
            LocalObjectPath, list[AnyFunctionDefinitionAstNode | ast.ClassDef]
        ],
        /,
        *,
        local_path: LocalObjectPath,
    ) -> None:
        self._definition_nodes, self._local_path = (
            definition_nodes,
            local_path,
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_any_function_node(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_any_function_node(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_local_path = self._local_path.join(node.name)
        self._definition_nodes.setdefault(class_local_path, []).append(node)
        body_visitor = DefinitionAstNodeParser(
            self._definition_nodes, local_path=class_local_path
        )
        for body_node in node.body:
            body_visitor.visit(body_node)

    def _visit_any_function_node(
        self, node: AnyFunctionDefinitionAstNode, /
    ) -> None:
        function_local_path = self._local_path.join(node.name)
        self._definition_nodes.setdefault(function_local_path, []).append(node)
        body_visitor = DefinitionAstNodeParser(
            self._definition_nodes, local_path=function_local_path
        )
        for body_node in node.body:
            body_visitor.visit(body_node)


def _invert_mapping(
    value: Mapping[_KT, _VT], /
) -> Mapping[_VT, Sequence[_KT]]:
    result: dict[_VT, list[_KT]] = {}
    for item_key, item_value in value.items():
        result.setdefault(item_value, []).append(item_key)
    return result


MODULE_NAMES: Final[Mapping[types.ModuleType, Sequence[str]]] = (
    _invert_mapping(sys.modules)
)


@singledispatch
def _locate_values(
    value: Any,  # noqa: ARG001
    value_path: ObjectPath | None,  # noqa: ARG001
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],  # noqa: ARG001
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],  # noqa: ARG001
    located_values: dict[ObjectPath, _NamespaceValue],  # noqa: ARG001
) -> None:
    return


@_locate_values.register(type)
def _(
    value: type[Any],
    value_path: ObjectPath | None,
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],
    located_values: dict[ObjectPath, _NamespaceValue],
) -> None:
    if value_path is None:
        value_module_path = None
    else:
        value_module_path, _ = value_path
    if isinstance(module_name := getattr(value, '__module__', None), str) and (
        (module := _checked_find_module_by_name(module_name)) is not None
    ):
        module_path = ModulePath.from_module_name(module_name)
        _register_module_path(
            mentioned_module_paths, module, (module_path, LocalObjectPath())
        )
        if value_module_path is None or module_path != value_module_path:
            _locate_values(
                module,
                (module_path, LocalObjectPath()),
                mentioned_module_paths,
                all_value_paths=all_value_paths,
                located_values=located_values,
            )
    if value_path is None:
        return
    value_module_path, value_local_path = value_path
    try:
        value_paths = all_value_paths[value]
    except KeyError:
        _set_absent_key(located_values, value_path, value)
        _set_absent_key(all_value_paths, value, [value_path])
    else:
        if value_path in value_paths:
            return
        if any(
            (
                module_path == value_module_path
                and value_local_path.starts_with(local_path)
            )
            for module_path, local_path in value_paths
        ):
            value_paths.append(value_path)
            return
        _set_absent_key(located_values, value_path, value)
        value_paths.append(value_path)
    for field_name in dir(value):
        try:
            field_value = getattr(value, field_name)
        except AttributeError:
            continue
        _locate_values(
            field_value,
            (value_module_path, value_local_path.join(field_name)),
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )
    for base in value.__bases__:
        _locate_values(
            base,
            None,
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )


@_locate_values.register(types.ModuleType)
def _(
    value: types.ModuleType,
    value_path: ObjectPath | None,
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],
    located_values: dict[ObjectPath, _NamespaceValue],
) -> None:
    if value_path is None:
        value_module_path = None
    else:
        value_module_path, _ = value_path
    self_module_name = value.__name__
    if (
        module := _checked_find_module_by_name(self_module_name)
    ) is not None and module is not value:
        _locate_values(
            module,
            (ModulePath.from_module_name(self_module_name), LocalObjectPath()),
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )
        for module_name in MODULE_NAMES.get(value, []):
            module_path = ModulePath.from_module_name(module_name)
            if value_module_path is None or module_path != value_module_path:
                _locate_values(
                    value,
                    (module_path, LocalObjectPath()),
                    mentioned_module_paths,
                    all_value_paths=all_value_paths,
                    located_values=located_values,
                )
    elif value in sys.modules.values():
        self_module_path = ModulePath.from_module_name(self_module_name)
        _register_module_path(
            mentioned_module_paths,
            value,
            (self_module_path, LocalObjectPath()),
        )
        if value_module_path is None or self_module_path != value_module_path:
            _locate_values(
                module,
                (self_module_path, LocalObjectPath()),
                mentioned_module_paths,
                all_value_paths=all_value_paths,
                located_values=located_values,
            )
    for module_name in MODULE_NAMES.get(value, []):
        _register_module_path(
            mentioned_module_paths,
            value,
            (ModulePath.from_module_name(module_name), LocalObjectPath()),
        )
    if value_path is None:
        return
    value_module_path, value_local_path = value_path
    _register_module_path(mentioned_module_paths, value, value_path)
    try:
        value_paths = all_value_paths[value]
    except KeyError:
        _set_absent_key(located_values, value_path, value)
        _set_absent_key(all_value_paths, value, [value_path])
    else:
        if value_path in value_paths:
            return
        if any(
            (
                module_path == value_module_path
                and value_local_path.starts_with(local_path)
            )
            for module_path, local_path in value_paths
        ):
            value_paths.append(value_path)
            return
        _set_absent_key(located_values, value_path, value)
        value_paths.append(value_path)
    if value in sys.modules.values() and len(value_local_path.components) > 0:
        return
    for field_name in dir(value):
        try:
            field_value = getattr(value, field_name)
        except Exception:
            continue
        _locate_values(
            field_value,
            (value_module_path, value_local_path.join(field_name)),
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )


@_locate_values.register(types.BuiltinFunctionType)
@_locate_values.register(types.BuiltinMethodType)
def _(
    value: types.BuiltinFunctionType | types.BuiltinMethodType,
    value_path: ObjectPath | None,  # noqa: ARG001
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],
    located_values: dict[ObjectPath, _NamespaceValue],
) -> None:
    instance = value.__self__
    if instance in all_value_paths:
        return
    _locate_values(
        instance,
        None,
        mentioned_module_paths,
        all_value_paths=all_value_paths,
        located_values=located_values,
    )


@_locate_values.register(types.GetSetDescriptorType)
@_locate_values.register(types.MemberDescriptorType)
@_locate_values.register(types.MethodDescriptorType)
@_locate_values.register(types.WrapperDescriptorType)
def _(
    value: _AnyDescriptorType,
    value_path: ObjectPath | None,  # noqa: ARG001
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],
    located_values: dict[ObjectPath, _NamespaceValue],
) -> None:
    object_class = value.__objclass__
    if object_class in all_value_paths:
        return
    _locate_values(
        object_class,
        None,
        mentioned_module_paths,
        all_value_paths=all_value_paths,
        located_values=located_values,
    )


@_locate_values.register(types.ClassMethodDescriptorType)
def _(
    value: types.ClassMethodDescriptorType,
    value_path: ObjectPath | None,  # noqa: ARG001
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],
    located_values: dict[ObjectPath, _NamespaceValue],
) -> None:
    if (wrapped := getattr(value, '__wrapped__', None)) is not None:
        if wrapped in all_value_paths:
            return
        _locate_values(
            wrapped,
            None,
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )
    else:
        object_class = value.__objclass__
        if object_class in all_value_paths:
            return
        _locate_values(
            object_class,
            None,
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )


@_locate_values.register(types.FunctionType)
def _(
    value: types.FunctionType,
    value_path: ObjectPath,  # noqa: ARG001
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    /,
    *,
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]],
    located_values: dict[ObjectPath, _NamespaceValue],
) -> None:
    if (
        isinstance(module_name := value.__module__, str)
        and (module := _checked_find_module_by_name(module_name)) is not None
    ):
        module_object_path = (
            ModulePath.from_module_name(module_name),
            LocalObjectPath(),
        )
        _register_module_path(
            mentioned_module_paths, module, module_object_path
        )
        _locate_values(
            module,
            module_object_path,
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )


def _register_module_path(
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]],
    module: types.ModuleType,
    module_object_path: ObjectPath,
    /,
) -> None:
    assert isinstance(module, types.ModuleType), module
    assert isinstance(module_object_path, tuple), module_object_path
    assert len(module_object_path) == 2, module_object_path
    mentioned_module_paths.setdefault(module, {}).setdefault(
        module_object_path, None
    )


def _checked_find_module_by_name(
    module_name: str, /
) -> types.ModuleType | None:
    try:
        return sys.modules[module_name]
    except KeyError:
        pass
    return next(
        (
            candidate
            for candidate in sys.modules.values()
            if candidate.__name__ == module_name
        ),
        None,
    )


def _parse_modules(
    *modules: types.ModuleType,
) -> MutableMapping[ModulePath, MutableObject]:
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]] = {}
    all_value_paths: dict[_NamespaceValue, list[ObjectPath]] = {}
    located_values: dict[ObjectPath, _NamespaceValue] = {}
    for module in modules:
        module_name = module.__name__
        assert module_name in MODULE_NAMES[module], (
            module_name,
            MODULE_NAMES[module],
        )
        _locate_values(
            module,
            (ModulePath.from_module_name(module_name), LocalObjectPath()),
            mentioned_module_paths,
            all_value_paths=all_value_paths,
            located_values=located_values,
        )
    module_origin_paths: dict[types.ModuleType, ObjectPath] = {}
    references: dict[ObjectPath, ObjectPath] = {}
    _locate_module_origins(
        [
            (module, list(module_paths))
            for module, module_paths in mentioned_module_paths.items()
        ],
        module_origin_paths=module_origin_paths,
        references=references,
    )
    module_definition_nodes: dict[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ] = {}
    value_origin_paths: dict[Any, ObjectPath] = module_origin_paths
    topologically_sorted_values = (
        _to_topologically_sorted_sequence_resolving_cycles_by_deletion(
            {
                value: {
                    parent_value
                    for path in value_paths
                    for parent_path in _to_parent_paths(path)
                    if (
                        (
                            (parent_value := located_values.get(parent_path))
                            is not None
                        )
                        and not inspect.ismodule(parent_value)
                    )
                }
                for value, value_paths in all_value_paths.items()
                if not inspect.ismodule(value)
            }
        )
    )
    for value in topologically_sorted_values:
        value_paths = all_value_paths[value]
        assert not inspect.ismodule(value)
        try:
            (origin_path,) = value_paths
        except ValueError:
            pass
        else:
            _set_absent_key(value_origin_paths, value, origin_path)
            continue
        assert len(value_paths) > 0
        value_paths = [
            (module_path, local_path)
            for module_path, local_path in value_paths
            if (
                value_origin_paths[
                    located_values[module_path, LocalObjectPath()]
                ]
                == (module_path, LocalObjectPath())
            )
        ]
        if inspect.isclass(value):
            alt_value_paths = [
                object_path
                for object_path in value_paths
                if all(
                    (
                        value_origin_paths.get(located_values[parent_path])
                        in (None, parent_path)
                    )
                    for parent_path in _to_parent_paths(object_path)
                )
            ]
            if value is ctypes.c_long:
                _ = 0
            origin_path = _to_cls_origin_path(
                value,
                alt_value_paths,
                located_values=located_values,
                module_definition_nodes=module_definition_nodes,
            )
            _set_absent_key(value_origin_paths, value, origin_path)
            for candidate_path in value_paths:
                if candidate_path == origin_path:
                    continue
                _add_reference(references, candidate_path, origin_path)
    assert len(value_origin_paths) == len(all_value_paths), [
        value for value in all_value_paths if value not in value_origin_paths
    ]
    for value_origin_path in value_origin_paths.values():
        origin_module_path, origin_local_path = value_origin_path
        if len(origin_local_path.components) == 0:
            continue
        parent_origin_path = (origin_module_path, origin_local_path.parent)
        assert (
            value_origin_paths[located_values[parent_origin_path]]
            == parent_origin_path
        ), (
            value_origin_paths[located_values[parent_origin_path]],
            parent_origin_path,
        )
    dependencies: dict[ObjectPath, set[ObjectPath]] = {}
    base_cls_paths: dict[ObjectPath, list[ObjectPath]] = {}
    instance_cls_paths: dict[ObjectPath, ObjectPath] = {}
    metacls_paths: dict[ObjectPath, ObjectPath] = {}
    method_component_paths: dict[
        ObjectPath, tuple[ObjectPath, ObjectPath]
    ] = {}
    for value, value_path in value_origin_paths.copy().items():
        if inspect.ismodule(value):
            dependencies[value_origin_paths[value]] = set()
            continue
        value_module_path, value_local_path = value_path
        value_dependencies = dependencies.setdefault(value_path, set())
        assert len(value_local_path.components) > 0, value_path
        value_parent_path = (value_module_path, value_local_path.parent)
        assert (
            parent_value := located_values.get(value_parent_path)  # noqa: RUF018
        ) is None or value_origin_paths[parent_value] == value_parent_path
        value_dependencies.add(value_parent_path)
        if inspect.isclass(value):
            base_cls_paths[value_path] = origin_base_cls_paths = []
            for base_cls in value.__bases__:
                try:
                    origin_base_cls_path = value_origin_paths[base_cls]
                except KeyError:
                    origin_base_cls_path = (
                        ModulePath.from_module_name(base_cls.__module__),
                        LocalObjectPath.from_object_name(
                            base_cls.__qualname__
                        ),
                    )
                else:
                    value_dependencies.add(origin_base_cls_path)
                origin_base_cls_paths.append(origin_base_cls_path)
            if (
                all(
                    base_cls_path is None
                    for base_cls_path in origin_base_cls_paths
                )
                and value is not builtins.object
            ):
                value_dependencies.add(value_origin_paths[builtins.object])
            if not _is_metaclass(value) and value is not builtins.object:
                metacls = type(value)
                try:
                    origin_metacls_path = value_origin_paths[metacls]
                except KeyError:
                    origin_metacls_path = (
                        ModulePath.from_module_name(metacls.__module__),
                        LocalObjectPath.from_object_name(metacls.__qualname__),
                    )
                else:
                    value_dependencies.add(origin_metacls_path)
                metacls_paths[value_path] = origin_metacls_path
        else:
            continue
            try:
                origin_cls_path = value_origin_paths[type(value)]
            except KeyError:
                pass
            else:
                instance_cls_paths[value_path] = origin_cls_path
                value_dependencies.add(origin_cls_path)
            if isinstance(value, types.MethodType):
                method_instance = value.__self__
                if inspect.isclass(method_instance):
                    if (
                        located_values[
                            value_module_path, value_local_path.parent
                        ]
                        is method_instance
                    ) and inspect.isroutine(value_callable := value.__func__):
                        value_origin_paths[value_callable] = (
                            value_origin_paths.pop(value)
                        )
                        located_values[value_path] = value_callable
                        continue
                    try:
                        method_instance_path = value_origin_paths[
                            method_instance
                        ]
                    except KeyError:
                        method_instance_path = (
                            ModulePath.from_module_name(
                                method_instance.__module__
                            ),
                            LocalObjectPath.from_object_name(
                                method_instance.__qualname__
                            ),
                        )
                    else:
                        value_dependencies.add(method_instance_path)
                else:
                    assert not inspect.ismodule(method_instance)
                    method_instance_cls = type(method_instance)
                    try:
                        method_instance_cls_path = value_origin_paths[
                            method_instance_cls
                        ]
                    except KeyError:
                        try:
                            method_instance_cls_path = (
                                ModulePath.from_module_name(
                                    method_instance_cls.__module__
                                ),
                                LocalObjectPath.from_object_name(
                                    method_instance_cls.__qualname__
                                ),
                            )
                        except ValueError:
                            method_instance_cls_path = (
                                value_module_path,
                                value_local_path.join('__self__', '__class__'),
                            )
                    else:
                        value_dependencies.add(method_instance_cls_path)
                    method_instance_path = (
                        value_module_path,
                        value_local_path.join('__self__'),
                    )
                    _set_absent_key(
                        instance_cls_paths,
                        method_instance_path,
                        method_instance_cls_path,
                    )
                    _set_absent_key(
                        located_values, method_instance_path, method_instance
                    )
                method_callable = value.__func__
                try:
                    method_callable_path = value_origin_paths[method_callable]
                except KeyError:
                    try:
                        method_callable_path = (
                            ModulePath.from_module_name(
                                method_callable.__module__
                            ),
                            LocalObjectPath.from_object_name(
                                method_callable.__qualname__
                            ),
                        )
                    except ValueError:
                        method_callable_path = (
                            value_module_path,
                            value_local_path.join('__func__'),
                        )
                else:
                    value_dependencies.add(method_callable_path)
                method_component_paths[value_path] = (
                    method_callable_path,
                    method_instance_path,
                )
            elif isinstance(value, types.BuiltinMethodType):
                method_instance = value.__self__
                if inspect.isclass(method_instance) or inspect.ismodule(
                    method_instance
                ):
                    try:
                        method_instance_path = value_origin_paths[
                            method_instance
                        ]
                    except KeyError:
                        pass
                    else:
                        value_dependencies.add(method_instance_path)
                else:
                    try:
                        method_instance_cls_path = value_origin_paths[
                            type(method_instance)
                        ]
                    except KeyError:
                        pass
                    else:
                        value_dependencies.add(method_instance_cls_path)
            elif inspect.isfunction(value):
                value_dependencies.add(value_origin_paths[builtins.dict])
                value_dependencies.add(value_origin_paths[builtins.tuple])
    topologically_sorted_value_paths = [
        *graphlib.TopologicalSorter(dependencies).static_order()
    ]
    result: dict[ModulePath, MutableObject] = {}
    for value_path in topologically_sorted_value_paths:
        value_module_path, value_local_path = value_path
        if len(value_local_path.components) == 0:
            result[value_module_path] = Module(
                Scope(
                    ScopeKind.STATIC_MODULE,
                    value_module_path,
                    value_local_path,
                )
            )
            continue
        value = located_values[value_path]
        origin_object: Object
        if inspect.isclass(value):
            origin_base_cls_paths = base_cls_paths[value_path]
            base_cls_objects = [
                _checked_get_object_by_path(
                    result, base_cls_module_path, base_cls_local_path
                )
                for base_cls_module_path, base_cls_local_path in (
                    origin_base_cls_paths
                )
            ]
            has_unknown_base = any(
                base_cls is None for base_cls in base_cls_objects
            )
            origin_object = Class(
                Scope(
                    (
                        ScopeKind.METACLASS
                        if _is_metaclass(value)
                        else ScopeKind.CLASS
                    ),
                    value_module_path,
                    value_local_path,
                ),
                *[
                    base_cls
                    for base_cls in base_cls_objects
                    if base_cls is not None
                ],
                *(
                    (
                        result[BUILTINS_MODULE_PATH].get_nested_attribute(
                            BUILTINS_OBJECT_LOCAL_OBJECT_PATH
                        ),
                    )
                    if value is not builtins.object
                    else ()
                ),
                *(
                    (
                        Class(
                            Scope(
                                ScopeKind.UNKNOWN_CLASS,
                                value_module_path,
                                value_local_path.join(
                                    f'__base_{len(origin_base_cls_paths) - 1}__'  # noqa: E501
                                ),
                            ),
                            metaclass=MISSING,
                        ),
                    )
                    if has_unknown_base
                    else ()
                ),
                metaclass=(
                    _path_to_object_or_unknown(
                        result, metacls_paths[value_path]
                    )
                    if (
                        not _is_metaclass(value)
                        and value is not builtins.object
                    )
                    else MISSING
                ),
            )
        elif (
            origin_method_component_paths := method_component_paths.get(
                value_path
            )
        ) is not None:
            method_callable_path, method_instance_path = (
                origin_method_component_paths
            )
            origin_object = Method(
                ensure_type(
                    _path_to_object_or_unknown(result, method_callable_path),
                    CALLABLE_OBJECT_CLASSES,
                ),
                _path_to_object_or_unknown(result, method_instance_path),
            )
        elif inspect.isroutine(value):
            value_ast_node: AnyFunctionDefinitionAstNode | None = None
            try:
                value_module_function_definition_nodes = (
                    module_definition_nodes[value_module_path]
                )
            except KeyError:
                pass
            else:
                value_ast_nodes = value_module_function_definition_nodes.get(
                    value_local_path, []
                )
                if len(value_ast_nodes) > 0:
                    (candidate_ast_node,) = value_ast_nodes
                    assert isinstance(
                        candidate_ast_node, AnyFunctionDefinitionAstNode
                    ), candidate_ast_node
                    value_ast_node = candidate_ast_node
                if (
                    value_ast_node is None
                    and value_module_function_definition_nodes
                ):
                    _ = 0
            value_base_cls: Class | UnknownObject
            try:
                instance_cls_path = instance_cls_paths[value_path]
            except KeyError:
                value_base_cls = UnknownObject(
                    value_module_path, value_local_path.join('__base__')
                )
            else:
                value_base_cls = ensure_type(
                    _path_to_object(result, instance_cls_path),
                    (Class, UnknownObject),
                )
            origin_object = Routine(
                value_module_path,
                value_local_path,
                value_base_cls,
                ast_node=value_ast_node,
            )
            if inspect.isfunction(value):
                origin_object.set_attribute(
                    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
                    PlainObject(
                        ObjectKind.INSTANCE,
                        value_module_path,
                        value_local_path.join(
                            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME
                        ),
                        _path_to_object(
                            result, value_origin_paths[builtins.tuple]
                        ),
                    ),
                )
                origin_object.set_value(
                    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
                    value.__defaults__ or (),
                )
                origin_object.set_attribute(
                    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
                    PlainObject(
                        ObjectKind.INSTANCE,
                        value_module_path,
                        value_local_path.join(
                            FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
                        ),
                        _path_to_object(
                            result, value_origin_paths[builtins.dict]
                        ),
                    ),
                )
                origin_object.set_value(
                    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
                    value.__kwdefaults__ or {},
                )
        elif (
            maybe_instance_cls_path := instance_cls_paths.get(value_path)
        ) is not None:
            origin_object = PlainObject(
                ObjectKind.INSTANCE,
                value_module_path,
                value_local_path,
                _path_to_object(result, maybe_instance_cls_path),
            )
        else:
            origin_object = UnknownObject(value_module_path, value_local_path)
        origin_module_object = result[value_module_path]
        origin_module_object.set_nested_attribute(
            value_local_path, origin_object
        )
        if is_safe(value):
            origin_module_object.set_nested_value(value_local_path, value)
    topologically_sorted_references = [
        (candidate_path, reference_path)
        for candidate_path in graphlib.TopologicalSorter(
            {
                referent_path: [reference_path]
                for referent_path, reference_path in references.items()
            }
        ).static_order()
        if (reference_path := references.get(candidate_path)) is not None
    ]
    for (
        referent_module_path,
        referent_local_path,
    ), reference_path in topologically_sorted_references:
        referent_object = _path_to_object(result, reference_path)
        if len(referent_local_path.components) == 0:
            assert isinstance(referent_object, MutableObject)
            result[referent_module_path] = referent_object
        else:
            result[referent_module_path].set_nested_attribute(
                referent_local_path, referent_object
            )
    return result


def _locate_module_origins(
    module_with_name_and_local_paths: Iterable[
        tuple[types.ModuleType, Sequence[ObjectPath]]
    ],
    /,
    *,
    module_origin_paths: MutableMapping[types.ModuleType, ObjectPath],
    references: MutableMapping[ObjectPath, ObjectPath],
) -> None:
    for module, module_object_paths in module_with_name_and_local_paths:
        assert inspect.ismodule(module)
        origin_path = _to_module_origin_path(module, module_object_paths)
        _set_absent_key(module_origin_paths, module, origin_path)
        for candidate_path in module_object_paths:
            if candidate_path == origin_path:
                continue
            _add_reference(references, candidate_path, origin_path)


def _to_parent_paths(
    object_path: ObjectPath, /
) -> Sequence[tuple[ModulePath, LocalObjectPath]]:
    module_path, local_path = object_path
    return [
        (module_path, LocalObjectPath(*local_path.components[:stop_index]))
        for stop_index in reversed(range(len(local_path.components)))
    ]


def _to_topologically_sorted_sequence_resolving_cycles_by_deletion(
    mapping: Mapping[_KT, set[_KT]], /
) -> Sequence[_KT]:
    while True:
        try:
            result = [*graphlib.TopologicalSorter(mapping).static_order()]
        except graphlib.CycleError as error:
            _, cycle = error.args
            assert len(cycle) > 1
            mapping[cycle[0]].remove(cycle[1])
        else:
            break
    return result


def _to_module_origin_path(
    module: types.ModuleType, module_object_paths: Sequence[ObjectPath], /
) -> ObjectPath:
    try:
        (origin_path,) = module_object_paths
    except ValueError:
        assert len(module_object_paths) > 0
        candidate_paths: Sequence[ObjectPath] = module_object_paths
        plain_module_paths = [
            (module_path, local_path)
            for module_path, local_path in module_object_paths
            if len(local_path.components) == 0
        ]
        if len(plain_module_paths) > 0:
            candidate_paths = plain_module_paths
        module_name_candidates = [
            (module_path, local_path)
            for module_path, local_path in candidate_paths
            if (
                '.'.join([*module_path.components, *local_path.components])
                == module.__name__
            )
        ]
        try:
            (origin_path,) = module_name_candidates
        except ValueError:
            assert len(module_name_candidates) == 0
            origin_path = candidate_paths[0]
    return origin_path


def _to_cls_origin_path(
    cls: type[Any],
    cls_paths: Sequence[ObjectPath],
    /,
    *,
    located_values: Mapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
) -> ObjectPath:
    assert len(cls_paths) > 0, cls
    # first we try to narrow down classes
    # based on the presence of class definition in the module file AST ...
    ast_candidate_paths = _to_cls_origin_candidate_paths_based_on_module_ast(
        cls_paths, located_values, module_definition_nodes
    )
    if len(ast_candidate_paths) == 1:
        (result,) = ast_candidate_paths
        return result
    # ... then by module names mentioned in the members ...
    candidate_paths = ast_candidate_paths or cls_paths
    candidate_module_paths = [
        module_path for module_path, _ in candidate_paths
    ]
    cls_member_module_paths: set[ModulePath] = set()
    try:
        self_local_path = LocalObjectPath.from_object_name(cls.__qualname__)
    except ValueError:
        self_local_path = None
    for cls_member in vars(cls).values():
        if inspect.isclass(cls_member):
            if (
                self_local_path is not None
                and not LocalObjectPath.from_object_name(
                    cls_member.__qualname__
                ).starts_with(self_local_path)
            ):
                continue
            nested_cls_module_name = cls_member.__module__
            try:
                nested_cls_module_path = ModulePath.from_module_name(
                    nested_cls_module_name
                )
            except ValueError:
                continue
            if nested_cls_module_path in candidate_module_paths:
                cls_member_module_paths.add(nested_cls_module_path)
        if not inspect.isfunction(cls_member):
            cls_member = getattr(cls_member, '__func__', cls_member)
        if not inspect.isfunction(cls_member):
            continue
        if (
            self_local_path is not None
            and not LocalObjectPath.from_object_name(
                cls_member.__qualname__
            ).starts_with(self_local_path)
        ):
            continue
        function_module_name = getattr(cls_member, '__module__', None)
        if not isinstance(function_module_name, str):
            continue
        try:
            function_module_path = ModulePath.from_module_name(
                function_module_name
            )
        except ValueError:
            continue
        if function_module_path in candidate_module_paths:
            cls_member_module_paths.add(function_module_path)
    if len(cls_member_module_paths) > 0:
        candidate_paths = [
            (module_path, local_path)
            for module_path, local_path in candidate_paths
            if module_path in cls_member_module_paths
        ]
    self_module_path = ModulePath.from_module_name(cls.__module__)
    if (
        len(
            self_module_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_module_path == self_module_path
            ]
        )
        > 0
    ):
        candidate_paths = self_module_path_candidate_paths
    if self_local_path is not None and (
        len(
            self_local_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_local_path == self_local_path
            ]
        )
        > 0
    ):
        candidate_paths = self_local_path_candidate_paths
    # ... if there are still more than one candidate left
    # -- we just pick the first one
    return candidate_paths[0]


def _to_cls_origin_candidate_paths_based_on_module_ast(
    value_paths: Sequence[ObjectPath],
    all_values: Mapping[ObjectPath, Any],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
) -> Sequence[ObjectPath]:
    result: list[ObjectPath] = []
    for value_path in value_paths:
        value_module_path, value_local_path = value_path
        value_module = all_values[value_module_path, LocalObjectPath()]
        if (
            file_path_string := getattr(value_module, '__file__', None)
        ) is not None:
            try:
                value_module_definition_nodes = module_definition_nodes[
                    value_module_path
                ]
            except KeyError:
                try:
                    module_source = Path(file_path_string).read_text(
                        encoding='utf-8'
                    )
                except (OSError, UnicodeDecodeError):
                    value_module_definition_nodes = {}
                else:
                    parsed_value_module_definition_nodes: MutableMapping[
                        LocalObjectPath,
                        list[AnyFunctionDefinitionAstNode | ast.ClassDef],
                    ] = {}
                    DefinitionAstNodeParser(
                        parsed_value_module_definition_nodes,
                        local_path=LocalObjectPath(),
                    ).visit(ast.parse(module_source))
                    value_module_definition_nodes = (
                        parsed_value_module_definition_nodes
                    )
                    _set_absent_key(
                        module_definition_nodes,
                        value_module_path,
                        value_module_definition_nodes,
                    )
            value_nodes = [
                node
                for node in value_module_definition_nodes.get(
                    value_local_path, []
                )
                if isinstance(node, ast.ClassDef)
            ]
            if len(value_nodes) > 0:
                result.append(value_path)
    return result


_NamespaceValue: TypeAlias = types.ModuleType | type[Any]


def _locate_objects(
    module_with_paths: Iterable[tuple[types.ModuleType, ObjectPath]],
    /,
    *,
    all_value_paths: MutableMapping[Any, list[ObjectPath]],
    located_values: MutableMapping[ObjectPath, Any],
    all_value_dependencies: MutableMapping[ObjectPath, set[ObjectPath]],
) -> None:
    queue: deque[tuple[types.ModuleType | type[Any], ObjectPath]] = deque(
        module_with_paths
    )
    while queue:
        value, value_path = queue.popleft()
        value_dependencies = all_value_dependencies.setdefault(
            value_path, set()
        )
        value_module_path, value_local_path = value_path
        if (value_paths := all_value_paths.get(value)) is not None:
            value_paths.append(value_path)
            _set_absent_key(located_values, value_path, value)
            continue
        _set_absent_key(located_values, value_path, value)
        _set_absent_key(all_value_paths, value, [value_path])
        field_names = dir(value)
        try:
            value_dict = vars(value)
        except Exception:
            pass
        else:
            field_names.sort(
                key=partial(
                    _call_or_else,
                    tuple(value_dict).index,
                    exception_classes=(ValueError,),
                    default=len(field_names),
                )
            )
        for field_name in field_names:
            try:
                field_value = getattr(value, field_name)
            except Exception:
                continue
            field_path = (value_module_path, value_local_path.join(field_name))
            if inspect.isclass(field_value) or inspect.ismodule(field_value):
                value_dependencies.add(field_path)
                queue.append((field_value, field_path))
            else:
                _set_absent_key(located_values, field_path, field_value)


def _checked_get_object_by_path(
    modules: Mapping[ModulePath, MutableObject],
    module_path: ModulePath,
    local_path: LocalObjectPath,
    /,
) -> Object | None:
    try:
        return modules[module_path].get_nested_attribute(local_path)
    except KeyError:
        return None


def _is_metaclass(value: Any, /) -> TypeIs[type[Any]]:
    return issubclass(value, type)


_ValueT = TypeVar('_ValueT')
_ResultT = TypeVar('_ResultT')


def _call_or_else(
    callable_: Callable[[_ValueT], _ResultT],
    value: _ValueT,
    /,
    *,
    exception_classes: tuple[type[Exception], ...],
    default: _ResultT,
) -> _ResultT:
    try:
        return callable_(value)
    except exception_classes:
        return default


_AnyDescriptorType: TypeAlias = (
    types.GetSetDescriptorType
    | types.MemberDescriptorType
    | types.MethodDescriptorType
    | types.WrapperDescriptorType
)


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


def _set_absent_key(
    mapping: MutableMapping[_KT, _VT], key: _KT, value: _VT, /
) -> None:
    assert key not in mapping, (mapping[key], value)
    mapping[key] = value


def _add_reference(
    references: MutableMapping[ObjectPath, ObjectPath],
    referent_path: ObjectPath,
    reference_path: ObjectPath,
    /,
) -> None:
    assert referent_path != reference_path, referent_path
    _set_absent_key(references, referent_path, reference_path)


def _path_to_object(
    modules: Mapping[ModulePath, MutableObject], object_path: ObjectPath, /
) -> Object:
    module_path, local_path = object_path
    return modules[module_path].get_nested_attribute(local_path)


def _path_to_object_or_unknown(
    modules: Mapping[ModulePath, MutableObject], object_path: ObjectPath, /
) -> Object:
    module_path, local_path = object_path
    try:
        return modules[module_path].get_nested_attribute(local_path)
    except KeyError:
        return UnknownObject(module_path, local_path)


MODULES: Final[MutableMapping[ModulePath, MutableObject]] = _parse_modules(
    builtins, sys, types
)
BUILTINS_MODULE: Final[Module] = ensure_type(
    MODULES[BUILTINS_MODULE_PATH], Module
)
TYPES_MODULE: Final[Object] = ensure_type(MODULES[TYPES_MODULE_PATH], Module)
Method.BASE_CLS = ensure_type(
    TYPES_MODULE.get_nested_attribute(TYPES_METHOD_TYPE_LOCAL_OBJECT_PATH),
    Class,
)


def _setup_builtin_classes() -> None:
    for cls in [
        builtins.bool,
        builtins.bytearray,
        builtins.bytes,
        builtins.dict,
        builtins.float,
        builtins.frozenset,
        builtins.int,
        builtins.list,
        builtins.set,
        builtins.str,
        builtins.tuple,
        builtins.type,
    ]:
        assert inspect.isclass(cls), cls
        BUILTINS_MODULE.set_nested_value(
            LocalObjectPath.from_object_name(cls.__qualname__), cls
        )


_setup_builtin_classes()
