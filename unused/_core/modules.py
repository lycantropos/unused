from __future__ import annotations

import builtins
import graphlib
import inspect
import sys
import types
from collections.abc import Collection, Mapping, MutableMapping
from typing import Any, Final, Literal, TypeAlias

from .dependency_node import DependencyNode
from .missing import MISSING
from .object_ import (
    CLASS_OBJECT_KINDS,
    Class,
    Module,
    Object,
    ObjectKind,
    PlainObject,
    Scope,
    ScopeKind,
)
from .object_path import (
    BUILTINS_DICT_LOCAL_OBJECT_PATH,
    BUILTINS_FROZENSET_LOCAL_OBJECT_PATH,
    BUILTINS_LIST_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    BUILTINS_SET_LOCAL_OBJECT_PATH,
    BUILTINS_TUPLE_LOCAL_OBJECT_PATH,
    LocalObjectPath,
    ModulePath,
    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH,
    TYPES_MODULE_PATH,
)
from .safety import is_safe
from .utils import ensure_type

ObjectPath: TypeAlias = tuple[ModulePath, LocalObjectPath]

MODULES: Final[dict[ModulePath, Object]] = {}


def parse_modules(*modules: types.ModuleType) -> None:
    dependency_graph: dict[DependencyNode, set[DependencyNode]] = {}
    instance_class_paths: dict[ObjectPath, ObjectPath] = {}
    base_class_paths: dict[ObjectPath, set[ObjectPath]] = {}
    metaclass_paths: dict[ObjectPath, ObjectPath] = {}
    for module in modules:
        assert inspect.ismodule(module)
        module_path = ModulePath.from_module_name(module.__name__)
        module_dependency_node = DependencyNode(
            ObjectKind.STATIC_MODULE,
            module_path,
            LocalObjectPath(),
            dependant_local_path=LocalObjectPath(),
            dependant_module_path=module_path,
            value=MISSING,
        )
        _collect_dependencies(
            dependency_graph,
            instance_class_paths,
            base_class_paths,
            metaclass_paths,
            module,
            module_dependency_node,
        )
    topologically_sorted_dependency_nodes = [
        *graphlib.TopologicalSorter(dependency_graph).static_order()
    ]
    for dependency_node in topologically_sorted_dependency_nodes:
        if len(dependency_node.dependant_local_path.components) == 0:
            continue
        for (
            submodule_path
        ) in dependency_node.dependant_module_path.submodule_paths():
            MODULES.setdefault(
                submodule_path,
                Module(
                    Scope(
                        ScopeKind.STATIC_MODULE,
                        submodule_path,
                        LocalObjectPath(),
                    )
                ),
            )
        dependant_module_object = MODULES[
            dependency_node.dependant_module_path
        ]
        if (
            dependency_node.module_path
            == dependency_node.dependant_module_path
        ) and (
            dependency_node.local_path == dependency_node.dependant_local_path
        ):
            try:
                dependant_module_object.get_nested_attribute(
                    dependency_node.dependant_local_path
                )
            except KeyError:
                dependant_module_object.set_nested_attribute(
                    dependency_node.dependant_local_path,
                    _dependency_node_to_object(
                        dependency_node,
                        base_class_paths=base_class_paths,
                        instance_class_paths=instance_class_paths,
                        metaclass_paths=metaclass_paths,
                    ),
                )
        else:
            dependency_module_object = MODULES[dependency_node.module_path]
            try:
                dependency_object = (
                    dependency_module_object.get_nested_attribute(
                        dependency_node.local_path
                    )
                )
            except KeyError:
                dependency_object = _dependency_node_to_object(
                    dependency_node,
                    base_class_paths=base_class_paths,
                    instance_class_paths=instance_class_paths,
                    metaclass_paths=metaclass_paths,
                )
                dependency_module_object.set_nested_attribute(
                    dependency_node.local_path, dependency_object
                )
            dependant_module_object.set_nested_attribute(
                dependency_node.dependant_local_path, dependency_object
            )
        if (dependency_value := dependency_node.value) is not MISSING:
            dependant_module_object.set_nested_value(
                dependency_node.dependant_local_path, dependency_value
            )
        elif dependency_node.object_kind in (
            ObjectKind.CLASS,
            ObjectKind.METACLASS,
            ObjectKind.STATIC_MODULE,
        ):
            dependant_module_object.set_nested_value(
                dependency_node.dependant_local_path,
                dependant_module_object.get_nested_attribute(
                    dependency_node.dependant_local_path
                ).as_object(),
            )


def _class_object_kind_to_scope_kind(
    object_kind: ObjectKind, /
) -> Literal[ScopeKind.CLASS, ScopeKind.METACLASS, ScopeKind.UNKNOWN_CLASS]:
    if object_kind is ObjectKind.CLASS:
        return ScopeKind.CLASS
    if object_kind is ObjectKind.METACLASS:
        return ScopeKind.METACLASS
    assert object_kind is ObjectKind.UNKNOWN_CLASS, object_kind
    return ScopeKind.UNKNOWN_CLASS


def _collect_dependencies(
    dependency_graph: MutableMapping[DependencyNode, set[DependencyNode]],
    instance_class_paths: MutableMapping[ObjectPath, ObjectPath],
    base_class_paths: MutableMapping[ObjectPath, set[ObjectPath]],
    metaclass_paths: MutableMapping[ObjectPath, ObjectPath],
    value: types.FunctionType | types.ModuleType | type[Any],
    value_dependency_node: DependencyNode,
    /,
) -> None:
    field_names = dir(value)
    if inspect.isclass(value):
        field_names = [
            name
            for name in field_names
            if name not in ('__base__', '__class__')
        ]
    dependency_graph.setdefault(value_dependency_node, set())
    for field_name in field_names:
        try:
            field_value = getattr(value, field_name)
        except AttributeError:
            field_dependency_node = DependencyNode(
                ObjectKind.UNKNOWN,
                value_dependency_node.module_path,
                value_dependency_node.local_path.join(field_name),
                dependant_local_path=(
                    value_dependency_node.dependant_local_path.join(field_name)
                ),
                dependant_module_path=(
                    value_dependency_node.dependant_module_path
                ),
                value=MISSING,
            )
            dependency_graph.setdefault(field_dependency_node, set()).add(
                value_dependency_node
            )
        else:
            if inspect.ismodule(field_value):
                submodule_name = field_value.__name__
                submodule_is_standalone = (
                    sys.modules.get(submodule_name) is field_value
                )
                if submodule_is_standalone:
                    submodule_path = ModulePath.from_module_name(
                        submodule_name
                    )
                    submodule_local_path = LocalObjectPath()
                else:
                    # case of pseudo-submodules like `sys.monitoring`
                    submodule_path = value_dependency_node.module_path
                    submodule_local_path = (
                        value_dependency_node.local_path.join(field_name)
                    )
                submodule_dependency_node = DependencyNode(
                    ObjectKind.STATIC_MODULE,
                    submodule_path,
                    submodule_local_path,
                    dependant_local_path=(
                        value_dependency_node.dependant_local_path.join(
                            field_name
                        )
                    ),
                    dependant_module_path=(
                        value_dependency_node.dependant_module_path
                    ),
                    value=MISSING,
                )
                if submodule_dependency_node not in dependency_graph:
                    _collect_dependencies(
                        dependency_graph,
                        instance_class_paths,
                        base_class_paths,
                        metaclass_paths,
                        field_value,
                        submodule_dependency_node,
                    )
                continue
            field_dependency_node = DependencyNode(
                _classify_value(field_value),
                value_dependency_node.module_path,
                value_dependency_node.local_path.join(field_name),
                dependant_local_path=(
                    value_dependency_node.dependant_local_path.join(field_name)
                ),
                dependant_module_path=(
                    value_dependency_node.dependant_module_path
                ),
                value=field_value if is_safe(field_value) else MISSING,
            )
            if isinstance(field_value, builtins.dict):
                dependency_graph.setdefault(
                    field_dependency_node, builtins.set()
                ).add(
                    DependencyNode(
                        _classify_value(builtins.tuple),
                        BUILTINS_MODULE_PATH,
                        BUILTINS_DICT_LOCAL_OBJECT_PATH,
                        dependant_local_path=BUILTINS_DICT_LOCAL_OBJECT_PATH,
                        dependant_module_path=BUILTINS_MODULE_PATH,
                        value=MISSING,
                    )
                )
                instance_class_paths[
                    field_dependency_node.module_path,
                    field_dependency_node.local_path,
                ] = (BUILTINS_MODULE_PATH, BUILTINS_DICT_LOCAL_OBJECT_PATH)
            elif isinstance(field_value, builtins.frozenset):
                dependency_graph.setdefault(
                    field_dependency_node, builtins.set()
                ).add(
                    DependencyNode(
                        _classify_value(builtins.tuple),
                        BUILTINS_MODULE_PATH,
                        BUILTINS_FROZENSET_LOCAL_OBJECT_PATH,
                        dependant_local_path=(
                            BUILTINS_FROZENSET_LOCAL_OBJECT_PATH
                        ),
                        dependant_module_path=BUILTINS_MODULE_PATH,
                        value=MISSING,
                    )
                )
                instance_class_paths[
                    field_dependency_node.module_path,
                    field_dependency_node.local_path,
                ] = (
                    BUILTINS_MODULE_PATH,
                    BUILTINS_FROZENSET_LOCAL_OBJECT_PATH,
                )
            elif isinstance(field_value, builtins.list):
                dependency_graph.setdefault(
                    field_dependency_node, builtins.set()
                ).add(
                    DependencyNode(
                        _classify_value(builtins.tuple),
                        BUILTINS_MODULE_PATH,
                        BUILTINS_LIST_LOCAL_OBJECT_PATH,
                        dependant_local_path=BUILTINS_LIST_LOCAL_OBJECT_PATH,
                        dependant_module_path=BUILTINS_MODULE_PATH,
                        value=MISSING,
                    )
                )
                instance_class_paths[
                    field_dependency_node.module_path,
                    field_dependency_node.local_path,
                ] = (BUILTINS_MODULE_PATH, BUILTINS_LIST_LOCAL_OBJECT_PATH)
            elif isinstance(field_value, builtins.set):
                dependency_graph.setdefault(
                    field_dependency_node, builtins.set()
                ).add(
                    DependencyNode(
                        _classify_value(builtins.tuple),
                        BUILTINS_MODULE_PATH,
                        BUILTINS_SET_LOCAL_OBJECT_PATH,
                        dependant_local_path=BUILTINS_SET_LOCAL_OBJECT_PATH,
                        dependant_module_path=BUILTINS_MODULE_PATH,
                        value=MISSING,
                    )
                )
                instance_class_paths[
                    field_dependency_node.module_path,
                    field_dependency_node.local_path,
                ] = (BUILTINS_MODULE_PATH, BUILTINS_SET_LOCAL_OBJECT_PATH)
            elif isinstance(field_value, builtins.tuple):
                dependency_graph.setdefault(
                    field_dependency_node, builtins.set()
                ).add(
                    DependencyNode(
                        _classify_value(builtins.tuple),
                        BUILTINS_MODULE_PATH,
                        BUILTINS_TUPLE_LOCAL_OBJECT_PATH,
                        dependant_local_path=(
                            BUILTINS_TUPLE_LOCAL_OBJECT_PATH
                        ),
                        dependant_module_path=BUILTINS_MODULE_PATH,
                        value=MISSING,
                    )
                )
                instance_class_paths[
                    field_dependency_node.module_path,
                    field_dependency_node.local_path,
                ] = (BUILTINS_MODULE_PATH, BUILTINS_TUPLE_LOCAL_OBJECT_PATH)
            elif inspect.isroutine(field_value):
                dependency_graph.setdefault(field_dependency_node, set()).add(
                    DependencyNode(
                        _classify_value(types.FunctionType),
                        TYPES_MODULE_PATH,
                        TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH,
                        dependant_local_path=(
                            TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                        ),
                        dependant_module_path=TYPES_MODULE_PATH,
                        value=MISSING,
                    )
                )
                instance_class_paths[
                    field_dependency_node.module_path,
                    field_dependency_node.local_path,
                ] = (TYPES_MODULE_PATH, TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH)
            elif inspect.isclass(field_value):
                prev_metacls = field_value
                prev_metacls_alias_local_path = (
                    field_dependency_node.dependant_local_path
                )
                prev_metacls_local_path = field_dependency_node.local_path
                prev_metacls_module_path = field_dependency_node.module_path
                while (metacls := type(prev_metacls)) is not prev_metacls:
                    metacls_alias_local_path = (
                        prev_metacls_alias_local_path.join('__class__')
                    )
                    metacls_module_path = ModulePath.from_module_name(
                        metacls.__module__
                    )
                    metacls_local_path = LocalObjectPath.from_object_name(
                        metacls.__qualname__
                    )
                    metacls_dependency_node = DependencyNode(
                        ObjectKind.METACLASS,
                        metacls_module_path,
                        metacls_local_path,
                        dependant_local_path=metacls_local_path,
                        dependant_module_path=metacls_module_path,
                        value=MISSING,
                    )
                    dependency_graph.setdefault(
                        field_dependency_node, set()
                    ).add(metacls_dependency_node)
                    metaclass_paths[
                        prev_metacls_module_path, prev_metacls_local_path
                    ] = (metacls_module_path, metacls_local_path)
                    metacls_module_dependency_node = DependencyNode(
                        ObjectKind.STATIC_MODULE,
                        metacls_dependency_node.module_path,
                        LocalObjectPath(),
                        dependant_local_path=LocalObjectPath(),
                        dependant_module_path=metacls_dependency_node.module_path,
                        value=MISSING,
                    )
                    if metacls_module_dependency_node not in dependency_graph:
                        metacls_module = sys.modules[
                            metacls_dependency_node.module_path.to_module_name()
                        ]
                        _collect_dependencies(
                            dependency_graph,
                            instance_class_paths,
                            base_class_paths,
                            metaclass_paths,
                            metacls_module,
                            metacls_module_dependency_node,
                        )
                    _collect_dependencies(
                        dependency_graph,
                        instance_class_paths,
                        base_class_paths,
                        metaclass_paths,
                        metacls,
                        metacls_dependency_node,
                    )
                    prev_metacls = metacls
                    prev_metacls_alias_local_path = metacls_alias_local_path
                    prev_metacls_local_path = metacls_local_path
                    prev_metacls_module_path = metacls_module_path
                for base_cls in field_value.__mro__[1:-1]:  # pyright: ignore[reportIndexIssue]
                    base_cls_module_path = ModulePath.from_module_name(
                        base_cls.__module__
                    )
                    base_cls_local_path = LocalObjectPath.from_object_name(
                        base_cls.__qualname__
                    )
                    base_cls_dependency_node = DependencyNode(
                        ObjectKind.CLASS,
                        base_cls_module_path,
                        base_cls_local_path,
                        dependant_local_path=base_cls_local_path,
                        dependant_module_path=base_cls_module_path,
                        value=MISSING,
                    )
                    base_cls_module_dependency_node = DependencyNode(
                        ObjectKind.STATIC_MODULE,
                        base_cls_dependency_node.module_path,
                        LocalObjectPath(),
                        dependant_local_path=LocalObjectPath(),
                        dependant_module_path=base_cls_dependency_node.module_path,
                        value=MISSING,
                    )
                    if base_cls_module_dependency_node not in dependency_graph:
                        base_cls_module = sys.modules[
                            base_cls_dependency_node.module_path.to_module_name()
                        ]
                        _collect_dependencies(
                            dependency_graph,
                            instance_class_paths,
                            base_class_paths,
                            metaclass_paths,
                            base_cls_module,
                            base_cls_module_dependency_node,
                        )
                    dependency_graph.setdefault(
                        field_dependency_node, set()
                    ).add(base_cls_dependency_node)
                    base_class_paths.setdefault(
                        (
                            field_dependency_node.module_path,
                            field_dependency_node.local_path,
                        ),
                        set(),
                    ).add((base_cls_module_path, base_cls_local_path))
                    _collect_dependencies(
                        dependency_graph,
                        instance_class_paths,
                        base_class_paths,
                        metaclass_paths,
                        base_cls,
                        base_cls_dependency_node,
                    )
                _collect_dependencies(
                    dependency_graph,
                    instance_class_paths,
                    base_class_paths,
                    metaclass_paths,
                    field_value,
                    field_dependency_node,
                )
            dependency_graph.setdefault(field_dependency_node, set()).add(
                value_dependency_node
            )


def _classify_value(
    value: Any, /
) -> Literal[
    ObjectKind.CLASS,
    ObjectKind.INSTANCE,
    ObjectKind.METACLASS,
    ObjectKind.ROUTINE,
    ObjectKind.UNKNOWN,
]:
    return (
        ObjectKind.ROUTINE
        if inspect.isroutine(value)
        else (
            (
                ObjectKind.METACLASS
                if issubclass(value, type)
                else ObjectKind.CLASS
            )
            if inspect.isclass(value)
            else (
                ObjectKind.INSTANCE
                if isinstance(value, dict | frozenset | list | set | tuple)
                else ObjectKind.UNKNOWN
            )
        )
    )


def _path_to_object(object_path: ObjectPath, /) -> Object:
    module_path, local_path = object_path
    return MODULES[module_path].get_nested_attribute(local_path)


def _dependency_node_to_object(
    dependency_node: DependencyNode,
    /,
    *,
    base_class_paths: Mapping[ObjectPath, Collection[ObjectPath]],
    instance_class_paths: Mapping[ObjectPath, ObjectPath],
    metaclass_paths: Mapping[ObjectPath, ObjectPath],
) -> Class | PlainObject:
    return (
        Class(
            Scope(
                _class_object_kind_to_scope_kind(dependency_node.object_kind),
                dependency_node.module_path,
                dependency_node.local_path,
            ),
            *map(
                _path_to_object,
                base_class_paths.get(
                    (dependency_node.module_path, dependency_node.local_path),
                    [],
                ),
            ),
            metaclass=(
                _path_to_object(metacls_full_path)
                if (
                    metacls_full_path := metaclass_paths.get(
                        (
                            dependency_node.module_path,
                            dependency_node.local_path,
                        )
                    )
                )
                is not None
                else None
            ),
        )
        if dependency_node.object_kind in CLASS_OBJECT_KINDS
        else PlainObject(
            dependency_node.object_kind,
            dependency_node.module_path,
            dependency_node.local_path,
            *(
                (_path_to_object(class_path),)
                if (
                    class_path := instance_class_paths.get(
                        (
                            dependency_node.module_path,
                            dependency_node.local_path,
                        )
                    )
                )
                is not None
                else ()
            ),
        )
    )


parse_modules(builtins, sys, types)

BUILTINS_MODULE: Final[Module] = ensure_type(
    MODULES[BUILTINS_MODULE_PATH], Module
)
TYPES_MODULE: Final[Object] = ensure_type(MODULES[TYPES_MODULE_PATH], Module)


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
