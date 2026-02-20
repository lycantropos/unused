from __future__ import annotations

import builtins
import graphlib
import inspect
import sys
import types
from collections.abc import MutableMapping
from typing import Any, Final

from .dependency_node import DependencyNode
from .missing import MISSING
from .namespace import Namespace, ObjectKind
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

MODULE_NAMESPACES: Final[dict[ModulePath, Namespace]] = {}


def parse_modules(*modules: types.ModuleType) -> None:
    dependency_graph: dict[DependencyNode, set[DependencyNode]] = {}
    sub_object_graph: dict[
        tuple[ModulePath, LocalObjectPath],
        set[tuple[ModulePath, LocalObjectPath]],
    ] = {}
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
            dependency_graph, sub_object_graph, module, module_dependency_node
        )
    topologically_sorted_dependency_nodes = [
        *graphlib.TopologicalSorter(dependency_graph).static_order()
    ]
    for dependency_node in topologically_sorted_dependency_nodes:
        if len(dependency_node.dependant_local_path.components) == 0:
            continue
        for base_submodule_path in (
            dependency_node.module_path,
            dependency_node.dependant_module_path,
        ):
            for submodule_path in base_submodule_path.submodule_paths():
                MODULE_NAMESPACES.setdefault(
                    submodule_path,
                    Namespace(
                        ObjectKind.STATIC_MODULE,
                        submodule_path,
                        LocalObjectPath(),
                    ),
                )
        dependant_module_namespace = MODULE_NAMESPACES[
            dependency_node.dependant_module_path
        ]
        if (
            dependency_node.module_path
            == dependency_node.dependant_module_path
        ) and (
            dependency_node.local_path == dependency_node.dependant_local_path
        ):
            try:
                dependant_namespace = (
                    dependant_module_namespace.get_namespace_by_path(
                        dependency_node.dependant_local_path
                    )
                )
            except KeyError:
                dependant_module_namespace.set_namespace_by_path(
                    dependency_node.dependant_local_path,
                    Namespace(
                        dependency_node.object_kind,
                        dependency_node.module_path,
                        dependency_node.local_path,
                    ),
                )
            else:
                assert (
                    dependant_namespace.kind is dependency_node.object_kind
                ), (dependant_namespace, dependency_node)
                del dependant_namespace
        else:
            dependency_module_namespace = MODULE_NAMESPACES[
                dependency_node.module_path
            ]
            try:
                dependency_namespace = (
                    dependency_module_namespace.get_namespace_by_path(
                        dependency_node.local_path
                    )
                )
            except KeyError:
                dependency_namespace = Namespace(
                    dependency_node.object_kind,
                    dependency_node.module_path,
                    dependency_node.local_path,
                )
                dependency_module_namespace.set_namespace_by_path(
                    dependency_node.local_path, dependency_namespace
                )
            dependant_module_namespace.set_namespace_by_path(
                dependency_node.dependant_local_path, dependency_namespace
            )
        if (dependency_value := dependency_node.value) is not MISSING:
            dependant_module_namespace.set_object_by_path(
                dependency_node.dependant_local_path, dependency_value
            )
        elif dependency_node.object_kind in (
            ObjectKind.CLASS,
            ObjectKind.METACLASS,
            ObjectKind.STATIC_MODULE,
        ):
            dependant_module_namespace.set_object_by_path(
                dependency_node.dependant_local_path,
                dependant_module_namespace.get_namespace_by_path(
                    dependency_node.dependant_local_path
                ).as_object(),
            )
    for (
        module_path,
        local_path,
    ), sub_object_paths in sub_object_graph.items():
        namespace = MODULE_NAMESPACES[module_path].get_namespace_by_path(
            local_path
        )
        for sub_module_path, sub_local_path in sub_object_paths:
            sub_namespace = MODULE_NAMESPACES[
                sub_module_path
            ].get_namespace_by_path(sub_local_path)
            namespace.append_sub_namespace(sub_namespace)


def _collect_dependencies(
    dependency_graph: MutableMapping[DependencyNode, set[DependencyNode]],
    sub_object_graph: MutableMapping[
        tuple[ModulePath, LocalObjectPath],
        set[tuple[ModulePath, LocalObjectPath]],
    ],
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
                        sub_object_graph,
                        field_value,
                        submodule_dependency_node,
                    )
                continue
            field_dependency_node = DependencyNode(
                (
                    ObjectKind.ROUTINE
                    if inspect.isroutine(field_value)
                    else (
                        (
                            ObjectKind.METACLASS
                            if issubclass(field_value, type)
                            else ObjectKind.CLASS
                        )
                        if inspect.isclass(field_value)
                        else (
                            ObjectKind.INSTANCE
                            if isinstance(
                                field_value,
                                dict | frozenset | list | set | tuple,
                            )
                            else ObjectKind.UNKNOWN
                        )
                    )
                ),
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
            if isinstance(field_value, dict):
                sub_object_graph.setdefault(
                    (
                        field_dependency_node.module_path,
                        field_dependency_node.local_path,
                    ),
                    set(),
                ).add((BUILTINS_MODULE_PATH, BUILTINS_DICT_LOCAL_OBJECT_PATH))
            elif isinstance(field_value, frozenset):
                sub_object_graph.setdefault(
                    (
                        field_dependency_node.module_path,
                        field_dependency_node.local_path,
                    ),
                    set(),
                ).add(
                    (
                        BUILTINS_MODULE_PATH,
                        BUILTINS_FROZENSET_LOCAL_OBJECT_PATH,
                    )
                )
            elif isinstance(field_value, list):
                sub_object_graph.setdefault(
                    (
                        field_dependency_node.module_path,
                        field_dependency_node.local_path,
                    ),
                    set(),
                ).add((BUILTINS_MODULE_PATH, BUILTINS_LIST_LOCAL_OBJECT_PATH))
            elif isinstance(field_value, set):
                sub_object_graph.setdefault(
                    (
                        field_dependency_node.module_path,
                        field_dependency_node.local_path,
                    ),
                    set(),
                ).add((BUILTINS_MODULE_PATH, BUILTINS_SET_LOCAL_OBJECT_PATH))
            elif isinstance(field_value, tuple):
                sub_object_graph.setdefault(
                    (
                        field_dependency_node.module_path,
                        field_dependency_node.local_path,
                    ),
                    set(),
                ).add((BUILTINS_MODULE_PATH, BUILTINS_TUPLE_LOCAL_OBJECT_PATH))
            elif inspect.isroutine(field_value):
                sub_object_graph.setdefault(
                    (
                        field_dependency_node.module_path,
                        field_dependency_node.local_path,
                    ),
                    set(),
                ).add(
                    (TYPES_MODULE_PATH, TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH)
                )
            elif inspect.isclass(field_value):
                prev_metacls = field_value
                prev_metacls_alias_local_path = (
                    field_dependency_node.dependant_local_path
                )
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
                    sub_object_graph.setdefault(
                        (
                            field_dependency_node.module_path,
                            field_dependency_node.local_path,
                        ),
                        set(),
                    ).add((metacls_module_path, metacls_local_path))
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
                            sub_object_graph,
                            metacls_module,
                            metacls_module_dependency_node,
                        )
                    _collect_dependencies(
                        dependency_graph,
                        sub_object_graph,
                        metacls,
                        metacls_dependency_node,
                    )
                    prev_metacls = metacls
                    prev_metacls_alias_local_path = metacls_alias_local_path
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
                            sub_object_graph,
                            base_cls_module,
                            base_cls_module_dependency_node,
                        )
                    dependency_graph.setdefault(
                        field_dependency_node, set()
                    ).add(base_cls_dependency_node)
                    sub_object_graph.setdefault(
                        (
                            field_dependency_node.module_path,
                            field_dependency_node.local_path,
                        ),
                        set(),
                    ).add((base_cls_module_path, base_cls_local_path))
                    _collect_dependencies(
                        dependency_graph,
                        sub_object_graph,
                        base_cls,
                        base_cls_dependency_node,
                    )
                _collect_dependencies(
                    dependency_graph,
                    sub_object_graph,
                    field_value,
                    field_dependency_node,
                )
            dependency_graph.setdefault(field_dependency_node, set()).add(
                value_dependency_node
            )


parse_modules(builtins, sys, types)
BUILTINS_MODULE_NAMESPACE: Final[Namespace] = MODULE_NAMESPACES[
    BUILTINS_MODULE_PATH
]
TYPES_MODULE_NAMESPACE: Final[Namespace] = MODULE_NAMESPACES[TYPES_MODULE_PATH]


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
        BUILTINS_MODULE_NAMESPACE.set_object_by_path(
            LocalObjectPath.from_object_name(cls.__qualname__), cls
        )


_setup_builtin_classes()
