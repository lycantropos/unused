from __future__ import annotations

import functools
import operator
from collections.abc import Iterator, MutableMapping
from enum import Enum
from typing import Any

from typing_extensions import Self

from .attribute_mapping import AttributeMapping
from .mapping_chain import MappingChain
from .missing import MISSING
from .object_path import LocalObjectPath, ModulePath


class ObjectKind(str, Enum):
    BUILTIN_MODULE = 'BUILTIN_MODULE'
    CLASS = 'CLASS'
    EXTENSION_MODULE = 'EXTENSION_MODULE'
    INSTANCE = 'INSTANCE'
    MODULE = 'MODULE'
    OBJECT = 'OBJECT'
    PROPERTY = 'PROPERTY'
    ROUTINE = 'ROUTINE'
    ROUTINE_CALL = 'ROUTINE_CALL'
    UNKNOWN = 'UNKNOWN'

    def __repr__(self, /) -> str:
        return f'{type(self).__qualname__}.{self.name}'


class Namespace(MutableMapping[str, 'Namespace']):
    @property
    def kind(self, /) -> ObjectKind:
        return self._kind

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    def append_sub_namespace(self, sub_namespace: Self, /) -> None:
        assert isinstance(sub_namespace, Namespace), self
        assert sub_namespace is not self, self
        assert sub_namespace not in self._sub_namespaces, self
        self._sub_namespaces.append(sub_namespace)

    def as_object(self, /) -> AttributeMapping:
        return AttributeMapping(
            MappingChain(
                self._objects,
                *[
                    sub_namespace._objects  # noqa: SLF001
                    for sub_namespace in self._sub_namespaces
                ],
            )
        )

    def get_namespace_by_path(self, local_path: LocalObjectPath, /) -> Self:
        return functools.reduce(
            operator.getitem,  # type: ignore[arg-type]
            local_path.components,
            self,
        )

    def get_object_by_name(self, name: str, /) -> Any:
        assert isinstance(name, str), name
        return self._objects[name]

    def set_namespace_by_path(
        self, local_path: LocalObjectPath, namespace: Self, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(namespace, type(self)), namespace
        parent_namespace = functools.reduce(
            operator.getitem,  # type: ignore[arg-type]
            local_path.components[:-1],
            self,
        )
        name = local_path.components[-1]
        if (old_namespace := parent_namespace.get(name)) is not None:
            assert old_namespace is namespace
            return
        parent_namespace[name] = namespace

    def set_object_by_name(self, name: str, value: Any, /) -> None:
        assert isinstance(name, str), name
        assert value is not MISSING
        assert name in self._children
        self._objects[name] = value

    def set_object_by_path(
        self, local_path: LocalObjectPath, value: Any, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(self.get_namespace_by_path(local_path), Namespace)
        assert value is not MISSING
        namespace = self
        for component in local_path.components[:-1]:
            if component not in namespace._objects:  # noqa: SLF001
                namespace._objects[component] = namespace[  # noqa: SLF001
                    component
                ].as_object()
            namespace = namespace[component]
        namespace._objects[local_path.components[-1]] = value  # noqa: SLF001

    _children: dict[str, Self]
    _kind: ObjectKind
    _module_path: ModulePath
    _local_path: LocalObjectPath
    _objects: dict[str, Any]
    _sub_namespaces: list[Self]

    __slots__ = (
        '_children',
        '_kind',
        '_local_path',
        '_module_path',
        '_objects',
        '_sub_namespaces',
    )

    def __delitem__(self, key: str, /) -> None:
        del self._children[key]

    def __eq__(self, other: Any) -> Any:
        return (
            (
                self._kind is other._kind
                and self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._children == other._children
                and self._objects == other._objects
                and self._sub_namespaces == other._sub_namespaces
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __getitem__(self, key: str, /) -> Self:
        try:
            return self._children[key]
        except KeyError:
            for sub_namespace in self._sub_namespaces:
                try:
                    return sub_namespace[key]
                except KeyError:
                    continue
            if self.kind in (
                ObjectKind.BUILTIN_MODULE,
                ObjectKind.EXTENSION_MODULE,
                ObjectKind.ROUTINE_CALL,
                ObjectKind.INSTANCE,
                ObjectKind.PROPERTY,
                ObjectKind.UNKNOWN,
            ):
                return type(self)(
                    ObjectKind.UNKNOWN,
                    self.module_path,
                    self.local_path.join(key),
                )
            raise

    def __init__(
        self,
        kind: ObjectKind,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
        *sub_namespaces: Self,
    ) -> None:
        (
            self._children,
            self._kind,
            self._module_path,
            self._local_path,
            self._sub_namespaces,
            self._objects,
        ) = {}, kind, module_path, local_path, list(sub_namespaces), {}

    def __iter__(self, /) -> Iterator[str]:
        return iter(self._children)

    def __len__(self, /) -> int:
        return len(self._children)

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._kind!r}, {self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._sub_namespaces)}'
            f'{", ".join(map(repr, self._sub_namespaces))}'
            ')'
        )

    def __setitem__(self, key: str, value: Self, /) -> None:
        assert isinstance(key, str), (key, value)
        assert isinstance(value, type(self)), (key, value)
        self._children[key] = value
