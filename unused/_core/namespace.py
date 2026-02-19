from __future__ import annotations

import functools
from enum import Enum
from typing import Any, TypeVar

from typing_extensions import Self

from .attribute_mapping import AttributeMapping
from .mapping_chain import MappingChain
from .missing import MISSING, Missing
from .object_path import LocalObjectPath, ModulePath


class ObjectKind(str, Enum):
    BUILTIN_MODULE = 'BUILTIN_MODULE'
    CLASS = 'CLASS'
    DYNAMIC_MODULE = 'DYNAMIC_MODULE'
    EXTENSION_MODULE = 'EXTENSION_MODULE'
    FUNCTION_SCOPE = 'FUNCTION_SCOPE'
    INSTANCE = 'INSTANCE'
    INSTANCE_ROUTINE = 'INSTANCE_ROUTINE'
    METACLASS = 'METACLASS'
    PROPERTY = 'PROPERTY'
    ROUTINE = 'ROUTINE'
    ROUTINE_CALL = 'ROUTINE_CALL'
    STATIC_MODULE = 'STATIC_MODULE'
    UNKNOWN = 'UNKNOWN'
    UNKNOWN_CLASS = 'UNKNOWN_CLASS'

    def __repr__(self, /) -> str:
        return f'{type(self).__qualname__}.{self.name}'


_T = TypeVar('_T')


class Namespace:
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
        assert sub_namespace.kind in (
            ObjectKind.CLASS,
            ObjectKind.INSTANCE,
            ObjectKind.BUILTIN_MODULE,
            ObjectKind.DYNAMIC_MODULE,
            ObjectKind.EXTENSION_MODULE,
            ObjectKind.METACLASS,
            ObjectKind.ROUTINE,
            ObjectKind.STATIC_MODULE,
            ObjectKind.UNKNOWN,
            ObjectKind.UNKNOWN_CLASS,
        ), (self, sub_namespace)
        assert self.kind is not ObjectKind.UNKNOWN, (self, sub_namespace)
        assert isinstance(sub_namespace, Namespace), (self, sub_namespace)
        assert sub_namespace is not self, (self, sub_namespace)
        assert sub_namespace not in self._sub_namespaces, (self, sub_namespace)
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
        assert isinstance(local_path, LocalObjectPath), local_path
        return functools.reduce(
            type(self).get_namespace_by_name, local_path.components, self
        )

    def get_object_by_name(self, name: str, /) -> Any:
        assert isinstance(name, str), name
        return self._objects[name]

    def get_object_by_name_or_else(
        self, name: str, /, *, default: _T
    ) -> Any | _T:
        assert isinstance(name, str), name
        return self._objects.get(name, default)

    def get_namespace_by_name(self, name: str, /) -> Self:
        try:
            candidate = self._children[name]
        except KeyError:
            for sub_namespace in self._sub_namespaces:
                try:
                    candidate = sub_namespace.get_namespace_by_name(name)
                except KeyError:
                    continue
                else:
                    if candidate.kind is ObjectKind.ROUTINE and (
                        (
                            self._kind is ObjectKind.CLASS
                            and sub_namespace.kind is ObjectKind.METACLASS
                        )
                        or (
                            self._kind is ObjectKind.INSTANCE
                            and sub_namespace.kind is ObjectKind.CLASS
                        )
                    ):
                        candidate = type(self)(
                            ObjectKind.INSTANCE_ROUTINE,
                            self._module_path,
                            self._local_path.join(name),
                            candidate,
                        )
                    return candidate
            if self.kind in (
                ObjectKind.BUILTIN_MODULE,
                ObjectKind.DYNAMIC_MODULE,
                ObjectKind.EXTENSION_MODULE,
                ObjectKind.ROUTINE_CALL,
                ObjectKind.INSTANCE,
                ObjectKind.PROPERTY,
                ObjectKind.UNKNOWN,
                ObjectKind.UNKNOWN_CLASS,
            ):
                assert name not in self._children
                self._children[name] = result = type(self)(
                    ObjectKind.UNKNOWN,
                    self.module_path,
                    self.local_path.join(name),
                )
                return result
            raise
        else:
            if (
                self._kind is ObjectKind.CLASS
                and candidate.kind is ObjectKind.ROUTINE
                and (
                    name
                    in (
                        object.__init_subclass__.__name__,
                        object.__new__.__name__,
                    )
                )
            ):
                candidate = type(self)(
                    ObjectKind.INSTANCE_ROUTINE,
                    self._module_path,
                    self._local_path.join(name),
                    candidate,
                )
            return candidate

    def mark_module_as_dynamic(self, /) -> None:
        assert self._kind is ObjectKind.STATIC_MODULE
        self._kind = ObjectKind.DYNAMIC_MODULE

    def instance_routine_to_routine(self, /) -> Self:
        assert self._kind is ObjectKind.INSTANCE_ROUTINE
        (result,) = self._sub_namespaces
        return result

    def safe_delete_object_by_name(self, name: str, /) -> bool:
        assert isinstance(name, str), name
        return self._objects.pop(name, MISSING) is not MISSING

    def safe_delete_object_by_path(
        self, local_path: LocalObjectPath, /
    ) -> bool:
        assert isinstance(local_path, LocalObjectPath), local_path
        namespace = self
        for component in local_path.components[:-1]:
            namespace = namespace.get_namespace_by_name(component)
        return namespace.safe_delete_object_by_name(local_path.components[-1])

    def set_namespace_by_name(self, name: str, value: Self, /) -> None:
        assert isinstance(name, str), (name, value)
        assert isinstance(value, type(self)), (name, value)
        self._children[name] = value

    def set_namespace_by_path(
        self, local_path: LocalObjectPath, namespace: Self, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(namespace, type(self)), namespace
        parent_namespace = functools.reduce(
            type(self).get_namespace_by_name, local_path.components[:-1], self
        )
        parent_namespace.set_namespace_by_name(
            local_path.components[-1], namespace
        )

    def set_object_by_name(self, name: str, value: Any | Missing, /) -> None:
        assert isinstance(name, str), name
        assert name in self._children
        if value is MISSING:
            assert name in self._objects
            self._objects.pop(name, None)
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
                namespace._objects[component] = (  # noqa: SLF001
                    namespace.get_namespace_by_name(component).as_object()
                )
            namespace = namespace.get_namespace_by_name(component)
        namespace._objects[local_path.components[-1]] = value  # noqa: SLF001

    def strict_get_namespace_by_name(self, name: str, /) -> Self:
        try:
            return self._children[name]
        except KeyError:
            for sub_namespace in self._sub_namespaces:
                try:
                    return sub_namespace.strict_get_namespace_by_name(name)
                except KeyError:
                    continue
            raise

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

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._kind!r}, {self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._sub_namespaces)}'
            f'{", ".join(map(repr, self._sub_namespaces))}'
            ')'
        )
