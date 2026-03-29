from __future__ import annotations

import functools
from typing import Any, TypeVar

from .enums import ObjectKind, ScopeKind
from .missing import MISSING
from .object_ import (
    MUTABLE_OBJECT_CLASSES,
    MutableObject,
    Object,
    UnknownObject,
    object_get_attribute,
)
from .object_path import LocalObjectPath, ModulePath, ObjectPath
from .utils import ensure_type

_T = TypeVar('_T')


class Scope:
    @property
    def kind(self, /) -> ScopeKind:
        return self._kind

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    def delete_object(self, name: str, /) -> None:
        assert isinstance(name, str), name
        del self._objects[name]

    def get_mutable_nested_object(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_object(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_mutable_object(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_object(name), MUTABLE_OBJECT_CLASSES)

    def get_nested_object(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert len(local_path.components) > 0, local_path
        first_component, *rest_components = local_path.components
        return functools.reduce(
            object_get_attribute,
            rest_components,
            self.get_object(first_component),
        )

    def get_object(self, name: str, /, *, strict: bool = False) -> Object:
        return self._get_object(
            name, strict=strict, visited_object_paths=set()
        )

    def _get_object(
        self,
        name: str,
        /,
        *,
        strict: bool,
        visited_object_paths: set[ObjectPath],
    ) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            for included_object in self._included_objects:
                try:
                    return included_object._get_attribute(  # noqa: SLF001
                        name,
                        strict=strict,
                        visited_object_paths=visited_object_paths,
                    )
                except KeyError:
                    continue
            if not strict and self.kind in (
                ScopeKind.BUILTIN_MODULE,
                ScopeKind.DYNAMIC_MODULE,
                ScopeKind.EXTENSION_MODULE,
                ScopeKind.UNKNOWN_CLASS,
            ):
                assert name not in self._objects
                self._objects[name] = result = UnknownObject(
                    self.module_path, self.local_path.join(name), value=MISSING
                )
                return result
            raise

    def include_object(self, object_: Object, /) -> None:
        assert object_.kind in (
            ObjectKind.CLASS,
            ObjectKind.BUILTIN_MODULE,
            ObjectKind.DYNAMIC_MODULE,
            ObjectKind.EXTENSION_MODULE,
            ObjectKind.METACLASS,
            ObjectKind.STATIC_MODULE,
            ObjectKind.UNKNOWN,
            ObjectKind.UNKNOWN_CLASS,
        ), (self, object_)
        assert isinstance(object_, MutableObject), (self, object_)
        self._included_objects.append(object_)

    def mark_module_as_dynamic(self, /) -> None:
        assert self._kind is ScopeKind.STATIC_MODULE
        self._kind = ScopeKind.DYNAMIC_MODULE

    def set_nested_object(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        *first_components, last_component = local_path.components
        if len(first_components) > 1:
            grandparent_object = functools.reduce(
                object_get_attribute,
                first_components[1:-1],
                self.get_object(first_components[0]),
            )
            grandparent_object.get_mutable_attribute(
                first_components[-1]
            ).set_attribute(last_component, object_)
        elif len(first_components) == 1:
            self.get_mutable_object(first_components[0]).set_attribute(
                last_component, object_
            )
        else:
            self.set_object(last_component, object_)

    def set_object(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._objects[name] = object_

    def _checked_get_object(self, name: str, /) -> Object | None:
        try:
            return self.get_object(name)
        except KeyError:
            return None

    _kind: ScopeKind
    _module_path: ModulePath
    _local_path: LocalObjectPath
    _objects: dict[str, Object]
    _included_objects: list[MutableObject]

    __slots__ = (
        '_included_objects',
        '_kind',
        '_local_path',
        '_module_path',
        '_objects',
    )

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._kind is other._kind
                and self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._objects == other._objects
                and self._included_objects == other._included_objects
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        kind: ScopeKind,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
    ) -> None:
        (
            self._included_objects,
            self._kind,
            self._local_path,
            self._module_path,
            self._objects,
        ) = [], kind, local_path, module_path, {}

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._kind!r}, {self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._included_objects)}'
            f'{", ".join(map(repr, self._included_objects))}'
            ')'
        )
