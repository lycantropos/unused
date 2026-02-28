from __future__ import annotations

import functools
from typing import Any, TypeVar

from .attribute_mapping import AttributeMapping
from .enums import ObjectKind, ScopeKind
from .mapping_chain import MappingChain
from .missing import MISSING, Missing
from .object_ import (
    MUTABLE_OBJECT_CLASSES,
    MutableObject,
    Object,
    UnknownObject,
    object_get_attribute,
    object_get_mutable_attribute,
)
from .object_path import LocalObjectPath, ModulePath
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
        assert isinstance(object_, Object), (self, object_)
        assert object_ not in self._included_objects, (self, object_)
        self._included_objects.append(object_)

    def as_object(self, /) -> AttributeMapping:
        return AttributeMapping(
            MappingChain(
                self._values,
                *[
                    included_object._values  # noqa: SLF001
                    for included_object in self._included_objects
                ],
            )
        )

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

    def get_value(self, name: str, /) -> Any:
        assert isinstance(name, str), name
        return self._values[name]

    def get_value_or_else(self, name: str, /, *, default: _T) -> Any | _T:
        assert isinstance(name, str), name
        return self._values.get(name, default)

    def get_object(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            for included_object in self._included_objects:
                try:
                    return included_object.get_attribute(name)
                except KeyError:
                    continue
            if self.kind in (
                ScopeKind.BUILTIN_MODULE,
                ScopeKind.DYNAMIC_MODULE,
                ScopeKind.EXTENSION_MODULE,
                ScopeKind.UNKNOWN_CLASS,
            ):
                assert name not in self._objects
                self._objects[name] = result = UnknownObject(
                    self.module_path, self.local_path.join(name)
                )
                return result
            raise

    def mark_module_as_dynamic(self, /) -> None:
        assert self._kind is ScopeKind.STATIC_MODULE
        self._kind = ScopeKind.DYNAMIC_MODULE

    def safe_delete_value(self, name: str, /) -> bool:
        assert isinstance(name, str), name
        return self._values.pop(name, MISSING) is not MISSING

    def safe_delete_nested_value(self, local_path: LocalObjectPath, /) -> bool:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert len(local_path.components) > 0, local_path
        if len(local_path.components) == 1:
            return self.safe_delete_value(local_path.components[-1])
        return self.get_mutable_nested_object(
            local_path.parent
        ).safe_delete_value(local_path.components[-1])

    def set_object(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._objects[name] = object_

    def set_nested_object(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        *first_components, last_component = local_path.components
        if len(first_components) > 0:
            parent_object = functools.reduce(
                object_get_mutable_attribute,
                first_components[1:],
                self.get_mutable_object(first_components[0]),
            )
            parent_object.set_attribute(last_component, object_)
        else:
            self.set_object(last_component, object_)

    def set_value(self, name: str, value: Any | Missing, /) -> None:
        assert isinstance(name, str), name
        assert name in self._objects
        if value is MISSING:
            assert name in self._values
            self._values.pop(name, None)
        else:
            self._values[name] = value

    def set_nested_value(
        self, local_path: LocalObjectPath, value: Any, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(self.get_nested_object(local_path), Object)
        assert value is not MISSING
        assert len(local_path.components) > 0
        *first_components, last_component = local_path.components
        if len(first_components) == 0:
            self._values[last_component] = value
        else:
            object_ = self.get_object(first_components[0])
            for component in first_components[1:]:
                if component not in object_._values:  # noqa: SLF001
                    object_._values[component] = (  # noqa: SLF001
                        object_.get_attribute(component).as_object()
                    )
                object_ = object_.get_attribute(component)
            object_._values[last_component] = value  # noqa: SLF001

    def strict_get_object(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            for included_object in self._included_objects:
                try:
                    return included_object.strict_get_attribute(name)
                except KeyError:
                    continue
            raise

    _kind: ScopeKind
    _module_path: ModulePath
    _local_path: LocalObjectPath
    _objects: dict[str, Object]
    _included_objects: list[Object]
    _values: dict[str, Any]

    __slots__ = (
        '_included_objects',
        '_kind',
        '_local_path',
        '_module_path',
        '_objects',
        '_values',
    )

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._kind is other._kind
                and self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._objects == other._objects
                and self._values == other._values
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
            self._objects,
            self._kind,
            self._module_path,
            self._local_path,
            self._included_objects,
            self._values,
        ) = {}, kind, module_path, local_path, [], {}

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._kind!r}, {self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._included_objects)}'
            f'{", ".join(map(repr, self._included_objects))}'
            ')'
        )
