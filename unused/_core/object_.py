from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Final, TypeAlias, TypeVar

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


class ScopeKind(str, Enum):
    BUILTIN_MODULE = 'BUILTIN_MODULE'
    CLASS = 'CLASS'
    EXTENSION_MODULE = 'EXTENSION_MODULE'
    DYNAMIC_MODULE = 'DYNAMIC_MODULE'
    FUNCTION = 'FUNCTION'
    METACLASS = 'METACLASS'
    STATIC_MODULE = 'STATIC_MODULE'
    UNKNOWN_CLASS = 'UNKNOWN_CLASS'

    def __repr__(self, /) -> str:
        return f'{type(self).__qualname__}.{self.name}'


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

    def get_nested_object(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert len(local_path.components) > 0, local_path
        first_component, *rest_components = local_path.components
        return functools.reduce(
            _object_get_attribute,
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
            candidate = self._objects[name]
        except KeyError:
            for included_object in self._included_objects:
                try:
                    candidate = included_object.get_attribute(name)
                except KeyError:
                    continue
                else:
                    if candidate.kind is ObjectKind.ROUTINE and (
                        self._kind is ScopeKind.CLASS
                        and included_object.kind is ObjectKind.METACLASS
                    ):
                        candidate = PlainObject(
                            ObjectKind.INSTANCE_ROUTINE,
                            self._module_path,
                            self._local_path.join(name),
                            candidate,
                        )
                    return candidate
            if self.kind in (
                ScopeKind.BUILTIN_MODULE,
                ScopeKind.DYNAMIC_MODULE,
                ScopeKind.EXTENSION_MODULE,
                ScopeKind.UNKNOWN_CLASS,
            ):
                assert name not in self._objects
                self._objects[name] = result = PlainObject(
                    ObjectKind.UNKNOWN,
                    self.module_path,
                    self.local_path.join(name),
                )
                return result
            raise
        else:
            if (
                self._kind is ScopeKind.CLASS
                and candidate.kind is ObjectKind.ROUTINE
                and (
                    name
                    in (
                        object.__init_subclass__.__name__,
                        object.__new__.__name__,
                    )
                )
            ):
                candidate = PlainObject(
                    ObjectKind.INSTANCE_ROUTINE,
                    self._module_path,
                    self._local_path.join(name),
                    candidate,
                )
            return candidate

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
        return self.get_nested_object(local_path.parent).safe_delete_value(
            local_path.components[-1]
        )

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
                _object_get_attribute,
                first_components[1:],
                self.get_object(first_components[0]),
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
            for sub_namespace in self._included_objects:
                try:
                    return sub_namespace.strict_get_attribute(name)
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


class PlainObject:
    @property
    def kind(self, /) -> ObjectKind:
        return self._kind

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    def as_object(self, /) -> AttributeMapping:
        return AttributeMapping(
            MappingChain(
                self._values,
                *[
                    sub_namespace._values  # noqa: SLF001
                    for sub_namespace in self._included_objects
                ],
            )
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            _object_get_attribute, local_path.components, initial_object
        )

    def get_value(self, name: str, /) -> Any:
        assert isinstance(name, str), name
        return self._values[name]

    def get_value_or_else(self, name: str, /, *, default: _T) -> Any | _T:
        assert isinstance(name, str), name
        return self._values.get(name, default)

    def get_attribute(self, name: str, /) -> Object:
        try:
            candidate = self._attributes[name]
        except KeyError:
            for included_object in self._included_objects:
                try:
                    candidate = included_object.get_attribute(name)
                except KeyError:
                    continue
                else:
                    if candidate.kind is ObjectKind.ROUTINE and (
                        (
                            self._kind is ObjectKind.CLASS
                            and included_object.kind is ObjectKind.METACLASS
                        )
                        or (
                            self._kind is ObjectKind.INSTANCE
                            and included_object.kind is ObjectKind.CLASS
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
                assert name not in self._attributes
                self._attributes[name] = result = type(self)(
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

    def instance_routine_to_routine(self, /) -> Object:
        assert self._kind is ObjectKind.INSTANCE_ROUTINE
        (result,) = self._included_objects
        return result

    def safe_delete_value(self, name: str, /) -> bool:
        assert isinstance(name, str), name
        return self._values.pop(name, MISSING) is not MISSING

    def safe_delete_nested_value(self, local_path: LocalObjectPath, /) -> bool:
        assert isinstance(local_path, LocalObjectPath), local_path
        return self.get_nested_attribute(local_path.parent).safe_delete_value(
            local_path.components[-1]
        )

    def set_attribute(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._attributes[name] = object_

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        self.get_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

    def set_value(self, name: str, value: Any | Missing, /) -> None:
        assert isinstance(name, str), name
        assert name in self._attributes
        if value is MISSING:
            assert name in self._values
            self._values.pop(name, None)
        else:
            self._values[name] = value

    def set_nested_value(
        self, local_path: LocalObjectPath, value: Any, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(self.get_nested_attribute(local_path), PlainObject)
        assert value is not MISSING
        object_: Object = self
        for component in local_path.components[:-1]:
            if component not in object_._values:  # noqa: SLF001
                object_._values[component] = (  # noqa: SLF001
                    object_.get_attribute(component).as_object()
                )
            object_ = object_.get_attribute(component)
        object_._values[local_path.components[-1]] = value  # noqa: SLF001

    def strict_get_attribute(self, name: str, /) -> Object:
        try:
            return self._attributes[name]
        except KeyError:
            for included_object in self._included_objects:
                try:
                    return included_object.strict_get_attribute(name)
                except KeyError:
                    continue
            raise

    _attributes: dict[str, Object]
    _included_objects: list[Object]
    _kind: ObjectKind
    _module_path: ModulePath
    _local_path: LocalObjectPath
    _values: dict[str, Any]

    __slots__ = (
        '_attributes',
        '_included_objects',
        '_kind',
        '_local_path',
        '_module_path',
        '_values',
    )

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._kind is other._kind
                and self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._attributes == other._attributes
                and self._values == other._values
                and self._included_objects == other._included_objects
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
        *included_object: Object,
    ) -> None:
        assert kind not in CLASS_OBJECT_KINDS
        (
            self._attributes,
            self._included_objects,
            self._kind,
            self._local_path,
            self._module_path,
            self._values,
        ) = {}, list(included_object), kind, local_path, module_path, {}

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._kind!r}, {self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._included_objects)}'
            f'{", ".join(map(repr, self._included_objects))}'
            ')'
        )


class Module:
    @property
    def kind(self, /) -> ObjectKind:
        scope_kind = self._scope.kind
        if scope_kind is ScopeKind.BUILTIN_MODULE:
            return ObjectKind.BUILTIN_MODULE
        if scope_kind is ScopeKind.DYNAMIC_MODULE:
            return ObjectKind.DYNAMIC_MODULE
        if scope_kind is ScopeKind.EXTENSION_MODULE:
            return ObjectKind.EXTENSION_MODULE
        assert scope_kind is ScopeKind.STATIC_MODULE
        return ObjectKind.STATIC_MODULE

    @property
    def module_path(self, /) -> ModulePath:
        return self._scope.module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._scope.local_path

    def to_scope(self, /) -> Scope:
        return self._scope

    @property
    def _values(self, /) -> dict[str, Any]:
        return self._scope._values  # noqa: SLF001

    def as_object(self, /) -> AttributeMapping:
        return self._scope.as_object()

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            _object_get_attribute, local_path.components, initial_object
        )

    def get_value(self, name: str, /) -> Any:
        return self._scope.get_value(name)

    def get_value_or_else(self, name: str, /, *, default: _T) -> Any | _T:
        return self._scope.get_value_or_else(name, default=default)

    def get_attribute(self, name: str, /) -> Object:
        try:
            return self._scope.get_object(name)
        except KeyError:
            for included_object in self._included_objects:
                try:
                    return included_object.get_attribute(name)
                except KeyError:
                    continue
            if self.kind in (
                ObjectKind.BUILTIN_MODULE,
                ObjectKind.DYNAMIC_MODULE,
                ObjectKind.EXTENSION_MODULE,
            ):
                result = PlainObject(
                    ObjectKind.UNKNOWN,
                    self.module_path,
                    self.local_path.join(name),
                )
                self._scope.set_object(name, result)
                return result
            raise

    def safe_delete_value(self, name: str, /) -> bool:
        return self._scope.safe_delete_value(name)

    def safe_delete_nested_value(self, local_path: LocalObjectPath, /) -> bool:
        return self._scope.safe_delete_nested_value(local_path)

    def set_attribute(self, name: str, object_: Object, /) -> None:
        self._scope.set_object(name, object_)

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        self._scope.set_nested_object(local_path, object_)

    def set_value(self, name: str, value: Any | Missing, /) -> None:
        self._scope.set_value(name, value)

    def set_nested_value(
        self, local_path: LocalObjectPath, value: Any, /
    ) -> None:
        self._scope.set_nested_value(local_path, value)

    def strict_get_attribute(self, name: str, /) -> Object:
        try:
            return self._scope.strict_get_object(name)
        except KeyError:
            for included_object in self._included_objects:
                try:
                    return included_object.strict_get_attribute(name)
                except KeyError:
                    continue
            raise

    _scope: Scope
    _included_objects: list[Object]

    __slots__ = '_included_objects', '_scope'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._scope == other._scope
                and self._included_objects == other._included_objects
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(self, scope: Scope, /) -> None:
        self._included_objects, self._scope = [], scope

    def __repr__(self, /) -> str:
        return f'{type(self).__qualname__}({self._scope!r})'


class Class:
    @property
    def kind(self, /) -> ObjectKind:
        if self._scope.kind is ScopeKind.CLASS:
            return ObjectKind.CLASS
        if self._scope.kind is ScopeKind.METACLASS:
            return ObjectKind.METACLASS
        assert self._scope.kind is ScopeKind.UNKNOWN_CLASS
        return ObjectKind.UNKNOWN_CLASS

    @property
    def module_path(self, /) -> ModulePath:
        return self._scope.module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._scope.local_path

    def as_object(self, /) -> AttributeMapping:
        return AttributeMapping(
            MappingChain(
                self._values,
                *[
                    base._values  # noqa: SLF001
                    for base in self._bases
                ],
            )
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            _object_get_attribute, local_path.components, initial_object
        )

    def get_value(self, name: str, /) -> Any:
        assert isinstance(name, str), name
        return self._values[name]

    def get_value_or_else(self, name: str, /, *, default: _T) -> Any | _T:
        assert isinstance(name, str), name
        return self._values.get(name, default)

    def get_attribute(self, name: str, /) -> Object:
        try:
            candidate = self._attributes[name]
        except KeyError:
            try:
                return self._scope.get_object(name)
            except KeyError:
                pass
            for base in self._bases:
                try:
                    candidate = base.get_attribute(name)
                except KeyError:
                    continue
                else:
                    if candidate.kind is ObjectKind.ROUTINE and (
                        self.kind is ObjectKind.CLASS
                        and base.kind is ObjectKind.METACLASS
                    ):
                        candidate = PlainObject(
                            ObjectKind.INSTANCE_ROUTINE,
                            self.module_path,
                            self.local_path.join(name),
                            candidate,
                        )
                    return candidate
            if (metaclass := self._metaclass) is not None:
                try:
                    return metaclass.get_attribute(name)
                except KeyError:
                    pass
            if self.kind is ObjectKind.UNKNOWN_CLASS:
                assert name not in self._attributes
                self._attributes[name] = result = PlainObject(
                    ObjectKind.UNKNOWN,
                    self.module_path,
                    self.local_path.join(name),
                )
                return result
            raise
        else:
            if (
                self.kind is ObjectKind.CLASS
                and candidate.kind is ObjectKind.ROUTINE
                and (
                    name
                    in (
                        object.__init_subclass__.__name__,
                        object.__new__.__name__,
                    )
                )
            ):
                candidate = PlainObject(
                    ObjectKind.INSTANCE_ROUTINE,
                    self.module_path,
                    self.local_path.join(name),
                    candidate,
                )
            return candidate

    def safe_delete_value(self, name: str, /) -> bool:
        assert isinstance(name, str), name
        return self._values.pop(name, MISSING) is not MISSING

    def safe_delete_nested_value(self, local_path: LocalObjectPath, /) -> bool:
        assert isinstance(local_path, LocalObjectPath), local_path
        object_: Object = self
        for component in local_path.components[:-1]:
            object_ = object_.get_attribute(component)
        return object_.safe_delete_value(local_path.components[-1])

    def set_attribute(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._attributes[name] = object_

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        self.get_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

    def set_value(self, name: str, value: Any | Missing, /) -> None:
        assert isinstance(name, str), name
        assert name in self._attributes
        if value is MISSING:
            assert name in self._values
            self._values.pop(name, None)
        self._values[name] = value

    def set_nested_value(
        self, local_path: LocalObjectPath, value: Any, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(self.get_nested_attribute(local_path), PlainObject)
        assert value is not MISSING
        assert len(local_path.components) > 0
        object_: Object = self
        for component in local_path.components[:-1]:
            if component not in object_._values:  # noqa: SLF001
                object_._values[component] = (  # noqa: SLF001
                    object_.get_attribute(component).as_object()
                )
            object_ = object_.get_attribute(component)
        object_._values[local_path.components[-1]] = value  # noqa: SLF001

    def strict_get_attribute(self, name: str, /) -> Object:
        try:
            return self._attributes[name]
        except KeyError:
            try:
                return self._scope.strict_get_object(name)
            except KeyError:
                for base in self._bases:
                    try:
                        return base.strict_get_attribute(name)
                    except KeyError:
                        continue
                if (metaclass := self._metaclass) is not None:
                    try:
                        return metaclass.strict_get_attribute(name)
                    except KeyError:
                        pass
            raise

    _attributes: dict[str, Object]
    _bases: list[Object]
    _metaclass: Object | None
    _scope: Scope
    _values: dict[str, Any]

    __slots__ = '_attributes', '_bases', '_metaclass', '_scope', '_values'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._attributes == other._attributes
                and self._scope == other._scope
                and self._values == other._values
                and self._bases == other._bases
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self, scope: Scope, /, *bases: Object, metaclass: Object | None
    ) -> None:
        assert scope.kind in CLASS_SCOPE_KINDS, scope
        (
            self._attributes,
            self._bases,
            self._metaclass,
            self._scope,
            self._values,
        ) = {}, list(bases), metaclass, scope, {}

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._scope!r}'
            f'{", " * bool(self._bases)}'
            f'{", ".join(map(repr, self._bases))}'
            ')'
        )


Object: TypeAlias = Class | Module | PlainObject


CLASS_OBJECT_KINDS: Final = (
    ObjectKind.CLASS,
    ObjectKind.METACLASS,
    ObjectKind.UNKNOWN_CLASS,
)
CLASS_SCOPE_KINDS: Final = (
    ScopeKind.CLASS,
    ScopeKind.METACLASS,
    ScopeKind.UNKNOWN_CLASS,
)


def _object_get_attribute(object_: Object, name: str, /) -> Object:
    return object_.get_attribute(name)
