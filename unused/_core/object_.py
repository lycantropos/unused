from __future__ import annotations

import ast
import functools
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    get_args,
)

from .enums import ObjectKind, ScopeKind
from .missing import MISSING, Missing
from .object_path import (
    CLASS_FIELD_NAME,
    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
)
from .utils import AnyFunctionDefinitionAstNode, ensure_type

if TYPE_CHECKING:
    from .scope import Scope

_T = TypeVar('_T')


class Class:
    @property
    def bases(self, /) -> Sequence[ClassObject]:
        return self._bases

    @property
    def kind(self, /) -> ClassObjectKind:
        if self._scope.kind is ScopeKind.CLASS:
            return ObjectKind.CLASS
        if self._scope.kind is ScopeKind.METACLASS:
            return ObjectKind.METACLASS
        assert self._scope.kind is ScopeKind.UNKNOWN_CLASS
        return ObjectKind.UNKNOWN_CLASS

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._scope.local_path

    @property
    def metacls(self, /) -> ClassObject | Missing:
        return self._metacls

    @property
    def module_path(self, /) -> ModulePath:
        return self._scope.module_path

    @property
    def value(self, /) -> Any:
        raise NameError(self._scope.local_path.components[-1])

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
                    return base.get_attribute(name)
                except KeyError:
                    continue
            if (metaclass := self._metacls) is not MISSING:
                assert self.kind is ObjectKind.CLASS, self
                try:
                    candidate = metaclass.get_attribute(name)
                except KeyError:
                    pass
                else:
                    if candidate.kind is ObjectKind.ROUTINE:
                        candidate = Method(candidate, self)
                    return candidate
            if self.kind is ObjectKind.UNKNOWN_CLASS:
                assert name not in self._attributes
                self._attributes[name] = result = UnknownObject(
                    self.module_path, self.local_path.join(name), value=MISSING
                )
                return result
            raise
        else:
            if candidate.kind is ObjectKind.DESCRIPTOR:
                return UnknownObject(
                    self.module_path, candidate.local_path, value=MISSING
                )
            return candidate

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
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
        self.get_mutable_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

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
                if (metaclass := self._metacls) is not MISSING:
                    try:
                        return metaclass.strict_get_attribute(name)
                    except KeyError:
                        pass
            raise

    _attributes: dict[str, Object]
    _bases: Sequence[ClassObject]
    _metacls: ClassObject | Missing
    _scope: Scope

    __slots__ = '_attributes', '_bases', '_metacls', '_scope'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._attributes == other._attributes
                and self._scope == other._scope
                and self._bases == other._bases
                and self._metacls == other._metacls
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        scope: Scope,
        /,
        *bases: ClassObject,
        metacls: ClassObject | Missing,
    ) -> None:
        assert scope.kind in CLASS_SCOPE_KINDS, scope
        assert [
            base_index
            for base_index, base in enumerate(bases)
            if base.kind in (ObjectKind.UNKNOWN, ObjectKind.UNKNOWN_CLASS)
        ] <= [len(bases) - 1]
        self._attributes, self._bases, self._metacls, self._scope = (
            {},
            bases,
            metacls,
            scope,
        )

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._scope!r}'
            f'{", " * bool(self._bases)}'
            f'{", ".join(map(repr, self._bases))}, '
            f'metaclass={self._metacls}'
            ')'
        )


class Instance:
    @property
    def cls(self, /) -> Class | UnknownObject:
        return self._cls

    @property
    def kind(self, /) -> Literal[ObjectKind.INSTANCE]:
        return ObjectKind.INSTANCE

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def value(self, /) -> Any:
        if self._value is MISSING:
            raise NameError(self._local_path.components[-1])
        return self._value

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def get_attribute(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            try:
                candidate = self._cls.get_attribute(name)
            except KeyError:
                pass
            else:
                if (
                    self._cls.kind is ObjectKind.CLASS
                    and candidate.kind is ObjectKind.ROUTINE
                ):
                    candidate = Method(candidate, self)
                if (
                    self._cls.kind is ObjectKind.CLASS
                    and candidate.kind is ObjectKind.DESCRIPTOR
                ):
                    candidate = UnknownObject(
                        self._module_path, candidate.local_path, value=MISSING
                    )
                return candidate
            assert name not in self._objects
            self._objects[name] = result = UnknownObject(
                self.module_path, self.local_path.join(name), value=MISSING
            )
            return result

    def set_attribute(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._objects[name] = object_

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        self.get_mutable_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

    def strict_get_attribute(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            try:
                return self._cls.strict_get_attribute(name)
            except KeyError:
                pass
            raise

    _cls: Class | UnknownObject
    _local_path: LocalObjectPath
    _module_path: ModulePath
    _objects: dict[str, Object]
    _value: Any | Missing

    __slots__ = '_cls', '_local_path', '_module_path', '_objects', '_value'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._objects == other._objects
                and self._value == other._value
                and self._cls == other._cls
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
        *,
        cls: Class | UnknownObject,
        value: Any | Missing,
    ) -> None:
        (
            self._cls,
            self._local_path,
            self._module_path,
            self._objects,
            self._value,
        ) = cls, local_path, module_path, {}, value

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._module_path!r}, {self._local_path!r}, cls={self._cls}'
            ')'
        )


class Call:
    @property
    def callable_(self, /) -> Object:
        return self._callable

    @property
    def kind(self, /) -> Literal[ObjectKind.ROUTINE_CALL]:
        return ObjectKind.ROUTINE_CALL

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def value(self, /) -> Any:
        raise NameError(self._local_path.components[-1])

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def get_attribute(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            assert name not in self._objects
            self._objects[name] = result = UnknownObject(
                self.module_path, self.local_path.join(name), value=MISSING
            )
            return result

    def set_attribute(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._objects[name] = object_

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        self.get_mutable_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

    def strict_get_attribute(self, name: str, /) -> Object:
        return self._objects[name]

    _keyword_arguments: Sequence[tuple[str | None, Object]]
    _local_path: LocalObjectPath
    _module_path: ModulePath
    _callable: Object
    _objects: dict[str, Object]
    _positional_arguments: Sequence[tuple[Object] | Object]

    __slots__ = (
        '_callable',
        '_keyword_arguments',
        '_local_path',
        '_module_path',
        '_objects',
        '_positional_arguments',
    )

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._module_path == other._module_path
                and self._positional_arguments == other._positional_arguments
                and self._callable == other._callable
                and self._keyword_arguments == other._keyword_arguments
                and self._objects == other._objects
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        callable_: Object,
        positional_arguments: Sequence[tuple[Object] | Object],
        keyword_arguments: Sequence[tuple[str | None, Object]],
        /,
    ) -> None:
        (
            self._keyword_arguments,
            self._local_path,
            self._module_path,
            self._objects,
            self._positional_arguments,
            self._callable,
        ) = (
            keyword_arguments,
            local_path,
            module_path,
            {},
            positional_arguments,
            callable_,
        )

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._module_path!r}, {self._local_path!r}'
            ')'
        )


class Method:
    @property
    def instance(self, /) -> Object:
        return self._instance

    @property
    def kind(self, /) -> Literal[ObjectKind.METHOD]:
        return ObjectKind.METHOD

    @property
    def module_path(self, /) -> ModulePath:
        return self._instance.module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._instance.local_path.join(
            self._routine.local_path.components[-1]
        )

    @property
    def routine(self, /) -> CallableObject:
        return self._routine

    @property
    def value(self, /) -> Any:
        raise NameError(self.local_path.components[-1])

    def get_attribute(self, name: str, /) -> Object:
        return self.strict_get_attribute(name)

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def strict_get_attribute(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            try:
                candidate = self.BASE_CLS.get_attribute(name)
            except KeyError:
                pass
            else:
                if candidate.kind is ObjectKind.ROUTINE:
                    candidate = type(self)(candidate, self)
                return candidate
            raise

    BASE_CLS: ClassVar[Class]

    _instance: Object
    _objects: dict[str, Object]
    _routine: CallableObject

    __slots__ = '_instance', '_objects', '_routine'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._routine == other._routine
                and self._instance == other._instance
                and self._objects == other._objects
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(self, routine: CallableObject, instance: Object, /) -> None:
        self._instance, self._objects, self._routine = (
            instance,
            {'__self__': instance, '__func__': routine},
            routine,
        )

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}({self._routine!r}, {self._instance!r})'
        )


class Routine:
    @property
    def ast_node(self, /) -> AnyFunctionDefinitionAstNode | ast.Lambda | None:
        return self._ast_node

    @property
    def kind(self, /) -> Literal[ObjectKind.ROUTINE]:
        return ObjectKind.ROUTINE

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def value(self, /) -> Any:
        raise NameError(self._local_path.components[-1])

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def get_attribute(self, name: str, /) -> Object:
        return self.strict_get_attribute(name)

    def set_attribute(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._objects[name] = object_

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        self.get_mutable_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

    def strict_get_attribute(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            for base_cls in self._base_classes:
                try:
                    candidate = base_cls.get_attribute(name)
                except KeyError:
                    continue
                else:
                    if candidate.kind is ObjectKind.ROUTINE:
                        candidate = Method(candidate, self)
                    return candidate
            raise

    _ast_node: AnyFunctionDefinitionAstNode | ast.Lambda | None
    _base_classes: Sequence[Class | UnknownObject]
    _module_path: ModulePath
    _local_path: LocalObjectPath
    _objects: dict[str, Object]

    __slots__ = (
        '_ast_node',
        '_base_classes',
        '_local_path',
        '_module_path',
        '_objects',
    )

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._objects == other._objects
                and self._base_classes == other._base_classes
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
        *base_classes: Class | UnknownObject,
        ast_node: AnyFunctionDefinitionAstNode | ast.Lambda | None,
        keyword_only_defaults: Mapping[Any, Any],
        positional_defaults: Sequence[Any],
    ) -> None:
        (
            self._ast_node,
            self._base_classes,
            self._local_path,
            self._module_path,
            self._objects,
        ) = (
            ast_node,
            base_classes,
            local_path,
            module_path,
            {
                FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME: UnknownObject(
                    module_path,
                    local_path.join(FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME),
                    value=positional_defaults,
                ),
                FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME: UnknownObject(
                    module_path,
                    local_path.join(FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME),
                    value=keyword_only_defaults,
                ),
            },
        )

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._base_classes)}'
            f'{", ".join(map(repr, self._base_classes))}'
            ')'
        )


class Descriptor:
    @property
    def ast_node(self, /) -> AnyFunctionDefinitionAstNode | None:
        return self._ast_node

    @property
    def kind(self, /) -> Literal[ObjectKind.DESCRIPTOR]:
        return ObjectKind.DESCRIPTOR

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def value(self, /) -> Any:
        raise NameError(self._local_path.components[-1])

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def get_attribute(self, name: str, /) -> Object:
        return self.strict_get_attribute(name)

    def strict_get_attribute(self, name: str, /) -> Object:
        for base_cls in self._base_classes:
            try:
                candidate = base_cls.get_attribute(name)
            except KeyError:
                continue
            else:
                if candidate.kind is ObjectKind.ROUTINE:
                    candidate = Method(candidate, self)
                return candidate
        raise KeyError(name)

    _ast_node: AnyFunctionDefinitionAstNode | None
    _base_classes: Sequence[Class | UnknownObject]
    _local_path: LocalObjectPath
    _module_path: ModulePath

    __slots__ = '_ast_node', '_base_classes', '_local_path', '_module_path'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._base_classes == other._base_classes
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
        *base_classes: Class | UnknownObject,
        ast_node: AnyFunctionDefinitionAstNode | None,
    ) -> None:
        (
            self._ast_node,
            self._base_classes,
            self._local_path,
            self._module_path,
        ) = ast_node, base_classes, local_path, module_path

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._module_path!r}, {self._local_path!r}'
            f'{", " * bool(self._base_classes)}'
            f'{", ".join(map(repr, self._base_classes))}'
            ')'
        )


class Module:
    BASE_CLS: ClassVar[Class | None] = None

    @property
    def kind(
        self, /
    ) -> Literal[
        ObjectKind.BUILTIN_MODULE,
        ObjectKind.DYNAMIC_MODULE,
        ObjectKind.EXTENSION_MODULE,
        ObjectKind.STATIC_MODULE,
    ]:
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
    def value(self, /) -> Any:
        raise NameError(self.local_path.components[-1])

    def get_attribute(self, name: str, /) -> Object:
        if name == CLASS_FIELD_NAME:
            assert self.BASE_CLS is not None, self
            return self.BASE_CLS
        assert isinstance(name, str), name
        try:
            return self._scope.get_object(name)
        except KeyError:
            if (base_cls := self.BASE_CLS) is not None:
                try:
                    candidate = base_cls.get_attribute(name)
                except KeyError:
                    pass
                else:
                    if candidate.kind is ObjectKind.DESCRIPTOR:
                        return UnknownObject(
                            self.module_path,
                            candidate.local_path,
                            value=MISSING,
                        )
                    if candidate.kind is ObjectKind.ROUTINE:
                        return Method(candidate, self)
                    return candidate
            if self.kind in (
                ObjectKind.BUILTIN_MODULE,
                ObjectKind.DYNAMIC_MODULE,
                ObjectKind.EXTENSION_MODULE,
            ):
                result = UnknownObject(
                    self.module_path, self.local_path.join(name), value=MISSING
                )
                self._scope.set_object(name, result)
                return result
            raise

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def set_attribute(self, name: str, object_: Object, /) -> None:
        self._scope.set_object(name, object_)

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        self._scope.set_nested_object(local_path, object_)

    def strict_get_attribute(self, name: str, /) -> Object:
        return self._scope.strict_get_object(name)

    _scope: Scope

    __slots__ = ('_scope',)

    def __eq__(self, other: Any, /) -> Any:
        return (
            self._scope == other._scope
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(self, scope: Scope, /) -> None:
        self._scope = scope

    def __repr__(self, /) -> str:
        return f'{type(self).__qualname__}({self._scope!r})'


class UnknownObject:
    @property
    def kind(self, /) -> Literal[ObjectKind.UNKNOWN]:
        return ObjectKind.UNKNOWN

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    @property
    def value(self, /) -> Any:
        if self._value is MISSING:
            raise NameError(self._local_path.components[-1])
        return self._value

    def get_mutable_attribute(self, name: str, /) -> MutableObject:
        return ensure_type(self.get_attribute(name), MUTABLE_OBJECT_CLASSES)

    def get_mutable_nested_attribute(
        self, local_path: LocalObjectPath, /
    ) -> MutableObject:
        return ensure_type(
            self.get_nested_attribute(local_path), MUTABLE_OBJECT_CLASSES
        )

    def get_nested_attribute(self, local_path: LocalObjectPath, /) -> Object:
        assert isinstance(local_path, LocalObjectPath), local_path
        initial_object: Object = self
        return functools.reduce(
            object_get_attribute, local_path.components, initial_object
        )

    def get_attribute(self, name: str, /) -> Object:
        try:
            return self._objects[name]
        except KeyError:
            assert name not in self._objects
            self._objects[name] = result = type(self)(
                self.module_path, self.local_path.join(name), value=MISSING
            )
            return result

    def set_attribute(self, name: str, object_: Object, /) -> None:
        assert isinstance(name, str), (name, object_)
        assert isinstance(object_, Object), (name, object_)
        self._objects[name] = object_

    def set_nested_attribute(
        self, local_path: LocalObjectPath, object_: Object, /
    ) -> None:
        assert isinstance(local_path, LocalObjectPath), local_path
        assert isinstance(object_, Object), object_
        self.get_mutable_nested_attribute(local_path.parent).set_attribute(
            local_path.components[-1], object_
        )

    def strict_get_attribute(self, name: str, /) -> Object:
        return self._objects[name]

    _module_path: ModulePath
    _local_path: LocalObjectPath
    _objects: dict[str, Object]
    _value: Any | Missing

    __slots__ = '_local_path', '_module_path', '_objects', '_value'

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._module_path == other._module_path
                and self._local_path == other._local_path
                and self._objects == other._objects
                and self._value == other._value
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __init__(
        self,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
        *,
        value: Any | Missing,
    ) -> None:
        self._local_path, self._module_path, self._objects, self._value = (
            local_path,
            module_path,
            {},
            value,
        )

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._module_path!r}, {self._local_path!r}'
            ')'
        )


ClassObject: TypeAlias = Call | Class | Instance | Routine | UnknownObject
CLASS_OBJECT_CLASSES: Final = (Call, Class, Instance, Routine, UnknownObject)
assert get_args(ClassObject) == CLASS_OBJECT_CLASSES
MutableObject: TypeAlias = (
    Call | Class | Instance | Module | Routine | UnknownObject
)
MUTABLE_OBJECT_CLASSES: Final = (
    Call,
    Class,
    Instance,
    Module,
    Routine,
    UnknownObject,
)
assert get_args(MutableObject) == MUTABLE_OBJECT_CLASSES
ImmutableObject: TypeAlias = Descriptor | Method
Object: TypeAlias = ImmutableObject | MutableObject
CallableObject: TypeAlias = (
    Call | Class | Descriptor | Instance | Method | Routine | UnknownObject
)
CALLABLE_OBJECT_CLASSES: Final = (
    Call,
    Class,
    Descriptor,
    Instance,
    Method,
    Routine,
    UnknownObject,
)
assert get_args(CallableObject) == CALLABLE_OBJECT_CLASSES


ClassObjectKind: TypeAlias = Literal[
    ObjectKind.CLASS, ObjectKind.METACLASS, ObjectKind.UNKNOWN_CLASS
]
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


def object_get_attribute(object_: Object, name: str, /) -> Object:
    return object_.get_attribute(name)


def to_object_value(object_: Object, /) -> Any:
    return object_.value
