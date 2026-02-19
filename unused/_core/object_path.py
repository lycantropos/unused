from __future__ import annotations

import builtins
import collections
import functools
import sys
import types
from collections.abc import Iterable, Sequence
from itertools import accumulate
from typing import Any, ClassVar, Final

from typing_extensions import Self


class ModulePath:
    COMPONENT_SEPARATOR: ClassVar = '.'

    @classmethod
    def from_module_name(cls, name: str, /) -> Self:
        return cls(*name.split(cls.COMPONENT_SEPARATOR))

    @classmethod
    def checked_from_module_name(cls, name: str, /) -> Self | None:
        components = name.split(cls.COMPONENT_SEPARATOR)
        try:
            return cls(*components)
        except ValueError:
            return None

    @property
    def components(self, /) -> Sequence[str]:
        return self._components

    def join(self, /, *components: str) -> Self:
        return type(self)(*self._components, *components)

    def submodule_paths(self, /) -> Iterable[Self]:
        cls = type(self)
        return accumulate(
            self.components[1:], cls.join, initial=cls(self.components[0])
        )

    def to_module_name(self, /) -> str:
        return self.COMPONENT_SEPARATOR.join(self.components)

    _components: tuple[str, ...]

    __slots__ = ('_components',)

    def __new__(cls, first_component: str, /, *rest_components: str) -> Self:
        components = (first_component, *rest_components)
        if (
            len(
                invalid_components := [
                    component
                    for component in components
                    if not _is_object_path_component_valid(component)
                ]
            )
            > 0
        ):
            raise ValueError(
                f'Following module path components are invalid: '
                f'{", ".join(map(repr, invalid_components))}.'
            )
        self = super().__new__(cls)
        self._components = components
        return self

    def __eq__(self, other: Any, /) -> Any:
        return (
            self._components == other._components
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __hash__(self, /) -> int:
        return hash(self._components)

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}'
            f'({", ".join(map(repr, self._components))})'
        )


class LocalObjectPath:
    COMPONENT_SEPARATOR: ClassVar = '.'

    @classmethod
    def from_object_name(cls, name: str, /) -> Self:
        return cls(*name.split(cls.COMPONENT_SEPARATOR))

    @property
    def components(self, /) -> Sequence[str]:
        return self._components

    @property
    def parent(self, /) -> Self:
        assert len(self._components) > 0, self
        return type(self)(*self._components[:-1])

    def join(self, /, *components: str) -> Self:
        return type(self)(*self._components, *components)

    def starts_with(self, other: Self, /) -> bool:
        return (
            len(other._components) <= len(self._components)  # noqa: SLF001
            and self._components[: len(other._components)] == other._components  # noqa: SLF001
        )

    def to_object_name(self, /) -> str:
        return self.COMPONENT_SEPARATOR.join(self.components)

    _components: tuple[str, ...]

    __slots__ = ('_components',)

    def __new__(cls, /, *components: str) -> Self:
        if (
            len(
                invalid_components := [
                    component
                    for component in components
                    if not _is_object_path_component_valid(component)
                ]
            )
            > 0
        ):
            raise ValueError(
                f'Following local object path components are invalid: '
                f'{", ".join(map(repr, invalid_components))}.'
            )
        self = super().__new__(cls)
        self._components = components
        return self

    def __eq__(self, other: Any, /) -> Any:
        return (
            self._components == other._components
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __hash__(self, /) -> int:
        return hash(self._components)

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}'
            f'({", ".join(map(repr, self._components))})'
        )


def _is_object_path_component_valid(component: str, /) -> bool:
    return (
        isinstance(component, str)
        and len(component) > 0
        and component.isidentifier()
    )


BUILTINS_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    builtins.__name__
)
COLLECTIONS_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    collections.__name__
)
SYS_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(sys.__name__)
TYPES_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    types.__name__
)

DICT_FIELD_NAME: Final = '__dict__'


def _search_local_path(
    local_path: LocalObjectPath, module: types.ModuleType, /
) -> Any:
    return functools.reduce(builtins.getattr, local_path.components, module)


BUILTINS_DICT_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'dict'
)
assert (
    _search_local_path(BUILTINS_DICT_LOCAL_OBJECT_PATH, builtins)
    is builtins.dict
)
BUILTINS_FROZENSET_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'frozenset'
)
assert (
    _search_local_path(BUILTINS_FROZENSET_LOCAL_OBJECT_PATH, builtins)
    is builtins.frozenset
)
BUILTINS_GLOBALS_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(builtins.globals.__qualname__)
)
assert (
    _search_local_path(BUILTINS_GLOBALS_LOCAL_OBJECT_PATH, builtins)
    is builtins.globals
)
BUILTINS_LIST_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'list'
)
assert (
    _search_local_path(BUILTINS_LIST_LOCAL_OBJECT_PATH, builtins)
    is builtins.list
)
BUILTINS_SET_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath('set')
assert (
    _search_local_path(BUILTINS_SET_LOCAL_OBJECT_PATH, builtins)
    is builtins.set
)
BUILTINS_TUPLE_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'tuple'
)
assert (
    _search_local_path(BUILTINS_TUPLE_LOCAL_OBJECT_PATH, builtins)
    is builtins.tuple
)
BUILTINS_TYPE_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(builtins.type.__qualname__)
)
assert (
    _search_local_path(BUILTINS_TYPE_LOCAL_OBJECT_PATH, builtins)
    is builtins.type
)
COLLECTIONS_NAMEDTUPLE_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(collections.namedtuple.__qualname__)
)
assert (
    _search_local_path(COLLECTIONS_NAMEDTUPLE_LOCAL_OBJECT_PATH, collections)
    is collections.namedtuple
)
SYS_MODULES_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'modules'
)
assert _search_local_path(SYS_MODULES_LOCAL_OBJECT_PATH, sys) is sys.modules
TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath('FunctionType')
)
assert (
    _search_local_path(TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH, types)
    is types.FunctionType
)
