from __future__ import annotations

import inspect
from typing import Any

from typing_extensions import Self

from .missing import Missing
from .object_ import ObjectKind
from .object_path import LocalObjectPath, ModulePath


class DependencyNode:
    @property
    def dependant_local_path(self, /) -> LocalObjectPath:
        return self._dependant_local_path

    @property
    def dependant_module_path(self, /) -> ModulePath:
        return self._dependant_module_path

    @property
    def local_path(self, /) -> LocalObjectPath:
        return self._local_path

    @property
    def module_path(self, /) -> ModulePath:
        return self._module_path

    @property
    def object_kind(self, /) -> ObjectKind:
        return self._object_kind

    @property
    def value(self, /) -> Any | Missing:
        return self._value

    _dependant_local_path: LocalObjectPath
    _dependant_module_path: ModulePath
    _local_path: LocalObjectPath
    _module_path: ModulePath
    _object_kind: ObjectKind
    _value: Any | Missing

    __slots__ = (
        '_dependant_local_path',
        '_dependant_module_path',
        '_local_path',
        '_module_path',
        '_object_kind',
        '_value',
    )

    def __new__(
        cls,
        object_kind: ObjectKind,
        module_path: ModulePath,
        local_path: LocalObjectPath,
        /,
        *,
        dependant_local_path: LocalObjectPath,
        dependant_module_path: ModulePath,
        value: Any | Missing,
    ) -> Self:
        assert not inspect.isclass(value)
        assert not inspect.ismodule(value)
        self = super().__new__(cls)
        (
            self._dependant_local_path,
            self._dependant_module_path,
            self._local_path,
            self._module_path,
            self._object_kind,
            self._value,
        ) = (
            dependant_local_path,
            dependant_module_path,
            local_path,
            module_path,
            object_kind,
            value,
        )
        return self

    def __eq__(self, other: Any, /) -> Any:
        return (
            (
                self._object_kind is other._object_kind
                and self._local_path == other._local_path
                and self._module_path == other._module_path
            )
            if isinstance(other, type(self))
            else NotImplemented
        )

    def __hash__(self, /) -> int:
        return hash((self._object_kind, self._local_path, self._module_path))

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._object_kind!r}, '
            f'{self._module_path!r}, '
            f'{self._local_path!r}, '
            f'{self._dependant_module_path!r}, '
            f'{self._dependant_local_path!r}'
            f')'
        )
