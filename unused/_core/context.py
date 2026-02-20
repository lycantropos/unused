from typing import TypeAlias

from typing_extensions import Self

from .object_path import ModulePath


class NullContext:
    __slots__ = ()


class FunctionCallContext:
    @property
    def caller_module_path(self, /) -> ModulePath:
        return self._caller_module_path

    _caller_module_path: ModulePath

    __slots__ = ('_caller_module_path',)

    def __new__(cls, caller_module_path: ModulePath, /) -> Self:
        self = super().__new__(cls)
        self._caller_module_path = caller_module_path
        return self


Context: TypeAlias = FunctionCallContext | NullContext
