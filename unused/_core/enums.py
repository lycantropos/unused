from __future__ import annotations

from enum import Enum


class ObjectKind(str, Enum):
    BUILTIN_MODULE = 'BUILTIN_MODULE'
    CLASS = 'CLASS'
    DYNAMIC_MODULE = 'DYNAMIC_MODULE'
    EXTENSION_MODULE = 'EXTENSION_MODULE'
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
