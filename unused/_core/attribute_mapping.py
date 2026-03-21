from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typing_extensions import Self


class AttributeMapping:
    _wrapped: Mapping[str, Any]

    __slots__ = ('_wrapped',)

    def __new__(cls, wrapped: Mapping[str, Any], /) -> Self:
        self = super().__new__(cls)
        self._wrapped = wrapped
        return self

    def __eq__(self, other: Any, /) -> Any:
        return (
            self._wrapped == other._wrapped
            if isinstance(other, AttributeMapping)
            else NotImplemented
        )

    def __format__(self, format_spec: str, /) -> str:
        raise TypeError

    def __getattr__(self, name: str, /) -> Any:
        try:
            return self._wrapped[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self, /) -> str:
        return f'{type(self).__qualname__}({self._wrapped!r})'
