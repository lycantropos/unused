from __future__ import annotations

import types
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

    @property
    def __dict__(self, /) -> Mapping[str, Any]:  # type: ignore[override]
        return types.MappingProxyType(self._wrapped)

    @property
    def __module__(self, /) -> str:  # type: ignore[override]
        result = self._wrapped['__module__']
        assert isinstance(result, str), result
        return result

    def __getattr__(self, name: str, /) -> Any:
        try:
            return self._wrapped[name]
        except KeyError:
            raise AttributeError(name) from None
