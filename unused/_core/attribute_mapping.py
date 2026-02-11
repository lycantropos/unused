from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AttributeMapping:
    def __init__(self, wrapped: Mapping[str, Any], /) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str, /) -> Any:
        try:
            return self._wrapped[name]
        except KeyError:
            raise AttributeError(name) from None
