from __future__ import annotations

from typing import Any, TypeVar

_T = TypeVar('_T')


def ensure_type(value: Any, cls: type[_T]) -> _T:
    assert isinstance(value, cls)
    return value
