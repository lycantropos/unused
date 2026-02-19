from __future__ import annotations

import builtins
import functools
import types
from typing import Any

from .missing import MISSING, Missing


@functools.singledispatch
def to_safe(_value: Any, /) -> Any | Missing:
    return MISSING


@to_safe.register(types.EllipsisType)
@to_safe.register(types.NoneType)
@to_safe.register(builtins.bool)
@to_safe.register(builtins.bytearray)
@to_safe.register(builtins.bytes)
@to_safe.register(builtins.float)
@to_safe.register(builtins.int)
@to_safe.register(builtins.slice)
@to_safe.register(builtins.str)
def _(value: Any, /) -> Any:
    return value


@to_safe.register(dict)
def _(value: dict[Any, Any], /) -> dict[Any, Any | Missing]:
    return {
        key: to_safe(item_value)
        for item_key, item_value in value.items()
        if (key := to_safe(item_key)) is not MISSING
    }


@to_safe.register(frozenset)
def _(value: frozenset[Any], /) -> frozenset[Any]:
    return frozenset(
        safe_element
        for element in value
        if (safe_element := to_safe(element)) is not MISSING
    )


@to_safe.register(list)
def _(value: list[Any], /) -> list[Any | Missing]:
    return [to_safe(element) for element in value]


@to_safe.register(set)
def _(value: set[Any], /) -> set[Any]:
    return {
        safe_element
        for element in value
        if (safe_element := to_safe(element)) is not MISSING
    }


@to_safe.register(tuple)
def _(value: tuple[Any, ...], /) -> tuple[Any | Missing, ...]:
    return tuple(to_safe(element) for element in value)
