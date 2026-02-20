from __future__ import annotations

import builtins
import functools
import types
from typing import Any


@functools.singledispatch
def is_safe(_value: Any, /) -> bool:
    return False


@is_safe.register(types.EllipsisType)
@is_safe.register(types.NoneType)
@is_safe.register(builtins.bool)
@is_safe.register(builtins.bytearray)
@is_safe.register(builtins.bytes)
@is_safe.register(builtins.float)
@is_safe.register(builtins.int)
@is_safe.register(builtins.slice)
@is_safe.register(builtins.str)
def _(_value: Any, /) -> bool:
    return True


@is_safe.register(dict)
def _(value: dict[Any, Any], /) -> bool:
    return all(
        is_safe(item_key) and is_safe(item_value)
        for item_key, item_value in value.items()
    )


@is_safe.register(frozenset)
@is_safe.register(list)
@is_safe.register(set)
@is_safe.register(tuple)
def _(value: list[Any], /) -> bool:
    return all(is_safe(element) for element in value)
