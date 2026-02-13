from __future__ import annotations

import builtins
import functools
import types
from typing import Any, Final

from typing_extensions import Self

_VALUES: Final[list[Any]] = []


class EscapedValue:
    _value_index: int

    __slots__ = ('_value_index',)

    def __new__(cls, value: Any, /) -> Self:
        self = super().__new__(cls)
        self._value_index = len(_VALUES)
        _VALUES.append(value)
        return self

    def __eq__(self, other: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index] == other)

    def __ge__(self, other: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index] >= other)

    def __gt__(self, other: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index] > other)

    def __le__(self, other: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index] <= other)

    def __lt__(self, other: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index] < other)

    def __ne__(self, other: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index] != other)

    def __hash__(self, /) -> int:
        result = hash(_VALUES[self._value_index])
        assert isinstance(result, int), result
        return result

    def __getattr__(self, name: Any, /) -> Any:
        return escape_value(getattr(_VALUES[self._value_index], name))

    def __getitem__(self, item: Any, /) -> Any:
        return escape_value(_VALUES[self._value_index][item])

    def __repr__(self, /) -> str:
        result = repr(_VALUES[self._value_index])
        assert isinstance(result, str)
        return result


@functools.singledispatch
def escape_value(value: Any, /) -> Any:
    return EscapedValue(value)


@escape_value.register(types.NoneType)
@escape_value.register(builtins.bool)
@escape_value.register(builtins.bytearray)
@escape_value.register(builtins.bytes)
@escape_value.register(builtins.float)
@escape_value.register(builtins.int)
@escape_value.register(builtins.str)
def _(value: Any, /) -> Any:
    return value


@escape_value.register(dict)
def _(value: dict[Any, Any], /) -> Any:
    return {
        escape_value(item_key): escape_value(item_value)
        for item_key, item_value in value.items()
    }


@escape_value.register(list)
def _(value: list[Any], /) -> Any:
    return [escape_value(element) for element in value]


@escape_value.register(tuple)
def _(value: tuple[Any, ...], /) -> Any:
    if type(value) is not tuple:
        return EscapedValue(value)
    return tuple(escape_value(element) for element in value)


@escape_value.register(set)
def _(value: set[Any], /) -> Any:
    return {escape_value(element) for element in value}
