from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

_KT = TypeVar('_KT')
_VT1 = TypeVar('_VT1')
_VT2 = TypeVar('_VT2')


class MappedMapping(Mapping[_KT, _VT2], Generic[_KT, _VT1, _VT2]):
    _function: Callable[[_VT1], _VT2]
    _mapping: Mapping[_KT, _VT1]

    def __getitem__(self, key: _KT, /) -> _VT2:
        return self._function(self._mapping[key])

    def __init__(
        self, function: Callable[[_VT1], _VT2], mapping: Mapping[_KT, _VT1], /
    ) -> None:
        self._function, self._mapping = function, mapping

    def __iter__(self, /) -> Iterator[_KT]:
        return iter(self._mapping)

    def __len__(self, /) -> int:
        return len(self._mapping)

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}('
            f'{self._function.__qualname__}, {self._mapping!r}'
            ')'
        )
