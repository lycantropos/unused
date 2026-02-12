from collections.abc import Iterator, Mapping, Sequence
from reprlib import recursive_repr
from typing import Any, TypeVar, overload

from typing_extensions import Self, override

_KT = TypeVar('_KT')
_T = TypeVar('_T')
_VT = TypeVar('_VT')


class MappingChain(Mapping[_KT, _VT]):
    @overload
    def get(self, key: _KT, /) -> _VT: ...

    @overload
    def get(self, key: _KT, /, default: _VT) -> _VT: ...

    @overload
    def get(self, key: _KT, /, default: _T = ...) -> _VT | _T: ...

    @override
    def get(self, key: _KT, /, default: _T | None = None) -> _VT | _T | None:
        return self[key] if key in self else default  # noqa: SIM401

    _mappings: Sequence[Mapping[_KT, _VT]]

    __slots__ = ('_mappings',)

    def __new__(
        cls,
        first_mapping: Mapping[_KT, _VT],
        /,
        *rest_mappings: Mapping[_KT, _VT],
    ) -> Self:
        self = super().__new__(cls)
        self._mappings = (first_mapping, *rest_mappings)
        return self

    def __bool__(self, /) -> bool:
        return any(self._mappings)

    @override
    def __contains__(self, key: Any, /) -> bool:
        return any(key in mapping for mapping in self._mappings)

    @override
    def __getitem__(self, key: _KT, /) -> _VT:
        for mapping in self._mappings:
            try:
                return mapping[key]
            except KeyError:
                continue
        raise KeyError(key)

    @override
    def __iter__(self, /) -> Iterator[_KT]:
        key_dict = {}
        for mapping in reversed(self._mappings):
            key_dict.update(dict.fromkeys(mapping))
        return iter(key_dict)

    @override
    def __len__(self, /) -> int:
        return len(set().union(*self._mappings))

    @recursive_repr()
    @override
    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}'
            '('
            f'{", ".join(map(repr, self._mappings))}'
            ')'
        )
