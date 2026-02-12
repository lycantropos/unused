from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from itertools import chain
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

_T = TypeVar('_T')


class BaseValuespace(ABC, Generic[_T]):
    @classmethod
    @abstractmethod
    def value_cls(cls, /) -> type[_T] | UnionType:
        pass

    @classmethod
    def values(cls, /) -> Iterable[_T]:
        yield from cls.__values_map.values()

    __values_map: ClassVar[
        Mapping[
            str,
            # TODO:
            #   change to `_T` once
            #   https://github.com/python/mypy/commit/ad570933924b3810ba61d2e4a13eac596f74672b
            #   is released
            Any,
        ]
    ]

    def __init_subclass__(cls, /) -> None:
        super().__init_subclass__()
        annotations = get_type_hints(cls, include_extras=True)
        value_cls = cls.value_cls()
        if (
            len(
                errors := list(
                    chain.from_iterable(
                        [
                            *(
                                (
                                    'field name '
                                    f'should be `{field_name.upper()}`, '
                                    f'but got `{field_name}`'
                                )
                                if not field_name.isupper()
                                else ()
                            ),
                            *(
                                (
                                    f'annotation for `{field_name}` should be '
                                    f'either `{ClassVar[type(field_value)]}` '
                                    f'or `{ClassVar}`, '
                                    f'but got `{annotation}`'
                                )
                                if not (
                                    (
                                        (
                                            annotation := annotations.get(
                                                field_name
                                            )
                                        )
                                        is not None
                                    )
                                    and (
                                        annotation is ClassVar
                                        or (
                                            get_origin(annotation) is ClassVar
                                            and len(
                                                field_value_types := get_args(
                                                    annotation
                                                )
                                            )
                                            == 1
                                            and isinstance(
                                                field_value,
                                                field_value_types[0],
                                            )
                                        )
                                    )
                                )
                                else ()
                            ),
                        ]
                        for field_name, field_value in vars(cls).items()
                        if isinstance(field_value, value_cls)
                    )
                )
            )
            > 0
        ):
            raise ValueError(
                f'Invalid {cls.__qualname__!r}: {"; ".join(errors)}.'
            )
        cls.__values_map = {
            field_name: field_value
            for node_cls in reversed(cls.mro())
            for field_name, field_value in vars(node_cls).items()
            if isinstance(field_value, value_cls)
        }
