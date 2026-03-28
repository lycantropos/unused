from __future__ import annotations

import ast
import uuid
from typing import Any, Final, TypeAlias, TypeVar, cast, overload

AnyFunctionDefinitionAstNode: TypeAlias = (
    ast.AsyncFunctionDef | ast.FunctionDef
)

EVALUATION_EXCEPTIONS: Final = (
    AttributeError,
    IndexError,
    KeyError,
    NameError,
    TypeError,
)

_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')
_T3 = TypeVar('_T3')
_T4 = TypeVar('_T4')
_T5 = TypeVar('_T5')
_T6 = TypeVar('_T6')
_T7 = TypeVar('_T7')


@overload
def ensure_type(value: Any, cls_or_union: type[_T], /) -> _T: ...


@overload
def ensure_type(
    value: Any, cls_or_union: tuple[type[_T1], type[_T2]], /
) -> _T1 | _T2: ...


@overload
def ensure_type(
    value: Any, cls_or_union: tuple[type[_T1], type[_T2], type[_T3]], /
) -> _T1 | _T2 | _T3: ...


@overload
def ensure_type(
    value: Any,
    cls_or_union: tuple[type[_T1], type[_T2], type[_T3], type[_T4]],
    /,
) -> _T1 | _T2 | _T3 | _T4: ...


@overload
def ensure_type(
    value: Any,
    cls_or_union: tuple[type[_T1], type[_T2], type[_T3], type[_T4], type[_T5]],
    /,
) -> _T1 | _T2 | _T3 | _T4 | _T5: ...


@overload
def ensure_type(
    value: Any,
    cls_or_union: tuple[
        type[_T1], type[_T2], type[_T3], type[_T4], type[_T5], type[_T6]
    ],
    /,
) -> _T1 | _T2 | _T3 | _T4 | _T5 | _T6: ...


@overload
def ensure_type(
    value: Any,
    cls_or_union: tuple[
        type[_T1],
        type[_T2],
        type[_T3],
        type[_T4],
        type[_T5],
        type[_T6],
        type[_T7],
    ],
    /,
) -> _T1 | _T2 | _T3 | _T4 | _T5 | _T6 | _T7: ...


def ensure_type(
    value: Any, cls_or_union: type[_T] | tuple[type[_T], ...], /
) -> _T:
    assert isinstance(value, cls_or_union), value
    return cast(_T, value)


def generate_random_identifier() -> str:
    return '__' + uuid.uuid4().hex
