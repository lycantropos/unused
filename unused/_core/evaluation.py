from __future__ import annotations

import ast
import builtins
import functools
import inspect
import operator
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from typing import Any, Final

from .context import Context
from .enums import ObjectKind
from .missing import MISSING, Missing
from .modules import BUILTINS_MODULE, TYPES_MODULE
from .object_ import Class, ClassObject, Instance, Object, UnknownObject
from .object_path import (
    BUILTINS_BOOL_LOCAL_OBJECT_PATH,
    BUILTINS_BYTES_LOCAL_OBJECT_PATH,
    BUILTINS_COMPLEX_LOCAL_OBJECT_PATH,
    BUILTINS_DICT_LOCAL_OBJECT_PATH,
    BUILTINS_FLOAT_LOCAL_OBJECT_PATH,
    BUILTINS_FROZENSET_LOCAL_OBJECT_PATH,
    BUILTINS_INT_LOCAL_OBJECT_PATH,
    BUILTINS_LIST_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    BUILTINS_OBJECT_LOCAL_OBJECT_PATH,
    BUILTINS_SET_LOCAL_OBJECT_PATH,
    BUILTINS_SLICE_LOCAL_OBJECT_PATH,
    BUILTINS_STR_LOCAL_OBJECT_PATH,
    BUILTINS_TUPLE_LOCAL_OBJECT_PATH,
    BUILTINS_TYPE_LOCAL_OBJECT_PATH,
    LocalObjectPath,
    ModulePath,
    TYPES_ELLIPSIS_TYPE_LOCAL_OBJECT_PATH,
    TYPES_NONE_TYPE_LOCAL_OBJECT_PATH,
)
from .scope import Scope
from .utils import ensure_type, generate_random_identifier

EVALUATION_EXCEPTIONS: Final = (
    AttributeError,
    IndexError,
    KeyError,
    NameError,
    TypeError,
)


@functools.singledispatch
def evaluate_expression_node(
    node: ast.expr,
    _scope: Scope,
    /,
    *_parent_scopes: Scope,
    context: Context,  # noqa: ARG001
) -> Object:
    raise TypeError(type(node))


@evaluate_expression_node.register(ast.Attribute)
def _(
    node: ast.Attribute,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object:
    return evaluate_expression_node(
        node.value, scope, *parent_scopes, context=context
    ).get_attribute(node.attr)


@evaluate_expression_node.register(ast.JoinedStr)
def _(
    node: ast.JoinedStr,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object:
    return value_to_object(
        ''.join(
            evaluate_expression_node(
                value_node, scope, *parent_scopes, context=context
            ).value
            for value_node in node.values
        ),
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


@evaluate_expression_node.register(ast.FormattedValue)
def _(
    node: ast.FormattedValue,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object:
    value = evaluate_expression_node(
        node.value, scope, *parent_scopes, context=context
    ).value
    if node.conversion == ord('r'):
        value = repr(value)
    elif node.conversion == ord('s'):
        value = str(value)
    elif node.conversion == ord('a'):
        value = ascii(value)
    return value_to_object(
        (
            format(
                value,
                evaluate_expression_node(
                    node.format_spec, scope, *parent_scopes, context=context
                ).value,
            )
            if node.format_spec is not None
            else format(value)
        ),
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


BUILTINS_GETATTR_LOCAL_OBJECT_PATH: Final = LocalObjectPath.from_object_name(
    builtins.getattr.__qualname__
)
BUILTINS_HASATTR_LOCAL_OBJECT_PATH: Final = LocalObjectPath.from_object_name(
    builtins.hasattr.__qualname__
)
BUILTINS_ISINSTANCE_LOCAL_OBJECT_PATH: Final = (
    LocalObjectPath.from_object_name(builtins.isinstance.__qualname__)
)
BUILTINS_ISSUBCLASS_LOCAL_OBJECT_PATH: Final = (
    LocalObjectPath.from_object_name(builtins.issubclass.__qualname__)
)
BUILTINS_LEN_LOCAL_OBJECT_PATH: Final = LocalObjectPath.from_object_name(
    builtins.len.__qualname__
)


@evaluate_expression_node.register(ast.Call)
def _(
    node: ast.Call, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    callable_object = evaluate_expression_node(
        node.func, scope, *parent_scopes, context=context
    )
    positional_argument_objects: list[tuple[bool, Object]] = []
    routine_object: Object
    if callable_object.kind is ObjectKind.METHOD:
        positional_argument_objects.append((False, callable_object.instance))
        routine_object = callable_object.routine
    else:
        routine_object = callable_object
    for positional_argument_node in node.args:
        if isinstance(positional_argument_node, ast.Starred):
            positional_argument_objects.append(
                (
                    True,
                    evaluate_expression_node(
                        positional_argument_node.value,
                        scope,
                        *parent_scopes,
                        context=context,
                    ),
                )
            )
        else:
            positional_argument_objects.append(
                (
                    False,
                    evaluate_expression_node(
                        positional_argument_node,
                        scope,
                        *parent_scopes,
                        context=context,
                    ),
                )
            )
    keyword_argument_objects: list[tuple[str | None, Object]] = []
    for keyword_argument_node in node.keywords:
        if (parameter_name := keyword_argument_node.arg) is not None:
            keyword_argument_objects.append(
                (
                    parameter_name,
                    evaluate_expression_node(
                        keyword_argument_node.value,
                        scope,
                        *parent_scopes,
                        context=context,
                    ),
                )
            )
        else:
            keyword_argument_objects.append(
                (
                    None,
                    evaluate_expression_node(
                        keyword_argument_node.value,
                        scope,
                        *parent_scopes,
                        context=context,
                    ),
                )
            )
    if routine_object.module_path == BUILTINS_MODULE_PATH:
        if routine_object.local_path == BUILTINS_HASATTR_LOCAL_OBJECT_PATH:
            (
                (subject_is_variadic, subject),
                (attribute_name_object_is_variadic, attribute_name_object),
            ) = positional_argument_objects
            if (
                len(keyword_argument_objects) > 0
                or subject_is_variadic
                or attribute_name_object_is_variadic
            ):
                raise TypeError(ast.unparse(node))
            attribute_name = attribute_name_object.value
            if not isinstance(attribute_name, str):
                raise TypeError(ast.unparse(node))
            try:
                subject.strict_get_attribute(attribute_name)
            except KeyError:
                value = False
            else:
                value = True
            return value_to_object(
                value,
                module_path=scope.module_path,
                local_path=scope.local_path.join(generate_random_identifier()),
            )
        if routine_object.local_path == BUILTINS_ISINSTANCE_LOCAL_OBJECT_PATH:
            (
                (subject_is_variadic, subject),
                (cls_or_tuple_is_variadic, cls_or_tuple),
            ) = positional_argument_objects
            if (
                len(keyword_argument_objects) > 0
                or subject_is_variadic
                or cls_or_tuple_is_variadic
                or cls_or_tuple.kind is not ObjectKind.CLASS
            ):
                pass
            elif (
                subject_cls := object_to_cls(subject)
            ).kind is ObjectKind.CLASS:
                return value_to_object(
                    any(
                        parent_cls is cls_or_tuple
                        for parent_cls in cls_to_mro(subject_cls)
                    ),
                    module_path=scope.module_path,
                    local_path=scope.local_path.join(
                        generate_random_identifier()
                    ),
                )
            raise TypeError(ast.unparse(node))
        if routine_object.local_path == BUILTINS_TYPE_LOCAL_OBJECT_PATH:
            if (
                len(positional_argument_objects) != 1
                or len(keyword_argument_objects) > 0
            ):
                raise TypeError(ast.unparse(node))
            ((subject_is_variadic, subject),) = positional_argument_objects
            if not subject_is_variadic:
                if (
                    subject.kind is ObjectKind.CLASS
                    and (metacls := subject.metacls) is not MISSING
                ):
                    return metacls
                if subject.kind is ObjectKind.INSTANCE:
                    return subject.cls
            raise TypeError(ast.unparse(node))
        routine = None
        if (
            routine_object.local_path.starts_with(
                BUILTINS_BYTES_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_COMPLEX_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_DICT_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_FLOAT_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_FROZENSET_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_INT_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path == BUILTINS_LEN_LOCAL_OBJECT_PATH
            or routine_object.local_path.starts_with(
                BUILTINS_LIST_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_SET_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_STR_LOCAL_OBJECT_PATH
            )
            or routine_object.local_path.starts_with(
                BUILTINS_TUPLE_LOCAL_OBJECT_PATH
            )
        ):
            routine = reduce(
                getattr, routine_object.local_path.components, builtins
            )
        if routine is None:
            raise TypeError(ast.unparse(node))
        assert not inspect.ismodule(routine), routine
        positional_arguments: list[Any] = []
        for (
            positional_argument_is_variadic,
            positional_argument_object,
        ) in positional_argument_objects:
            if positional_argument_is_variadic:
                positional_arguments.extend(
                    [*positional_argument_object.value]
                )
            else:
                positional_arguments.append(positional_argument_object.value)
        keyword_arguments: dict[Any, Any] = {}
        for (
            keyword_argument_name,
            keyword_argument_object,
        ) in keyword_argument_objects:
            if keyword_argument_name is None:
                keyword_arguments.update({**keyword_argument_object.value})
            else:
                keyword_arguments[keyword_argument_name] = (
                    keyword_argument_object.value
                )
        return value_to_object(
            routine(*positional_arguments, **keyword_arguments),  # pyright: ignore[reportCallIssue]
            module_path=scope.module_path,
            local_path=scope.local_path.join(generate_random_identifier()),
        )
    raise TypeError(ast.unparse(node))


def object_to_cls(object_: Object, /) -> ClassObject:
    if object_.kind is ObjectKind.CLASS:
        metacls = object_.metacls
        if metacls is MISSING:
            raise TypeError(object_)
        return metacls
    if (
        object_.kind is ObjectKind.DESCRIPTOR
        or object_.kind is ObjectKind.INSTANCE
        or object_.kind is ObjectKind.ROUTINE
    ):
        return object_.cls
    if (
        object_.kind is ObjectKind.BUILTIN_MODULE
        or object_.kind is ObjectKind.DYNAMIC_MODULE
        or object_.kind is ObjectKind.EXTENSION_MODULE
        or object_.kind is ObjectKind.METHOD
        or object_.kind is ObjectKind.STATIC_MODULE
    ):
        return object_.CLS
    assert object_.kind in (
        ObjectKind.ROUTINE_CALL,
        ObjectKind.UNKNOWN,
        ObjectKind.UNKNOWN_CLASS,
    )
    raise TypeError(object_)


def cls_to_mro(cls: ClassObject, /) -> Sequence[Class]:
    if cls.kind is not ObjectKind.CLASS:
        raise TypeError(cls.kind)
    result = [cls]
    if (
        cls.module_path == BUILTINS_MODULE_PATH
        and cls.local_path == BUILTINS_OBJECT_LOCAL_OBJECT_PATH
    ):
        return result
    parent_chains: list[Sequence[Any]] = [
        cls_to_mro(base_cls) for base_cls in cls.bases
    ]
    parent_chains.append([*cls.bases])
    while parent_chains:
        next_parent = None
        for parent_chain in parent_chains:
            candidate = parent_chain[0]
            if not any(candidate in chain[1:] for chain in parent_chains):
                next_parent = candidate
                break
        if next_parent is None:
            raise TypeError('MRO resolution error.')
        if next_parent.kind is not ObjectKind.CLASS:
            raise TypeError(next_parent.kind)
        result.append(next_parent)
        parent_chains = [
            new_chain
            for chain in parent_chains
            if len(
                new_chain := (chain[1:] if chain[0] is next_parent else chain)
            )
            > 0
        ]
    return result


_binary_operators_by_operator_type: Mapping[
    type[ast.operator], Callable[[Any, Any], Any]
] = {
    ast.Add: operator.add,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.LShift: operator.lshift,
    ast.MatMult: operator.matmul,
    ast.Mod: operator.mod,
    ast.Mult: operator.mul,
    ast.Pow: operator.pow,
    ast.RShift: operator.rshift,
    ast.Sub: operator.sub,
}


@evaluate_expression_node.register(ast.BinOp)
def _(
    node: ast.BinOp, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    return value_to_object(
        _binary_operators_by_operator_type[type(node.op)](
            evaluate_expression_node(
                node.left, scope, *parent_scopes, context=context
            ).value,
            evaluate_expression_node(
                node.right, scope, *parent_scopes, context=context
            ).value,
        ),
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


@evaluate_expression_node.register(ast.BoolOp)
def _(
    node: ast.BoolOp, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Any:
    if isinstance(node.op, ast.And):
        try:
            return next(
                candidate
                for value_node in node.values[:-1]
                if not (
                    candidate := evaluate_expression_node(
                        value_node, scope, *parent_scopes, context=context
                    )
                ).value
            )
        except StopIteration:
            return evaluate_expression_node(
                node.values[-1], scope, *parent_scopes, context=context
            )
    assert isinstance(node.op, ast.Or), ast.unparse(node)
    try:
        return next(
            candidate
            for value_node in node.values[:-1]
            if (
                candidate := evaluate_expression_node(
                    value_node, scope, *parent_scopes, context=context
                )
            ).value
        )
    except StopIteration:
        return evaluate_expression_node(
            node.values[-1], scope, *parent_scopes, context=context
        )


_binary_comparison_operators_by_operator_node_type: Mapping[
    type[ast.cmpop], Callable[[Any, Any], bool]
] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


@evaluate_expression_node.register(ast.Compare)
def _(
    node: ast.Compare, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    value = evaluate_expression_node(
        node.left, scope, *parent_scopes, context=context
    ).value
    for operator_node, next_value in zip(
        node.ops,
        (
            evaluate_expression_node(
                operand_node, scope, *parent_scopes, context=context
            ).value
            for operand_node in node.comparators
        ),
        strict=True,
    ):
        if not _binary_comparison_operators_by_operator_node_type[
            type(operator_node)
        ](value, next_value):
            return value_to_object(
                False,  # noqa: FBT003
                module_path=scope.module_path,
                local_path=scope.local_path.join(generate_random_identifier()),
            )
        value = next_value
    return value_to_object(
        True,  # noqa: FBT003
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


@evaluate_expression_node.register(ast.Constant)
def _(
    node: ast.Constant,
    scope: Scope,
    /,
    *parent_scopes: Scope,  # noqa: ARG001
    context: Context,  # noqa: ARG001
) -> Object:
    return value_to_object(
        node.value,
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


@evaluate_expression_node.register(ast.Dict)
def _(
    node: ast.Dict, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    value: dict[Any, Any] = {}
    for item_key_node, item_value_node in zip(
        node.keys, node.values, strict=True
    ):
        item_value = evaluate_expression_node(
            item_value_node, scope, *parent_scopes, context=context
        ).value
        if item_key_node is None:
            value.update({**item_value})
        else:
            value[
                evaluate_expression_node(
                    item_key_node, scope, *parent_scopes, context=context
                ).value
            ] = item_value
    return Instance(
        scope.module_path,
        scope.local_path.join(generate_random_identifier()),
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_DICT_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )


@evaluate_expression_node.register(ast.List)
def _(
    node: ast.List, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    value = []
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            value.extend(
                [
                    *evaluate_expression_node(
                        element_node.value,
                        scope,
                        *parent_scopes,
                        context=context,
                    ).value
                ]
            )
        else:
            value.append(
                evaluate_expression_node(
                    element_node, scope, *parent_scopes, context=context
                ).value
            )
    return Instance(
        scope.module_path,
        scope.local_path.join(generate_random_identifier()),
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_LIST_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )


@evaluate_expression_node.register(ast.Name)
def _(
    node: ast.Name,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,  # noqa: ARG001
) -> Object:
    name = node.id
    try:
        return scope.get_object(name)
    except KeyError:
        for parent_scope in parent_scopes:
            try:
                return parent_scope.get_object(name)
            except KeyError:
                continue
        raise NameError(name) from None


@evaluate_expression_node.register(ast.Set)
def _(
    node: ast.Set, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    value = set()
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            value.update(
                [
                    *evaluate_expression_node(
                        element_node.value,
                        scope,
                        *parent_scopes,
                        context=context,
                    ).value
                ]
            )
        else:
            value.add(
                evaluate_expression_node(
                    element_node, scope, *parent_scopes, context=context
                ).value
            )
    return Instance(
        scope.module_path,
        scope.local_path.join(generate_random_identifier()),
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_SET_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=value,
    )


@evaluate_expression_node.register(ast.Subscript)
def _(
    node: ast.Subscript,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Object:
    return value_to_object(
        evaluate_expression_node(
            node.value, scope, *parent_scopes, context=context
        ).value[
            evaluate_expression_node(
                node.slice, scope, *parent_scopes, context=context
            ).value
        ],
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


@evaluate_expression_node.register(ast.Slice)
def _(
    node: ast.Slice, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    start = (
        evaluate_expression_node(
            start_node, scope, *parent_scopes, context=context
        ).value
        if (start_node := node.lower) is not None
        else None
    )
    stop = (
        evaluate_expression_node(
            stop_node, scope, *parent_scopes, context=context
        ).value
        if (stop_node := node.upper) is not None
        else None
    )
    step = (
        evaluate_expression_node(
            step_node, scope, *parent_scopes, context=context
        ).value
        if (step_node := node.step) is not None
        else None
    )
    return Instance(
        scope.module_path,
        scope.local_path.join(generate_random_identifier()),
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_SLICE_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=slice(start, stop, step),
    )


@evaluate_expression_node.register(ast.Tuple)
def _(
    node: ast.Tuple, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    value = []
    for element_node in node.elts:
        if isinstance(element_node, ast.Starred):
            value.extend(
                [
                    *evaluate_expression_node(
                        element_node.value,
                        scope,
                        *parent_scopes,
                        context=context,
                    ).value
                ]
            )
        else:
            value.append(
                evaluate_expression_node(
                    element_node, scope, *parent_scopes, context=context
                ).value
            )
    return Instance(
        scope.module_path,
        scope.local_path.join(generate_random_identifier()),
        cls=ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_TUPLE_LOCAL_OBJECT_PATH
            ),
            Class,
        ),
        value=tuple(value),
    )


_unary_operators_by_operator_type: Mapping[
    type[ast.unaryop], Callable[[Any], Any]
] = {
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@evaluate_expression_node.register(ast.UnaryOp)
def _(
    node: ast.UnaryOp, scope: Scope, /, *parent_scopes: Scope, context: Context
) -> Object:
    return value_to_object(
        _unary_operators_by_operator_type[type(node.op)](
            evaluate_expression_node(
                node.operand, scope, *parent_scopes, context=context
            ).value
        ),
        module_path=scope.module_path,
        local_path=scope.local_path.join(generate_random_identifier()),
    )


def function_node_to_keyword_only_defaults(
    signature_node: ast.arguments,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Mapping[Any, Any]:
    result: dict[Any, Any] = {}
    for keyword_parameter_node, keyword_default_node in zip(
        signature_node.kwonlyargs, signature_node.kw_defaults, strict=True
    ):
        if keyword_default_node is None:
            continue
        try:
            keyword_only_default_value = evaluate_expression_node(
                keyword_default_node, scope, *parent_scopes, context=context
            ).value
        except EVALUATION_EXCEPTIONS:
            keyword_only_default_value = MISSING
        result[keyword_parameter_node.arg] = keyword_only_default_value
    return result


def function_node_to_positional_defaults(
    signature_node: ast.arguments,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
) -> Sequence[Any]:
    result: list[Any] = []
    for positional_default_node in signature_node.defaults:
        try:
            positional_default_value = evaluate_expression_node(
                positional_default_node, scope, *parent_scopes, context=context
            ).value
        except EVALUATION_EXCEPTIONS:
            positional_default_value = MISSING
        result.append(positional_default_value)
    return result


def value_to_object(
    value: Any | Missing,
    /,
    *,
    module_path: ModulePath,
    local_path: LocalObjectPath,
) -> Instance | UnknownObject:
    return (
        Instance(module_path, local_path, cls=cls, value=value)
        if (
            value is not MISSING
            and (cls := _value_to_cls_object(value)) is not None
        )
        else UnknownObject(module_path, local_path, value=value)
    )


def _value_to_cls_object(value: Any, /) -> Class | None:
    value_cls = type(value)
    if value_cls is bool:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_BOOL_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is bytes:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_BYTES_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is complex:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_COMPLEX_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is float:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_FLOAT_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is int:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_INT_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is str:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_STR_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value is None:
        return ensure_type(
            TYPES_MODULE.get_nested_attribute(
                TYPES_NONE_TYPE_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value is Ellipsis:
        return ensure_type(
            TYPES_MODULE.get_nested_attribute(
                TYPES_ELLIPSIS_TYPE_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is dict:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_DICT_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is frozenset:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_FROZENSET_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is list:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_LIST_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value_cls is set:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_SET_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value is slice:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_SLICE_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    if value is tuple:
        return ensure_type(
            BUILTINS_MODULE.get_nested_attribute(
                BUILTINS_TUPLE_LOCAL_OBJECT_PATH
            ),
            Class,
        )
    return None
