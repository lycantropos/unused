from __future__ import annotations

import ast
import builtins
import functools
import inspect
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from typing import Any, Final

from typing_extensions import Self, override

from .enums import ObjectKind, ScopeKind
from .missing import MISSING, Missing
from .modules import BUILTINS_MODULE, MODULES, TYPES_MODULE
from .object_ import (
    Call,
    Class,
    ClassObject,
    Instance,
    Object,
    Routine,
    UnknownObject,
    is_subclass,
)
from .object_path import (
    BUILTINS_BOOL_LOCAL_OBJECT_PATH,
    BUILTINS_BYTES_LOCAL_OBJECT_PATH,
    BUILTINS_COMPLEX_LOCAL_OBJECT_PATH,
    BUILTINS_DICT_LOCAL_OBJECT_PATH,
    BUILTINS_FLOAT_LOCAL_OBJECT_PATH,
    BUILTINS_FROZENSET_LOCAL_OBJECT_PATH,
    BUILTINS_GLOBALS_LOCAL_OBJECT_PATH,
    BUILTINS_INT_LOCAL_OBJECT_PATH,
    BUILTINS_LIST_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    BUILTINS_OBJECT_LOCAL_OBJECT_PATH,
    BUILTINS_SET_LOCAL_OBJECT_PATH,
    BUILTINS_SLICE_LOCAL_OBJECT_PATH,
    BUILTINS_STR_LOCAL_OBJECT_PATH,
    BUILTINS_TUPLE_LOCAL_OBJECT_PATH,
    BUILTINS_TYPE_LOCAL_OBJECT_PATH,
    COLLECTIONS_MODULE_PATH,
    COLLECTIONS_NAMEDTUPLE_LOCAL_OBJECT_PATH,
    DICT_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    SYS_MODULES_LOCAL_OBJECT_PATH,
    SYS_MODULE_PATH,
    TYPES_ELLIPSIS_TYPE_LOCAL_OBJECT_PATH,
    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH,
    TYPES_NONE_TYPE_LOCAL_OBJECT_PATH,
)
from .scope import Scope
from .utils import (
    EVALUATION_EXCEPTIONS,
    ensure_type,
    generate_random_identifier,
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


class Context(ABC):
    @property
    @abstractmethod
    def local_path(self, /) -> LocalObjectPath:
        raise NotImplementedError

    @property
    @abstractmethod
    def module_path(self, /) -> ModulePath:
        raise NotImplementedError

    @functools.singledispatchmethod
    def construct_object_from_expression_node(
        self,
        node: ast.expr,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        try:
            value = self.evaluate_expression_node(node).value
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        return value_to_object(
            value, module_path=module_path, local_path=local_path
        )

    @construct_object_from_expression_node.register(ast.Attribute)
    def _(
        self,
        node: ast.Attribute,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        if (
            value_object := self.lookup_object_by_expression_node(node.value)
        ) is not None:
            attribute_name = node.attr
            try:
                return value_object.get_attribute(attribute_name)
            except KeyError:
                raise AttributeError(attribute_name) from None
        return UnknownObject(module_path, local_path, value=MISSING)

    @construct_object_from_expression_node.register(ast.Call)
    def _(
        self,
        node: ast.Call,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        callable_object = self.lookup_object_by_expression_node(node.func)
        if callable_object is None:
            return UnknownObject(module_path, local_path, value=MISSING)
        if callable_object.module_path == BUILTINS_MODULE_PATH and (
            callable_object.local_path == BUILTINS_TYPE_LOCAL_OBJECT_PATH
        ):
            assert callable_object.kind is ObjectKind.METACLASS, (
                callable_object
            )
            first_argument_object = self.construct_object_from_expression_node(
                node.args[0],
                local_path=local_path.join('__args_0__'),
                module_path=module_path,
            )
            return (
                Class(
                    Scope(ScopeKind.METACLASS, module_path, local_path),
                    ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_TYPE_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    metacls=MISSING,
                )
                if (
                    len(node.args) == 1
                    and first_argument_object is not None
                    and first_argument_object.kind is ObjectKind.CLASS
                )
                else Class(
                    Scope(
                        ScopeKind.UNKNOWN_CLASS
                        if (
                            first_argument_object is None
                            or (
                                first_argument_object.kind
                                in (
                                    ObjectKind.UNKNOWN_CLASS,
                                    ObjectKind.UNKNOWN,
                                )
                            )
                        )
                        else ScopeKind.CLASS,
                        module_path,
                        local_path,
                    ),
                    ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_OBJECT_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    metacls=MISSING,
                )
            )
        if callable_object.kind is ObjectKind.CLASS:
            return Instance(
                module_path, local_path, cls=callable_object, value=MISSING
            )
        if callable_object.kind is ObjectKind.METACLASS:
            return Class(
                Scope(ScopeKind.CLASS, module_path, local_path),
                metacls=callable_object,
            )
        if (
            callable_object.module_path == BUILTINS_MODULE_PATH
            and callable_object.local_path
            == BUILTINS_GLOBALS_LOCAL_OBJECT_PATH
        ):
            assert callable_object.kind is ObjectKind.ROUTINE, callable_object
            return MODULES[self.module_path].get_attribute(DICT_FIELD_NAME)
        if (
            callable_object.kind is ObjectKind.ROUTINE
            and callable_object.module_path == BUILTINS_MODULE_PATH
            and (
                callable_object.local_path
                == LocalObjectPath.from_object_name(builtins.vars.__qualname__)
            )
        ):
            (argument_node,) = node.args
            argument_object = self.lookup_object_by_expression_node(
                argument_node
            )
            assert argument_object is not None
            return argument_object.get_attribute(DICT_FIELD_NAME)
        if callable_object.module_path == COLLECTIONS_MODULE_PATH and (
            callable_object.local_path
            == COLLECTIONS_NAMEDTUPLE_LOCAL_OBJECT_PATH
        ):
            _, namedtuple_field_name_node = node.args
            try:
                named_tuple_field_names = self.evaluate_expression_node(
                    namedtuple_field_name_node
                ).value
            except EVALUATION_EXCEPTIONS:
                return UnknownObject(module_path, local_path, value=MISSING)
            if isinstance(named_tuple_field_names, str):
                named_tuple_field_names = named_tuple_field_names.replace(
                    ',', ' '
                ).split()
            assert isinstance(named_tuple_field_names, tuple | list), (
                ast.unparse(node)
            )
            named_tuple_object = Class(
                Scope(ScopeKind.CLASS, module_path, local_path),
                ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_TUPLE_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_OBJECT_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                metacls=MISSING,
            )
            for field_name in named_tuple_field_names:
                assert isinstance(field_name, str), field_name
                named_tuple_object.set_attribute(
                    field_name,
                    UnknownObject(
                        named_tuple_object.module_path,
                        named_tuple_object.local_path.join(field_name),
                        value=MISSING,
                    ),
                )
            return named_tuple_object
        return Call(
            module_path,
            local_path,
            callable_object,
            [
                (
                    isinstance(argument_node, ast.Starred),
                    self.construct_object_from_expression_node(
                        (
                            argument_node.value
                            if isinstance(argument_node, ast.Starred)
                            else argument_node
                        ),
                        local_path=local_path.join(
                            f'__args_{argument_index}__'
                        ),
                        module_path=module_path,
                    ),
                )
                for argument_index, argument_node in enumerate(node.args)
            ],
            [
                (
                    argument_node.arg,
                    self.construct_object_from_expression_node(
                        argument_node.value,
                        local_path=local_path.join(
                            f'__args_{argument_index}__'
                        ),
                        module_path=module_path,
                    ),
                )
                for argument_index, argument_node in enumerate(node.keywords)
            ],
        )

    @construct_object_from_expression_node.register(ast.Dict)
    @construct_object_from_expression_node.register(ast.DictComp)
    def _(
        self,
        node: ast.Dict | ast.DictComp,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        try:
            value = self.evaluate_expression_node(node).value
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        return Instance(
            module_path,
            local_path,
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_DICT_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @construct_object_from_expression_node.register(ast.Lambda)
    def _(
        self,
        node: ast.Lambda,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        return Routine(
            module_path,
            local_path,
            ast_node=node,
            cls=ensure_type(
                TYPES_MODULE.get_nested_attribute(
                    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            keyword_only_defaults=self.function_node_to_keyword_only_defaults(
                node.args
            ),
            positional_defaults=self.function_node_to_positional_defaults(
                node.args
            ),
        )

    @construct_object_from_expression_node.register(ast.List)
    @construct_object_from_expression_node.register(ast.ListComp)
    def _(
        self,
        node: ast.List | ast.ListComp,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        try:
            value = self.evaluate_expression_node(node).value
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        return Instance(
            module_path,
            local_path,
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_LIST_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @construct_object_from_expression_node.register(ast.Name)
    def _(
        self,
        node: ast.Name,
        /,
        *,
        local_path: LocalObjectPath,  # noqa: ARG002
        module_path: ModulePath,  # noqa: ARG002
    ) -> Object:
        return self.lookup_object_by_name(node.id)

    @construct_object_from_expression_node.register(ast.NamedExpr)
    def _(
        self,
        node: ast.NamedExpr,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        return (
            result
            if (result := self.lookup_object_by_expression_node(node.value))
            is not None
            else UnknownObject(module_path, local_path, value=MISSING)
        )

    @construct_object_from_expression_node.register(ast.Set)
    @construct_object_from_expression_node.register(ast.SetComp)
    def _(
        self,
        node: ast.Set | ast.SetComp,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        try:
            value = self.evaluate_expression_node(node).value
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        return Instance(
            module_path,
            local_path,
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_SET_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @construct_object_from_expression_node.register(ast.Tuple)
    def _(
        self,
        node: ast.Tuple,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        try:
            value = self.evaluate_expression_node(node).value
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        return Instance(
            module_path,
            local_path,
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_TUPLE_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @abstractmethod
    def evaluate_expression_node(self, node: ast.expr, /) -> Object:
        raise NotImplementedError

    @abstractmethod
    def function_node_to_keyword_only_defaults(
        self, signature_node: ast.arguments, /
    ) -> Mapping[Any, Any] | Missing:
        raise NotImplementedError

    @abstractmethod
    def function_node_to_positional_defaults(
        self, signature_node: ast.arguments, /
    ) -> Sequence[Any] | Missing:
        raise NotImplementedError

    def lookup_object_by_expression_node(
        self, node: ast.expr, /
    ) -> Object | None:
        return self._lookup_object_by_expression_node(node)

    @functools.singledispatchmethod
    def _lookup_object_by_expression_node(
        self, _node: ast.expr, /
    ) -> Object | None:
        return None

    @_lookup_object_by_expression_node.register(ast.Attribute)
    def _(self, node: ast.Attribute, /) -> Object | None:
        assert isinstance(node.ctx, ast.Load), ast.unparse(node)
        value_object = self._lookup_object_by_expression_node(node.value)
        if value_object is None:
            return None
        attribute_name = node.attr
        try:
            return value_object.get_attribute(attribute_name)
        except KeyError:
            raise AttributeError(attribute_name) from None

    @_lookup_object_by_expression_node.register(ast.Call)
    def _(self, node: ast.Call, /) -> Object | None:
        callable_object = self._lookup_object_by_expression_node(node.func)
        if callable_object is None:
            return None
        if (
            callable_object.module_path == BUILTINS_MODULE_PATH
            and callable_object.local_path
            == BUILTINS_GLOBALS_LOCAL_OBJECT_PATH
        ):
            return MODULES[self.module_path].get_attribute(DICT_FIELD_NAME)
        if callable_object.kind is ObjectKind.CLASS:
            return Instance(
                callable_object.module_path,
                callable_object.local_path,
                cls=callable_object,
                value=MISSING,
            )
        if callable_object.kind is ObjectKind.METACLASS:
            local_path = self.local_path.join(generate_random_identifier())
            return Class(
                Scope(ScopeKind.CLASS, self.module_path, local_path),
                metacls=callable_object,
            )
        if callable_object.kind is ObjectKind.ROUTINE:
            local_path = self.local_path.join(generate_random_identifier())
            return Call(
                self.module_path,
                local_path,
                callable_object,
                [
                    (
                        isinstance(argument_node, ast.Starred),
                        self.construct_object_from_expression_node(
                            (
                                argument_node.value
                                if isinstance(argument_node, ast.Starred)
                                else argument_node
                            ),
                            local_path=local_path.join(
                                f'__args_{argument_index}__'
                            ),
                            module_path=self.module_path,
                        ),
                    )
                    for argument_index, argument_node in enumerate(node.args)
                ],
                [
                    (
                        argument_node.arg,
                        self.construct_object_from_expression_node(
                            argument_node.value,
                            local_path=self.local_path.join(
                                f'__args_{argument_index}__'
                            ),
                            module_path=self.module_path,
                        ),
                    )
                    for argument_index, argument_node in enumerate(
                        node.keywords
                    )
                ],
            )
        return None

    @_lookup_object_by_expression_node.register(ast.Name)
    def _(self, node: ast.Name, /) -> Object | None:
        assert isinstance(node.ctx, ast.Load)
        return self.lookup_object_by_name(node.id)

    @_lookup_object_by_expression_node.register(ast.NamedExpr)
    def _(self, node: ast.NamedExpr, /) -> Object | None:
        return self._lookup_object_by_expression_node(node.value)

    @_lookup_object_by_expression_node.register(ast.Subscript)
    def _(self, node: ast.Subscript, /) -> Object | None:
        return self._lookup_object_by_subscript(node)

    @abstractmethod
    def lookup_object_by_local_path(
        self, local_path: LocalObjectPath, /
    ) -> Object:
        raise NotImplementedError

    @abstractmethod
    def lookup_object_by_name(self, name: str, /) -> Object:
        raise NotImplementedError

    @abstractmethod
    def _lookup_object_by_subscript(
        self, node: ast.Subscript, /
    ) -> Object | None:
        raise NotImplementedError


class NonEvaluatingContext(Context):
    @property
    @override
    def local_path(self, /) -> LocalObjectPath:
        return self._scopes[0].local_path

    @property
    @override
    def module_path(self, /) -> ModulePath:
        return self._scopes[0].module_path

    @override
    def evaluate_expression_node(self, node: ast.expr, /) -> Object:
        raise TypeError(type(node))

    @override
    def function_node_to_keyword_only_defaults(
        self, signature_node: ast.arguments, /
    ) -> Mapping[Any, Any] | Missing:
        return MISSING

    @override
    def function_node_to_positional_defaults(
        self, signature_node: ast.arguments, /
    ) -> Sequence[Any] | Missing:
        return MISSING

    @override
    def lookup_object_by_local_path(
        self, local_path: LocalObjectPath, /
    ) -> Object:
        try:
            return self._scopes[0].get_nested_object(local_path)
        except KeyError:
            for parent_scope in self._scopes[1:]:
                try:
                    return parent_scope.get_nested_object(local_path)
                except KeyError:
                    continue
            raise

    @override
    def lookup_object_by_name(self, name: str, /) -> Object:
        return _lookup_object_by_name(name, *self._scopes)

    @override
    def _lookup_object_by_subscript(
        self, node: ast.Subscript, /
    ) -> Object | None:
        return None

    _scopes: Sequence[Scope]

    __slots__ = ('_scopes',)

    def __new__(cls, scope: Scope, /, *parent_scopes: Scope) -> Self:
        self = super().__new__(cls)
        self._scopes = scope, *parent_scopes
        return self


class EvaluatingContext(Context):
    @override
    def evaluate_expression_node(self, node: ast.expr, /) -> Object:
        return self._evaluate_expression_node(node)

    @functools.singledispatchmethod
    def _evaluate_expression_node(self, node: ast.expr, /) -> Object:
        raise TypeError(type(node))

    @_evaluate_expression_node.register(ast.Attribute)
    def _(self, node: ast.Attribute, /) -> Object:
        return self._evaluate_expression_node(node.value).get_attribute(
            node.attr
        )

    @_evaluate_expression_node.register(ast.JoinedStr)
    def _(self, node: ast.JoinedStr, /) -> Object:
        return value_to_object(
            ''.join(
                self._evaluate_expression_node(value_node).value
                for value_node in node.values
            ),
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @_evaluate_expression_node.register(ast.FormattedValue)
    def _(self, node: ast.FormattedValue, /) -> Object:
        value = self._evaluate_expression_node(node.value).value
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
                    self._evaluate_expression_node(node.format_spec).value,
                )
                if node.format_spec is not None
                else format(value)
            ),
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @_evaluate_expression_node.register(ast.Call)
    def _(self, node: ast.Call, /) -> Object:
        callable_object = self._evaluate_expression_node(node.func)
        positional_argument_objects: list[tuple[bool, Object]] = []
        routine_object: Object
        if callable_object.kind is ObjectKind.METHOD:
            positional_argument_objects.append(
                (False, callable_object.instance)
            )
            routine_object = callable_object.routine
        else:
            routine_object = callable_object
        for positional_argument_node in node.args:
            if isinstance(positional_argument_node, ast.Starred):
                positional_argument_objects.append(
                    (
                        True,
                        self._evaluate_expression_node(
                            positional_argument_node.value
                        ),
                    )
                )
            else:
                positional_argument_objects.append(
                    (
                        False,
                        self._evaluate_expression_node(
                            positional_argument_node
                        ),
                    )
                )
        keyword_argument_objects: list[tuple[str | None, Object]] = []
        for keyword_argument_node in node.keywords:
            if (parameter_name := keyword_argument_node.arg) is not None:
                keyword_argument_objects.append(
                    (
                        parameter_name,
                        self._evaluate_expression_node(
                            keyword_argument_node.value
                        ),
                    )
                )
            else:
                keyword_argument_objects.append(
                    (
                        None,
                        self._evaluate_expression_node(
                            keyword_argument_node.value
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
                    pass
                elif isinstance(
                    attribute_name := attribute_name_object.value, str
                ):
                    try:
                        subject.get_attribute(attribute_name, strict=True)
                    except KeyError:
                        value = False
                    else:
                        value = True
                    return value_to_object(
                        value,
                        module_path=self.module_path,
                        local_path=self.local_path.join(
                            generate_random_identifier()
                        ),
                    )
                raise TypeError(ast.unparse(node))
            if (
                routine_object.local_path
                == BUILTINS_ISINSTANCE_LOCAL_OBJECT_PATH
            ):
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
                elif (subject_cls := object_to_cls(subject)).kind in (
                    ObjectKind.METACLASS,
                    ObjectKind.CLASS,
                ):
                    assert isinstance(subject_cls, Class), subject_cls
                    return value_to_object(
                        is_subclass(subject_cls, cls_or_tuple),
                        module_path=self.module_path,
                        local_path=self.local_path.join(
                            generate_random_identifier()
                        ),
                    )
                raise TypeError(ast.unparse(node))
            if (
                routine_object.local_path
                == BUILTINS_ISSUBCLASS_LOCAL_OBJECT_PATH
            ):
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
                elif subject.kind in (ObjectKind.METACLASS, ObjectKind.CLASS):
                    assert isinstance(subject, Class), subject
                    return value_to_object(
                        is_subclass(subject, cls_or_tuple),
                        module_path=self.module_path,
                        local_path=self.local_path.join(
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
                    positional_arguments.append(
                        positional_argument_object.value
                    )
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
                module_path=self.module_path,
                local_path=self.local_path.join(generate_random_identifier()),
            )
        if callable_object.kind is ObjectKind.CLASS:
            return Instance(
                self.module_path,
                self.local_path.join(generate_random_identifier()),
                cls=callable_object,
                value=MISSING,
            )
        if callable_object.kind is ObjectKind.METACLASS:
            return Class(
                Scope(
                    ScopeKind.CLASS,
                    self.module_path,
                    self.local_path.join(generate_random_identifier()),
                ),
                metacls=callable_object,
            )
        return Call(
            self.module_path,
            self.local_path.join(generate_random_identifier()),
            callable_object,
            positional_argument_objects,
            keyword_argument_objects,
        )

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

    @_evaluate_expression_node.register(ast.BinOp)
    def _(self, node: ast.BinOp, /) -> Object:
        return value_to_object(
            self._binary_operators_by_operator_type[type(node.op)](
                self._evaluate_expression_node(node.left).value,
                self._evaluate_expression_node(node.right).value,
            ),
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @_evaluate_expression_node.register(ast.BoolOp)
    def _(self, node: ast.BoolOp, /) -> Object:
        if isinstance(node.op, ast.And):
            try:
                candidate: Object
                return next(
                    candidate
                    for value_node in node.values[:-1]
                    if not (
                        candidate := self._evaluate_expression_node(value_node)
                    ).value
                )
            except StopIteration:
                return self._evaluate_expression_node(node.values[-1])
        assert isinstance(node.op, ast.Or), ast.unparse(node)
        try:
            return next(
                candidate
                for value_node in node.values[:-1]
                if (
                    candidate := self._evaluate_expression_node(value_node)
                ).value
            )
        except StopIteration:
            return self._evaluate_expression_node(node.values[-1])

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

    @_evaluate_expression_node.register(ast.Compare)
    def _(self, node: ast.Compare, /) -> Object:
        value = self._evaluate_expression_node(node.left).value
        for operator_node, next_value in zip(
            node.ops,
            (
                self._evaluate_expression_node(operand_node).value
                for operand_node in node.comparators
            ),
            strict=True,
        ):
            if not self._binary_comparison_operators_by_operator_node_type[
                type(operator_node)
            ](value, next_value):
                return value_to_object(
                    False,  # noqa: FBT003
                    module_path=self.module_path,
                    local_path=self.local_path.join(
                        generate_random_identifier()
                    ),
                )
            value = next_value
        return value_to_object(
            True,  # noqa: FBT003
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @_evaluate_expression_node.register(ast.Constant)
    def _(self, node: ast.Constant, /) -> Object:
        return value_to_object(
            node.value,
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @_evaluate_expression_node.register(ast.Dict)
    def _(self, node: ast.Dict, /) -> Object:
        value: dict[Any, Any] = {}
        for item_key_node, item_value_node in zip(
            node.keys, node.values, strict=True
        ):
            item_value = self._evaluate_expression_node(item_value_node).value
            if item_key_node is None:
                value.update({**item_value})
            else:
                value[self._evaluate_expression_node(item_key_node).value] = (
                    item_value
                )
        return Instance(
            self.module_path,
            self.local_path.join(generate_random_identifier()),
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_DICT_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @_evaluate_expression_node.register(ast.List)
    def _(self, node: ast.List, /) -> Object:
        value = []
        for element_node in node.elts:
            if isinstance(element_node, ast.Starred):
                value.extend(
                    [*self._evaluate_expression_node(element_node.value).value]
                )
            else:
                value.append(
                    self._evaluate_expression_node(element_node).value
                )
        return Instance(
            self.module_path,
            self.local_path.join(generate_random_identifier()),
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_LIST_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @_evaluate_expression_node.register(ast.Name)
    def _(self, node: ast.Name, /) -> Object:
        return self.lookup_object_by_name(node.id)

    @_evaluate_expression_node.register(ast.Set)
    def _(self, node: ast.Set, /) -> Object:
        value = set()
        for element_node in node.elts:
            if isinstance(element_node, ast.Starred):
                value.update(
                    [*self._evaluate_expression_node(element_node.value).value]
                )
            else:
                value.add(self._evaluate_expression_node(element_node).value)
        return Instance(
            self.module_path,
            self.local_path.join(generate_random_identifier()),
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_SET_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=value,
        )

    @_evaluate_expression_node.register(ast.Subscript)
    def _(self, node: ast.Subscript, /) -> Object:
        return value_to_object(
            self._evaluate_expression_node(node.value).value[
                self._evaluate_expression_node(node.slice).value
            ],
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @_evaluate_expression_node.register(ast.Slice)
    def _(self, node: ast.Slice, /) -> Object:
        start = (
            self._evaluate_expression_node(start_node).value
            if (start_node := node.lower) is not None
            else None
        )
        stop = (
            self._evaluate_expression_node(stop_node).value
            if (stop_node := node.upper) is not None
            else None
        )
        step = (
            self._evaluate_expression_node(step_node).value
            if (step_node := node.step) is not None
            else None
        )
        return Instance(
            self.module_path,
            self.local_path.join(generate_random_identifier()),
            cls=ensure_type(
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_SLICE_LOCAL_OBJECT_PATH
                ),
                Class,
            ),
            value=slice(start, stop, step),
        )

    @_evaluate_expression_node.register(ast.Tuple)
    def _(self, node: ast.Tuple, /) -> Object:
        value = []
        for element_node in node.elts:
            if isinstance(element_node, ast.Starred):
                value.extend(
                    [*self._evaluate_expression_node(element_node.value).value]
                )
            else:
                value.append(
                    self._evaluate_expression_node(element_node).value
                )
        return Instance(
            self.module_path,
            self.local_path.join(generate_random_identifier()),
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

    @_evaluate_expression_node.register(ast.UnaryOp)
    def _(self, node: ast.UnaryOp, /) -> Object:
        return value_to_object(
            self._unary_operators_by_operator_type[type(node.op)](
                self._evaluate_expression_node(node.operand).value
            ),
            module_path=self.module_path,
            local_path=self.local_path.join(generate_random_identifier()),
        )

    @override
    def function_node_to_keyword_only_defaults(
        self, signature_node: ast.arguments, /
    ) -> Mapping[Any, Any]:
        result: dict[Any, Any] = {}
        for keyword_parameter_node, keyword_default_node in zip(
            signature_node.kwonlyargs, signature_node.kw_defaults, strict=True
        ):
            if keyword_default_node is None:
                continue
            try:
                keyword_only_default_value = self._evaluate_expression_node(
                    keyword_default_node
                ).value
            except EVALUATION_EXCEPTIONS:
                keyword_only_default_value = MISSING
            result[keyword_parameter_node.arg] = keyword_only_default_value
        return result

    @override
    def function_node_to_positional_defaults(
        self, signature_node: ast.arguments, /
    ) -> Sequence[Any]:
        result: list[Any] = []
        for positional_default_node in signature_node.defaults:
            try:
                positional_default_value = self._evaluate_expression_node(
                    positional_default_node
                ).value
            except EVALUATION_EXCEPTIONS:
                positional_default_value = MISSING
            result.append(positional_default_value)
        return result


class StaticContext(EvaluatingContext):
    @property
    @override
    def local_path(self, /) -> LocalObjectPath:
        return self._scopes[0].local_path

    @property
    @override
    def module_path(self, /) -> ModulePath:
        return self._scopes[0].module_path

    @override
    def lookup_object_by_name(self, name: str, /) -> Object:
        return _lookup_object_by_name(name, *self._scopes)

    @override
    def lookup_object_by_local_path(
        self, local_path: LocalObjectPath, /
    ) -> Object:
        try:
            return self._scopes[0].get_nested_object(local_path)
        except KeyError:
            for parent_scope in self._scopes[1:]:
                try:
                    return parent_scope.get_nested_object(local_path)
                except KeyError:
                    continue
            raise

    @override
    def _lookup_object_by_subscript(
        self, node: ast.Subscript, /
    ) -> Object | None:
        value_object = self.lookup_object_by_expression_node(node.value)
        if value_object is None:
            return None
        if (
            value_object.module_path == SYS_MODULE_PATH
            and value_object.local_path == SYS_MODULES_LOCAL_OBJECT_PATH
        ):
            assert value_object.kind is ObjectKind.INSTANCE, value_object
            try:
                module_name = self.evaluate_expression_node(node.slice).value
            except EVALUATION_EXCEPTIONS:
                pass
            else:
                assert isinstance(module_name, str), module_name
                return MODULES[ModulePath.from_module_name(module_name)]
        return None

    _scopes: Sequence[Scope]

    __slots__ = ('_scopes',)

    def __new__(cls, scope: Scope, /, *parent_scopes: Scope) -> Self:
        self = super().__new__(cls)
        self._scopes = (scope, *parent_scopes)
        return self


class FunctionCallContext(EvaluatingContext):
    @property
    def caller_module_path(self, /) -> ModulePath:
        return self._caller_module_path

    @property
    @override
    def local_path(self, /) -> LocalObjectPath:
        return self._scopes[0].local_path

    @property
    @override
    def module_path(self, /) -> ModulePath:
        return self._scopes[0].module_path

    @override
    def lookup_object_by_local_path(
        self, local_path: LocalObjectPath, /
    ) -> Object:
        try:
            return self._scopes[0].get_nested_object(local_path)
        except KeyError:
            for parent_scope in self._scopes[1:]:
                try:
                    return parent_scope.get_nested_object(local_path)
                except KeyError:
                    continue
            raise

    @override
    def lookup_object_by_name(self, name: str, /) -> Object:
        return _lookup_object_by_name(name, *self._scopes)

    @override
    def _lookup_object_by_subscript(
        self, node: ast.Subscript, /
    ) -> Object | None:
        value_object = self.lookup_object_by_expression_node(node.value)
        if value_object is None:
            return None
        if (
            value_object.module_path == SYS_MODULE_PATH
            and value_object.local_path == SYS_MODULES_LOCAL_OBJECT_PATH
        ):
            assert value_object.kind is ObjectKind.INSTANCE, value_object
            try:
                module_name = self.evaluate_expression_node(node.slice).value
            except EVALUATION_EXCEPTIONS:
                # assume that caller module is affected
                return MODULES[self.caller_module_path]
            else:
                assert isinstance(module_name, str), module_name
                return MODULES[ModulePath.from_module_name(module_name)]
        return None

    _caller_module_path: ModulePath
    _scopes: Sequence[Scope]

    __slots__ = '_caller_module_path', '_scopes'

    def __new__(
        cls,
        scope: Scope,
        /,
        *parent_scopes: Scope,
        caller_module_path: ModulePath,
    ) -> Self:
        self = super().__new__(cls)
        self._caller_module_path, self._scopes = (
            caller_module_path,
            (scope, *parent_scopes),
        )
        return self


def _lookup_object_by_name(name: str, /, *scopes: Scope) -> Object:
    for scope in scopes:
        try:
            return scope.get_object(name, strict=True)
        except KeyError:
            continue
    for scope in scopes:
        try:
            return scope.get_object(name)
        except KeyError:
            continue
    raise NameError(name) from None
