from __future__ import annotations

import ast
import builtins
import contextlib
import enum
import functools
import operator
import pkgutil
import sys
import tempfile
import types
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from importlib.machinery import (
    EXTENSION_SUFFIXES,
    ExtensionFileLoader,
    SOURCE_SUFFIXES,
    SourceFileLoader,
)
from itertools import chain, repeat, takewhile
from pathlib import Path
from typing import Any, Final

from typing_extensions import override

from .construction import construct_object_from_expression_node
from .context import Context, FunctionCallContext, NullContext
from .evaluation import EVALUATION_EXCEPTIONS, evaluate_expression_node
from .lookup import lookup_object_by_expression_node
from .missing import MISSING, Missing
from .modules import BUILTINS_MODULE, MODULES, TYPES_MODULE, parse_modules
from .object_ import (
    CLASS_SCOPE_KINDS,
    Class,
    Module,
    Object,
    ObjectKind,
    PlainObject,
    Scope,
    ScopeKind,
)
from .object_path import (
    BUILTINS_DICT_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    DICT_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH,
)
from .resolution import (
    ResolvedAssignmentTarget,
    ResolvedAssignmentTargetSplitPath,
    combine_resolved_assignment_target_with_value,
    resolve_assignment_target,
)
from .utils import ensure_type

FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME: Final = '__defaults__'
FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME: Final = '__kwdefaults__'
MODULE_FIELD_NAME: Final = '__module__'
NAME_FIELD_NAME: Final = '__name__'
QUALNAME_FIELD_NAME: Final = '__qualname__'

TYPES_MODULE_TYPE_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'ModuleType'
)
EMPTY_MODULE_FILE_PATH: Final[Path] = Path(
    tempfile.NamedTemporaryFile(delete=False).name  # noqa: SIM115
)
BUILTINS_STR_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(builtins.str.__qualname__)
)


def _does_function_modify_caller_global_state(
    function_namespace: Object,
    /,
    *,
    cache: dict[tuple[ModulePath, LocalObjectPath], bool] = {},  # noqa: B006
    caller_module_scope: Scope,
    function_definition_nodes: MutableMapping[
        tuple[ModulePath, LocalObjectPath],
        ast.AsyncFunctionDef | ast.FunctionDef,
    ],
    keyword_arguments: Mapping[str, Any],
    module_file_paths: Mapping[ModulePath, Path | None],
    positional_arguments: Sequence[Any | Missing | Starred],
) -> bool:
    cache_key = (function_namespace.module_path, function_namespace.local_path)
    try:
        return cache[cache_key]
    except KeyError:
        try:
            function_definition_node = function_definition_nodes[
                function_namespace.module_path, function_namespace.local_path
            ]
        except KeyError:
            assert module_file_paths[function_namespace.module_path] is None, (
                function_namespace
            )
            cache[cache_key] = result = (
                function_namespace.module_path == BUILTINS_MODULE_PATH
                and (
                    function_namespace.local_path
                    in (
                        LocalObjectPath.from_object_name(
                            builtins.eval.__qualname__
                        ),
                        LocalObjectPath.from_object_name(
                            builtins.exec.__qualname__
                        ),
                    )
                )
                and (
                    sum(
                        argument is not Starred.UNKNOWN
                        for argument in positional_arguments
                    )
                    < 2
                )
                and 'globals' not in keyword_arguments
            )
            return result
        # for recursive cases
        keyword_argument_dict = dict(keyword_arguments)
        signature_node = function_definition_node.args
        function_scope = Scope(
            ScopeKind.FUNCTION,
            function_namespace.module_path,
            function_namespace.local_path,
        )
        positional_parameter_nodes = list(
            chain(signature_node.posonlyargs, signature_node.args)
        )
        first_non_starred_positional_arguments = list(
            takewhile(
                functools.partial(operator.is_not, Starred.UNKNOWN),
                positional_arguments,
            )
        )
        for positional_argument, positional_parameter_node in zip(
            first_non_starred_positional_arguments,
            positional_parameter_nodes[
                : len(first_non_starred_positional_arguments)
            ],
            strict=False,
        ):
            positional_parameter_name = positional_parameter_node.arg
            function_scope.set_object(
                positional_parameter_name,
                PlainObject(
                    ObjectKind.UNKNOWN,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        positional_parameter_name
                    ),
                ),
            )
            if positional_argument is not MISSING:
                function_scope.set_value(
                    positional_parameter_name, positional_argument
                )
        positional_defaults = function_namespace.get_value(
            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME
        )
        defaulted_positional_parameter_nodes = positional_parameter_nodes[
            len(first_non_starred_positional_arguments) :
        ]
        for defaulted_positional_parameter_node, positional_default in zip(
            defaulted_positional_parameter_nodes[::-1],
            chain(positional_defaults[::-1], repeat(MISSING)),
            strict=False,
        ):
            defaulted_positional_parameter_name = (
                defaulted_positional_parameter_node.arg
            )
            function_scope.set_object(
                defaulted_positional_parameter_name,
                PlainObject(
                    ObjectKind.UNKNOWN,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        defaulted_positional_parameter_name
                    ),
                ),
            )
            if positional_default is not MISSING:
                function_scope.set_value(
                    defaulted_positional_parameter_name, positional_default
                )
        for keyword_parameter_node in signature_node.kwonlyargs:
            keyword_parameter_name = keyword_parameter_node.arg
            function_scope.set_object(
                keyword_parameter_name,
                PlainObject(
                    ObjectKind.UNKNOWN,
                    function_namespace.module_path,
                    function_namespace.local_path.join(keyword_parameter_name),
                ),
            )
            try:
                keyword_argument = keyword_argument_dict.pop(
                    keyword_parameter_name
                )
            except KeyError:
                keyword_argument = function_namespace.get_value(
                    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
                )[keyword_parameter_name]
            if keyword_argument is not MISSING:
                function_scope.set_value(
                    keyword_parameter_name, keyword_argument
                )
        if (
            variadic_positional_parameter_node := signature_node.vararg
        ) is not None:
            variadic_positional_parameter_name = (
                variadic_positional_parameter_node.arg
            )
            function_scope.set_object(
                variadic_positional_parameter_name,
                PlainObject(
                    ObjectKind.INSTANCE,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        variadic_positional_parameter_name
                    ),
                    BUILTINS_MODULE.get_nested_attribute(
                        LocalObjectPath.from_object_name(tuple.__qualname__)
                    ),
                ),
            )
            function_scope.set_value(
                variadic_positional_parameter_name,
                tuple(positional_arguments[len(positional_parameter_nodes) :]),
            )
        if (
            variadic_keyword_parameter_node := signature_node.kwarg
        ) is not None:
            variadic_keyword_parameter_name = (
                variadic_keyword_parameter_node.arg
            )
            function_scope.set_object(
                variadic_keyword_parameter_name,
                PlainObject(
                    ObjectKind.INSTANCE,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        variadic_keyword_parameter_name
                    ),
                    BUILTINS_MODULE.get_nested_attribute(
                        LocalObjectPath.from_object_name(dict.__qualname__)
                    ),
                ),
            )
            function_scope.set_value(
                variadic_keyword_parameter_name, keyword_argument_dict
            )
        function_body_parser = ScopeParser(
            function_scope,
            ensure_type(
                MODULES[function_namespace.module_path], Module
            ).to_scope(),
            BUILTINS_MODULE.to_scope(),
            context=FunctionCallContext(caller_module_scope.module_path),
            function_definition_nodes=function_definition_nodes,
            module_file_paths=module_file_paths,
        )
        cache[cache_key] = result = False
        for function_body_node in function_definition_node.body:
            function_body_parser.visit(function_body_node)
            if caller_module_scope.kind is ScopeKind.DYNAMIC_MODULE:
                result = True
                break
        if len(positional_arguments) > 0 or len(keyword_arguments) > 0:
            del cache[cache_key]
        return result


class Starred(enum.Enum):
    UNKNOWN = enum.auto()


class ScopeParser(ast.NodeVisitor):
    def __init__(
        self,
        scope: Scope,
        /,
        *parent_scopes: Scope,
        context: Context,
        function_definition_nodes: MutableMapping[
            tuple[ModulePath, LocalObjectPath],
            ast.AsyncFunctionDef | ast.FunctionDef,
        ],
        module_file_paths: Mapping[ModulePath, Path | None],
    ) -> None:
        super().__init__()
        (
            self._context,
            self._function_definition_nodes,
            self._module_file_paths,
            self._scope,
            self._parent_scopes,
        ) = (
            context,
            function_definition_nodes,
            module_file_paths,
            scope,
            parent_scopes,
        )
        self._name_scopes: MutableMapping[str, Scope] = {}

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if (value_node := node.value) is not None:
            try:
                value = self._evaluate_expression_node(value_node)
            except EVALUATION_EXCEPTIONS:
                value = MISSING
            self._process_assignment(node.target, value_node, value)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        value_node = node.value
        try:
            value = self._evaluate_expression_node(value_node)
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        for target_node in node.targets:
            self._process_assignment(target_node, value_node, value)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_any_function_def(node)

    @override
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            for operand_node in node.values:
                try:
                    operand_value = self._evaluate_expression_node(
                        operand_node
                    )
                except EVALUATION_EXCEPTIONS:
                    pass
                else:
                    if not operand_value:
                        break
                self.visit(operand_node)
            return
        assert isinstance(node.op, ast.Or), ast.unparse(node)
        for operand_node in node.values:
            try:
                operand_value = self._evaluate_expression_node(operand_node)
            except EVALUATION_EXCEPTIONS:
                pass
            else:
                if operand_value:
                    break
            self.visit(operand_node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        callable_namespace = self._lookup_object_by_expression_node(node.func)
        if callable_namespace is None:
            return
        if (
            callable_namespace.local_path
            == LocalObjectPath(DICT_FIELD_NAME, 'update')
        ) and (
            module_scope := ensure_type(
                MODULES[callable_namespace.module_path], Module
            ).to_scope()
        ).kind is ScopeKind.STATIC_MODULE:
            module_scope.mark_module_as_dynamic()
            return
        if (
            (
                callable_namespace.kind
                in (ObjectKind.INSTANCE_ROUTINE, ObjectKind.ROUTINE)
            )
            and _does_function_modify_caller_global_state(
                _to_plain_routine_object(callable_namespace),
                caller_module_scope=self._get_module_scope(),
                function_definition_nodes=self._function_definition_nodes,
                keyword_arguments=self._evaluate_keyword_arguments(
                    node.keywords
                ),
                module_file_paths=self._module_file_paths,
                positional_arguments=self._to_complete_positional_arguments(
                    node.args, callable_namespace
                ),
            )
        ) and (
            module_scope := self._get_module_scope()
        ).kind is ScopeKind.STATIC_MODULE:
            module_scope.mark_module_as_dynamic()
            return

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_name = node.name
        class_module_path = self._scope.module_path
        class_local_path = self._scope.local_path.join(class_name)
        base_cls_objects: list[Object] = []
        for index, base_node in reversed([*enumerate(node.bases)]):
            self.visit(base_node)
            base_cls = self._construct_object_from_expression_node(
                base_node,
                local_path=class_local_path.join(
                    # FIXME: add support for indexing?
                    f'__bases___{index}'
                ),
                module_path=class_module_path,
            )
            if base_cls is None:
                continue
            base_cls_objects.append(base_cls)
        class_scope = Scope(
            (
                ScopeKind.METACLASS
                if any(
                    base_cls_object.kind is ObjectKind.METACLASS
                    for base_cls_object in base_cls_objects
                )
                else (
                    ScopeKind.UNKNOWN_CLASS
                    if any(
                        (
                            base_cls_object.kind
                            in (ObjectKind.UNKNOWN, ObjectKind.UNKNOWN_CLASS)
                        )
                        for base_cls_object in base_cls_objects
                    )
                    else ScopeKind.CLASS
                )
            ),
            self._scope.module_path,
            class_local_path,
        )
        class_parser = ScopeParser(
            class_scope,
            *self._get_inherited_scopes(),
            context=self._context,
            function_definition_nodes=self._function_definition_nodes,
            module_file_paths=self._module_file_paths,
        )
        for body_node in node.body:
            class_parser.visit(body_node)
        metacls_object: Object | None = None
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                metacls_object = self._construct_object_from_expression_node(
                    keyword.value,
                    local_path=class_local_path.join('__class__'),
                    module_path=class_module_path,
                )
                if metacls_object is None:
                    continue
                assert metacls_object.kind in (
                    ObjectKind.METACLASS,
                    ObjectKind.UNKNOWN_CLASS,
                )
        class_object = Class(
            class_scope,
            *(
                [
                    BUILTINS_MODULE.get_nested_attribute(
                        LocalObjectPath.from_object_name(object.__qualname__)
                    )
                ]
                if len(node.bases) == 0
                else base_cls_objects
            ),
            metaclass=metacls_object,
        )
        class_object.set_attribute(
            DICT_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                class_module_path,
                class_local_path.join(DICT_FIELD_NAME),
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_DICT_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_object.set_attribute(
            MODULE_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                class_module_path,
                class_local_path.join(MODULE_FIELD_NAME),
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_STR_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_object.set_value(
            MODULE_FIELD_NAME, class_module_path.to_module_name()
        )
        class_object.set_attribute(
            NAME_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                class_module_path,
                class_local_path.join(NAME_FIELD_NAME),
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_STR_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_object.set_value(NAME_FIELD_NAME, class_name)
        class_object.set_attribute(
            QUALNAME_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                class_module_path,
                class_local_path.join(QUALNAME_FIELD_NAME),
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_STR_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_object.set_value(
            QUALNAME_FIELD_NAME, class_local_path.to_object_name()
        )
        for decorator_node in node.decorator_list:
            self.visit(decorator_node)
            decorator_object = self._lookup_object_by_expression_node(
                decorator_node
            )
            assert isinstance(decorator_object, PlainObject), decorator_object
            if decorator_object is None:
                continue
            if (
                (
                    decorator_object.kind
                    in (ObjectKind.INSTANCE_ROUTINE, ObjectKind.ROUTINE)
                )
                and _does_function_modify_caller_global_state(
                    (
                        decorator_object.instance_routine_to_routine()
                        if (
                            decorator_object.kind
                            is ObjectKind.INSTANCE_ROUTINE
                        )
                        else decorator_object
                    ),
                    caller_module_scope=self._get_module_scope(),
                    positional_arguments=[
                        *self._to_complete_positional_arguments(
                            [], decorator_object
                        ),
                        class_scope.as_object(),
                    ],
                    keyword_arguments={},
                    function_definition_nodes=self._function_definition_nodes,
                    module_file_paths=self._module_file_paths,
                )
                and (
                    (module_scope := self._get_module_scope()).kind
                    is ScopeKind.STATIC_MODULE
                )
            ):
                module_scope.mark_module_as_dynamic()
                continue
        self._scope.set_object(class_name, class_object)
        self._scope.set_value(class_name, class_scope.as_object())

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    @override
    def visit_For(self, node: ast.For) -> None:
        resolved_target = self._resolve_assignment_target(node.target)
        assert resolved_target is not None
        for target_object_split_path in _flatten_resolved_assignment_target(
            resolved_target
        ):
            self._set_target_object_split_path(
                target_object_split_path,
                PlainObject(
                    ObjectKind.UNKNOWN,
                    self._scope.module_path,
                    target_object_split_path.combine_local(),
                ),
            )
        try:
            iterable = iter(self._evaluate_expression_node(node.iter))
        except EVALUATION_EXCEPTIONS:
            self.generic_visit(node)
        else:
            for element in iterable:
                for (
                    object_split_path,
                    value,
                ) in combine_resolved_assignment_target_with_value(
                    resolved_target, element
                ):
                    if object_split_path is not None and value is not MISSING:
                        self._resolve_absolute_local_path(
                            object_split_path.module,
                            object_split_path.absolute,
                        ).set_nested_value(object_split_path.relative, value)
                for body_node in node.body:
                    self.visit(body_node)
            for else_node in node.orelse:
                self.visit(else_node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_any_function_def(node)

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    @override
    def visit_Global(self, node: ast.Global) -> None:
        self._name_scopes.update(
            zip(node.names, repeat(self._get_module_scope()))
        )

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (module_alias := alias.asname) is not None:
                submodule_object = self._resolve_absolute_module_path(
                    ModulePath.from_module_name(alias.name)
                )
                self._scope.set_object(module_alias, submodule_object)
                self._scope.set_value(
                    module_alias, submodule_object.as_object()
                )
            else:
                submodule_paths = iter(
                    ModulePath.from_module_name(alias.name).submodule_paths()
                )
                first_submodule_path = next(submodule_paths)
                submodule_object = self._resolve_absolute_module_path(
                    first_submodule_path
                )
                assert len(first_submodule_path.components) == 1, (
                    first_submodule_path
                )
                self._scope.set_object(
                    first_submodule_path.components[0], submodule_object
                )
                self._scope.set_value(
                    first_submodule_path.components[0],
                    submodule_object.as_object(),
                )
                for submodule_path in submodule_paths:
                    next_submodule_object = self._resolve_absolute_module_path(
                        submodule_path
                    )
                    submodule_last_name = submodule_path.components[-1]
                    submodule_object.set_attribute(
                        submodule_last_name, next_submodule_object
                    )
                    submodule_object.set_value(
                        submodule_last_name, next_submodule_object.as_object()
                    )
                    submodule_object = next_submodule_object

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        import_is_relative = node.level > 0
        if import_is_relative:
            is_package = (
                module_file_path := self._module_file_paths.get(
                    self._scope.module_path
                )
            ) is not None and module_file_path.stem == '__init__'
            depth = (
                len(self._scope.module_path.components)
                + is_package
                - node.level
            ) or None
            components = list(self._scope.module_path.components)[:depth]
            if (submodule_relative_name := node.module) is not None:
                submodule_relative_path_components = (
                    ModulePath.from_module_name(
                        submodule_relative_name
                    ).components
                )
                components += submodule_relative_path_components
                if is_package:
                    self._scope.set_object(
                        submodule_relative_path_components[0],
                        self._resolve_absolute_module_path(
                            self._scope.module_path.join(
                                submodule_relative_path_components[0]
                            )
                        ),
                    )
            top_submodule_path = ModulePath(*components)
        else:
            assert node.module is not None, ast.unparse(node)
            assert node.level == 0, ast.unparse(node)
            top_submodule_path = ModulePath.from_module_name(node.module)
        top_submodule_object = self._resolve_absolute_module_path(
            top_submodule_path
        )
        for alias in node.names:
            if alias.name == '*':
                assert self._scope.kind in (
                    ScopeKind.DYNAMIC_MODULE,
                    ScopeKind.STATIC_MODULE,
                ), (
                    'Star imports are only allowed on top module level, '
                    f'but found inside {self._scope.kind} '
                    f'with path {self._scope.module_path!r} '
                    f'{self._scope.local_path!r}'
                )
                self._scope.include_object(top_submodule_object)
                continue
            value: Any | Missing
            try:
                object_namespace = top_submodule_object.get_attribute(
                    alias.name
                )
            except KeyError:
                if (
                    submodule_path := top_submodule_object.module_path.join(
                        alias.name
                    )
                ) in self._module_file_paths:
                    object_namespace = self._resolve_absolute_module_path(
                        submodule_path
                    )
                    value = object_namespace.as_object()
                else:
                    object_namespace = PlainObject(
                        ObjectKind.UNKNOWN,
                        top_submodule_object.module_path,
                        LocalObjectPath(alias.name),
                    )
                    top_submodule_object.set_attribute(
                        alias.name, object_namespace
                    )
                    value = MISSING
            else:
                try:
                    value = top_submodule_object.get_value(alias.name)
                except KeyError:
                    value = MISSING
            object_alias_or_name = alias.asname or alias.name
            self._scope.set_object(object_alias_or_name, object_namespace)
            if value is not MISSING:
                self._scope.set_value(object_alias_or_name, value)

    @override
    def visit_If(self, node: ast.If) -> None:
        self.generic_visit(node.test)
        try:
            condition_satisfied = self._evaluate_expression_node(node.test)
        except EVALUATION_EXCEPTIONS:
            with contextlib.suppress(*EVALUATION_EXCEPTIONS):
                for body_node in chain(node.body, node.orelse):
                    self.visit(body_node)
        else:
            for body_node in node.body if condition_satisfied else node.orelse:
                self.visit(body_node)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.generic_visit(node.args)

    @override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    @override
    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        value_object = self._lookup_object_by_expression_node(node.value)
        target_node = node.target
        assert isinstance(target_node.ctx, ast.Store)
        target_name = target_node.id
        self._scope.set_object(
            target_name,
            (
                value_object
                if value_object is not None
                else PlainObject(
                    ObjectKind.UNKNOWN,
                    self._scope.module_path,
                    self._scope.local_path.join(target_name),
                )
            ),
        )

    @override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    @override
    def visit_Try(self, node: ast.Try) -> None:
        try:
            for body_node in node.body:
                self.visit(body_node)
        except Exception as error:
            for handler in node.handlers:
                exception_type_node = handler.type
                exception_type_namespace = (
                    self._lookup_object_by_expression_node(exception_type_node)
                    if exception_type_node is not None
                    else None
                )
                if exception_type_namespace is None or (
                    any(
                        (
                            (
                                exception_type_namespace.module_path
                                == ModulePath.from_module_name(
                                    exception_cls.__module__
                                )
                            )
                            and (
                                exception_type_namespace.local_path
                                == LocalObjectPath.from_object_name(
                                    exception_cls.__qualname__
                                )
                            )
                        )
                        for exception_cls in type(error).mro()[:-1]
                    )
                ):
                    exception_name = handler.name
                    if exception_name is not None:
                        assert exception_type_namespace is not None
                        self._scope.set_object(
                            exception_name,
                            PlainObject(
                                ObjectKind.INSTANCE,
                                self._scope.module_path,
                                self._scope.local_path.join(exception_name),
                                exception_type_namespace,
                            ),
                        )
                    for handler_node in handler.body:
                        self.visit(handler_node)
                    if exception_name is not None:
                        self._scope.delete_object(exception_name)
                    break
            else:
                raise
        else:
            for else_node in node.orelse:
                self.visit(else_node)

    @override
    def visit_With(self, node: ast.With) -> None:
        for item_node in node.items:
            if (target_node := item_node.optional_vars) is not None:
                value_node = item_node.context_expr
                try:
                    value = self._evaluate_expression_node(value_node)
                except EVALUATION_EXCEPTIONS:
                    value = MISSING
                self._process_assignment(target_node, value_node, value)
        try:
            self.generic_visit(node)
        except EVALUATION_EXCEPTIONS as error:
            for item_node in node.items:
                item_expression_node = item_node.context_expr
                if isinstance(item_expression_node, ast.Call):
                    callable_namespace = (
                        self._lookup_object_by_expression_node(
                            item_expression_node.func
                        )
                    )
                    if callable_namespace is None:
                        continue
                    if (
                        callable_namespace.module_path
                        == CONTEXTLIB_MODULE_PATH
                    ) and (
                        callable_namespace.local_path
                        == CONTEXTLIB_SUPPRESS_LOCAL_OBJECT_PATH
                    ):
                        exception_namespaces = [
                            exception_namespace
                            for argument_node in item_expression_node.args
                            if (
                                (
                                    exception_namespace
                                    := self._lookup_object_by_expression_node(
                                        argument_node
                                    )
                                )
                                is not None
                            )
                        ]
                        if any(
                            (
                                (
                                    exception_namespace.module_path
                                    == ModulePath.from_module_name(
                                        exception_cls.__module__
                                    )
                                )
                                and (
                                    exception_namespace.local_path
                                    == (
                                        LocalObjectPath.from_object_name(
                                            exception_cls.__qualname__
                                        )
                                    )
                                )
                                for exception_cls in type(error).mro()[:-1]
                            )
                            for exception_namespace in exception_namespaces
                        ):
                            return
            raise

    def _construct_object_from_expression_node(
        self,
        node: ast.expr,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Object:
        return construct_object_from_expression_node(
            node,
            self._scope,
            *self._parent_scopes,
            context=self._context,
            local_path=local_path,
            module_path=module_path,
        )

    def _evaluate_keyword_arguments(
        self, keyword_argument_nodes: Sequence[ast.keyword], /
    ) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for keyword_argument_node in keyword_argument_nodes:
            if keyword_argument_node.arg is None:
                try:
                    keyword_argument_values = self._evaluate_expression_node(
                        keyword_argument_node.value
                    )
                except EVALUATION_EXCEPTIONS:
                    pass
                else:
                    result.update(keyword_argument_values)
            else:
                try:
                    keyword_argument_value = self._evaluate_expression_node(
                        keyword_argument_node.value
                    )
                except EVALUATION_EXCEPTIONS:
                    keyword_argument_value = MISSING
                result[keyword_argument_node.arg] = keyword_argument_value
        return result

    def _evaluate_expression_node(self, node: ast.expr, /) -> Any:
        return evaluate_expression_node(
            node, self._scope, *self._parent_scopes, context=self._context
        )

    def _get_inherited_scopes(self, /) -> Sequence[Scope]:
        result = (
            self._parent_scopes
            if self._scope.kind in CLASS_SCOPE_KINDS
            else (self._scope, *self._parent_scopes)
        )
        assert (
            len(
                invalid_scopes := [
                    scope
                    for scope in result
                    if scope.kind in CLASS_SCOPE_KINDS
                ]
            )
            == 0
        ), invalid_scopes
        return result

    def _get_module_scope(self, /) -> Scope:
        *_, result, _ = self._scope, *self._parent_scopes
        assert result.kind in (
            ScopeKind.DYNAMIC_MODULE,
            ScopeKind.STATIC_MODULE,
        ), result
        return result

    def _lookup_object_by_expression_node(
        self, node: ast.expr, /
    ) -> Object | None:
        return lookup_object_by_expression_node(
            node, self._scope, *self._parent_scopes, context=self._context
        )

    def _process_assignment(
        self,
        target_node: ast.expr,
        value_node: ast.expr,
        value: Any | Missing,
        /,
    ) -> None:
        assert (
            ctx := getattr(target_node, 'ctx', None)
        ) is None or isinstance(ctx, ast.Store), ast.unparse(target_node)
        resolved_target = self._resolve_assignment_target(target_node)
        for target_object_split_path in _flatten_resolved_assignment_target(
            resolved_target
        ):
            self._set_target_object_split_path(
                target_object_split_path,
                (
                    self._construct_object_from_expression_node(
                        value_node,
                        local_path=target_object_split_path.combine_local(),
                        module_path=target_object_split_path.module,
                    )
                    if isinstance(
                        resolved_target, ResolvedAssignmentTargetSplitPath
                    )
                    else PlainObject(
                        ObjectKind.UNKNOWN,
                        target_object_split_path.module,
                        target_object_split_path.combine_local(),
                    )
                ),
            )
        if value is not MISSING:
            for (
                maybe_target_object_split_path,
                sub_value,
            ) in combine_resolved_assignment_target_with_value(
                resolved_target, value
            ):
                if maybe_target_object_split_path is None:
                    continue
                if (
                    len(maybe_target_object_split_path.relative.components)
                    == 0
                ):
                    assert (
                        len(maybe_target_object_split_path.absolute.components)
                        == 0
                    )
                    continue
                self._resolve_absolute_local_path(
                    maybe_target_object_split_path.module,
                    maybe_target_object_split_path.absolute,
                ).set_nested_value(
                    maybe_target_object_split_path.relative, sub_value
                )
        else:
            for (
                target_object_split_path
            ) in _flatten_resolved_assignment_target(resolved_target):
                if len(target_object_split_path.relative.components) == 0:
                    assert (
                        len(target_object_split_path.absolute.components) == 0
                    )
                    continue
                self._resolve_absolute_local_path(
                    target_object_split_path.module,
                    target_object_split_path.absolute,
                ).safe_delete_nested_value(target_object_split_path.relative)

    def _set_target_object_split_path(
        self,
        target_object_split_path: ResolvedAssignmentTargetSplitPath,
        value_object: Class | Module | PlainObject,
    ) -> None:
        if (
            len(target_object_split_path.absolute.components)
            == len(target_object_split_path.relative.components)
            == 0
        ):
            MODULES[target_object_split_path.module] = value_object
            return
        target_object_or_scope = self._resolve_absolute_local_path(
            target_object_split_path.module, target_object_split_path.absolute
        )
        (
            target_object_or_scope.set_nested_object
            if isinstance(target_object_or_scope, Scope)
            else target_object_or_scope.set_nested_attribute
        )(target_object_split_path.relative, value_object)

    def _resolve_absolute_local_path(
        self, module_path: ModulePath, local_path: LocalObjectPath, /
    ) -> Scope | Object:
        if module_path != self._scope.module_path:
            return self._resolve_absolute_module_path(
                module_path
            ).get_nested_attribute(local_path)
        if local_path.starts_with(self._scope.local_path):
            relative_local_path = LocalObjectPath(
                *local_path.components[
                    len(self._scope.local_path.components) :
                ]
            )
            return (
                self._scope
                if len(relative_local_path.components) == 0
                else self._scope.get_nested_object(relative_local_path)
            )
        module_scope = self._get_module_scope()
        return (
            module_scope
            if len(local_path.components) == 0
            else module_scope.get_nested_object(local_path)
        )

    def _resolve_assignment_target(
        self, node: ast.expr, /
    ) -> ResolvedAssignmentTarget:
        return resolve_assignment_target(
            node,
            self._scope,
            *self._parent_scopes,
            context=self._context,
            name_scopes=self._name_scopes,
        )

    def _resolve_absolute_module_path(
        self, module_path: ModulePath, /
    ) -> Object:
        return resolve_module_path(
            module_path,
            function_definition_nodes=self._function_definition_nodes,
            module_file_paths=self._module_file_paths,
        )

    def _to_complete_positional_arguments(
        self,
        positional_argument_nodes: Sequence[ast.expr],
        callable_namespace: Object,
        /,
    ) -> Sequence[Any | Missing | Starred]:
        result: list[Any] = []
        if callable_namespace.kind is ObjectKind.INSTANCE_ROUTINE:
            instance_local_path = callable_namespace.local_path.parent
            result.append(
                self._resolve_absolute_local_path(
                    callable_namespace.module_path, instance_local_path.parent
                ).get_value_or_else(
                    instance_local_path.components[-1], default=MISSING
                )
            )
        for positional_argument_node in positional_argument_nodes:
            if isinstance(positional_argument_node, ast.Starred):
                try:
                    positional_argument_values = (
                        self._evaluate_expression_node(
                            positional_argument_node.value
                        )
                    )
                except EVALUATION_EXCEPTIONS:
                    result.append(Starred.UNKNOWN)
                else:
                    result.extend(positional_argument_values)
            else:
                try:
                    positional_argument_value = self._evaluate_expression_node(
                        positional_argument_node
                    )
                except EVALUATION_EXCEPTIONS:
                    positional_argument_value = MISSING
                result.append(positional_argument_value)
        return result

    def _visit_any_function_def(
        self, node: ast.AsyncFunctionDef | ast.FunctionDef, /
    ) -> None:
        function_name = node.name
        for decorator_node in node.decorator_list:
            decorator_namespace = self._lookup_object_by_expression_node(
                decorator_node
            )
            if decorator_namespace is None:
                continue
            if decorator_namespace.module_path == BUILTINS_MODULE_PATH and (
                decorator_namespace.local_path
                == BUILTINS_PROPERTY_LOCAL_OBJECT_PATH
            ):
                function_namespace = PlainObject(
                    ObjectKind.PROPERTY,
                    self._scope.module_path,
                    self._scope.local_path.join(function_name),
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_PROPERTY_LOCAL_OBJECT_PATH
                    ),
                )
                break
            if decorator_namespace.module_path == BUILTINS_MODULE_PATH and (
                decorator_namespace.local_path
                in (
                    LocalObjectPath.from_object_name(
                        property.deleter.__qualname__
                    ),
                    LocalObjectPath.from_object_name(
                        property.setter.__qualname__
                    ),
                )
            ):
                return
            if decorator_namespace.module_path == FUNCTOOLS_MODULE_PATH and (
                decorator_namespace.local_path
                == FUNCTOOLS_SINGLEDISPATCH_LOCAL_OBJECT_PATH
            ):
                function_namespace = PlainObject(
                    ObjectKind.ROUTINE,
                    self._scope.module_path,
                    self._scope.local_path.join(function_name),
                    TYPES_MODULE.get_nested_attribute(
                        TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                    ),
                    PlainObject(
                        ObjectKind.UNKNOWN,
                        decorator_namespace.module_path,
                        decorator_namespace.local_path,
                    ),
                )
                break
            if decorator_namespace.module_path == FUNCTOOLS_MODULE_PATH and (
                decorator_namespace.local_path.starts_with(
                    FUNCTOOLS_SINGLEDISPATCH_LOCAL_OBJECT_PATH
                )
            ):
                return
            if decorator_namespace.kind is ObjectKind.CLASS:
                function_namespace = PlainObject(
                    ObjectKind.INSTANCE,
                    self._scope.module_path,
                    self._scope.local_path.join(function_name),
                    decorator_namespace,
                )
                break
        else:
            function_namespace = PlainObject(
                ObjectKind.ROUTINE,
                self._scope.module_path,
                self._scope.local_path.join(function_name),
                TYPES_MODULE.get_nested_attribute(
                    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                ),
            )
        if (
            function_name == '__getattr__'
            and self._scope.kind is ScopeKind.STATIC_MODULE
        ):
            self._scope.mark_module_as_dynamic()
        positional_defaults = []
        for positional_default_node in node.args.defaults:
            try:
                positional_default_value = self._evaluate_expression_node(
                    positional_default_node
                )
            except EVALUATION_EXCEPTIONS:
                positional_default_value = MISSING
            positional_defaults.append(positional_default_value)
        keyword_only_defaults = {}
        for keyword_parameter_node, keyword_default_node in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        ):
            if keyword_default_node is None:
                continue
            try:
                keyword_only_default_value = self._evaluate_expression_node(
                    keyword_default_node
                )
            except EVALUATION_EXCEPTIONS:
                keyword_only_default_value = MISSING
            keyword_only_defaults[keyword_parameter_node.arg] = (
                keyword_only_default_value
            )
        function_namespace.set_attribute(
            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                function_namespace.module_path,
                function_namespace.local_path.join(
                    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME
                ),
                BUILTINS_MODULE.get_nested_attribute(
                    LocalObjectPath.from_object_name(
                        builtins.tuple.__qualname__
                    )
                ),
            ),
        )
        function_namespace.set_value(
            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME, positional_defaults
        )
        function_namespace.set_attribute(
            FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                function_namespace.module_path,
                function_namespace.local_path.join(
                    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
                ),
                BUILTINS_MODULE.get_nested_attribute(
                    LocalObjectPath.from_object_name(
                        builtins.dict.__qualname__
                    )
                ),
            ),
        )
        function_namespace.set_value(
            FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME, keyword_only_defaults
        )
        self._scope.set_object(function_name, function_namespace)
        self._function_definition_nodes[
            self._scope.module_path, self._scope.local_path.join(function_name)
        ] = node


def _to_plain_routine_object(callable_namespace: Object) -> Object:
    if callable_namespace.kind is ObjectKind.INSTANCE_ROUTINE:
        assert isinstance(callable_namespace, PlainObject), callable_namespace
        return callable_namespace.instance_routine_to_routine()
    return callable_namespace


def load_module_file_paths(
    *source_directories: Path,
) -> Mapping[ModulePath, Path | None]:
    result: dict[ModulePath, Path | None] = {
        module_path: None
        for module_name in sys.stdlib_module_names
        if (
            (module_path := ModulePath.checked_from_module_name(module_name))
            is not None
        )
    }
    for module_info in chain(
        pkgutil.iter_modules(),
        pkgutil.iter_modules(map(Path.as_posix, source_directories)),
    ):
        if (
            module_path := ModulePath.checked_from_module_name(
                module_info.name
            )
        ) is not None:
            result[module_path] = module_file_path = (
                _checked_module_file_path_from_module_info(module_info)
            )
            if module_info.ispkg:
                assert module_file_path is not None, module_path
                package_directory_path = module_file_path.parent
                package_module_path = ModulePath.from_module_name(
                    module_info.name
                )
                assert package_directory_path.is_dir(), module_path
                for module_file_path_suffix in (
                    SOURCE_SUFFIXES + EXTENSION_SUFFIXES
                ):
                    for submodule_file_path in package_directory_path.rglob(
                        '*' + module_file_path_suffix
                    ):
                        if submodule_file_path == module_file_path:
                            continue
                        submodule_relative_file_path = (
                            submodule_file_path.relative_to(
                                package_directory_path
                            )
                        )
                        for interim_module_relative_file_path in list(
                            submodule_relative_file_path.parents
                        )[:-1]:
                            try:
                                interim_module_path = package_module_path.join(
                                    *interim_module_relative_file_path.parts
                                )
                            except ValueError:
                                continue
                            if not (
                                package_directory_path
                                / interim_module_relative_file_path
                                / '__init__.py'
                            ).is_file():
                                result[interim_module_path] = (
                                    EMPTY_MODULE_FILE_PATH
                                )
                        try:
                            submodule_path = package_module_path.join(
                                *submodule_relative_file_path.parent.parts,
                                *(
                                    ()
                                    if (
                                        (
                                            submodule_file_name_without_suffix
                                            := (
                                                submodule_relative_file_path.name.removesuffix(
                                                    module_file_path_suffix
                                                )
                                            )
                                        )
                                        == '__init__'
                                    )
                                    else (submodule_file_name_without_suffix,)
                                ),
                            )
                        except ValueError:
                            continue
                        result[submodule_path] = submodule_file_path
    return result


def resolve_module_path(
    module_path: ModulePath,
    /,
    *,
    function_definition_nodes: MutableMapping[
        tuple[ModulePath, LocalObjectPath],
        ast.AsyncFunctionDef | ast.FunctionDef,
    ],
    module_file_paths: Mapping[ModulePath, Path | None],
) -> Object:
    root_component, *rest_components = module_path.components
    root_module_path = ModulePath(root_component)
    result = _load_module_path_namespace(
        root_module_path,
        function_definition_nodes=function_definition_nodes,
        module_file_paths=module_file_paths,
    )
    for component in rest_components:
        try:
            submodule_path = result.module_path.join(component)
            result = _load_module_path_namespace(
                submodule_path,
                function_definition_nodes=function_definition_nodes,
                module_file_paths=module_file_paths,
            )
        except ModuleNotFoundError as error:
            try:
                result = result.get_attribute(component)
            except KeyError:
                raise error from None
    return result


def _checked_module_file_path_from_module_info(
    value: pkgutil.ModuleInfo, /
) -> Path | None:
    module_spec = value.module_finder.find_spec(value.name, None)
    if module_spec is None:
        return None
    module_loader = module_spec.loader
    if not isinstance(module_loader, ExtensionFileLoader | SourceFileLoader):
        return None
    return Path(module_loader.path).resolve(strict=True)


def _flatten_resolved_assignment_target(
    target: ResolvedAssignmentTarget, /
) -> Iterable[ResolvedAssignmentTargetSplitPath]:
    if target is None:
        return
    queue: list[ResolvedAssignmentTarget] = (
        [target]
        if isinstance(target, ResolvedAssignmentTargetSplitPath)
        else list(target)
    )
    while queue:
        candidate = queue.pop()
        if candidate is None:
            continue
        if not isinstance(candidate, ResolvedAssignmentTargetSplitPath):
            queue.extend(candidate)
            continue
        yield candidate


BUILTINS_PROPERTY_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(builtins.property.__qualname__)
)
CONTEXTLIB_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    contextlib.__name__
)
CONTEXTLIB_SUPPRESS_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(contextlib.suppress.__qualname__)
)
FUNCTOOLS_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    functools.__name__
)
FUNCTOOLS_SINGLEDISPATCH_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(functools.singledispatch.__qualname__)
)


def _load_module_path_namespace(
    module_path: ModulePath,
    /,
    *,
    function_definition_nodes: MutableMapping[
        tuple[ModulePath, LocalObjectPath],
        ast.AsyncFunctionDef | ast.FunctionDef,
    ],
    module_file_paths: Mapping[ModulePath, Path | None],
) -> Object:
    try:
        result = MODULES[module_path]
    except KeyError:
        try:
            module_file_path = module_file_paths[module_path]
        except KeyError:
            raise ModuleNotFoundError(module_path) from None
        if module_file_path is None:
            return Module(
                Scope(ScopeKind.BUILTIN_MODULE, module_path, LocalObjectPath())
            )
        if module_file_path.name.endswith(tuple(EXTENSION_SUFFIXES)):
            return Module(
                Scope(
                    ScopeKind.EXTENSION_MODULE, module_path, LocalObjectPath()
                )
            )
        module_source_text = module_file_path.read_text(encoding='utf-8')
        module_node = ast.parse(module_source_text)
        module = types.ModuleType(
            module_path.to_module_name(), ast.get_docstring(module_node)
        )
        module.__file__ = str(module_file_path)
        if module_file_path.name.startswith('__init__.'):
            module.__package__ = module_path.to_module_name()
            module.__path__ = [str(module_file_path.parent)]
        assert module_path not in MODULES
        parse_modules(module)
        result = MODULES[module_path]
        assert isinstance(result, Module), result
        result.set_attribute(
            DICT_FIELD_NAME,
            PlainObject(
                ObjectKind.INSTANCE,
                result.module_path,
                result.local_path.join(DICT_FIELD_NAME),
                BUILTINS_MODULE.get_nested_attribute(
                    BUILTINS_DICT_LOCAL_OBJECT_PATH
                ),
            ),
        )
        result.set_attribute(
            '__class__',
            TYPES_MODULE.get_nested_attribute(
                TYPES_MODULE_TYPE_LOCAL_OBJECT_PATH
            ),
        )
        namespace_parser = ScopeParser(
            result.to_scope(),
            BUILTINS_MODULE.to_scope(),
            context=NullContext(),
            function_definition_nodes=function_definition_nodes,
            module_file_paths=module_file_paths,
        )
        try:
            namespace_parser.visit(module_node)
        except Exception:
            del MODULES[module_path]
            raise
    return result
