from __future__ import annotations

import ast
import builtins
import contextlib
import enum
import functools
import operator
from collections.abc import Mapping, MutableMapping, Sequence
from importlib.machinery import EXTENSION_SUFFIXES
from itertools import chain, repeat, takewhile
from pathlib import Path
from typing import Any, Final

from typing_extensions import override

from .construction import construct_object_from_expression_node
from .context import Context, FunctionCallContext, NullContext
from .enums import ObjectKind, ScopeKind
from .evaluation import (
    EVALUATION_EXCEPTIONS,
    evaluate_expression_node,
    function_node_to_keyword_only_defaults,
    function_node_to_positional_defaults,
    value_to_object,
)
from .lookup import lookup_object_by_expression_node
from .missing import MISSING, Missing
from .modules import BUILTINS_MODULE, BUILTINS_OBJECT, MODULES, TYPES_MODULE
from .object_ import (
    CLASS_OBJECT_CLASSES,
    CLASS_SCOPE_KINDS,
    Class,
    ClassObject,
    Descriptor,
    Instance,
    Module,
    MutableObject,
    Object,
    Routine,
    UnknownObject,
)
from .object_path import (
    BUILTINS_DICT_LOCAL_OBJECT_PATH,
    BUILTINS_LIST_LOCAL_OBJECT_PATH,
    BUILTINS_MODULE_PATH,
    BUILTINS_STR_LOCAL_OBJECT_PATH,
    BUILTINS_TUPLE_LOCAL_OBJECT_PATH,
    BUILTINS_TYPE_LOCAL_OBJECT_PATH,
    DICT_FIELD_NAME,
    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    TYPES_CODE_TYPE_LOCAL_OBJECT_PATH,
    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH,
    TYPES_NONE_TYPE_LOCAL_OBJECT_PATH,
)
from .resolution import (
    ResolvedAssignmentTarget,
    ResolvedAssignmentTargetSplitPath,
    checked_combine_resolved_assignment_target_with_value,
    flatten_resolved_assignment_target,
    resolve_assignment_target,
)
from .scope import Scope
from .utils import AnyFunctionDefinitionAstNode, ensure_type

MODULE_FIELD_NAME: Final = '__module__'
NAME_FIELD_NAME: Final = '__name__'
QUALNAME_FIELD_NAME: Final = '__qualname__'

TYPES_MODULE_TYPE_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = LocalObjectPath(
    'ModuleType'
)


def _does_function_modify_caller_global_state(
    function_object: Routine,
    /,
    *function_call_scopes: Scope,
    cache: dict[tuple[ModulePath, LocalObjectPath], bool] = {},  # noqa: B006
    caller_module_scope: Scope,
    keyword_arguments: Mapping[str, Any],
    module_file_paths: Mapping[ModulePath, Path | None],
    positional_arguments: Sequence[Any | Missing | Starred],
) -> bool:
    cache_key = (function_object.module_path, function_object.local_path)
    try:
        return cache[cache_key]
    except KeyError:
        function_definition_node = (
            function_object.ast_node
            if isinstance(function_object, Routine)
            else None
        )
        if function_definition_node is None:
            cache[cache_key] = result = (
                function_object.module_path == BUILTINS_MODULE_PATH
                and (
                    function_object.local_path
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
            function_object.module_path,
            function_object.local_path,
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
                value_to_object(
                    positional_argument,
                    module_path=function_object.module_path,
                    local_path=function_object.local_path.join(
                        positional_parameter_name
                    ),
                ),
            )
        positional_defaults = function_object.get_attribute(
            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME
        ).value
        assert isinstance(positional_defaults, Sequence), positional_defaults
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
                value_to_object(
                    positional_default,
                    module_path=function_object.module_path,
                    local_path=function_object.local_path.join(
                        defaulted_positional_parameter_name
                    ),
                ),
            )
        keyword_only_defaults = function_object.get_attribute(
            FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
        ).value
        assert isinstance(keyword_only_defaults, Mapping), (
            keyword_only_defaults
        )
        for keyword_parameter_node in signature_node.kwonlyargs:
            keyword_parameter_name = keyword_parameter_node.arg
            try:
                keyword_argument = keyword_argument_dict.pop(
                    keyword_parameter_name
                )
            except KeyError:
                keyword_argument = keyword_only_defaults[
                    keyword_parameter_name
                ]
            function_scope.set_object(
                keyword_parameter_name,
                value_to_object(
                    keyword_argument,
                    module_path=function_object.module_path,
                    local_path=function_object.local_path.join(
                        keyword_parameter_name
                    ),
                ),
            )
        if (
            variadic_positional_parameter_node := signature_node.vararg
        ) is not None:
            variadic_positional_parameter_name = (
                variadic_positional_parameter_node.arg
            )
            function_scope.set_object(
                variadic_positional_parameter_name,
                Instance(
                    function_object.module_path,
                    function_object.local_path.join(
                        variadic_positional_parameter_name
                    ),
                    cls=ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_TUPLE_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=tuple(
                        positional_arguments[len(positional_parameter_nodes) :]
                    ),
                ),
            )
        if (
            variadic_keyword_parameter_node := signature_node.kwarg
        ) is not None:
            variadic_keyword_parameter_name = (
                variadic_keyword_parameter_node.arg
            )
            function_scope.set_object(
                variadic_keyword_parameter_name,
                Instance(
                    function_object.module_path,
                    function_object.local_path.join(
                        variadic_keyword_parameter_name
                    ),
                    cls=ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_DICT_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=keyword_argument_dict,
                ),
            )
        function_body_parser = ScopeParser(
            function_scope,
            BUILTINS_MODULE.to_scope(),
            *function_call_scopes,
            context=FunctionCallContext(caller_module_scope.module_path),
            module_file_paths=module_file_paths,
        )
        cache[cache_key] = result = False
        for function_body_node in (
            [function_definition_node.body]
            if isinstance(function_definition_node, ast.Lambda)
            else function_definition_node.body
        ):
            function_body_parser.visit(function_body_node)
            if caller_module_scope.kind is ScopeKind.DYNAMIC_MODULE:
                result = True
                break
        if len(positional_arguments) > 0 or len(keyword_arguments) > 0:
            del cache[cache_key]
        return result


class Starred(enum.Enum):
    UNKNOWN = enum.auto()


def _is_package_module_path(
    module_path: ModulePath,
    /,
    *,
    module_file_paths: Mapping[ModulePath, Path | None],
) -> bool:
    return (
        module_file_path := module_file_paths.get(module_path)
    ) is not None and module_file_path.name.startswith('__init__.')


class ScopeParser(ast.NodeVisitor):
    def __init__(
        self,
        scope: Scope,
        /,
        *parent_scopes: Scope,
        context: Context,
        module_file_paths: Mapping[ModulePath, Path | None],
    ) -> None:
        super().__init__()
        (
            self._context,
            self._module_file_paths,
            self._scope,
            self._parent_scopes,
        ) = context, module_file_paths, scope, parent_scopes
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
                except EVALUATION_EXCEPTIONS as error:
                    with contextlib.suppress(type(error)):
                        self.visit(operand_node)
                else:
                    self.visit(operand_node)
                    if not operand_value:
                        break
            return
        assert isinstance(node.op, ast.Or), ast.unparse(node)
        for operand_node in node.values:
            try:
                operand_value = self._evaluate_expression_node(operand_node)
            except EVALUATION_EXCEPTIONS as error:
                with contextlib.suppress(type(error)):
                    self.visit(operand_node)
            else:
                self.visit(operand_node)
                if operand_value:
                    break

    @override
    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        callable_object = self._lookup_object_by_expression_node(node.func)
        if callable_object is None:
            return
        if callable_object.module_path == BUILTINS_MODULE_PATH and (
            callable_object.local_path
            == LocalObjectPath.from_object_name('__import__')
        ):
            try:
                module_name = self._evaluate_expression_node(node.args[0])
            except EVALUATION_EXCEPTIONS:
                pass
            else:
                if isinstance(module_name, str):
                    self._resolve_absolute_module_path(
                        ModulePath.from_module_name(module_name)
                    )
            return
        if (
            callable_object.local_path
            == LocalObjectPath(DICT_FIELD_NAME, 'update')
        ) and (
            module_scope := ensure_type(
                MODULES[callable_object.module_path], Module
            ).to_scope()
        ).kind is ScopeKind.STATIC_MODULE:
            module_scope.mark_module_as_dynamic()
            return
        if callable_object.kind in (ObjectKind.METHOD, ObjectKind.ROUTINE):
            function_object = _to_plain_routine_object(callable_object)
            if (
                (self._get_module_scope().kind is ScopeKind.STATIC_MODULE)
                and _does_function_modify_caller_global_state(
                    function_object,
                    *(
                        self._get_inherited_scopes()
                        if (
                            function_object.module_path
                            == self._scope.module_path
                            and (
                                function_object.local_path.parent
                                == self._scope.local_path
                            )
                        )
                        else (
                            ensure_type(
                                self._resolve_absolute_module_path(
                                    function_object.module_path
                                ),
                                Module,
                            ).to_scope(),
                            BUILTINS_MODULE.to_scope(),
                        )
                    ),
                    caller_module_scope=self._get_module_scope(),
                    keyword_arguments=self._evaluate_keyword_arguments(
                        node.keywords
                    ),
                    module_file_paths=self._module_file_paths,
                    positional_arguments=self._to_complete_positional_arguments(
                        node.args, callable_object
                    ),
                )
                and (
                    (module_scope := self._get_module_scope()).kind
                    is ScopeKind.STATIC_MODULE
                )
            ):
                module_scope.mark_module_as_dynamic()
                return

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        cls_name = node.name
        cls_module_path = self._scope.module_path
        cls_local_path = self._scope.local_path.join(cls_name)
        bases: list[ClassObject] = []
        for index, base_node in enumerate(node.bases):
            self.visit(base_node)
            base_cls = ensure_type(
                self._construct_object_from_expression_node(
                    base_node,
                    local_path=cls_local_path.join(
                        # FIXME: add support for indexing?
                        f'__bases_{index}__'
                    ),
                    module_path=cls_module_path,
                ),
                CLASS_OBJECT_CLASSES,
            )
            if base_cls is None:
                continue
            bases.append(base_cls)
        cls_scope = Scope(
            (
                ScopeKind.METACLASS
                if any(
                    base_cls_object.kind is ObjectKind.METACLASS
                    for base_cls_object in bases
                )
                else (
                    ScopeKind.UNKNOWN_CLASS
                    if any(
                        (
                            base_cls_object.kind
                            in (ObjectKind.UNKNOWN, ObjectKind.UNKNOWN_CLASS)
                        )
                        for base_cls_object in bases
                    )
                    else ScopeKind.CLASS
                )
            ),
            self._scope.module_path,
            cls_local_path,
        )
        cls_parser = ScopeParser(
            cls_scope,
            *self._get_inherited_scopes(),
            context=self._context,
            module_file_paths=self._module_file_paths,
        )
        for body_node in node.body:
            cls_parser.visit(body_node)
        metacls: Class | Missing = MISSING
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                candidate_metacls = ensure_type(
                    self._construct_object_from_expression_node(
                        keyword.value,
                        local_path=cls_local_path.join('__class__'),
                        module_path=cls_module_path,
                    ),
                    Class,
                )
                if candidate_metacls is None:
                    continue
                assert candidate_metacls.kind in (
                    ObjectKind.METACLASS,
                    ObjectKind.UNKNOWN_CLASS,
                )
                metacls = candidate_metacls
        cls_object = Class(
            cls_scope,
            *([BUILTINS_OBJECT] if len(node.bases) == 0 else bases),
            metacls=(
                ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_TYPE_LOCAL_OBJECT_PATH
                    ),
                    Class,
                )
                if metacls is MISSING and len(node.bases) == 0
                else metacls
            ),
        )
        cls_object.set_attribute(
            DICT_FIELD_NAME,
            Instance(
                cls_module_path,
                cls_local_path.join(DICT_FIELD_NAME),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_DICT_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=MISSING,
            ),
        )
        cls_object.set_attribute(
            MODULE_FIELD_NAME,
            Instance(
                cls_module_path,
                cls_local_path.join(MODULE_FIELD_NAME),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_STR_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=cls_module_path.to_module_name(),
            ),
        )
        cls_object.set_attribute(
            NAME_FIELD_NAME,
            Instance(
                cls_module_path,
                cls_local_path.join(NAME_FIELD_NAME),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_STR_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=cls_name,
            ),
        )
        cls_object.set_attribute(
            QUALNAME_FIELD_NAME,
            Instance(
                cls_module_path,
                cls_local_path.join(QUALNAME_FIELD_NAME),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_STR_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=cls_local_path.to_object_name(),
            ),
        )
        for decorator_node in node.decorator_list:
            self.visit(decorator_node)
            decorator_object = self._lookup_object_by_expression_node(
                decorator_node
            )
            assert decorator_object is not None
            if decorator_object.kind in (
                ObjectKind.METHOD,
                ObjectKind.ROUTINE,
            ):
                function_object = _to_plain_routine_object(decorator_object)
                if (
                    self._get_module_scope().kind is ScopeKind.STATIC_MODULE
                    and _does_function_modify_caller_global_state(
                        function_object,
                        *(
                            self._get_inherited_scopes()
                            if (
                                function_object.module_path
                                == self._scope.module_path
                                and (
                                    function_object.local_path.parent
                                    == self._scope.local_path
                                )
                            )
                            else (
                                ensure_type(
                                    self._resolve_absolute_module_path(
                                        function_object.module_path
                                    ),
                                    Module,
                                ).to_scope(),
                                BUILTINS_MODULE.to_scope(),
                            )
                        ),
                        caller_module_scope=self._get_module_scope(),
                        keyword_arguments={},
                        module_file_paths=self._module_file_paths,
                        positional_arguments=[
                            *self._to_complete_positional_arguments(
                                [], decorator_object
                            ),
                            MISSING,
                        ],
                    )
                    and (
                        (module_scope := self._get_module_scope()).kind
                        is ScopeKind.STATIC_MODULE
                    )
                ):
                    module_scope.mark_module_as_dynamic()
                    continue
        self._scope.set_object(cls_name, cls_object)

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    @override
    def visit_Expr(self, node: ast.Expr) -> None:
        value_node = node.value
        if isinstance(value_node, ast.Name):
            assert isinstance(value_node.ctx, ast.Load), ast.unparse(node)
            self._lookup_object_by_expression_node(node.value)
        else:
            self.generic_visit(node)

    @override
    def visit_For(self, node: ast.For) -> None:
        resolved_target = self._resolve_assignment_target(node.target)
        assert resolved_target is not None
        try:
            iterable = [*self._evaluate_expression_node(node.iter)]
        except EVALUATION_EXCEPTIONS:
            for target_object_split_path in flatten_resolved_assignment_target(
                resolved_target
            ):
                self._set_target_object_split_path(
                    target_object_split_path,
                    UnknownObject(
                        self._scope.module_path,
                        target_object_split_path.combine_local(),
                        value=MISSING,
                    ),
                )
            self.generic_visit(node)
        else:
            for element in iterable:
                for (
                    maybe_target_object_split_path,
                    value,
                ) in checked_combine_resolved_assignment_target_with_value(
                    resolved_target, element
                ):
                    if maybe_target_object_split_path is None:
                        continue
                    parent_scope_or_object = (
                        self._resolve_absolute_local_path_of_mutable_object(
                            maybe_target_object_split_path.module,
                            maybe_target_object_split_path.absolute,
                        )
                    )
                    (
                        parent_scope_or_object.set_nested_object
                        if isinstance(parent_scope_or_object, Scope)
                        else parent_scope_or_object.set_nested_attribute
                    )(
                        maybe_target_object_split_path.relative,
                        value_to_object(
                            value,
                            module_path=self._scope.module_path,
                            local_path=maybe_target_object_split_path.combine_local(),
                        ),
                    )
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
                for submodule_path in submodule_paths:
                    next_submodule_object = self._resolve_absolute_module_path(
                        submodule_path
                    )
                    submodule_last_name = submodule_path.components[-1]
                    submodule_object.set_attribute(
                        submodule_last_name, next_submodule_object
                    )
                    submodule_object = next_submodule_object

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        import_is_relative = node.level > 0
        if import_is_relative:
            is_package = _is_package_module_path(
                self._scope.module_path,
                module_file_paths=self._module_file_paths,
            )
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
                if is_package and node.level == 1:
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
            try:
                object_ = top_submodule_object.get_attribute(alias.name)
            except KeyError:
                if (
                    submodule_path := top_submodule_object.module_path.join(
                        alias.name
                    )
                ) in self._module_file_paths:
                    object_ = self._resolve_absolute_module_path(
                        submodule_path
                    )
                else:
                    object_ = UnknownObject(
                        top_submodule_object.module_path,
                        LocalObjectPath(alias.name),
                        value=MISSING,
                    )
                    top_submodule_object.set_attribute(alias.name, object_)
            self._scope.set_object(alias.asname or alias.name, object_)

    @override
    def visit_If(self, node: ast.If) -> None:
        self.generic_visit(node.test)
        try:
            condition_satisfied = self._evaluate_expression_node(node.test)
        except EVALUATION_EXCEPTIONS:
            for body_node in chain(node.body, node.orelse):
                with contextlib.suppress(
                    ModuleNotFoundError, *EVALUATION_EXCEPTIONS
                ):
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
                else UnknownObject(
                    self._scope.module_path,
                    self._scope.local_path.join(target_name),
                    value=MISSING,
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
                exception_cls_object = (
                    self._lookup_object_by_expression_node(exception_type_node)
                    if exception_type_node is not None
                    else None
                )
                if exception_cls_object is None or (
                    any(
                        (
                            (
                                exception_cls_object.module_path
                                == ModulePath.from_module_name(
                                    exception_cls.__module__
                                )
                            )
                            and (
                                exception_cls_object.local_path
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
                        assert exception_cls_object is not None
                        self._scope.set_object(
                            exception_name,
                            Instance(
                                self._scope.module_path,
                                self._scope.local_path.join(exception_name),
                                cls=ensure_type(
                                    exception_cls_object,
                                    (Class, UnknownObject),
                                ),
                                value=MISSING,
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
                    callable_object = self._lookup_object_by_expression_node(
                        item_expression_node.func
                    )
                    if callable_object is None:
                        continue
                    if (
                        callable_object.module_path == CONTEXTLIB_MODULE_PATH
                    ) and (
                        callable_object.local_path
                        == CONTEXTLIB_SUPPRESS_LOCAL_OBJECT_PATH
                    ):
                        exception_objects = [
                            exception_object
                            for argument_node in item_expression_node.args
                            if (
                                (
                                    exception_object
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
                                    exception_object.module_path
                                    == ModulePath.from_module_name(
                                        exception_cls.__module__
                                    )
                                )
                                and (
                                    exception_object.local_path
                                    == (
                                        LocalObjectPath.from_object_name(
                                            exception_cls.__qualname__
                                        )
                                    )
                                )
                                for exception_cls in type(error).mro()[:-1]
                            )
                            for exception_object in exception_objects
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
                    keyword_argument_values = {
                        **self._evaluate_expression_node(
                            keyword_argument_node.value
                        )
                    }
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
        ).value

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
        for (
            target_object_split_path,
            sub_value,
        ) in checked_combine_resolved_assignment_target_with_value(
            resolved_target, value
        ):
            if target_object_split_path is None:
                continue
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
                    else value_to_object(
                        sub_value,
                        module_path=target_object_split_path.module,
                        local_path=target_object_split_path.combine_local(),
                    )
                ),
            )

    def _set_target_object_split_path(
        self,
        target_object_split_path: ResolvedAssignmentTargetSplitPath,
        value_object: Object,
    ) -> None:
        if (
            len(target_object_split_path.absolute.components)
            == len(target_object_split_path.relative.components)
            == 0
        ):
            assert isinstance(value_object, MutableObject), value_object
            MODULES[target_object_split_path.module] = value_object
            return
        target_object_or_scope = (
            self._resolve_absolute_local_path_of_mutable_object(
                target_object_split_path.module,
                target_object_split_path.absolute,
            )
        )
        (
            target_object_or_scope.set_nested_object
            if isinstance(target_object_or_scope, Scope)
            else target_object_or_scope.set_nested_attribute
        )(target_object_split_path.relative, value_object)

    def _resolve_absolute_local_path_of_mutable_object(
        self, module_path: ModulePath, local_path: LocalObjectPath, /
    ) -> Scope | MutableObject:
        if module_path != self._scope.module_path:
            return self._resolve_absolute_module_path(
                module_path
            ).get_mutable_nested_attribute(local_path)
        if local_path.starts_with(self._scope.local_path):
            relative_local_path = LocalObjectPath(
                *local_path.components[
                    len(self._scope.local_path.components) :
                ]
            )
            return (
                self._scope
                if len(relative_local_path.components) == 0
                else self._scope.get_mutable_nested_object(relative_local_path)
            )
        module_scope = self._get_module_scope()
        return (
            module_scope
            if len(local_path.components) == 0
            else module_scope.get_mutable_nested_object(local_path)
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
    ) -> MutableObject:
        return resolve_module_path(
            module_path, module_file_paths=self._module_file_paths
        )

    def _to_complete_positional_arguments(
        self,
        positional_argument_nodes: Sequence[ast.expr],
        callable_object: Object,
        /,
    ) -> Sequence[Any | Missing | Starred]:
        result: list[Any] = []
        if callable_object.kind is ObjectKind.METHOD:
            result.append(callable_object.instance)
        for positional_argument_node in positional_argument_nodes:
            if isinstance(positional_argument_node, ast.Starred):
                try:
                    positional_argument_values = [
                        *self._evaluate_expression_node(
                            positional_argument_node.value
                        )
                    ]
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
        self, node: AnyFunctionDefinitionAstNode, /
    ) -> None:
        function_name = node.name
        function_object: Object
        function_local_path = self._scope.local_path.join(function_name)
        signature_node = node.args
        keyword_only_defaults = function_node_to_keyword_only_defaults(
            signature_node,
            self._scope,
            *self._parent_scopes,
            context=self._context,
        )
        positional_defaults = function_node_to_positional_defaults(
            signature_node,
            self._scope,
            *self._parent_scopes,
            context=self._context,
        )
        for decorator_node in node.decorator_list:
            decorator_object = self._lookup_object_by_expression_node(
                decorator_node
            )
            if decorator_object is None:
                continue
            if decorator_object.module_path == BUILTINS_MODULE_PATH and (
                decorator_object.local_path
                == BUILTINS_PROPERTY_LOCAL_OBJECT_PATH
            ):
                assert decorator_object.kind is ObjectKind.CLASS, (
                    decorator_object
                )
                function_object = Descriptor(
                    self._scope.module_path,
                    function_local_path,
                    ast_node=node,
                    cls=ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_PROPERTY_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                )
                break
            if (
                decorator_object.kind is ObjectKind.METHOD
                and (
                    decorator_object.routine.module_path
                    == BUILTINS_MODULE_PATH
                )
                and (
                    decorator_object.routine.local_path
                    in (
                        LocalObjectPath.from_object_name(
                            property.deleter.__qualname__
                        ),
                        LocalObjectPath.from_object_name(
                            property.setter.__qualname__
                        ),
                    )
                )
            ):
                return
            if decorator_object.module_path == FUNCTOOLS_MODULE_PATH and (
                decorator_object.local_path
                == FUNCTOOLS_SINGLEDISPATCH_LOCAL_OBJECT_PATH
            ):
                function_object = Routine(
                    self._scope.module_path,
                    function_local_path,
                    ast_node=node,
                    cls=Class(
                        Scope(
                            ScopeKind.UNKNOWN_CLASS,
                            decorator_object.module_path,
                            decorator_object.local_path,
                        ),
                        ensure_type(
                            TYPES_MODULE.get_nested_attribute(
                                TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                            ),
                            Class,
                        ),
                        metacls=MISSING,
                    ),
                    keyword_only_defaults=keyword_only_defaults,
                    positional_defaults=positional_defaults,
                )
                break
            if (
                decorator_object.kind is ObjectKind.ROUTINE_CALL
                and (
                    decorator_object.callable_.module_path
                    == FUNCTOOLS_MODULE_PATH
                )
                and (
                    decorator_object.callable_.local_path.starts_with(
                        FUNCTOOLS_SINGLEDISPATCH_LOCAL_OBJECT_PATH
                    )
                )
            ):
                return
            if decorator_object.kind is ObjectKind.CLASS:
                function_object = Instance(
                    self._scope.module_path,
                    function_local_path,
                    cls=decorator_object,
                    value=MISSING,
                )
                if decorator_object.module_path == BUILTINS_MODULE_PATH and (
                    decorator_object.local_path
                    == LocalObjectPath.from_object_name(
                        builtins.classmethod.__qualname__
                    )
                ):
                    wrapped_object = Routine(
                        self._scope.module_path,
                        function_local_path.join('__func__'),
                        ast_node=node,
                        cls=ensure_type(
                            TYPES_MODULE.get_nested_attribute(
                                TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                            ),
                            Class,
                        ),
                        keyword_only_defaults=keyword_only_defaults,
                        positional_defaults=positional_defaults,
                    )
                    function_object.set_attribute('__func__', wrapped_object)
                break
        else:
            function_object = Routine(
                self._scope.module_path,
                function_local_path,
                ast_node=node,
                cls=ensure_type(
                    TYPES_MODULE.get_nested_attribute(
                        TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                keyword_only_defaults=keyword_only_defaults,
                positional_defaults=positional_defaults,
            )
            function_object.set_attribute(
                '__code__',
                Instance(
                    self._scope.module_path,
                    function_local_path.join('__code__'),
                    cls=ensure_type(
                        TYPES_MODULE.get_nested_attribute(
                            TYPES_CODE_TYPE_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=MISSING,
                ),
            )
        if (
            function_name == '__getattr__'
            and self._scope.kind is ScopeKind.STATIC_MODULE
        ):
            self._scope.mark_module_as_dynamic()
        self._scope.set_object(function_name, function_object)


def _to_plain_routine_object(callable_object: Object, /) -> Routine:
    if callable_object.kind is ObjectKind.METHOD:
        result = callable_object.routine
        assert isinstance(result, Routine)
        return result
    assert isinstance(callable_object, Routine)
    return callable_object


def resolve_module_path(
    module_path: ModulePath,
    /,
    *,
    module_file_paths: Mapping[ModulePath, Path | None],
) -> MutableObject:
    root_component, *rest_components = module_path.components
    root_module_path = ModulePath(root_component)
    result = _load_module_by_path(
        root_module_path, module_file_paths=module_file_paths
    )
    for component in rest_components:
        submodule_path = result.module_path.join(component)
        try:
            next_result = _load_module_by_path(
                submodule_path, module_file_paths=module_file_paths
            )
        except ModuleNotFoundError as error:
            try:
                result = result.get_mutable_attribute(component)
            except KeyError:
                raise error from None
        else:
            result.set_attribute(component, next_result)
            result = next_result
    return result


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


def _load_module_by_path(
    module_path: ModulePath,
    /,
    *,
    module_file_paths: Mapping[ModulePath, Path | None],
) -> MutableObject:
    try:
        return MODULES[module_path]
    except KeyError:
        pass
    try:
        module_file_path = module_file_paths[module_path]
    except KeyError:
        raise ModuleNotFoundError(module_path) from None
    if module_file_path is None:
        MODULES[module_path] = result = Module(
            Scope(ScopeKind.BUILTIN_MODULE, module_path, LocalObjectPath())
        )
    elif module_file_path.name.endswith(tuple(EXTENSION_SUFFIXES)):
        MODULES[module_path] = result = Module(
            Scope(ScopeKind.EXTENSION_MODULE, module_path, LocalObjectPath())
        )
    else:
        module_source_text = module_file_path.read_text(encoding='utf-8')
        module_node = ast.parse(module_source_text)
        assert module_path not in MODULES
        module_scope = Scope(
            ScopeKind.STATIC_MODULE, module_path, LocalObjectPath()
        )
        result = MODULES[module_path] = Module(module_scope)
        result.set_attribute(
            '__file__',
            Instance(
                module_path,
                LocalObjectPath('__file__'),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_STR_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=str(module_file_path),
            ),
        )
        module_docstring = ast.get_docstring(module_node)
        if module_docstring is not None:
            assert isinstance(module_docstring, str), module_docstring
            result.set_attribute(
                '__doc__',
                Instance(
                    module_path,
                    LocalObjectPath('__doc__'),
                    cls=ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_STR_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=module_docstring,
                ),
            )
        else:
            result.set_attribute(
                '__doc__',
                Instance(
                    module_path,
                    LocalObjectPath('__doc__'),
                    cls=ensure_type(
                        TYPES_MODULE.get_nested_attribute(
                            TYPES_NONE_TYPE_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=module_docstring,
                ),
            )
        if module_file_path.name.startswith('__init__.'):
            result.set_attribute(
                '__package__',
                Instance(
                    module_path,
                    LocalObjectPath('__package__'),
                    cls=ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_STR_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=module_path.to_module_name(),
                ),
            )
            result.set_attribute(
                '__path__',
                Instance(
                    module_path,
                    LocalObjectPath('__path__'),
                    cls=ensure_type(
                        BUILTINS_MODULE.get_nested_attribute(
                            BUILTINS_LIST_LOCAL_OBJECT_PATH
                        ),
                        Class,
                    ),
                    value=[str(module_file_path.parent)],
                ),
            )
        assert isinstance(result, Module), result
        result.set_attribute(
            DICT_FIELD_NAME,
            Instance(
                result.module_path,
                result.local_path.join(DICT_FIELD_NAME),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_DICT_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=MISSING,
            ),
        )
        result.set_attribute(
            NAME_FIELD_NAME,
            Instance(
                result.module_path,
                result.local_path.join(NAME_FIELD_NAME),
                cls=ensure_type(
                    BUILTINS_MODULE.get_nested_attribute(
                        BUILTINS_STR_LOCAL_OBJECT_PATH
                    ),
                    Class,
                ),
                value=module_path.to_module_name(),
            ),
        )
        scope_parser = ScopeParser(
            module_scope,
            BUILTINS_MODULE.to_scope(),
            context=NullContext(),
            module_file_paths=module_file_paths,
        )
        try:
            scope_parser.visit(module_node)
        except Exception:
            del MODULES[module_path]
            raise
    return result
