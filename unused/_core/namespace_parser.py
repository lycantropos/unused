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
from collections.abc import (
    Container,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
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

from .construction import construct_namespace_from_expression_node
from .context import Context, FunctionCallContext, NullContext
from .evaluation import EVALUATION_EXCEPTIONS, evaluate_expression_node
from .lookup import lookup_namespace_by_expression_node
from .missing import MISSING, Missing
from .module_namespaces import (
    BUILTINS_MODULE_NAMESPACE,
    MODULE_NAMESPACES,
    TYPES_MODULE_NAMESPACE,
    parse_modules,
)
from .namespace import Namespace, ObjectKind
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

FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME: Final = '__defaults__'
FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME: Final = '__kwdefaults__'
MODULE_FIELD_NAME: Final = '__module__'
NAME_FIELD_NAME: Final = '__name__'
QUALNAME_FIELD_NAME: Final = '__qualname__'

CLASS_OBJECT_KINDS: Final[Container[ObjectKind]] = (
    ObjectKind.CLASS,
    ObjectKind.METACLASS,
    ObjectKind.UNKNOWN_CLASS,
)
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
    function_namespace: Namespace,
    /,
    *,
    cache: dict[tuple[ModulePath, LocalObjectPath], bool] = {},  # noqa: B006
    caller_module_namespace: Namespace,
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
        function_scope_namespace = Namespace(
            ObjectKind.FUNCTION_SCOPE,
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
            function_scope_namespace.set_namespace_by_name(
                positional_parameter_name,
                Namespace(
                    ObjectKind.UNKNOWN,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        positional_parameter_name
                    ),
                ),
            )
            if positional_argument is not MISSING:
                function_scope_namespace.set_object_by_name(
                    positional_parameter_name, positional_argument
                )
        positional_defaults = function_namespace.get_object_by_name(
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
            function_scope_namespace.set_namespace_by_name(
                defaulted_positional_parameter_name,
                Namespace(
                    ObjectKind.UNKNOWN,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        defaulted_positional_parameter_name
                    ),
                ),
            )
            if positional_default is not MISSING:
                function_scope_namespace.set_object_by_name(
                    defaulted_positional_parameter_name, positional_default
                )
        for keyword_parameter_node in signature_node.kwonlyargs:
            keyword_parameter_name = keyword_parameter_node.arg
            function_scope_namespace.set_namespace_by_name(
                keyword_parameter_name,
                Namespace(
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
                keyword_argument = function_namespace.get_object_by_name(
                    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
                )[keyword_parameter_name]
            if keyword_argument is not MISSING:
                function_scope_namespace.set_object_by_name(
                    keyword_parameter_name, keyword_argument
                )
        if (
            variadic_positional_parameter_node := signature_node.vararg
        ) is not None:
            variadic_positional_parameter_name = (
                variadic_positional_parameter_node.arg
            )
            function_scope_namespace.set_namespace_by_name(
                variadic_positional_parameter_name,
                Namespace(
                    ObjectKind.INSTANCE,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        variadic_positional_parameter_name
                    ),
                    BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                        LocalObjectPath.from_object_name(tuple.__qualname__)
                    ),
                ),
            )
            function_scope_namespace.set_object_by_name(
                variadic_positional_parameter_name,
                tuple(positional_arguments[len(positional_parameter_nodes) :]),
            )
        if (
            variadic_keyword_parameter_node := signature_node.kwarg
        ) is not None:
            variadic_keyword_parameter_name = (
                variadic_keyword_parameter_node.arg
            )
            function_scope_namespace.set_namespace_by_name(
                variadic_keyword_parameter_name,
                Namespace(
                    ObjectKind.INSTANCE,
                    function_namespace.module_path,
                    function_namespace.local_path.join(
                        variadic_keyword_parameter_name
                    ),
                    BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                        LocalObjectPath.from_object_name(dict.__qualname__)
                    ),
                ),
            )
            function_scope_namespace.set_object_by_name(
                variadic_keyword_parameter_name, keyword_argument_dict
            )
        function_body_parser = NamespaceParser(
            function_scope_namespace,
            MODULE_NAMESPACES[function_namespace.module_path],
            BUILTINS_MODULE_NAMESPACE,
            context=FunctionCallContext(caller_module_namespace.module_path),
            function_definition_nodes=function_definition_nodes,
            module_file_paths=module_file_paths,
        )
        cache[cache_key] = result = False
        for function_body_node in function_definition_node.body:
            function_body_parser.visit(function_body_node)
            if caller_module_namespace.kind is ObjectKind.DYNAMIC_MODULE:
                result = True
                break
        if len(positional_arguments) > 0 or len(keyword_arguments) > 0:
            del cache[cache_key]
        return result


class Starred(enum.Enum):
    UNKNOWN = enum.auto()


class NamespaceParser(ast.NodeVisitor):
    def __init__(
        self,
        namespace: Namespace,
        /,
        *parent_namespaces: Namespace,
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
            self._namespace,
            self._parent_namespaces,
        ) = (
            context,
            function_definition_nodes,
            module_file_paths,
            namespace,
            parent_namespaces,
        )

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        target_node = node.target
        assert isinstance(target_node, ast.Name), ast.unparse(node)
        assert isinstance(target_node.ctx, ast.Store), ast.unparse(node)
        target_name = target_node.id
        value_local_path = self._namespace.local_path.join(target_name)
        value_module_path = self._namespace.module_path
        self._namespace.set_namespace_by_name(
            target_name,
            self._construct_namespace_from_expression_node(
                value_node,
                local_path=value_local_path,
                module_path=value_module_path,
            )
            if (value_node := node.value) is not None
            else Namespace(
                ObjectKind.UNKNOWN, value_module_path, value_local_path
            ),
        )
        if (value_node := node.value) is not None:
            try:
                value = self._evaluate_expression_node(value_node)
            except EVALUATION_EXCEPTIONS:
                self._namespace.safe_delete_object_by_name(target_name)
            else:
                self._namespace.set_object_by_name(target_name, value)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        try:
            value = self._evaluate_expression_node(node.value)
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        self.generic_visit(node)
        for target_node in node.targets:
            assert (
                ctx := getattr(target_node, 'ctx', None)
            ) is None or isinstance(ctx, ast.Store), ast.unparse(node)
            self._process_assignment(target_node, node.value, value)

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
        callable_namespace = self._lookup_expression_node_namespace(node.func)
        if callable_namespace is None:
            return
        if (
            callable_namespace.local_path
            == LocalObjectPath(DICT_FIELD_NAME, 'update')
        ) and (
            module_namespace := self._resolve_module_path(
                callable_namespace.module_path
            )
        ).kind is ObjectKind.STATIC_MODULE:
            module_namespace.mark_module_as_dynamic()
            return
        if (
            (
                callable_namespace.kind
                in (ObjectKind.INSTANCE_ROUTINE, ObjectKind.ROUTINE)
            )
            and _does_function_modify_caller_global_state(
                (
                    callable_namespace.instance_routine_to_routine()
                    if callable_namespace.kind is ObjectKind.INSTANCE_ROUTINE
                    else callable_namespace
                ),
                caller_module_namespace=self._get_module_namespace(),
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
            module_namespace := self._get_module_namespace()
        ).kind is ObjectKind.STATIC_MODULE:
            module_namespace.mark_module_as_dynamic()
            return

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_name = node.name
        class_module_path = self._namespace.module_path
        class_local_path = self._namespace.local_path.join(class_name)
        base_namespaces: list[Namespace] = []
        for index, base_node in reversed([*enumerate(node.bases)]):
            self.visit(base_node)
            base_namespace = self._construct_namespace_from_expression_node(
                base_node,
                local_path=class_local_path.join(
                    # FIXME: add support for indexing?
                    f'__bases___{index}'
                ),
                module_path=class_module_path,
            )
            if base_namespace is None:
                continue
            base_namespaces.append(base_namespace)
        class_namespace = Namespace(
            (
                ObjectKind.METACLASS
                if any(
                    base_namespace.kind is ObjectKind.METACLASS
                    for base_namespace in base_namespaces
                )
                else (
                    ObjectKind.UNKNOWN_CLASS
                    if any(
                        (
                            base_namespace.kind
                            in (ObjectKind.UNKNOWN, ObjectKind.UNKNOWN_CLASS)
                        )
                        for base_namespace in base_namespaces
                    )
                    else ObjectKind.CLASS
                )
            ),
            self._namespace.module_path,
            class_local_path,
        )
        class_namespace.set_namespace_by_name(
            DICT_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                class_namespace.module_path,
                class_namespace.local_path.join(DICT_FIELD_NAME),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    BUILTINS_DICT_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_namespace.set_namespace_by_name(
            MODULE_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                class_namespace.module_path,
                class_namespace.local_path.join(MODULE_FIELD_NAME),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    BUILTINS_STR_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_namespace.set_object_by_name(
            MODULE_FIELD_NAME, class_module_path.to_module_name()
        )
        class_namespace.set_namespace_by_name(
            NAME_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                class_namespace.module_path,
                class_namespace.local_path.join(NAME_FIELD_NAME),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    BUILTINS_STR_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_namespace.set_object_by_name(NAME_FIELD_NAME, class_name)
        class_namespace.set_namespace_by_name(
            QUALNAME_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                class_namespace.module_path,
                class_namespace.local_path.join(QUALNAME_FIELD_NAME),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    BUILTINS_STR_LOCAL_OBJECT_PATH
                ),
            ),
        )
        class_namespace.set_object_by_name(
            QUALNAME_FIELD_NAME, class_local_path.to_object_name()
        )
        class_parser = NamespaceParser(
            class_namespace,
            *self._get_inherited_namespaces(),
            context=self._context,
            function_definition_nodes=self._function_definition_nodes,
            module_file_paths=self._module_file_paths,
        )
        for body_node in node.body:
            class_parser.visit(body_node)
        for base_namespace in base_namespaces:
            class_namespace.append_sub_namespace(base_namespace)
        if len(node.bases) == 0:
            class_namespace.append_sub_namespace(
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    LocalObjectPath.from_object_name(object.__qualname__)
                )
            )
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                metaclass_namespace = (
                    self._construct_namespace_from_expression_node(
                        keyword.value,
                        local_path=class_local_path.join('__class__'),
                        module_path=class_module_path,
                    )
                )
                if metaclass_namespace is None:
                    continue
                assert metaclass_namespace.kind in (
                    ObjectKind.METACLASS,
                    ObjectKind.UNKNOWN_CLASS,
                )
                class_namespace.append_sub_namespace(metaclass_namespace)
        for decorator_node in node.decorator_list:
            self.visit(decorator_node)
            decorator_namespace = self._lookup_expression_node_namespace(
                decorator_node
            )
            if decorator_namespace is None:
                continue
            if (
                (
                    decorator_namespace.kind
                    in (ObjectKind.INSTANCE_ROUTINE, ObjectKind.ROUTINE)
                )
                and _does_function_modify_caller_global_state(
                    (
                        decorator_namespace.instance_routine_to_routine()
                        if (
                            decorator_namespace.kind
                            is ObjectKind.INSTANCE_ROUTINE
                        )
                        else decorator_namespace
                    ),
                    caller_module_namespace=self._get_module_namespace(),
                    positional_arguments=[
                        *self._to_complete_positional_arguments(
                            [], decorator_namespace
                        ),
                        class_namespace.as_object(),
                    ],
                    keyword_arguments={},
                    function_definition_nodes=self._function_definition_nodes,
                    module_file_paths=self._module_file_paths,
                )
                and (
                    (module_namespace := self._get_module_namespace()).kind
                    is ObjectKind.STATIC_MODULE
                )
            ):
                module_namespace.mark_module_as_dynamic()
                continue
        self._namespace.set_namespace_by_name(class_name, class_namespace)
        self._namespace.set_object_by_name(
            class_name, class_namespace.as_object()
        )

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
            self._resolve_absolute_local_path(
                target_object_split_path.module,
                target_object_split_path.absolute,
            ).set_namespace_by_path(
                target_object_split_path.relative,
                Namespace(
                    ObjectKind.UNKNOWN,
                    self._namespace.module_path,
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
                        ).set_object_by_path(object_split_path.relative, value)
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
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (module_alias := alias.asname) is not None:
                module_namespace = self._resolve_module_path(
                    ModulePath.from_module_name(alias.name)
                )
                self._namespace.set_namespace_by_name(
                    module_alias, module_namespace
                )
                self._namespace.set_object_by_name(
                    module_alias, module_namespace.as_object()
                )
            else:
                namespace = self._namespace
                for submodule_path in ModulePath.from_module_name(
                    alias.name
                ).submodule_paths():
                    submodule_namespace = self._resolve_module_path(
                        submodule_path
                    )
                    submodule_last_name = submodule_path.components[-1]
                    namespace.set_namespace_by_name(
                        submodule_last_name, submodule_namespace
                    )
                    namespace.set_object_by_name(
                        submodule_last_name, submodule_namespace.as_object()
                    )
                    namespace = submodule_namespace

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        import_is_relative = node.level > 0
        if import_is_relative:
            is_package = (
                module_file_path := self._module_file_paths.get(
                    self._namespace.module_path
                )
            ) is not None and module_file_path.stem == '__init__'
            depth = (
                len(self._namespace.module_path.components)
                + is_package
                - node.level
            ) or None
            components = list(self._namespace.module_path.components)[:depth]
            if (submodule_relative_name := node.module) is not None:
                submodule_relative_path_components = (
                    ModulePath.from_module_name(
                        submodule_relative_name
                    ).components
                )
                components += submodule_relative_path_components
                if is_package:
                    self._namespace.set_namespace_by_name(
                        submodule_relative_path_components[0],
                        self._resolve_module_path(
                            self._namespace.module_path.join(
                                submodule_relative_path_components[0]
                            )
                        ),
                    )
            top_submodule_path = ModulePath(*components)
        else:
            assert node.module is not None, ast.unparse(node)
            assert node.level == 0, ast.unparse(node)
            top_submodule_path = ModulePath.from_module_name(node.module)
        top_submodule_namespace = self._resolve_module_path(top_submodule_path)
        for alias in node.names:
            if alias.name == '*':
                assert self._namespace.kind is ObjectKind.STATIC_MODULE, (
                    'Star imports are only allowed on top module level, '
                    f'but found inside {self._namespace.kind} '
                    f'with path {self._namespace.module_path!r} '
                    f'{self._namespace.local_path!r}'
                )
                self._namespace.append_sub_namespace(top_submodule_namespace)
                continue
            value: Any | Missing
            try:
                object_namespace = (
                    top_submodule_namespace.get_namespace_by_name(alias.name)
                )
            except KeyError:
                if (
                    submodule_path := top_submodule_namespace.module_path.join(
                        alias.name
                    )
                ) in self._module_file_paths:
                    object_namespace = self._resolve_module_path(
                        submodule_path
                    )
                    value = object_namespace.as_object()
                else:
                    object_namespace = Namespace(
                        ObjectKind.UNKNOWN,
                        top_submodule_namespace.module_path,
                        LocalObjectPath(alias.name),
                    )
                    top_submodule_namespace.set_namespace_by_name(
                        alias.name, object_namespace
                    )
                    value = MISSING
            else:
                try:
                    value = top_submodule_namespace.get_object_by_name(
                        alias.name
                    )
                except KeyError:
                    value = MISSING
            object_alias_or_name = alias.asname or alias.name
            self._namespace.set_namespace_by_name(
                object_alias_or_name, object_namespace
            )
            if value is not MISSING:
                self._namespace.set_object_by_name(object_alias_or_name, value)

    @override
    def visit_If(self, node: ast.If) -> None:
        self.generic_visit(node.test)
        try:
            condition_satisfied = self._evaluate_expression_node(node.test)
        except EVALUATION_EXCEPTIONS:
            with contextlib.suppress(
                AttributeError, ModuleNotFoundError, NameError
            ):
                self.generic_visit(node)
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
        value_namespace = self._lookup_expression_node_namespace(node.value)
        target_node = node.target
        assert isinstance(target_node.ctx, ast.Store)
        target_name = target_node.id
        self._namespace.set_namespace_by_name(
            target_name,
            (
                value_namespace
                if value_namespace is not None
                else Namespace(
                    ObjectKind.UNKNOWN,
                    self._namespace.module_path,
                    self._namespace.local_path.join(target_name),
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
                    self._lookup_expression_node_namespace(exception_type_node)
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
                        self._namespace.set_namespace_by_name(
                            exception_name,
                            Namespace(
                                ObjectKind.INSTANCE,
                                self._namespace.module_path,
                                self._namespace.local_path.join(
                                    exception_name
                                ),
                                exception_type_namespace,
                            ),
                        )
                    for handler_node in handler.body:
                        self.visit(handler_node)
                    if exception_name is not None:
                        self._namespace.delete_namespace_by_name(
                            exception_name
                        )
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
                        self._lookup_expression_node_namespace(
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
                                    := self._lookup_expression_node_namespace(
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

    def _construct_namespace_from_expression_node(
        self,
        node: ast.expr,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Namespace:
        return construct_namespace_from_expression_node(
            node,
            self._namespace,
            *self._parent_namespaces,
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
            node,
            self._namespace,
            *self._parent_namespaces,
            context=self._context,
        )

    def _get_inherited_namespaces(self, /) -> Sequence[Namespace]:
        result = (
            self._parent_namespaces
            if self._namespace.kind in CLASS_OBJECT_KINDS
            else (self._namespace, *self._parent_namespaces)
        )
        assert (
            len(
                invalid_namespaces := [
                    namespace
                    for namespace in result
                    if namespace.kind in CLASS_OBJECT_KINDS
                ]
            )
            == 0
        ), invalid_namespaces
        return result

    def _get_module_namespace(self, /) -> Namespace:
        *_, result, _ = self._namespace, *self._parent_namespaces
        assert result.kind in (
            ObjectKind.DYNAMIC_MODULE,
            ObjectKind.STATIC_MODULE,
        ), result
        return result

    def _lookup_expression_node_namespace(
        self, node: ast.expr, /
    ) -> Namespace | None:
        return lookup_namespace_by_expression_node(
            node,
            self._namespace,
            *self._parent_namespaces,
            context=self._context,
        )

    def _process_assignment(
        self,
        target_node: ast.expr,
        value_node: ast.expr,
        value: Any | Missing,
        /,
    ) -> None:
        resolved_target = self._resolve_assignment_target(target_node)
        for target_object_split_path in _flatten_resolved_assignment_target(
            resolved_target
        ):
            target_namespace = self._resolve_absolute_local_path(
                target_object_split_path.module,
                target_object_split_path.absolute,
            )
            target_namespace.set_namespace_by_path(
                target_object_split_path.relative,
                (
                    self._construct_namespace_from_expression_node(
                        value_node,
                        local_path=target_object_split_path.combine_local(),
                        module_path=target_namespace.module_path,
                    )
                    if isinstance(
                        resolved_target, ResolvedAssignmentTargetSplitPath
                    )
                    else Namespace(
                        ObjectKind.UNKNOWN,
                        target_namespace.module_path,
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
                self._resolve_absolute_local_path(
                    maybe_target_object_split_path.module,
                    maybe_target_object_split_path.absolute,
                ).set_object_by_path(
                    maybe_target_object_split_path.relative, sub_value
                )
        else:
            for (
                target_object_split_path
            ) in _flatten_resolved_assignment_target(resolved_target):
                self._resolve_absolute_local_path(
                    target_object_split_path.module,
                    target_object_split_path.absolute,
                ).safe_delete_object_by_path(target_object_split_path.relative)

    def _resolve_absolute_local_path(
        self, module_path: ModulePath, local_path: LocalObjectPath, /
    ) -> Namespace:
        if module_path != self._namespace.module_path:
            return self._resolve_module_path(
                module_path
            ).get_namespace_by_path(local_path)
        if local_path.starts_with(self._namespace.local_path):
            return self._namespace.get_namespace_by_path(
                LocalObjectPath(
                    *local_path.components[
                        len(self._namespace.local_path.components) :
                    ]
                )
            )
        return self._get_module_namespace().get_namespace_by_path(local_path)

    def _resolve_assignment_target(
        self, node: ast.expr, /
    ) -> ResolvedAssignmentTarget:
        return resolve_assignment_target(
            node,
            self._namespace,
            *self._parent_namespaces,
            context=self._context,
        )

    def _resolve_module_path(self, module_path: ModulePath, /) -> Namespace:
        return resolve_module_path(
            module_path,
            function_definition_nodes=self._function_definition_nodes,
            module_file_paths=self._module_file_paths,
        )

    def _to_complete_positional_arguments(
        self,
        positional_argument_nodes: Sequence[ast.expr],
        callable_namespace: Namespace,
        /,
    ) -> Sequence[Any | Missing | Starred]:
        result: list[Any] = []
        if callable_namespace.kind is ObjectKind.INSTANCE_ROUTINE:
            instance_local_path = callable_namespace.local_path.parent
            result.append(
                self._resolve_absolute_local_path(
                    callable_namespace.module_path, instance_local_path.parent
                ).get_object_by_name_or_else(
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
            decorator_namespace = self._lookup_expression_node_namespace(
                decorator_node
            )
            if decorator_namespace is None:
                continue
            if decorator_namespace.module_path == BUILTINS_MODULE_PATH and (
                decorator_namespace.local_path
                == BUILTINS_PROPERTY_LOCAL_OBJECT_PATH
            ):
                function_namespace = Namespace(
                    ObjectKind.PROPERTY,
                    self._namespace.module_path,
                    self._namespace.local_path.join(function_name),
                    BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
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
                function_namespace = Namespace(
                    ObjectKind.ROUTINE,
                    self._namespace.module_path,
                    self._namespace.local_path.join(function_name),
                    TYPES_MODULE_NAMESPACE.get_namespace_by_path(
                        TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                    ),
                    Namespace(
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
                function_namespace = Namespace(
                    ObjectKind.INSTANCE,
                    self._namespace.module_path,
                    self._namespace.local_path.join(function_name),
                    decorator_namespace,
                )
                break
        else:
            function_namespace = Namespace(
                ObjectKind.ROUTINE,
                self._namespace.module_path,
                self._namespace.local_path.join(function_name),
                TYPES_MODULE_NAMESPACE.get_namespace_by_path(
                    TYPES_FUNCTION_TYPE_LOCAL_OBJECT_PATH
                ),
            )
        if (
            function_name == '__getattr__'
            and self._namespace.kind is ObjectKind.STATIC_MODULE
        ):
            self._namespace.mark_module_as_dynamic()
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
        function_namespace.set_namespace_by_name(
            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                function_namespace.module_path,
                function_namespace.local_path.join(
                    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME
                ),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    LocalObjectPath.from_object_name(
                        builtins.tuple.__qualname__
                    )
                ),
            ),
        )
        function_namespace.set_object_by_name(
            FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME, positional_defaults
        )
        function_namespace.set_namespace_by_name(
            FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                function_namespace.module_path,
                function_namespace.local_path.join(
                    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
                ),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    LocalObjectPath.from_object_name(
                        builtins.dict.__qualname__
                    )
                ),
            ),
        )
        function_namespace.set_object_by_name(
            FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME, keyword_only_defaults
        )
        self._namespace.set_namespace_by_name(
            function_name, function_namespace
        )
        self._function_definition_nodes[
            self._namespace.module_path,
            self._namespace.local_path.join(function_name),
        ] = node


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
) -> Namespace:
    root_component, *rest_components = module_path.components
    root_module_path = ModulePath(root_component)
    result = _load_module_path_namespace(
        root_module_path,
        function_definition_nodes=function_definition_nodes,
        module_file_paths=module_file_paths,
    )
    for component in rest_components:
        try:
            result = result.get_namespace_by_name(component)
        except KeyError:
            submodule_path = result.module_path.join(component)
            result = _load_module_path_namespace(
                submodule_path,
                function_definition_nodes=function_definition_nodes,
                module_file_paths=module_file_paths,
            )
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
) -> Namespace:
    try:
        result = MODULE_NAMESPACES[module_path]
    except KeyError:
        try:
            module_file_path = module_file_paths[module_path]
        except KeyError:
            raise ModuleNotFoundError(module_path) from None
        if module_file_path is None:
            return Namespace(
                ObjectKind.BUILTIN_MODULE, module_path, LocalObjectPath()
            )
        if module_file_path.name.endswith(tuple(EXTENSION_SUFFIXES)):
            return Namespace(
                ObjectKind.EXTENSION_MODULE, module_path, LocalObjectPath()
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
        assert module_path not in MODULE_NAMESPACES
        parse_modules(module)
        result = MODULE_NAMESPACES[module_path]
        result.set_namespace_by_name(
            DICT_FIELD_NAME,
            Namespace(
                ObjectKind.INSTANCE,
                result.module_path,
                result.local_path.join(DICT_FIELD_NAME),
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    BUILTINS_DICT_LOCAL_OBJECT_PATH
                ),
            ),
        )
        result.set_namespace_by_name(
            '__class__',
            TYPES_MODULE_NAMESPACE.get_namespace_by_path(
                TYPES_MODULE_TYPE_LOCAL_OBJECT_PATH
            ),
        )
        namespace_parser = NamespaceParser(
            result,
            BUILTINS_MODULE_NAMESPACE,
            context=NullContext(),
            function_definition_nodes=function_definition_nodes,
            module_file_paths=module_file_paths,
        )
        try:
            namespace_parser.visit(module_node)
        except Exception:
            del MODULE_NAMESPACES[module_path]
            raise
    return result
