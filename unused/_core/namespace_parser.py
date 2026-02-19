from __future__ import annotations

import ast
import builtins
import contextlib
import enum
import functools
import inspect
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
from itertools import chain
from pathlib import Path
from typing import Any, Final

from typing_extensions import override

from .attribute_mapping import AttributeMapping
from .construction import construct_namespace_from_expression_node
from .evaluation import EVALUATION_EXCEPTIONS, evaluate_node
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
    BUILTINS_MODULE_PATH,
    FUNCTION_TYPE_LOCAL_OBJECT_PATH,
    GLOBALS_LOCAL_OBJECT_PATH,
    LocalObjectPath,
    ModulePath,
)
from .resolution import (
    ResolvedAssignmentTarget,
    ResolvedAssignmentTargetSplitPath,
    resolve_assignment_target,
)

FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME = '__defaults__'
FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME = '__kwdefaults__'

EMPTY_MODULE_FILE_PATH: Final[Path] = Path(
    tempfile.NamedTemporaryFile(delete=False).name  # noqa: SIM115
)
ENUM_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    enum.__name__
)
SIMPLE_ENUM_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(enum._simple_enum.__qualname__)  # type: ignore[attr-defined] # noqa: SLF001
    if sys.version_info >= (3, 11)
    else LocalObjectPath()
)
GLOBAL_ENUM_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(enum.global_enum.__qualname__)
    if sys.version_info >= (3, 11)
    else LocalObjectPath()
)
ENUM_META_CONVERT_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(
        enum.EnumMeta._convert_.__qualname__  # type: ignore[attr-defined]
    )
)


def _combine_resolved_assignment_target_with_value(
    target: ResolvedAssignmentTarget, value: Any, /
) -> Iterable[tuple[ResolvedAssignmentTargetSplitPath | None, Any]]:
    if target is None or isinstance(target, ResolvedAssignmentTargetSplitPath):
        yield target, value
        return
    if isinstance(value, AttributeMapping):
        # e.g.: a case of `enum.Enum` class unpacking
        return
    try:
        iter(value)
    except TypeError:
        return
    yield from chain.from_iterable(
        map(_combine_resolved_assignment_target_with_value, target, value)
    )


def _does_function_modify_global_state(
    function_namespace: Namespace,
    /,
    *,
    positional_arguments: Sequence[Any],
    keyword_arguments: Mapping[str, Any],
    cache: dict[tuple[ModulePath, LocalObjectPath], bool] = {},  # noqa: B006
    function_definition_nodes: MutableMapping[
        tuple[ModulePath, LocalObjectPath],
        ast.AsyncFunctionDef | ast.FunctionDef,
    ],
    module_file_paths: Mapping[ModulePath, Path | None],
) -> bool:
    cache_key = function_namespace.module_path, function_namespace.local_path
    try:
        return cache[cache_key]
    except KeyError:
        try:
            function_definition_node = function_definition_nodes[cache_key]
        except KeyError:
            assert module_file_paths[function_namespace.module_path] is None, (
                function_namespace
            )
            return (
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
                and len(positional_arguments) < 2
                and 'globals' not in keyword_arguments
            )
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
        for positional_argument, positional_parameter_node in zip(
            positional_arguments,
            positional_parameter_nodes[: len(positional_arguments)],
            strict=True,
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
            len(positional_arguments) :
        ]
        for defaulted_positional_parameter_node, positional_default in zip(
            defaulted_positional_parameter_nodes[::-1],
            positional_defaults[::-1],
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
        module_namespace = Namespace(
            ObjectKind.STATIC_MODULE,
            function_namespace.module_path,
            LocalObjectPath(),
            MODULE_NAMESPACES[function_namespace.module_path],
        )
        cache[cache_key] = False
        function_body_parser = NamespaceParser(
            function_scope_namespace,
            module_namespace,
            BUILTINS_MODULE_NAMESPACE,
            function_definition_nodes=function_definition_nodes,
            module_file_paths=module_file_paths,
        )
        for function_body_node in function_definition_node.body:
            function_body_parser.visit(function_body_node)
        del cache[cache_key]
        return module_namespace.kind is ObjectKind.DYNAMIC_MODULE


class BaseNamespaceParser(ast.NodeVisitor):
    def __init__(
        self,
        namespace: Namespace,
        /,
        *parent_namespaces: Namespace,
        function_definition_nodes: MutableMapping[
            tuple[ModulePath, LocalObjectPath],
            ast.AsyncFunctionDef | ast.FunctionDef,
        ],
        module_file_paths: Mapping[ModulePath, Path | None],
    ) -> None:
        super().__init__()
        (
            self._function_definition_nodes,
            self._module_file_paths,
            self._namespace,
            self._parent_namespaces,
        ) = (
            function_definition_nodes,
            module_file_paths,
            namespace,
            parent_namespaces,
        )

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_any_function_def(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_any_function_def(node)

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

    def _lookup_expression_node_namespace(
        self, node: ast.expr, /
    ) -> Namespace | None:
        return lookup_namespace_by_expression_node(
            node, self._namespace, *self._parent_namespaces
        )

    def _resolve_module_path(self, module_path: ModulePath, /) -> Namespace:
        return resolve_module_path(
            module_path,
            function_definition_nodes=self._function_definition_nodes,
            module_file_paths=self._module_file_paths,
        )

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
                decorator_namespace.local_path == PROPERTY_LOCAL_OBJECT_PATH
            ):
                function_namespace = Namespace(
                    ObjectKind.PROPERTY,
                    self._namespace.module_path,
                    self._namespace.local_path.join(function_name),
                    BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                        PROPERTY_LOCAL_OBJECT_PATH
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
                == SINGLEDISPATCH_LOCAL_OBJECT_PATH
            ):
                function_namespace = Namespace(
                    ObjectKind.ROUTINE,
                    self._namespace.module_path,
                    self._namespace.local_path.join(function_name),
                    TYPES_MODULE_NAMESPACE.get_namespace_by_path(
                        FUNCTION_TYPE_LOCAL_OBJECT_PATH
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
                    SINGLEDISPATCH_LOCAL_OBJECT_PATH
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
                    FUNCTION_TYPE_LOCAL_OBJECT_PATH
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
                positional_default_value = evaluate_node(
                    positional_default_node,
                    self._namespace,
                    *self._parent_namespaces,
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
                keyword_only_default_value = evaluate_node(
                    keyword_default_node,
                    self._namespace,
                    *self._parent_namespaces,
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


class NamespaceParser(BaseNamespaceParser):
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
                value = self._evaluate_node(value_node)
            except EVALUATION_EXCEPTIONS:
                pass
            else:
                self._namespace.set_object_by_name(target_name, value)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        try:
            value = self._evaluate_node(node.value)
        except EVALUATION_EXCEPTIONS:
            value = MISSING
        self.generic_visit(node)
        for target_node in node.targets:
            assert (
                ctx := getattr(target_node, 'ctx', None)
            ) is None or isinstance(ctx, ast.Store), ast.unparse(node)
            self._process_assignment(target_node, node.value, value)

    @override
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if isinstance(node.op, ast.And):
            for operand_node in node.values:
                try:
                    operand_value = self._evaluate_node(operand_node)
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
                operand_value = self._evaluate_node(operand_node)
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
            callable_namespace.module_path == BUILTINS_MODULE_PATH
            and (
                callable_namespace.local_path
                == GLOBALS_LOCAL_OBJECT_PATH.join('update')
            )
        ) and (
            (module_namespace := self._get_module_namespace()).kind
            is ObjectKind.STATIC_MODULE
        ):
            module_namespace.mark_module_as_dynamic()
            return
        function_namespace = (
            callable_namespace.instance_routine_to_routine()
            if callable_namespace.kind is ObjectKind.INSTANCE_ROUTINE
            else callable_namespace
        )
        if (
            (
                callable_namespace.kind
                in (ObjectKind.INSTANCE_ROUTINE, ObjectKind.ROUTINE)
            )
            and _does_function_modify_global_state(
                function_namespace,
                positional_arguments=self._to_complete_positional_arguments(
                    node.args, callable_namespace
                ),
                keyword_arguments=self._evaluate_keyword_arguments(
                    node.keywords
                ),
                function_definition_nodes=self._function_definition_nodes,
                module_file_paths=self._module_file_paths,
            )
        ) and (
            module_namespace := self._get_module_namespace()
        ).kind is ObjectKind.STATIC_MODULE:
            module_namespace.mark_module_as_dynamic()
        if function_namespace.module_path == ENUM_MODULE_PATH and (
            function_namespace.local_path
            == ENUM_META_CONVERT_LOCAL_OBJECT_PATH
        ):
            enum_name_node, enum_module_name_node, *_ = node.args
            enum_name = self._evaluate_node(enum_name_node)
            assert isinstance(enum_name, str), ast.unparse(node)
            enum_module_namespace = self._lookup_expression_node_namespace(
                enum_module_name_node
            )
            assert enum_module_namespace is not None, ast.unparse(node)
            assert (
                enum_module_namespace.module_path
                == self._namespace.module_path
            ), ast.unparse(node)
            assert (
                enum_module_namespace.local_path
                == self._namespace.local_path.join('__name__')
            ), ast.unparse(node)
            self._namespace.set_namespace_by_name(
                enum_name,
                Namespace(
                    ObjectKind.UNKNOWN,
                    self._namespace.module_path,
                    self._namespace.local_path.join(enum_name),
                    BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                        LocalObjectPath.from_object_name(object.__qualname__)
                    ),
                ),
            )

    def _to_complete_positional_arguments(
        self,
        positional_argument_nodes: Sequence[ast.expr],
        callable_namespace: Namespace,
        /,
    ) -> Sequence[Any | Missing]:
        result = self._evaluate_positional_arguments(positional_argument_nodes)
        if callable_namespace.kind is ObjectKind.INSTANCE_ROUTINE:
            try:
                instance_local_path = callable_namespace.local_path.parent
            except Exception:
                raise
            result = [
                self._resolve_absolute_local_path(
                    callable_namespace.module_path, instance_local_path.parent
                ).get_object_by_name_or_else(
                    instance_local_path.components[-1], default=MISSING
                ),
                *result,
            ]
        return result

    def _evaluate_positional_arguments(
        self, positional_argument_nodes: Sequence[ast.expr], /
    ) -> Sequence[Any]:
        result: list[Any] = []
        for positional_argument_node in positional_argument_nodes:
            if isinstance(positional_argument_node, ast.Starred):
                try:
                    positional_argument_values = self._evaluate_node(
                        positional_argument_node.value
                    )
                except EVALUATION_EXCEPTIONS:
                    pass
                else:
                    result.extend(positional_argument_values)
            else:
                try:
                    positional_argument_value = self._evaluate_node(
                        positional_argument_node
                    )
                except EVALUATION_EXCEPTIONS:
                    positional_argument_value = MISSING
                result.append(positional_argument_value)
        return result

    def _evaluate_keyword_arguments(
        self, keyword_argument_nodes: Sequence[ast.keyword], /
    ) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for keyword_argument_node in keyword_argument_nodes:
            if keyword_argument_node.arg is None:
                try:
                    keyword_argument_values = self._evaluate_node(
                        keyword_argument_node.value
                    )
                except EVALUATION_EXCEPTIONS:
                    pass
                else:
                    result.update(keyword_argument_values)
            else:
                try:
                    keyword_argument_value = self._evaluate_node(
                        keyword_argument_node.value
                    )
                except EVALUATION_EXCEPTIONS:
                    keyword_argument_value = MISSING
                result[keyword_argument_node.arg] = keyword_argument_value
        return result

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
                        base_namespace.kind is ObjectKind.UNKNOWN
                        for base_namespace in base_namespaces
                    )
                    else ObjectKind.CLASS
                )
            ),
            self._namespace.module_path,
            class_local_path,
        )
        class_parser = NamespaceParser(
            class_namespace,
            *self._get_inherited_namespaces(),
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
                and isinstance(decorator_node, ast.Call)
                and _does_function_modify_global_state(
                    (
                        decorator_namespace.instance_routine_to_routine()
                        if (
                            decorator_namespace.kind
                            is ObjectKind.INSTANCE_ROUTINE
                        )
                        else decorator_namespace
                    ),
                    positional_arguments=self._to_complete_positional_arguments(
                        decorator_node.args, decorator_namespace
                    ),
                    keyword_arguments=self._evaluate_keyword_arguments(
                        decorator_node.keywords
                    ),
                    function_definition_nodes=self._function_definition_nodes,
                    module_file_paths=self._module_file_paths,
                )
                and (
                    (module_namespace := self._get_module_namespace()).kind
                    is ObjectKind.STATIC_MODULE
                )
            ):
                module_namespace.mark_module_as_dynamic()
            if decorator_namespace.module_path == ENUM_MODULE_PATH and (
                decorator_namespace.local_path == GLOBAL_ENUM_LOCAL_OBJECT_PATH
            ):
                # FIXME: be more precise and append only enumeration members
                self._get_module_namespace().append_sub_namespace(
                    class_namespace
                )
            elif decorator_namespace.module_path == ENUM_MODULE_PATH and (
                decorator_namespace.local_path == SIMPLE_ENUM_LOCAL_OBJECT_PATH
            ):
                assert isinstance(decorator_node, ast.Call), ast.unparse(node)
                (base_enum_node,) = decorator_node.args
                base_enum_namespace = self._lookup_expression_node_namespace(
                    base_enum_node
                )
                assert base_enum_namespace is not None
                class_namespace.append_sub_namespace(base_enum_namespace)
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
            iterable = iter(self._evaluate_node(node.iter))
        except EVALUATION_EXCEPTIONS:
            self.generic_visit(node)
        else:
            for element in iterable:
                for (
                    object_split_path,
                    value,
                ) in _combine_resolved_assignment_target_with_value(
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
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    @override
    def visit_If(self, node: ast.If) -> None:
        self.generic_visit(node.test)
        try:
            condition_satisfied = self._evaluate_node(node.test)
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
        except NameError:
            for handler in node.handlers:
                exception_type_node = handler.type
                if exception_type_node is None or (
                    (
                        (
                            exception_type_namespace
                            := self._lookup_expression_node_namespace(
                                exception_type_node
                            )
                        )
                        is not None
                    )
                    and (
                        exception_type_namespace.module_path
                        == BUILTINS_MODULE_PATH
                    )
                    and (
                        exception_type_namespace.local_path
                        == LocalObjectPath.from_object_name(
                            NameError.__qualname__
                        )
                    )
                ):
                    for handler_node in handler.body:
                        self.visit(handler_node)
                    break
            else:
                raise
        except ModuleNotFoundError:
            for handler in node.handlers:
                exception_type_node = handler.type
                if exception_type_node is None or (
                    (
                        (
                            exception_type_namespace
                            := self._lookup_expression_node_namespace(
                                exception_type_node
                            )
                        )
                        is not None
                    )
                    and (
                        exception_type_namespace.module_path
                        == BUILTINS_MODULE_PATH
                    )
                    and (
                        exception_type_namespace.local_path
                        in (
                            LocalObjectPath.from_object_name(
                                exception_cls.__qualname__
                            )
                            for exception_cls in ModuleNotFoundError.mro()[:-1]
                        )
                    )
                ):
                    for handler_node in handler.body:
                        self.visit(handler_node)
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
                    value = self._evaluate_node(value_node)
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
                        == SUPPRESS_LOCAL_OBJECT_PATH
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
            local_path=local_path,
            module_path=module_path,
        )

    def _evaluate_node(self, node: ast.expr, /) -> Any:
        return evaluate_node(node, self._namespace, *self._parent_namespaces)

    def _get_inherited_namespaces(self, /) -> Sequence[Namespace]:
        result = (
            self._parent_namespaces
            if self._namespace.kind is ObjectKind.CLASS
            else (self._namespace, *self._parent_namespaces)
        )
        assert (
            len(
                invalid_namespaces := [
                    namespace
                    for namespace in result
                    if (
                        namespace.kind
                        in (
                            ObjectKind.CLASS,
                            ObjectKind.UNKNOWN_CLASS,
                            ObjectKind.METACLASS,
                        )
                    )
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
            ) in _combine_resolved_assignment_target_with_value(
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
            node, self._namespace, *self._parent_namespaces
        )


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


def _setup_builtin_cls() -> None:
    for cls in [
        builtins.dict,
        builtins.frozenset,
        builtins.list,
        builtins.set,
        builtins.tuple,
        builtins.type,
    ]:
        assert inspect.isclass(cls), cls
        BUILTINS_MODULE_NAMESPACE.set_object_by_path(
            LocalObjectPath.from_object_name(cls.__qualname__), cls
        )


_setup_builtin_cls()
FUNCTOOLS_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    functools.__name__
)
SINGLEDISPATCH_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(functools.singledispatch.__qualname__)
)
PROPERTY_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(property.__qualname__)
)
CONTEXTLIB_MODULE_PATH: Final[ModulePath] = ModulePath.from_module_name(
    contextlib.__name__
)
SUPPRESS_LOCAL_OBJECT_PATH: Final[LocalObjectPath] = (
    LocalObjectPath.from_object_name(contextlib.suppress.__qualname__)
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
        assert module_path not in MODULE_NAMESPACES
        module = types.ModuleType(
            module_path.to_module_name(), ast.get_docstring(module_node)
        )
        module.__file__ = str(module_file_path)
        if module_file_path.name.startswith('__init__.'):
            module.__package__ = module_path.to_module_name()
            module.__path__ = [str(module_file_path.parent)]
        parse_modules(module)
        result = MODULE_NAMESPACES[module_path]
        namespace_parser = NamespaceParser(
            result,
            BUILTINS_MODULE_NAMESPACE,
            function_definition_nodes=function_definition_nodes,
            module_file_paths=module_file_paths,
        )
        try:
            namespace_parser.visit(module_node)
        except Exception:
            del MODULE_NAMESPACES[module_path]
            raise
    return result
