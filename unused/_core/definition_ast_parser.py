from __future__ import annotations

import ast
import builtins
import operator
import sys
import types
import typing
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from functools import reduce, singledispatchmethod
from pathlib import Path
from typing import Any, ClassVar

from typing_extensions import override

from .object_path import (
    BUILTINS_MODULE_PATH,
    LocalObjectPath,
    ModulePath,
    ObjectPath,
    SYS_MODULE_PATH,
    TYPING_MODULE_PATH,
)
from .resolution import (
    ResolvedAssignmentTarget,
    ResolvedAssignmentTargetSplitPath,
    combine_resolved_assignment_target_with_value,
)
from .utils import AnyFunctionDefinitionAstNode


class DefinitionAstNodeParser(ast.NodeVisitor):
    _BUILTINS_SCOPE_PATHS: ClassVar[Mapping[str, ObjectPath]] = {
        name: (BUILTINS_MODULE_PATH, LocalObjectPath(name))
        for name in vars(builtins)
    }

    def __init__(
        self,
        definition_nodes: MutableMapping[
            LocalObjectPath, list[AnyFunctionDefinitionAstNode | ast.ClassDef]
        ],
        /,
        *parent_scope_paths: Mapping[str, ObjectPath],
        file_path: Path,
        is_class_scope: bool,
        local_path: LocalObjectPath,
        module_path: ModulePath,
        values: MutableMapping[str, Any],
    ) -> None:
        (
            self._definition_nodes,
            self._file_path,
            self._is_class_scope,
            self._local_path,
            self._module_path,
            self._values,
        ) = (
            definition_nodes,
            file_path,
            is_class_scope,
            local_path,
            module_path,
            values,
        )
        self._scope_paths: dict[str, ObjectPath] = {}
        self._parent_scope_paths: Sequence[Mapping[str, ObjectPath]] = (
            *parent_scope_paths,
            self._BUILTINS_SCOPE_PATHS,
        )

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        try:
            value = self._evaluate_expression_node(node.value)
        except _NonStaticallyEvaluatableAstNodeError:
            return
        for target_node in node.targets:
            for (
                resolved_target_split_path,
                sub_value,
            ) in combine_resolved_assignment_target_with_value(
                self._resolve_assignment_target(target_node), value
            ):
                if resolved_target_split_path is None:
                    continue
                assert resolved_target_split_path.module == self._module_path
                assert resolved_target_split_path.absolute == self._local_path
                self._values[
                    resolved_target_split_path.relative.components[0]
                ] = sub_value

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_any_function_node(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_local_path = self._local_path.join(node.name)
        self._definition_nodes.setdefault(class_local_path, []).append(node)
        body_visitor = DefinitionAstNodeParser(
            self._definition_nodes,
            *(() if self._is_class_scope else (self._scope_paths,)),
            *self._parent_scope_paths,
            file_path=self._file_path,
            is_class_scope=True,
            local_path=class_local_path,
            module_path=self._module_path,
            values={**self._values},
        )
        for body_node in node.body:
            body_visitor.visit(body_node)

    @override
    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node, ast.Name):
            assert isinstance(node.ctx, ast.Load), ast.unparse(node)
            self._lookup_object_path_by_name(node.id)
        else:
            self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_any_function_node(node)

    @override
    def visit_If(self, node: ast.If) -> None:
        try:
            condition = self._evaluate_expression_node(node.test)
        except _NonStaticallyEvaluatableAstNodeError:
            self.generic_visit(node)
        else:
            for body_node in node.body if condition else node.orelse:
                self.visit(body_node)

    @override
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (module_alias := alias.asname) is not None:
                self._scope_paths[module_alias] = (
                    ModulePath.from_module_name(alias.name),
                    LocalObjectPath(),
                )
            else:
                submodule_paths = iter(
                    ModulePath.from_module_name(alias.name).submodule_paths()
                )
                first_submodule_path = next(submodule_paths)
                assert len(first_submodule_path.components) == 1, (
                    first_submodule_path
                )
                self._scope_paths[first_submodule_path.components[0]] = (
                    first_submodule_path,
                    LocalObjectPath(),
                )
                for submodule_path in submodule_paths:
                    self._scope_paths[submodule_path.components[-1]] = (
                        submodule_path,
                        LocalObjectPath(),
                    )

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        import_is_relative = node.level > 0
        if import_is_relative:
            is_package = self._file_path.name.startswith('__init__.')
            depth = (
                len(self._module_path.components) + is_package - node.level
            ) or None
            components = list(self._module_path.components)[:depth]
            if (submodule_relative_name := node.module) is not None:
                submodule_relative_path_components = (
                    ModulePath.from_module_name(
                        submodule_relative_name
                    ).components
                )
                components += submodule_relative_path_components
                if is_package and node.level == 1:
                    self._scope_paths[
                        submodule_relative_path_components[0]
                    ] = (
                        self._module_path.join(
                            submodule_relative_path_components[0]
                        ),
                        LocalObjectPath(),
                    )
            top_submodule_path = ModulePath(*components)
        else:
            assert node.module is not None, ast.unparse(node)
            assert node.level == 0, ast.unparse(node)
            top_submodule_path = ModulePath.from_module_name(node.module)
        for alias in node.names:
            if alias.name == '*':
                continue
            self._scope_paths[alias.name] = (
                top_submodule_path,
                LocalObjectPath(alias.name),
            )

    @singledispatchmethod
    def _evaluate_expression_node(self, node: ast.expr, /) -> Any:
        raise _NonStaticallyEvaluatableAstNodeError(node)

    @_evaluate_expression_node.register(ast.Attribute)
    def _(self, node: ast.Attribute, /) -> Any:
        value_path = self._resolve_expression_node(node.value)
        if value_path is None:
            raise _NonStaticallyEvaluatableAstNodeError(node)
        value_module_path, value_local_path = value_path
        attribute_local_path = value_local_path.join(node.attr)
        if value_module_path != self._module_path:
            module: types.ModuleType | None = None
            if value_module_path == SYS_MODULE_PATH:
                module = sys
            elif value_module_path == TYPING_MODULE_PATH:
                module = typing
            if module is None:
                raise _NonStaticallyEvaluatableAstNodeError(node)
            try:
                return reduce(getattr, attribute_local_path.components, module)
            except AttributeError:
                raise _NonStaticallyEvaluatableAstNodeError(node) from None
        try:
            initial_value = self._values[attribute_local_path.components[0]]
        except KeyError:
            raise _NonStaticallyEvaluatableAstNodeError(node) from None
        try:
            return reduce(
                getattr, attribute_local_path.components[1:], initial_value
            )
        except (AttributeError, KeyError):
            raise _NonStaticallyEvaluatableAstNodeError(node) from None

    @_evaluate_expression_node.register(ast.Call)
    def _(self, node: ast.Call, /) -> Any:
        callable_path = self._resolve_expression_node(node.func)
        if callable_path is None:
            raise _NonStaticallyEvaluatableAstNodeError(node)
        if callable_path == (
            BUILTINS_MODULE_PATH,
            LocalObjectPath.from_object_name(builtins.hasattr.__qualname__),
        ):
            object_argument_node, attribute_argument_node = node.args
            object_argument_path = self._resolve_expression_node(
                object_argument_node
            )
            if object_argument_path is None:
                raise _NonStaticallyEvaluatableAstNodeError(node)
            object_module_path, object_local_path = object_argument_path
            attribute_name = self._evaluate_expression_node(
                attribute_argument_node
            )
            return hasattr(
                reduce(
                    getattr,
                    object_local_path.components,
                    sys.modules[object_module_path.to_module_name()],
                ),
                attribute_name,
            )
        raise _NonStaticallyEvaluatableAstNodeError(node)

    @_evaluate_expression_node.register(ast.Constant)
    def _(self, node: ast.Constant, /) -> Any:
        return node.value

    @_evaluate_expression_node.register(ast.List)
    def _(self, node: ast.List, /) -> Any:
        return list(map(self._evaluate_expression_node, node.elts))

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
    def _(self, node: ast.Compare, /) -> Any:
        value = self._evaluate_expression_node(node.left)
        for operator_node, next_value in zip(
            node.ops,
            (
                self._evaluate_expression_node(operand_node)
                for operand_node in node.comparators
            ),
            strict=True,
        ):
            if not self._binary_comparison_operators_by_operator_node_type[
                type(operator_node)
            ](value, next_value):
                return False
            value = next_value
        return True

    @_evaluate_expression_node.register(ast.Name)
    def _(self, node: ast.Name, /) -> Any:
        assert isinstance(node.ctx, ast.Load), ast.unparse(node)
        try:
            return self._values[node.id]
        except KeyError:
            raise _NonStaticallyEvaluatableAstNodeError(node) from None

    @singledispatchmethod
    def _resolve_assignment_target(
        self, _node: ast.expr, /
    ) -> ResolvedAssignmentTarget:
        return None

    @_resolve_assignment_target.register(ast.Name)
    def _(self, node: ast.Name, /) -> ResolvedAssignmentTarget:
        object_name = node.id
        if isinstance(node.ctx, ast.Load):
            object_module_path, object_local_path = (
                self._lookup_object_path_by_name(object_name)
            )
            if (
                object_module_path == self._module_path
                and object_local_path.starts_with(self._local_path)
            ):
                return ResolvedAssignmentTargetSplitPath(
                    self._module_path,
                    self._local_path,
                    LocalObjectPath(
                        *object_local_path.components[
                            len(self._local_path.components) :
                        ]
                    ),
                )
            return ResolvedAssignmentTargetSplitPath(
                object_module_path, object_local_path, LocalObjectPath()
            )
        return ResolvedAssignmentTargetSplitPath(
            self._module_path, self._local_path, LocalObjectPath(object_name)
        )

    def _lookup_object_path_by_name(self, object_name: str, /) -> ObjectPath:
        try:
            return self._scope_paths[object_name]
        except KeyError:
            for parent_scope_paths in self._parent_scope_paths:
                try:
                    return parent_scope_paths[object_name]
                except KeyError:
                    continue
            raise

    @singledispatchmethod
    def _resolve_expression_node(
        self, _node: ast.expr, /
    ) -> ObjectPath | None:
        return None

    @_resolve_expression_node.register(ast.Attribute)
    def _(self, node: ast.Attribute, /) -> ObjectPath | None:
        value_path = self._resolve_expression_node(node.value)
        if value_path is None:
            return None
        value_module_path, value_local_path = value_path
        return value_module_path, value_local_path.join(node.attr)

    @_resolve_expression_node.register(ast.Name)
    def _(self, node: ast.Name, /) -> ObjectPath | None:
        assert isinstance(node.ctx, ast.Load), ast.unparse(node)
        try:
            return self._lookup_object_path_by_name(node.id)
        except KeyError:
            return None

    def _visit_any_function_node(
        self, node: AnyFunctionDefinitionAstNode, /
    ) -> None:
        function_local_path = self._local_path.join(node.name)
        self._definition_nodes.setdefault(function_local_path, []).append(node)


class _NonStaticallyEvaluatableAstNodeError(Exception):
    pass
