from __future__ import annotations

import ast
import builtins
import graphlib
import inspect
import operator
import sys
import types
import typing
from ast import AsyncFunctionDef, ClassDef, FunctionDef
from collections import deque
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from functools import partial, reduce, singledispatch, singledispatchmethod
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any, ClassVar, Final, NewType, TypeAlias, TypeVar

from typing_extensions import TypeIs, override

from .enums import ObjectKind, ScopeKind
from .missing import MISSING
from .object_ import (
    CALLABLE_OBJECT_CLASSES,
    Class,
    Method,
    Module,
    MutableObject,
    Object,
    PlainObject,
    Routine,
    UnknownObject,
)
from .object_path import (
    BUILTINS_MODULE_PATH,
    FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
    FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    NAME_FIELD_NAME,
    SYS_MODULE_PATH,
    TYPES_METHOD_TYPE_LOCAL_OBJECT_PATH,
    TYPES_MODULE_PATH,
    TYPING_MODULE_PATH,
)
from .resolution import (
    ResolvedAssignmentTarget,
    ResolvedAssignmentTargetSplitPath,
    combine_resolved_assignment_target_with_value,
)
from .safety import is_safe
from .scope import Scope
from .utils import AnyFunctionDefinitionAstNode, ensure_type

ObjectPath: TypeAlias = tuple[ModulePath, LocalObjectPath]


class _NonStaticallyEvaluatableAstNodeError(Exception):
    pass


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


def _invert_mapping(
    value: Mapping[_KT, _VT], /
) -> Mapping[_VT, Sequence[_KT]]:
    result: dict[_VT, list[_KT]] = {}
    for item_key, item_value in value.items():
        result.setdefault(item_value, []).append(item_key)
    return result


MODULE_NAMES: Final[Mapping[types.ModuleType, Sequence[str]]] = (
    _invert_mapping(sys.modules)
)


@singledispatch
def _locate_values(
    value: Any,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[  # noqa: ARG001
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],  # noqa: ARG001
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],  # noqa: ARG001
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],  # noqa: ARG001
) -> None:
    if value_path is not None:
        _set_absent_key(located_rest_values, value_path, value)


@_locate_values.register(type)
def _(
    value: type[Any],
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    if value_path is None:
        value_module_path = None
    else:
        value_module_path, _ = value_path
    if isinstance(module_name := getattr(value, '__module__', None), str) and (
        (module := _checked_find_module_by_name(module_name)) is not None
    ):
        module_path = ModulePath.from_module_name(module_name)
        _register_module_path(
            mentioned_module_paths, module, (module_path, LocalObjectPath())
        )
        if value_module_path is None or module_path != value_module_path:
            _locate_values(
                module,
                (module_path, LocalObjectPath()),
                mentioned_module_paths,
                located_namespace_values=located_namespace_values,
                located_rest_values=located_rest_values,
                namespace_value_id_paths=namespace_value_id_paths,
                namespace_value_id_values=namespace_value_id_values,
            )
    if value_path is None:
        return
    value_module_path, value_local_path = value_path
    try:
        value_paths = namespace_value_id_paths[_namespace_value_id(value)]
    except KeyError:
        _set_absent_key(located_namespace_values, value_path, value)
        _set_absent_key(
            namespace_value_id_paths, _namespace_value_id(value), [value_path]
        )
        _set_absent_key(
            namespace_value_id_values, _namespace_value_id(value), value
        )
    else:
        if value_path in value_paths:
            return
        if any(
            (
                module_path == value_module_path
                and value_local_path.starts_with(local_path)
            )
            for module_path, local_path in value_paths
        ):
            value_paths.append(value_path)
            return
        _set_absent_key(located_namespace_values, value_path, value)
        value_paths.append(value_path)
    value_dict = vars(value)
    for field_name in dir(value):
        try:
            field_value = value_dict[field_name]
        except KeyError:
            try:
                field_value = getattr(value, field_name)
            except AttributeError:
                continue
            is_dynamic_field = field_value is not getattr(value, field_name)
            if is_dynamic_field:
                continue
        _locate_values(
            field_value,
            (
                None
                if field_name in ('__base__', '__class__')
                else (value_module_path, value_local_path.join(field_name))
            ),
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )
    for base in value.__bases__:
        _locate_values(
            base,
            None,
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )


@_locate_values.register(types.ModuleType)
def _(
    value: types.ModuleType,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    if value_path is None:
        value_module_path = None
    else:
        value_module_path, _ = value_path
    self_module_name = value.__name__
    if (
        module := _checked_find_module_by_name(self_module_name)
    ) is not None and module is not value:
        _locate_values(
            module,
            (ModulePath.from_module_name(self_module_name), LocalObjectPath()),
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )
        for module_name in MODULE_NAMES.get(value, []):
            module_path = ModulePath.from_module_name(module_name)
            if value_module_path is None or module_path != value_module_path:
                _locate_values(
                    value,
                    (module_path, LocalObjectPath()),
                    mentioned_module_paths,
                    located_namespace_values=located_namespace_values,
                    located_rest_values=located_rest_values,
                    namespace_value_id_paths=namespace_value_id_paths,
                    namespace_value_id_values=namespace_value_id_values,
                )
    elif value in sys.modules.values():
        self_module_path = ModulePath.from_module_name(self_module_name)
        _register_module_path(
            mentioned_module_paths,
            value,
            (self_module_path, LocalObjectPath()),
        )
        if value_module_path is None or self_module_path != value_module_path:
            _locate_values(
                module,
                (self_module_path, LocalObjectPath()),
                mentioned_module_paths,
                located_namespace_values=located_namespace_values,
                located_rest_values=located_rest_values,
                namespace_value_id_paths=namespace_value_id_paths,
                namespace_value_id_values=namespace_value_id_values,
            )
    for module_name in MODULE_NAMES.get(value, []):
        _register_module_path(
            mentioned_module_paths,
            value,
            (ModulePath.from_module_name(module_name), LocalObjectPath()),
        )
    if value_path is None:
        return
    value_module_path, value_local_path = value_path
    _register_module_path(mentioned_module_paths, value, value_path)
    value_id = _namespace_value_id(value)
    try:
        value_paths = namespace_value_id_paths[value_id]
    except KeyError:
        _set_absent_key(located_namespace_values, value_path, value)
        _set_absent_key(namespace_value_id_paths, value_id, [value_path])
        _set_absent_key(namespace_value_id_values, value_id, value)
    else:
        if value_path in value_paths:
            return
        if any(
            (
                module_path == value_module_path
                and value_local_path.starts_with(local_path)
            )
            for module_path, local_path in value_paths
        ):
            value_paths.append(value_path)
            return
        _set_absent_key(located_namespace_values, value_path, value)
        value_paths.append(value_path)
    if value in sys.modules.values() and len(value_local_path.components) > 0:
        return
    value_dict = vars(value)
    for field_name in dir(value):
        try:
            field_value = value_dict[field_name]
        except KeyError:
            try:
                field_value = getattr(value, field_name)
            except AttributeError:
                continue
            is_dynamic_field = field_value is not getattr(value, field_name)
            if is_dynamic_field:
                continue
        _locate_values(
            field_value,
            (value_module_path, value_local_path.join(field_name)),
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )


@_locate_values.register(types.BuiltinFunctionType)
@_locate_values.register(types.BuiltinMethodType)
def _(
    value: types.BuiltinFunctionType | types.BuiltinMethodType,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    instance = value.__self__
    if value_path is not None:
        value_module_path, value_local_path = value_path
        parent_value = located_namespace_values[
            value_module_path, value_local_path.parent
        ]
        if (
            not inspect.isclass(parent_value)
            or instance not in parent_value.__mro__[1:]
        ):
            try:
                value_paths = namespace_value_id_paths[
                    _namespace_value_id(value)
                ]
            except KeyError:
                _set_absent_key(located_namespace_values, value_path, value)
                _set_absent_key(
                    namespace_value_id_paths,
                    _namespace_value_id(value),
                    [value_path],
                )
                _set_absent_key(
                    namespace_value_id_values,
                    _namespace_value_id(value),
                    value,
                )
            else:
                if value_path in value_paths:
                    return
                if any(
                    (
                        module_path == value_module_path
                        and value_local_path.starts_with(local_path)
                    )
                    for module_path, local_path in value_paths
                ):
                    value_paths.append(value_path)
                    return
                _set_absent_key(located_namespace_values, value_path, value)
                value_paths.append(value_path)
    _locate_values(
        instance,
        None,
        mentioned_module_paths,
        located_namespace_values=located_namespace_values,
        located_rest_values=located_rest_values,
        namespace_value_id_paths=namespace_value_id_paths,
        namespace_value_id_values=namespace_value_id_values,
    )


@_locate_values.register(types.MethodType)
def _(
    value: types.MethodType,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    instance = value.__self__
    if value_path is not None:
        value_module_path, value_local_path = value_path
        parent_value = located_namespace_values[
            value_module_path, value_local_path.parent
        ]
        if (
            not inspect.isclass(parent_value)
            or instance not in parent_value.__mro__[1:]
        ):
            try:
                value_paths = namespace_value_id_paths[
                    _namespace_value_id(value)
                ]
            except KeyError:
                _set_absent_key(located_namespace_values, value_path, value)
                _set_absent_key(
                    namespace_value_id_paths,
                    _namespace_value_id(value),
                    [value_path],
                )
                _set_absent_key(
                    namespace_value_id_values,
                    _namespace_value_id(value),
                    value,
                )
            else:
                if value_path in value_paths:
                    return
                if any(
                    (
                        module_path == value_module_path
                        and value_local_path.starts_with(local_path)
                    )
                    for module_path, local_path in value_paths
                ):
                    value_paths.append(value_path)
                    return
                _set_absent_key(located_namespace_values, value_path, value)
                value_paths.append(value_path)
    _locate_values(
        instance,
        None,
        mentioned_module_paths,
        located_namespace_values=located_namespace_values,
        located_rest_values=located_rest_values,
        namespace_value_id_paths=namespace_value_id_paths,
        namespace_value_id_values=namespace_value_id_values,
    )


@_locate_values.register(types.GetSetDescriptorType)
@_locate_values.register(types.MemberDescriptorType)
@_locate_values.register(types.MethodDescriptorType)
@_locate_values.register(types.WrapperDescriptorType)
def _(
    value: _AnyDescriptorType,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    object_class = value.__objclass__
    if value_path is not None:
        value_module_path, value_local_path = value_path
        parent_value = located_namespace_values[
            value_module_path, value_local_path.parent
        ]
        if (
            not inspect.isclass(parent_value)
            or object_class not in parent_value.__mro__[1:]
        ):
            try:
                value_paths = namespace_value_id_paths[
                    _namespace_value_id(value)
                ]
            except KeyError:
                _set_absent_key(located_namespace_values, value_path, value)
                _set_absent_key(
                    namespace_value_id_paths,
                    _namespace_value_id(value),
                    [value_path],
                )
                _set_absent_key(
                    namespace_value_id_values,
                    _namespace_value_id(value),
                    value,
                )
            else:
                if value_path in value_paths:
                    return
                if any(
                    (
                        module_path == value_module_path
                        and value_local_path.starts_with(local_path)
                    )
                    for module_path, local_path in value_paths
                ):
                    value_paths.append(value_path)
                    return
                _set_absent_key(located_namespace_values, value_path, value)
                value_paths.append(value_path)
    _locate_values(
        object_class,
        None,
        mentioned_module_paths,
        located_namespace_values=located_namespace_values,
        located_rest_values=located_rest_values,
        namespace_value_id_paths=namespace_value_id_paths,
        namespace_value_id_values=namespace_value_id_values,
    )


@_locate_values.register(types.ClassMethodDescriptorType)
def _(
    value: types.ClassMethodDescriptorType,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    if value_path is not None:
        value_module_path, value_local_path = value_path
        try:
            value_paths = namespace_value_id_paths[_namespace_value_id(value)]
        except KeyError:
            _set_absent_key(located_namespace_values, value_path, value)
            _set_absent_key(
                namespace_value_id_paths,
                _namespace_value_id(value),
                [value_path],
            )
            _set_absent_key(
                namespace_value_id_values, _namespace_value_id(value), value
            )
        else:
            if value_path in value_paths:
                return
            assert not any(
                (
                    module_path == value_module_path
                    and value_local_path.starts_with(local_path)
                )
                for module_path, local_path in value_paths
            )
            _set_absent_key(located_namespace_values, value_path, value)
            value_paths.append(value_path)
    if (callable_ := getattr(value, '__func__', None)) is not None:
        _locate_values(
            callable_,
            None,
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )
    else:
        _locate_values(
            value.__objclass__,
            None,
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )


def _namespace_value_id(value: _NamespaceValue, /) -> _Id:
    assert _is_namespace_value(value), value
    return _Id(id(value))


@_locate_values.register(types.FunctionType)
def _(
    value: types.FunctionType,
    value_path: ObjectPath | None,
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: MutableMapping[ObjectPath, _NamespaceValue],
    namespace_value_id_paths: MutableMapping[_Id, list[ObjectPath]],
    namespace_value_id_values: MutableMapping[_Id, _NamespaceValue],
) -> None:
    if value_path is not None:
        value_module_path, value_local_path = value_path
        try:
            value_paths = namespace_value_id_paths[_namespace_value_id(value)]
        except KeyError:
            _set_absent_key(located_namespace_values, value_path, value)
            _set_absent_key(
                namespace_value_id_paths,
                _namespace_value_id(value),
                [value_path],
            )
            _set_absent_key(
                namespace_value_id_values, _namespace_value_id(value), value
            )
        else:
            if value_path in value_paths:
                return
            assert not any(
                (
                    module_path == value_module_path
                    and value_local_path.starts_with(local_path)
                )
                for module_path, local_path in value_paths
            )
            _set_absent_key(located_namespace_values, value_path, value)
            value_paths.append(value_path)
    if (
        isinstance(module_name := value.__module__, str)
        and (module := _checked_find_module_by_name(module_name)) is not None
    ):
        module_object_path = (
            ModulePath.from_module_name(module_name),
            LocalObjectPath(),
        )
        _register_module_path(
            mentioned_module_paths, module, module_object_path
        )
        _locate_values(
            module,
            module_object_path,
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )


def _register_module_path(
    mentioned_module_paths: MutableMapping[
        types.ModuleType, dict[ObjectPath, None]
    ],
    module: types.ModuleType,
    module_object_path: ObjectPath,
    /,
) -> None:
    assert isinstance(module, types.ModuleType), module
    assert isinstance(module_object_path, tuple), module_object_path
    assert len(module_object_path) == 2, module_object_path
    mentioned_module_paths.setdefault(module, {}).setdefault(
        module_object_path, None
    )


def _checked_find_module_by_name(
    module_name: str, /
) -> types.ModuleType | None:
    try:
        return sys.modules[module_name]
    except KeyError:
        pass
    return next(
        (
            candidate
            for candidate in sys.modules.values()
            if candidate.__name__ == module_name
        ),
        None,
    )


_AnyDescriptorType: TypeAlias = (
    types.GetSetDescriptorType
    | types.MemberDescriptorType
    | types.MethodDescriptorType
    | types.WrapperDescriptorType
)
_NamespaceValue: TypeAlias = (
    _AnyDescriptorType
    | types.BuiltinFunctionType
    | types.BuiltinMethodType
    | types.ClassMethodDescriptorType
    | types.FunctionType
    | types.MethodType
    | types.ModuleType
    | type[Any]
)
_Id = NewType('_Id', int)


def _parse_modules(
    *modules: types.ModuleType,
) -> MutableMapping[ModulePath, MutableObject]:
    mentioned_module_paths: dict[types.ModuleType, dict[ObjectPath, None]] = {}
    namespace_value_id_paths: dict[_Id, list[ObjectPath]] = {}
    namespace_value_id_values: dict[_Id, _NamespaceValue] = {}
    located_namespace_values: dict[ObjectPath, _NamespaceValue] = {}
    located_rest_values: dict[ObjectPath, Any] = {}
    for module in modules:
        module_name = module.__name__
        assert module_name in MODULE_NAMES[module], (
            module_name,
            MODULE_NAMES[module],
        )
        _locate_values(
            module,
            (ModulePath.from_module_name(module_name), LocalObjectPath()),
            mentioned_module_paths,
            located_namespace_values=located_namespace_values,
            located_rest_values=located_rest_values,
            namespace_value_id_paths=namespace_value_id_paths,
            namespace_value_id_values=namespace_value_id_values,
        )
    namespace_value_id_origin_paths: dict[_Id, ObjectPath] = {}
    references: dict[ObjectPath, ObjectPath] = {}
    _locate_module_origins(
        [
            (module, list(module_paths))
            for module, module_paths in mentioned_module_paths.items()
        ],
        module_origin_id_paths=namespace_value_id_origin_paths,
        references=references,
    )
    module_definition_nodes: dict[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ] = {}
    _locate_non_module_namespace_objects(
        namespace_value_id_paths,
        namespace_value_id_values,
        located_namespace_values=located_namespace_values,
        module_definition_nodes=module_definition_nodes,
        namespace_value_id_origin_paths=namespace_value_id_origin_paths,
        references=references,
    )
    assert len(namespace_value_id_origin_paths) == len(
        namespace_value_id_paths
    ), [
        value
        for value in namespace_value_id_paths
        if value not in namespace_value_id_origin_paths
    ]
    for (
        value_origin_module_path,
        value_origin_local_path,
    ) in namespace_value_id_origin_paths.values():
        if len(value_origin_local_path.components) == 0:
            continue
        parent_origin_path = (
            value_origin_module_path,
            value_origin_local_path.parent,
        )
        assert (
            namespace_value_id_origin_paths[
                _namespace_value_id(
                    located_namespace_values[parent_origin_path]
                )
            ]
            == parent_origin_path
        ), (
            namespace_value_id_origin_paths[
                _namespace_value_id(
                    located_namespace_values[parent_origin_path]
                )
            ],
            parent_origin_path,
        )
    dependencies: dict[ObjectPath, set[ObjectPath]] = {}
    base_cls_paths: dict[ObjectPath, list[ObjectPath]] = {}
    instance_cls_paths: dict[ObjectPath, ObjectPath] = {}
    metacls_paths: dict[ObjectPath, ObjectPath] = {}
    method_component_paths: dict[
        ObjectPath, tuple[ObjectPath, ObjectPath]
    ] = {}
    _collect_namespace_object_dependencies(
        dependencies,
        base_cls_paths=base_cls_paths,
        instance_cls_paths=instance_cls_paths,
        located_namespace_values=located_namespace_values,
        metacls_paths=metacls_paths,
        method_component_paths=method_component_paths,
        namespace_value_id_origin_paths=namespace_value_id_origin_paths,
        namespace_value_id_values=namespace_value_id_values,
    )
    _collect_rest_object_dependencies(
        dependencies,
        instance_cls_paths=instance_cls_paths,
        located_namespace_values=located_namespace_values,
        located_rest_values=located_rest_values,
        namespace_value_id_origin_paths=namespace_value_id_origin_paths,
    )
    topologically_sorted_value_paths = [
        *graphlib.TopologicalSorter(dependencies).static_order()
    ]
    result: dict[ModulePath, MutableObject] = {}
    for value_path in topologically_sorted_value_paths:
        value_module_path, value_local_path = value_path
        value_object: Object
        try:
            value = located_namespace_values[value_path]
        except KeyError:
            value = located_rest_values[value_path]
            if (
                maybe_instance_cls_path := instance_cls_paths.get(value_path)
            ) is not None:
                value_object = PlainObject(
                    ObjectKind.INSTANCE,
                    value_module_path,
                    value_local_path,
                    _path_to_object(result, maybe_instance_cls_path),
                )
            else:
                value_object = UnknownObject(
                    value_module_path, value_local_path
                )
        else:
            if inspect.ismodule(value):
                value_object = Module(
                    Scope(
                        ScopeKind.STATIC_MODULE,
                        value_module_path,
                        value_local_path,
                    )
                )
                if len(value_local_path.components) == 0:
                    result[value_module_path] = value_object
                else:
                    result[value_module_path].set_nested_attribute(
                        value_local_path, value_object
                    )
                value_object.set_attribute(
                    NAME_FIELD_NAME,
                    UnknownObject(
                        value_module_path,
                        value_local_path.join(NAME_FIELD_NAME),
                    ),
                )
                value_object.set_value(NAME_FIELD_NAME, value.__name__)
                continue
            if inspect.isclass(value):
                origin_base_cls_paths = base_cls_paths[value_path]
                base_cls_objects = [
                    _checked_get_object_by_path(
                        result, base_cls_module_path, base_cls_local_path
                    )
                    for base_cls_module_path, base_cls_local_path in (
                        origin_base_cls_paths
                    )
                ]
                has_unknown_base = any(
                    base_cls is None for base_cls in base_cls_objects
                )
                value_object = Class(
                    Scope(
                        (
                            ScopeKind.METACLASS
                            if _is_metaclass(value)
                            else ScopeKind.CLASS
                        ),
                        value_module_path,
                        value_local_path,
                    ),
                    *[
                        base_cls
                        for base_cls in base_cls_objects
                        if base_cls is not None
                    ],
                    *(
                        (
                            Class(
                                Scope(
                                    ScopeKind.UNKNOWN_CLASS,
                                    value_module_path,
                                    value_local_path.join(
                                        f'__base_{len(origin_base_cls_paths) - 1}__'  # noqa: E501
                                    ),
                                ),
                                metaclass=MISSING,
                            ),
                        )
                        if has_unknown_base
                        else ()
                    ),
                    metaclass=(
                        _path_to_object_or_unknown(
                            result, metacls_paths[value_path]
                        )
                        if (
                            not _is_metaclass(value)
                            and value is not builtins.object
                        )
                        else MISSING
                    ),
                )
                value_object.set_attribute(
                    NAME_FIELD_NAME,
                    UnknownObject(value_module_path, value_local_path),
                )
                value_object.set_value(NAME_FIELD_NAME, value.__name__)
            elif (
                origin_method_component_paths := method_component_paths.get(
                    value_path
                )
            ) is not None:
                method_callable_path, method_instance_path = (
                    origin_method_component_paths
                )
                value_object = Method(
                    ensure_type(
                        _path_to_object_or_unknown(
                            result, method_callable_path
                        ),
                        CALLABLE_OBJECT_CLASSES,
                    ),
                    _path_to_object_or_unknown(result, method_instance_path),
                )
            else:
                assert inspect.isroutine(value) or isinstance(
                    value, _AnyDescriptorType
                ), value
                value_ast_node: AnyFunctionDefinitionAstNode | None = None
                try:
                    value_module_function_definition_nodes = (
                        module_definition_nodes[value_module_path]
                    )
                except KeyError:
                    pass
                else:
                    value_ast_nodes = (
                        value_module_function_definition_nodes.get(
                            value_local_path, []
                        )
                    )
                    if len(value_ast_nodes) > 0:
                        candidate_ast_node = value_ast_nodes[-1]
                        assert isinstance(
                            candidate_ast_node, AnyFunctionDefinitionAstNode
                        ), candidate_ast_node
                        value_ast_node = candidate_ast_node
                value_base_cls: Class | UnknownObject
                try:
                    instance_cls_path = instance_cls_paths[value_path]
                except KeyError:
                    value_base_cls = UnknownObject(
                        value_module_path, value_local_path.join('__base__')
                    )
                else:
                    value_base_cls = ensure_type(
                        _path_to_object(result, instance_cls_path),
                        (Class, UnknownObject),
                    )
                value_object = Routine(
                    value_module_path,
                    value_local_path,
                    value_base_cls,
                    ast_node=value_ast_node,
                )
                if inspect.isfunction(value):
                    value_object.set_attribute(
                        FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
                        PlainObject(
                            ObjectKind.INSTANCE,
                            value_module_path,
                            value_local_path.join(
                                FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME
                            ),
                            _path_to_object(
                                result,
                                namespace_value_id_origin_paths[
                                    _namespace_value_id(builtins.tuple)
                                ],
                            ),
                        ),
                    )
                    value_object.set_value(
                        FUNCTION_POSITIONAL_DEFAULTS_FIELD_NAME,
                        value.__defaults__ or (),
                    )
                    value_object.set_attribute(
                        FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
                        PlainObject(
                            ObjectKind.INSTANCE,
                            value_module_path,
                            value_local_path.join(
                                FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME
                            ),
                            _path_to_object(
                                result,
                                namespace_value_id_origin_paths[
                                    _namespace_value_id(builtins.dict)
                                ],
                            ),
                        ),
                    )
                    value_object.set_value(
                        FUNCTION_KEYWORD_ONLY_DEFAULTS_FIELD_NAME,
                        value.__kwdefaults__ or {},
                    )
                    value_object.set_attribute(
                        '__code__',
                        PlainObject(
                            ObjectKind.INSTANCE,
                            value_module_path,
                            value_local_path.join('__code__'),
                            _path_to_object(
                                result,
                                namespace_value_id_origin_paths[
                                    _namespace_value_id(types.CodeType)
                                ],
                            ),
                        ),
                    )
        value_module_object = result[value_module_path]
        value_module_object.set_nested_attribute(
            value_local_path, value_object
        )
        if is_safe(value):
            value_module_object.set_nested_value(value_local_path, value)
    topologically_sorted_references = [
        (candidate_path, reference_path)
        for candidate_path in graphlib.TopologicalSorter(
            {
                referent_path: [
                    reference_path,
                    *_to_parent_paths(referent_path),
                ]
                for referent_path, reference_path in references.items()
            }
        ).static_order()
        if (reference_path := references.get(candidate_path)) is not None
    ]
    for (
        referent_module_path,
        referent_local_path,
    ), reference_path in topologically_sorted_references:
        referent_object = _path_to_object(result, reference_path)
        if len(referent_local_path.components) == 0:
            assert isinstance(referent_object, MutableObject)
            result[referent_module_path] = referent_object
        else:
            result[referent_module_path].set_nested_attribute(
                referent_local_path, referent_object
            )
    return result


def _locate_non_module_namespace_objects(
    namespace_value_id_paths: Mapping[_Id, Sequence[ObjectPath]],
    namespace_value_id_values: Mapping[_Id, _NamespaceValue],
    /,
    *,
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: dict[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AsyncFunctionDef | FunctionDef | ClassDef],
        ],
    ],
    namespace_value_id_origin_paths: MutableMapping[_Id, ObjectPath],
    references: dict[
        tuple[ModulePath, LocalObjectPath], tuple[ModulePath, LocalObjectPath]
    ],
) -> None:
    topologically_sorted_value_ids = (
        _to_topologically_sorted_sequence_resolving_cycles_by_deletion(
            {
                value_id: {
                    _namespace_value_id(parent_value)
                    for path in value_paths
                    for parent_path in _to_parent_paths(path)
                    if (
                        (
                            (
                                parent_value := located_namespace_values.get(
                                    parent_path
                                )
                            )
                            is not None
                        )
                        and not inspect.ismodule(parent_value)
                    )
                }
                for value_id, value_paths in namespace_value_id_paths.items()
                if not inspect.ismodule(namespace_value_id_values[value_id])
            }
        )
    )
    for value_id in topologically_sorted_value_ids:
        value_paths = namespace_value_id_paths[value_id]
        value = namespace_value_id_values[value_id]
        assert not inspect.ismodule(value)
        assert len(value_paths) > 0
        original_module_value_paths = [
            (module_path, local_path)
            for module_path, local_path in value_paths
            if (
                namespace_value_id_origin_paths[
                    _namespace_value_id(
                        located_namespace_values[
                            module_path, LocalObjectPath()
                        ]
                    )
                ]
                == (module_path, LocalObjectPath())
            )
        ]
        original_parent_subvalue_paths = [
            object_path
            for object_path in original_module_value_paths
            if all(
                (
                    namespace_value_id_origin_paths.get(
                        _namespace_value_id(
                            located_namespace_values[parent_path]
                        )
                    )
                    in (None, parent_path)
                )
                for parent_path in _to_parent_paths(object_path)
            )
        ]
        try:
            (origin_path,) = original_parent_subvalue_paths
        except ValueError:
            pass
        else:
            _set_absent_key(
                namespace_value_id_origin_paths, value_id, origin_path
            )
            continue
        if isinstance(value, classmethod):
            value = value.__func__
        if inspect.isclass(value):
            origin_path = _to_cls_origin_path(
                value,
                original_parent_subvalue_paths,
                located_values=located_namespace_values,
                module_definition_nodes=module_definition_nodes,
            )
        elif inspect.isfunction(value):
            origin_path = _to_function_origin_path(
                value,
                original_parent_subvalue_paths,
                located_values=located_namespace_values,
                module_definition_nodes=module_definition_nodes,
            )
        elif inspect.isbuiltin(value):
            origin_path = _to_builtin_function_origin_path(
                value,
                original_parent_subvalue_paths,
                located_values=located_namespace_values,
            )
        elif inspect.ismethod(value):
            instance = value.__self__
            if _is_namespace_value(instance):
                instance_origin_path = namespace_value_id_origin_paths[
                    _namespace_value_id(instance)
                ]
                instance_origin_module_path, instance_origin_local_path = (
                    instance_origin_path
                )
                origin_path = (
                    instance_origin_module_path,
                    instance_origin_local_path.join(value.__func__.__name__),
                )
            else:
                origin_path = original_parent_subvalue_paths[0]
        else:
            assert isinstance(value, _AnyDescriptorType), value_paths
            origin_path = _to_any_descriptor_origin_path(
                value,
                original_parent_subvalue_paths,
                located_values=located_namespace_values,
                module_definition_nodes=module_definition_nodes,
            )
        _set_absent_key(namespace_value_id_origin_paths, value_id, origin_path)
        for candidate_path in value_paths:
            if candidate_path == origin_path:
                continue
            _add_reference(references, candidate_path, origin_path)


def _collect_rest_object_dependencies(
    dependencies: MutableMapping[ObjectPath, set[ObjectPath]],
    /,
    *,
    instance_cls_paths: MutableMapping[ObjectPath, ObjectPath],
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    located_rest_values: dict[tuple[ModulePath, LocalObjectPath], Any],
    namespace_value_id_origin_paths: dict[
        Any, tuple[ModulePath, LocalObjectPath]
    ],
) -> None:
    for value_path, value in located_rest_values.items():
        assert not _is_namespace_value(value), value_path
        value_module_path, value_local_path = value_path
        value_parent_path = (value_module_path, value_local_path.parent)
        if (
            namespace_value_id_origin_paths[
                _namespace_value_id(
                    located_namespace_values[value_parent_path]
                )
            ]
            != value_parent_path
        ):
            continue
        value_dependencies = {value_parent_path}
        _set_absent_key(dependencies, value_path, value_dependencies)
        if (
            instance_cls_path := namespace_value_id_origin_paths.get(
                _namespace_value_id(type(value))
            )
        ) is not None:
            instance_cls_paths[value_path] = instance_cls_path
            value_dependencies.add(instance_cls_path)


def _collect_namespace_object_dependencies(
    dependencies: MutableMapping[ObjectPath, set[ObjectPath]],
    /,
    *,
    base_cls_paths: MutableMapping[ObjectPath, list[ObjectPath]],
    instance_cls_paths: MutableMapping[ObjectPath, ObjectPath],
    located_namespace_values: MutableMapping[ObjectPath, _NamespaceValue],
    metacls_paths: MutableMapping[ObjectPath, ObjectPath],
    method_component_paths: MutableMapping[
        ObjectPath, tuple[ObjectPath, ObjectPath]
    ],
    namespace_value_id_origin_paths: dict[_Id, ObjectPath],
    namespace_value_id_values: Mapping[_Id, _NamespaceValue],
) -> None:
    for value_id, value_path in namespace_value_id_origin_paths.copy().items():
        value = namespace_value_id_values[value_id]
        if inspect.ismodule(value):
            dependencies[namespace_value_id_origin_paths[value_id]] = set()
            continue
        value_module_path, value_local_path = value_path
        value_dependencies = dependencies.setdefault(value_path, set())
        assert len(value_local_path.components) > 0, value_path
        value_parent_path = (value_module_path, value_local_path.parent)
        assert (
            parent_value := located_namespace_values.get(value_parent_path)
        ) is None or namespace_value_id_origin_paths[
            _namespace_value_id(parent_value)
        ] == value_parent_path
        value_dependencies.add(value_parent_path)
        if inspect.isclass(value):
            base_cls_paths[value_path] = origin_base_cls_paths = []
            for base_cls in value.__bases__:
                try:
                    origin_base_cls_path = namespace_value_id_origin_paths[
                        _namespace_value_id(base_cls)
                    ]
                except KeyError:
                    origin_base_cls_path = (
                        ModulePath.from_module_name(base_cls.__module__),
                        LocalObjectPath.from_object_name(
                            base_cls.__qualname__
                        ),
                    )
                else:
                    value_dependencies.add(origin_base_cls_path)
                origin_base_cls_paths.append(origin_base_cls_path)
            if not _is_metaclass(value) and value is not builtins.object:
                metacls = type(value)
                try:
                    origin_metacls_path = namespace_value_id_origin_paths[
                        _namespace_value_id(metacls)
                    ]
                except KeyError:
                    origin_metacls_path = (
                        ModulePath.from_module_name(metacls.__module__),
                        LocalObjectPath.from_object_name(metacls.__qualname__),
                    )
                else:
                    value_dependencies.add(origin_metacls_path)
                metacls_paths[value_path] = origin_metacls_path
        else:
            try:
                origin_cls_path = namespace_value_id_origin_paths[
                    _namespace_value_id(type(value))
                ]
            except KeyError:
                pass
            else:
                instance_cls_paths[value_path] = origin_cls_path
                value_dependencies.add(origin_cls_path)
            if isinstance(value, types.MethodType):
                method_instance = value.__self__
                if inspect.isclass(method_instance):
                    if (
                        located_namespace_values[
                            value_module_path, value_local_path.parent
                        ]
                        is method_instance
                    ) and inspect.isroutine(value_callable := value.__func__):
                        namespace_value_id_origin_paths[
                            _namespace_value_id(value_callable)
                        ] = namespace_value_id_origin_paths.pop(value_id)
                        located_namespace_values[value_path] = value_callable
                        continue
                    try:
                        method_instance_path = namespace_value_id_origin_paths[
                            _namespace_value_id(method_instance)
                        ]
                    except KeyError:
                        method_instance_path = (
                            ModulePath.from_module_name(
                                method_instance.__module__
                            ),
                            LocalObjectPath.from_object_name(
                                method_instance.__qualname__
                            ),
                        )
                    else:
                        value_dependencies.add(method_instance_path)
                else:
                    assert not inspect.ismodule(method_instance)
                    method_instance_path = None
                method_callable = value.__func__
                if _is_namespace_value(method_callable):
                    method_callable_path = namespace_value_id_origin_paths[
                        _namespace_value_id(method_callable)
                    ]
                    value_dependencies.add(method_callable_path)
                else:
                    method_callable_path = None
                if (
                    method_callable_path is not None
                    and method_instance_path is not None
                ):
                    method_component_paths[value_path] = (
                        method_callable_path,
                        method_instance_path,
                    )
            elif inspect.isbuiltin(value):
                method_instance = value.__self__
                if inspect.isclass(method_instance) or inspect.ismodule(
                    method_instance
                ):
                    try:
                        method_instance_path = namespace_value_id_origin_paths[
                            _namespace_value_id(method_instance)
                        ]
                    except KeyError:
                        pass
                    else:
                        value_dependencies.add(method_instance_path)
                else:
                    try:
                        method_instance_cls_path = (
                            namespace_value_id_origin_paths[
                                _namespace_value_id(type(method_instance))
                            ]
                        )
                    except KeyError:
                        pass
                    else:
                        value_dependencies.add(method_instance_cls_path)
            elif inspect.isfunction(value):
                value_dependencies.add(
                    namespace_value_id_origin_paths[
                        _namespace_value_id(builtins.dict)
                    ]
                )
                value_dependencies.add(
                    namespace_value_id_origin_paths[
                        _namespace_value_id(builtins.tuple)
                    ]
                )
            elif isinstance(value, types.ClassMethodDescriptorType):
                try:
                    object_class = value.__objclass__
                except AttributeError:
                    assert isinstance(value, classmethod), value
                else:
                    value_dependencies.add(
                        namespace_value_id_origin_paths[
                            _namespace_value_id(object_class)
                        ]
                    )
            else:
                assert isinstance(value, _AnyDescriptorType), (
                    value,
                    value_path,
                )
                value_dependencies.add(
                    namespace_value_id_origin_paths[
                        _namespace_value_id(value.__objclass__)
                    ]
                )


def _is_namespace_value(value: Any, /) -> TypeIs[_NamespaceValue]:
    return isinstance(
        value,
        (
            _AnyDescriptorType
            | types.BuiltinFunctionType
            | types.BuiltinMethodType
            | types.ClassMethodDescriptorType
            | types.FunctionType
            | types.MethodType
            | types.ModuleType
            | type
        ),
    )


def _locate_module_origins(
    module_with_name_and_local_paths: Iterable[
        tuple[types.ModuleType, Sequence[ObjectPath]]
    ],
    /,
    *,
    module_origin_id_paths: MutableMapping[_Id, ObjectPath],
    references: MutableMapping[ObjectPath, ObjectPath],
) -> None:
    for module, module_object_paths in module_with_name_and_local_paths:
        assert inspect.ismodule(module)
        origin_path = _to_module_origin_path(module, module_object_paths)
        _set_absent_key(
            module_origin_id_paths, _namespace_value_id(module), origin_path
        )
        for candidate_path in module_object_paths:
            if candidate_path == origin_path:
                continue
            _add_reference(references, candidate_path, origin_path)


def _to_parent_paths(
    object_path: ObjectPath, /
) -> Sequence[tuple[ModulePath, LocalObjectPath]]:
    module_path, local_path = object_path
    return [
        (module_path, LocalObjectPath(*local_path.components[:stop_index]))
        for stop_index in reversed(range(len(local_path.components)))
    ]


def _to_topologically_sorted_sequence_resolving_cycles_by_deletion(
    mapping: Mapping[_KT, set[_KT]], /
) -> Sequence[_KT]:
    while True:
        try:
            result = [*graphlib.TopologicalSorter(mapping).static_order()]
        except graphlib.CycleError as error:
            _, cycle = error.args
            assert len(cycle) > 1
            mapping[cycle[0]].remove(cycle[1])
        else:
            break
    return result


def _to_module_origin_path(
    module: types.ModuleType, module_object_paths: Sequence[ObjectPath], /
) -> ObjectPath:
    try:
        (origin_path,) = module_object_paths
    except ValueError:
        assert len(module_object_paths) > 0
        candidate_paths: Sequence[ObjectPath] = module_object_paths
        plain_module_paths = [
            (module_path, local_path)
            for module_path, local_path in module_object_paths
            if len(local_path.components) == 0
        ]
        if len(plain_module_paths) > 0:
            candidate_paths = plain_module_paths
        module_name_candidates = [
            (module_path, local_path)
            for module_path, local_path in candidate_paths
            if (
                '.'.join([*module_path.components, *local_path.components])
                == module.__name__
            )
        ]
        try:
            (origin_path,) = module_name_candidates
        except ValueError:
            assert len(module_name_candidates) == 0
            origin_path = candidate_paths[0]
    return origin_path


def _to_any_descriptor_origin_path(
    descriptor: _AnyDescriptorType,
    descriptor_paths: Sequence[ObjectPath],
    /,
    *,
    located_values: Mapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
) -> ObjectPath:
    assert len(descriptor_paths) > 0, descriptor
    # first we try to narrow down paths
    # based on the presence of function definition in the module file AST ...
    ast_candidate_paths = (
        _to_any_descriptor_origin_candidate_paths_based_on_module_ast(
            descriptor_paths, located_values, module_definition_nodes
        )
    )
    if len(ast_candidate_paths) == 1:
        (result,) = ast_candidate_paths
        return result
    # ... then by module names mentioned in the members ...
    candidate_paths = ast_candidate_paths or descriptor_paths
    candidate_module_paths = [
        module_path for module_path, _ in candidate_paths
    ]
    object_class_member_module_paths: set[ModulePath] = set()
    self_local_path = LocalObjectPath.checked_from_object_name(
        descriptor.__qualname__
    )
    object_class = descriptor.__objclass__
    for object_class_member in vars(object_class).values():
        if inspect.isclass(object_class_member):
            if (
                self_local_path is not None
                and not LocalObjectPath.from_object_name(
                    object_class_member.__qualname__
                ).starts_with(self_local_path)
            ):
                continue
            cls_module_name = object_class_member.__module__
            try:
                cls_module_path = ModulePath.from_module_name(cls_module_name)
            except ValueError:
                continue
            if cls_module_path in candidate_module_paths:
                object_class_member_module_paths.add(cls_module_path)
        if not inspect.isfunction(object_class_member):
            object_class_member = getattr(
                object_class_member, '__func__', object_class_member
            )
        if not inspect.isfunction(object_class_member):
            continue
        if (
            self_local_path is not None
            and not LocalObjectPath.from_object_name(
                object_class_member.__qualname__
            ).starts_with(self_local_path)
        ):
            continue
        function_module_name = getattr(object_class_member, '__module__', None)
        if not isinstance(function_module_name, str):
            continue
        try:
            function_module_path = ModulePath.from_module_name(
                function_module_name
            )
        except ValueError:
            continue
        if function_module_path in candidate_module_paths:
            object_class_member_module_paths.add(function_module_path)
    if len(object_class_member_module_paths) > 0:
        candidate_paths = [
            (module_path, local_path)
            for module_path, local_path in candidate_paths
            if module_path in object_class_member_module_paths
        ]
    assert isinstance(object_class, type | types.ModuleType)
    object_class_module_path = ModulePath.from_module_name(
        object_class.__module__
        if inspect.isclass(object_class)
        else object_class.__name__
    )
    if (
        len(
            self_module_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_module_path == object_class_module_path
            ]
        )
        > 0
    ):
        candidate_paths = self_module_path_candidate_paths
    if (
        len(
            self_local_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_local_path == self_local_path
            ]
        )
        > 0
    ):
        candidate_paths = self_local_path_candidate_paths
    # ... if there are still more than one candidate left
    # -- we just pick the first one
    return candidate_paths[0]


def _to_cls_origin_path(
    cls: type[Any],
    cls_paths: Sequence[ObjectPath],
    /,
    *,
    located_values: Mapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
) -> ObjectPath:
    assert len(cls_paths) > 0, cls
    # first we try to narrow down classes
    # based on the presence of class definition in the module file AST ...
    ast_candidate_paths = _to_cls_origin_candidate_paths_based_on_module_ast(
        cls_paths, located_values, module_definition_nodes
    )
    if len(ast_candidate_paths) == 1:
        (result,) = ast_candidate_paths
        return result
    # ... then by module names mentioned in the members ...
    candidate_paths = ast_candidate_paths or cls_paths
    candidate_module_paths = [
        module_path for module_path, _ in candidate_paths
    ]
    cls_member_module_paths: set[ModulePath] = set()
    self_local_path = LocalObjectPath.checked_from_object_name(
        cls.__qualname__
    )
    for cls_member in vars(cls).values():
        if inspect.isclass(cls_member):
            if (
                self_local_path is not None
                and not LocalObjectPath.from_object_name(
                    cls_member.__qualname__
                ).starts_with(self_local_path)
            ):
                continue
            nested_cls_module_name = cls_member.__module__
            try:
                nested_cls_module_path = ModulePath.from_module_name(
                    nested_cls_module_name
                )
            except ValueError:
                continue
            if nested_cls_module_path in candidate_module_paths:
                cls_member_module_paths.add(nested_cls_module_path)
        if not inspect.isfunction(cls_member):
            cls_member = getattr(cls_member, '__func__', cls_member)
        if not inspect.isfunction(cls_member):
            continue
        if (
            self_local_path is not None
            and (
                (
                    cls_member_local_path
                    := LocalObjectPath.checked_from_object_name(
                        cls_member.__qualname__
                    )
                )
                is not None
            )
            and not cls_member_local_path.starts_with(self_local_path)
        ):
            continue
        function_module_name = getattr(cls_member, '__module__', None)
        if not isinstance(function_module_name, str):
            continue
        try:
            function_module_path = ModulePath.from_module_name(
                function_module_name
            )
        except ValueError:
            continue
        if function_module_path in candidate_module_paths:
            cls_member_module_paths.add(function_module_path)
    if len(cls_member_module_paths) > 0:
        candidate_paths = [
            (module_path, local_path)
            for module_path, local_path in candidate_paths
            if module_path in cls_member_module_paths
        ]
    self_module_path = ModulePath.from_module_name(cls.__module__)
    if (
        len(
            self_module_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_module_path == self_module_path
            ]
        )
        > 0
    ):
        candidate_paths = self_module_path_candidate_paths
    if self_local_path is not None and (
        len(
            self_local_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_local_path == self_local_path
            ]
        )
        > 0
    ):
        candidate_paths = self_local_path_candidate_paths
    # ... if there are still more than one candidate left
    # -- we just pick the first one
    return candidate_paths[0]


def _to_function_origin_path(
    function: types.FunctionType,
    function_paths: Sequence[ObjectPath],
    /,
    *,
    located_values: Mapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
) -> ObjectPath:
    assert len(function_paths) > 0, function
    ast_candidate_paths = (
        _to_function_origin_candidate_paths_based_on_module_ast(
            function_paths, located_values, module_definition_nodes
        )
    )
    if len(ast_candidate_paths) == 1:
        (result,) = ast_candidate_paths
        return result
    candidate_paths = ast_candidate_paths or function_paths
    self_module_name = function.__module__
    self_module_path = (
        ModulePath.from_module_name(self_module_name)
        if self_module_name is not None
        else None
    )
    if (
        self_module_path is not None
        and len(
            self_module_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_module_path == self_module_path
            ]
        )
        > 0
    ):
        candidate_paths = self_module_path_candidate_paths
    self_local_path = LocalObjectPath.from_object_name(function.__qualname__)
    if (
        len(
            self_local_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_local_path == self_local_path
            ]
        )
        > 0
    ):
        candidate_paths = self_local_path_candidate_paths
    return candidate_paths[0]


def _to_builtin_function_origin_path(
    function: types.BuiltinFunctionType,
    function_paths: Sequence[ObjectPath],
    /,
    *,
    located_values: Mapping[ObjectPath, _NamespaceValue],
) -> ObjectPath:
    assert len(function_paths) > 0, function
    candidate_paths = function_paths
    builtin_module_candidate_paths = []
    for object_path in candidate_paths:
        module_path, _ = object_path
        module = ensure_type(
            located_values[module_path, LocalObjectPath()], types.ModuleType
        )
        if (
            file_path_string := getattr(module, '__file__', None)
        ) is None or Path(file_path_string).name.endswith(
            tuple(EXTENSION_SUFFIXES)
        ):
            builtin_module_candidate_paths.append(object_path)
    if len(builtin_module_candidate_paths) > 0:
        candidate_paths = builtin_module_candidate_paths
    self_module_path = (
        ModulePath.from_module_name(self_module_name)
        if (self_module_name := function.__module__) is not None
        else None
    )
    self_local_path = LocalObjectPath.from_object_name(function.__qualname__)
    if self_module_path is not None and (
        len(
            self_module_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_module_path == self_module_path
            ]
        )
        > 0
    ):
        candidate_paths = self_module_path_candidate_paths
    if (
        len(
            self_local_path_candidate_paths := [
                (candidate_module_path, candidate_local_path)
                for candidate_module_path, candidate_local_path in (
                    candidate_paths
                )
                if candidate_local_path == self_local_path
            ]
        )
        > 0
    ):
        candidate_paths = self_local_path_candidate_paths
    return candidate_paths[0]


def _to_cls_origin_candidate_paths_based_on_module_ast(
    value_paths: Sequence[ObjectPath],
    located_values: Mapping[ObjectPath, Any],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
    /,
) -> Sequence[ObjectPath]:
    ast_paths: list[ObjectPath] = []
    builtin_paths: list[ObjectPath] = []
    for value_path in value_paths:
        value_module_path, value_local_path = value_path
        value_module = located_values[value_module_path, LocalObjectPath()]
        if (
            file_path_string := getattr(value_module, '__file__', None)
        ) is not None:
            try:
                value_module_definition_nodes = module_definition_nodes[
                    value_module_path
                ]
            except KeyError:
                try:
                    module_source = Path(file_path_string).read_text(
                        encoding='utf-8'
                    )
                except (OSError, UnicodeDecodeError):
                    value_module_definition_nodes = {}
                else:
                    parsed_value_module_definition_nodes: MutableMapping[
                        LocalObjectPath,
                        list[AnyFunctionDefinitionAstNode | ast.ClassDef],
                    ] = {}
                    parser = DefinitionAstNodeParser(
                        parsed_value_module_definition_nodes,
                        file_path=Path(file_path_string),
                        is_class_scope=False,
                        local_path=LocalObjectPath(),
                        module_path=value_module_path,
                        values={},
                    )
                    parser.visit(ast.parse(module_source))
                    value_module_definition_nodes = (
                        parsed_value_module_definition_nodes
                    )
                    _set_absent_key(
                        module_definition_nodes,
                        value_module_path,
                        value_module_definition_nodes,
                    )
            value_nodes = [
                node
                for node in value_module_definition_nodes.get(
                    value_local_path, []
                )
                if isinstance(node, ast.ClassDef)
            ]
            if len(value_nodes) > 0:
                ast_paths.append(value_path)
        else:
            builtin_paths.append(value_path)
    return ast_paths or builtin_paths


def _to_any_descriptor_origin_candidate_paths_based_on_module_ast(
    object_paths: Sequence[ObjectPath],
    located_values: Mapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
    /,
) -> Sequence[ObjectPath]:
    result: list[ObjectPath] = []
    for object_path in object_paths:
        module_path, value_local_path = object_path
        module = ensure_type(
            located_values[module_path, LocalObjectPath()], types.ModuleType
        )
        value_module_definition_nodes = _load_module_definition_nodes(
            module, module_path, module_definition_nodes
        )
        value_nodes = [
            node
            for node in value_module_definition_nodes.get(value_local_path, [])
            if isinstance(node, AnyFunctionDefinitionAstNode)
        ]
        if len(value_nodes) > 0:
            result.append(object_path)
    return result


def _to_function_origin_candidate_paths_based_on_module_ast(
    object_paths: Sequence[ObjectPath],
    located_values: Mapping[ObjectPath, _NamespaceValue],
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AnyFunctionDefinitionAstNode | ast.ClassDef],
        ],
    ],
    /,
) -> Sequence[ObjectPath]:
    result: list[ObjectPath] = []
    for object_path in object_paths:
        module_path, value_local_path = object_path
        module = ensure_type(
            located_values[module_path, LocalObjectPath()], types.ModuleType
        )
        value_module_definition_nodes = _load_module_definition_nodes(
            module, module_path, module_definition_nodes
        )
        value_nodes = [
            node
            for node in value_module_definition_nodes.get(value_local_path, [])
            if isinstance(node, AnyFunctionDefinitionAstNode)
        ]
        if len(value_nodes) > 0:
            result.append(object_path)
    return result


def _load_module_definition_nodes(
    module: types.ModuleType,
    module_path: ModulePath,
    module_definition_nodes: MutableMapping[
        ModulePath,
        Mapping[
            LocalObjectPath,
            Sequence[AsyncFunctionDef | FunctionDef | ClassDef],
        ],
    ],
) -> Mapping[
    LocalObjectPath, Sequence[AsyncFunctionDef | FunctionDef | ClassDef]
]:
    if (file_path_string := getattr(module, '__file__', None)) is not None:
        try:
            value_module_definition_nodes = module_definition_nodes[
                module_path
            ]
        except KeyError:
            try:
                module_source = Path(file_path_string).read_text(
                    encoding='utf-8'
                )
            except (OSError, UnicodeDecodeError):
                value_module_definition_nodes = {}
            else:
                parsed_value_module_definition_nodes: MutableMapping[
                    LocalObjectPath,
                    list[AnyFunctionDefinitionAstNode | ast.ClassDef],
                ] = {}
                DefinitionAstNodeParser(
                    parsed_value_module_definition_nodes,
                    file_path=Path(file_path_string),
                    is_class_scope=False,
                    local_path=LocalObjectPath(),
                    module_path=module_path,
                    values={},
                ).visit(ast.parse(module_source))
                value_module_definition_nodes = (
                    parsed_value_module_definition_nodes
                )
                _set_absent_key(
                    module_definition_nodes,
                    module_path,
                    value_module_definition_nodes,
                )
    else:
        value_module_definition_nodes = {}
    return value_module_definition_nodes


def _locate_objects(
    module_with_paths: Iterable[tuple[types.ModuleType, ObjectPath]],
    /,
    *,
    namespace_value_id_paths: MutableMapping[Any, list[ObjectPath]],
    located_values: MutableMapping[ObjectPath, Any],
    all_value_dependencies: MutableMapping[ObjectPath, set[ObjectPath]],
) -> None:
    queue: deque[tuple[types.ModuleType | type[Any], ObjectPath]] = deque(
        module_with_paths
    )
    while queue:
        value, value_path = queue.popleft()
        value_dependencies = all_value_dependencies.setdefault(
            value_path, set()
        )
        value_module_path, value_local_path = value_path
        if (value_paths := namespace_value_id_paths.get(value)) is not None:
            value_paths.append(value_path)
            _set_absent_key(located_values, value_path, value)
            continue
        _set_absent_key(located_values, value_path, value)
        _set_absent_key(namespace_value_id_paths, value, [value_path])
        field_names = dir(value)
        try:
            value_dict = vars(value)
        except Exception:
            pass
        else:
            field_names.sort(
                key=partial(
                    _call_or_else,
                    tuple(value_dict).index,
                    exception_classes=(ValueError,),
                    default=len(field_names),
                )
            )
        for field_name in field_names:
            try:
                field_value = getattr(value, field_name)
            except Exception:
                continue
            field_path = (value_module_path, value_local_path.join(field_name))
            if inspect.isclass(field_value) or inspect.ismodule(field_value):
                value_dependencies.add(field_path)
                queue.append((field_value, field_path))
            else:
                _set_absent_key(located_values, field_path, field_value)


def _checked_get_object_by_path(
    modules: Mapping[ModulePath, MutableObject],
    module_path: ModulePath,
    local_path: LocalObjectPath,
    /,
) -> Object | None:
    try:
        return modules[module_path].get_nested_attribute(local_path)
    except KeyError:
        return None


def _is_metaclass(value: Any, /) -> TypeIs[type[Any]]:
    return issubclass(value, type)


_ValueT = TypeVar('_ValueT')
_ResultT = TypeVar('_ResultT')


def _call_or_else(
    callable_: Callable[[_ValueT], _ResultT],
    value: _ValueT,
    /,
    *,
    exception_classes: tuple[type[Exception], ...],
    default: _ResultT,
) -> _ResultT:
    try:
        return callable_(value)
    except exception_classes:
        return default


_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


def _set_absent_key(
    mapping: MutableMapping[_KT, _VT], key: _KT, value: _VT, /
) -> None:
    assert key not in mapping, (mapping[key], value)
    mapping[key] = value


def _add_reference(
    references: MutableMapping[ObjectPath, ObjectPath],
    referent_path: ObjectPath,
    reference_path: ObjectPath,
    /,
) -> None:
    assert referent_path != reference_path, referent_path
    _set_absent_key(references, referent_path, reference_path)


def _path_to_object(
    modules: Mapping[ModulePath, MutableObject], object_path: ObjectPath, /
) -> Object:
    module_path, local_path = object_path
    return modules[module_path].get_nested_attribute(local_path)


def _path_to_object_or_unknown(
    modules: Mapping[ModulePath, MutableObject], object_path: ObjectPath, /
) -> Object:
    module_path, local_path = object_path
    try:
        return modules[module_path].get_nested_attribute(local_path)
    except KeyError:
        return UnknownObject(module_path, local_path)


MODULES: Final[MutableMapping[ModulePath, MutableObject]] = _parse_modules(
    builtins, sys, types
)
BUILTINS_MODULE: Final[Module] = ensure_type(
    MODULES[BUILTINS_MODULE_PATH], Module
)
TYPES_MODULE: Final[Object] = ensure_type(MODULES[TYPES_MODULE_PATH], Module)
Method.BASE_CLS = ensure_type(
    TYPES_MODULE.get_nested_attribute(TYPES_METHOD_TYPE_LOCAL_OBJECT_PATH),
    Class,
)


def _setup_builtin_classes() -> None:
    for cls in [
        builtins.bool,
        builtins.bytearray,
        builtins.bytes,
        builtins.dict,
        builtins.float,
        builtins.frozenset,
        builtins.int,
        builtins.list,
        builtins.set,
        builtins.str,
        builtins.tuple,
        builtins.type,
    ]:
        assert inspect.isclass(cls), cls
        BUILTINS_MODULE.set_nested_value(
            LocalObjectPath.from_object_name(cls.__qualname__), cls
        )


_setup_builtin_classes()
