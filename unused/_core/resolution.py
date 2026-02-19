from __future__ import annotations

import ast
import functools
from collections.abc import Sequence
from typing import TypeAlias

from typing_extensions import Self

from .evaluation import EVALUATION_EXCEPTIONS, evaluate_node
from .lookup import (
    lookup_namespace_by_expression_node,
    lookup_namespace_by_object_name,
)
from .namespace import Namespace, ObjectKind
from .object_path import (
    BUILTINS_MODULE_PATH,
    GLOBALS_LOCAL_OBJECT_PATH,
    LocalObjectPath,
    ModulePath,
)


class ResolvedAssignmentTargetSplitPath:
    @property
    def absolute(self, /) -> LocalObjectPath:
        return self._absolute

    @property
    def module(self, /) -> ModulePath:
        return self._module

    @property
    def relative(self, /) -> LocalObjectPath:
        return self._relative

    def join(self, /, *components: str) -> Self:
        return type(self)(
            self._module, self._absolute, self._relative.join(*components)
        )

    def combine_local(self, /) -> LocalObjectPath:
        return self._absolute.join(*self._relative.components)

    _absolute: LocalObjectPath
    _module: ModulePath
    _relative: LocalObjectPath

    __slots__ = '_absolute', '_module', '_relative'

    def __new__(
        cls,
        module: ModulePath,
        absolute: LocalObjectPath,
        relative: LocalObjectPath,
    ) -> Self:
        assert isinstance(absolute, LocalObjectPath), absolute
        assert isinstance(relative, LocalObjectPath), relative
        self = super().__new__(cls)
        self._absolute, self._module, self._relative = (
            absolute,
            module,
            relative,
        )
        return self

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}'
            f'({self._module!r}, {self._absolute!r}, {self._relative!r})'
        )


ResolvedAssignmentTarget: TypeAlias = (
    Sequence['ResolvedAssignmentTarget']
    | ResolvedAssignmentTargetSplitPath
    | None
)


@functools.singledispatch
def resolve_assignment_target(
    _node: ast.expr, _namespace: Namespace, /, *_parent_namespaces: Namespace
) -> ResolvedAssignmentTarget:
    return None


@resolve_assignment_target.register(ast.Attribute)
def _(
    node: ast.Attribute, namespace: Namespace, /, *parent_namespaces: Namespace
) -> ResolvedAssignmentTarget:
    if (
        object_path := resolve_assignment_target(
            node.value, namespace, *parent_namespaces
        )
    ) is not None:
        assert isinstance(object_path, ResolvedAssignmentTargetSplitPath)
        return object_path.join(node.attr)
    return None


@resolve_assignment_target.register(ast.List)
@resolve_assignment_target.register(ast.Tuple)
def _(
    node: ast.List | ast.Tuple,
    namespace: Namespace,
    /,
    *parent_namespaces: Namespace,
) -> ResolvedAssignmentTarget:
    return [
        resolve_assignment_target(element_node, namespace, *parent_namespaces)
        for element_node in node.elts
    ]


@resolve_assignment_target.register(ast.Name)
def _(
    node: ast.Name, namespace: Namespace, /, *parent_namespaces: Namespace
) -> ResolvedAssignmentTarget:
    object_name = node.id
    if isinstance(node.ctx, ast.Load):
        object_namespace = lookup_namespace_by_object_name(
            object_name, namespace, *parent_namespaces
        )
        if (
            object_namespace.module_path == namespace.module_path
            and object_namespace.local_path.starts_with(namespace.local_path)
        ):
            return ResolvedAssignmentTargetSplitPath(
                namespace.module_path,
                namespace.local_path,
                LocalObjectPath(
                    *object_namespace.local_path.components[
                        len(namespace.local_path.components) :
                    ]
                ),
            )
        return ResolvedAssignmentTargetSplitPath(
            object_namespace.module_path,
            object_namespace.local_path,
            LocalObjectPath(),
        )
    return ResolvedAssignmentTargetSplitPath(
        namespace.module_path,
        namespace.local_path,
        LocalObjectPath(object_name),
    )


@resolve_assignment_target.register(ast.NamedExpr)
def _(
    node: ast.NamedExpr, namespace: Namespace, /, *parent_namespaces: Namespace
) -> ResolvedAssignmentTarget:
    return resolve_assignment_target(node.value, namespace, *parent_namespaces)


@resolve_assignment_target.register(ast.Subscript)
def _(
    node: ast.Subscript, namespace: Namespace, /, *parent_namespaces: Namespace
) -> ResolvedAssignmentTarget:
    value_namespace = lookup_namespace_by_expression_node(
        node.value, namespace, *parent_namespaces
    )
    if value_namespace is None:
        return None
    if not (
        value_namespace.kind is ObjectKind.ROUTINE_CALL
        and value_namespace.module_path == BUILTINS_MODULE_PATH
        and value_namespace.local_path == GLOBALS_LOCAL_OBJECT_PATH
    ):
        return None
    try:
        slice_value = evaluate_node(node.slice, namespace, *parent_namespaces)
    except EVALUATION_EXCEPTIONS:
        return None
    assert isinstance(slice_value, str), ast.unparse(node)
    return ResolvedAssignmentTargetSplitPath(
        namespace.module_path, LocalObjectPath(), LocalObjectPath(slice_value)
    )
