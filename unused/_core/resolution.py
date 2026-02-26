from __future__ import annotations

import ast
import functools
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, TypeAlias

from typing_extensions import Self

from unused._core.context import Context, FunctionCallContext

from .attribute_mapping import AttributeMapping
from .enums import ObjectKind
from .evaluation import EVALUATION_EXCEPTIONS, evaluate_expression_node
from .lookup import lookup_object_by_expression_node, lookup_object_by_name
from .object_path import (
    DICT_FIELD_NAME,
    LocalObjectPath,
    ModulePath,
    SYS_MODULES_LOCAL_OBJECT_PATH,
    SYS_MODULE_PATH,
)
from .scope import Scope


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


def combine_resolved_assignment_target_with_value(
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
        map(combine_resolved_assignment_target_with_value, target, value)
    )


@functools.singledispatch
def resolve_assignment_target(
    _node: ast.expr,
    _scope: Scope,
    /,
    *_parent_scopes: Scope,
    context: Context,  # noqa: ARG001
    name_scopes: Mapping[str, Scope],  # noqa: ARG001
) -> ResolvedAssignmentTarget:
    return None


@resolve_assignment_target.register(ast.Attribute)
def _(
    node: ast.Attribute,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    name_scopes: Mapping[str, Scope],
) -> ResolvedAssignmentTarget:
    if (
        object_path := resolve_assignment_target(
            node.value,
            scope,
            *parent_scopes,
            context=context,
            name_scopes=name_scopes,
        )
    ) is not None:
        assert isinstance(object_path, ResolvedAssignmentTargetSplitPath)
        return object_path.join(node.attr)
    return None


@resolve_assignment_target.register(ast.List)
@resolve_assignment_target.register(ast.Tuple)
def _(
    node: ast.List | ast.Tuple,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    name_scopes: Mapping[str, Scope],
) -> ResolvedAssignmentTarget:
    return [
        resolve_assignment_target(
            element_node,
            scope,
            *parent_scopes,
            context=context,
            name_scopes=name_scopes,
        )
        for element_node in node.elts
    ]


@resolve_assignment_target.register(ast.Name)
def _(
    node: ast.Name,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,  # noqa: ARG001
    name_scopes: Mapping[str, Scope],
) -> ResolvedAssignmentTarget:
    object_name = node.id
    if isinstance(node.ctx, ast.Load):
        object_ = lookup_object_by_name(object_name, scope, *parent_scopes)
        if (
            object_.module_path == scope.module_path
            and object_.local_path.starts_with(scope.local_path)
        ):
            return ResolvedAssignmentTargetSplitPath(
                scope.module_path,
                scope.local_path,
                LocalObjectPath(
                    *object_.local_path.components[
                        len(scope.local_path.components) :
                    ]
                ),
            )
        return ResolvedAssignmentTargetSplitPath(
            object_.module_path, object_.local_path, LocalObjectPath()
        )
    name_scope = name_scopes.get(object_name, scope)
    return ResolvedAssignmentTargetSplitPath(
        name_scope.module_path,
        name_scope.local_path,
        LocalObjectPath(object_name),
    )


@resolve_assignment_target.register(ast.Subscript)
def _(
    node: ast.Subscript,
    scope: Scope,
    /,
    *parent_scopes: Scope,
    context: Context,
    name_scopes: Mapping[str, Scope],  # noqa: ARG001
) -> ResolvedAssignmentTarget:
    value_object = lookup_object_by_expression_node(
        node.value, scope, *parent_scopes, context=context
    )
    if value_object is None:
        return None
    if (
        value_object.kind is ObjectKind.INSTANCE
        and value_object.module_path == SYS_MODULE_PATH
        and value_object.local_path == SYS_MODULES_LOCAL_OBJECT_PATH
    ):
        try:
            module_name = evaluate_expression_node(
                node.slice, scope, *parent_scopes, context=context
            )
        except EVALUATION_EXCEPTIONS:
            assert not isinstance(context, FunctionCallContext)
        else:
            assert isinstance(module_name, str), module_name
            return ResolvedAssignmentTargetSplitPath(
                ModulePath.from_module_name(module_name),
                LocalObjectPath(),
                LocalObjectPath(),
            )
        return None
    if not (
        value_object.kind is ObjectKind.INSTANCE
        and value_object.module_path == scope.module_path
        and value_object.local_path == LocalObjectPath(DICT_FIELD_NAME)
    ):
        return None
    try:
        slice_value = evaluate_expression_node(
            node.slice, scope, *parent_scopes, context=context
        )
    except EVALUATION_EXCEPTIONS:
        return None
    assert isinstance(slice_value, str), ast.unparse(node)
    return ResolvedAssignmentTargetSplitPath(
        scope.module_path, LocalObjectPath(), LocalObjectPath(slice_value)
    )
