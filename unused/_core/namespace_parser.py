from __future__ import annotations

import ast
import builtins
import contextlib
import enum
import functools
import pkgutil
import sys
import tempfile
import types
from collections.abc import Iterable, Mapping, Sequence
from importlib.machinery import (
    EXTENSION_SUFFIXES,
    ExtensionFileLoader,
    SOURCE_SUFFIXES,
    SourceFileLoader,
)
from itertools import chain
from pathlib import Path
from typing import Any, Final, TypeAlias

from typing_extensions import Self, override

from .attribute_mapping import AttributeMapping
from .evaluation import EVALUATION_EXCEPTIONS, evaluate_node
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
from .resolution import lookup_node, resolve_node

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


class ResolvedAssignmentTargetSplitPath:
    @property
    def absolute(self, /) -> LocalObjectPath:
        return self._absolute

    @property
    def relative(self, /) -> LocalObjectPath:
        return self._relative

    def join(self, /, *components: str) -> Self:
        return type(self)(self._absolute, self._relative.join(*components))

    def unsplit(self, /) -> LocalObjectPath:
        return self._absolute.join(*self._relative.components)

    _absolute: LocalObjectPath
    _relative: LocalObjectPath

    __slots__ = '_absolute', '_relative'

    def __new__(
        cls, absolute: LocalObjectPath, relative: LocalObjectPath
    ) -> Self:
        assert isinstance(absolute, LocalObjectPath), absolute
        assert isinstance(relative, LocalObjectPath), relative
        self = super().__new__(cls)
        self._absolute, self._relative = absolute, relative
        return self

    def __repr__(self, /) -> str:
        return (
            f'{type(self).__qualname__}'
            f'({self._absolute!r}, {self._relative!r})'
        )


_ResolvedAssignmentTarget: TypeAlias = (
    Sequence['_ResolvedAssignmentTarget']
    | ResolvedAssignmentTargetSplitPath
    | None
)


def _combine_resolved_assignment_target_with_value(
    target: _ResolvedAssignmentTarget, value: Any, /
) -> Iterable[tuple[ResolvedAssignmentTargetSplitPath | None, Any]]:
    if target is None or isinstance(target, ResolvedAssignmentTargetSplitPath):
        yield target, value
        return
    if isinstance(value, AttributeMapping):
        # e.g.: a case of `enum.Enum` class unpacking
        return
    yield from chain.from_iterable(
        map(_combine_resolved_assignment_target_with_value, target, value)
    )


class NamespaceParser(ast.NodeVisitor):
    def __init__(
        self,
        namespace: Namespace,
        /,
        *parent_namespaces: Namespace,
        module_file_paths: Mapping[ModulePath, Path | None],
        module_paths: Sequence[ModulePath],
    ) -> None:
        super().__init__()
        self._module_file_paths, self._namespace, self._parent_namespaces = (
            module_file_paths,
            namespace,
            parent_namespaces,
        )
        self._module_paths = module_paths

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
            self._resolve_node(
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
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        function_name = node.name
        self._namespace.set_namespace_by_name(
            function_name,
            Namespace(
                ObjectKind.ROUTINE,
                self._namespace.module_path,
                self._namespace.local_path.join(function_name),
            ),
        )

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
        callable_namespace = self._lookup_node(node.func)
        if callable_namespace is None:
            return
        if callable_namespace.module_path == BUILTINS_MODULE_PATH and (
            callable_namespace.local_path
            == GLOBALS_LOCAL_OBJECT_PATH.join('update')
        ):
            module_namespace = self._get_module_namespace()
            module_namespace.append_sub_namespace(
                Namespace(
                    ObjectKind.UNKNOWN,
                    module_namespace.module_path,
                    LocalObjectPath(),
                )
            )
        if callable_namespace.module_path == ENUM_MODULE_PATH and (
            callable_namespace.local_path
            == ENUM_META_CONVERT_LOCAL_OBJECT_PATH
        ):
            enum_name_node, enum_module_name_node, *_ = node.args
            enum_name = self._evaluate_node(enum_name_node)
            assert isinstance(enum_name, str), ast.unparse(node)
            enum_module_namespace = self._lookup_node(enum_module_name_node)
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

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_name = node.name
        class_namespace = Namespace(
            ObjectKind.CLASS,
            self._namespace.module_path,
            self._namespace.local_path.join(class_name),
        )
        class_parser = NamespaceParser(
            class_namespace,
            *self._get_inherited_namespaces(),
            module_file_paths=self._module_file_paths,
            module_paths=self._module_paths,
        )
        for body_node in node.body:
            class_parser.visit(body_node)
        for decorator_node in node.decorator_list:
            decorator_namespace = self._lookup_node(decorator_node)
            if decorator_namespace is None:
                continue
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
                base_enum_namespace = self._lookup_node(base_enum_node)
                assert base_enum_namespace is not None
                class_namespace.append_sub_namespace(base_enum_namespace)
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                metaclass_namespace = self._lookup_node(keyword.value)
                if metaclass_namespace is None:
                    continue
                class_namespace.append_sub_namespace(metaclass_namespace)
        for index, base_node in reversed([*enumerate(node.bases)]):
            class_parser.visit(base_node)
            base_namespace = self._resolve_node(
                base_node,
                local_path=class_namespace.local_path.join(
                    # FIXME: add support for indexing?
                    f'__bases___{index}'
                ),
                module_path=class_namespace.module_path,
            )
            if base_namespace is None:
                continue
            class_namespace.append_sub_namespace(base_namespace)
        if len(node.bases) == 0:
            class_namespace.append_sub_namespace(
                BUILTINS_MODULE_NAMESPACE.get_namespace_by_path(
                    LocalObjectPath.from_object_name(object.__qualname__)
                )
            )
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
        for target_local_path in _flatten_resolved_assignment_target(
            resolved_target
        ):
            self._resolve_absolute_local_path(
                target_local_path.absolute
            ).set_namespace_by_path(
                target_local_path.relative,
                Namespace(
                    ObjectKind.UNKNOWN,
                    self._namespace.module_path,
                    target_local_path.unsplit(),
                ),
            )
        try:
            iterable = self._evaluate_node(node.iter)
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
                    if object_split_path is not None:
                        self._resolve_absolute_local_path(
                            object_split_path.absolute
                        ).set_object_by_path(object_split_path.relative, value)
                for body_node in node.body:
                    self.visit(body_node)
            for else_node in node.orelse:
                self.visit(else_node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        function_name = node.name
        for decorator_node in node.decorator_list:
            decorator_namespace = self._lookup_node(decorator_node)
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
            if decorator_namespace.module_path == BUILTINS_MODULE_PATH and any(
                (
                    decorator_namespace.local_path
                    == PROPERTY_LOCAL_OBJECT_PATH.join(property_modifier_name)
                )
                for property_modifier_name in ['deleter', 'setter']
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
                    decorator_namespace.module_path,
                    decorator_namespace.local_path,
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
        if function_name == '__getattr__' and self._namespace.kind in (
            ObjectKind.CLASS,
            ObjectKind.MODULE,
        ):
            self._namespace.append_sub_namespace(
                Namespace(
                    ObjectKind.UNKNOWN,
                    self._namespace.module_path,
                    self._namespace.local_path,
                )
            )
        self._namespace.set_namespace_by_name(
            function_name, function_namespace
        )

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
            return
        for body_node in node.body if condition_satisfied else node.orelse:
            self.visit(body_node)

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
                module_path = ModulePath.from_module_name(alias.name)
                module_paths = list(self._module_paths)
                for submodule_path in module_path.submodule_paths():
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
                    module_paths.append(submodule_path)
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
                assert self._namespace.kind is ObjectKind.MODULE, (
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
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.generic_visit(node.args)

    @override
    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        value_namespace = self._lookup_node(node.value)
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
    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

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
                            exception_type_namespace := self._lookup_node(
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
                            exception_type_namespace := self._lookup_node(
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
                    callable_namespace = self._lookup_node(
                        item_expression_node.func
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
                                    exception_namespace := self._lookup_node(
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
                    if namespace.kind is ObjectKind.CLASS
                ]
            )
            == 0
        ), invalid_namespaces
        return result

    def _get_module_namespace(self, /) -> Namespace:
        *_, result, _ = self._namespace, *self._parent_namespaces
        assert result.kind is ObjectKind.MODULE, result
        return result

    def _lookup_node(self, node: ast.expr, /) -> Namespace | None:
        return lookup_node(node, self._namespace, *self._parent_namespaces)

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
                target_object_split_path.absolute
            )
            target_namespace.set_namespace_by_path(
                target_object_split_path.relative,
                (
                    self._resolve_node(
                        value_node,
                        local_path=target_object_split_path.unsplit(),
                        module_path=target_namespace.module_path,
                    )
                    if isinstance(
                        resolved_target, ResolvedAssignmentTargetSplitPath
                    )
                    else Namespace(
                        ObjectKind.UNKNOWN,
                        target_namespace.module_path,
                        target_object_split_path.unsplit(),
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
                    maybe_target_object_split_path.absolute
                ).set_object_by_path(
                    maybe_target_object_split_path.relative, sub_value
                )

    def _resolve_absolute_local_path(
        self, path: LocalObjectPath, /
    ) -> Namespace:
        if path == self._namespace.local_path:
            return self._namespace
        return self._get_module_namespace().get_namespace_by_path(path)

    @functools.singledispatchmethod
    def _resolve_assignment_target(
        self, _node: ast.expr, /
    ) -> _ResolvedAssignmentTarget:
        return None

    @_resolve_assignment_target.register(ast.Attribute)
    def _(self, node: ast.Attribute, /) -> _ResolvedAssignmentTarget:
        if (
            object_path := self._resolve_assignment_target(node.value)
        ) is not None:
            assert isinstance(object_path, ResolvedAssignmentTargetSplitPath)
            return object_path.join(node.attr)
        return None

    @_resolve_assignment_target.register(ast.List)
    @_resolve_assignment_target.register(ast.Tuple)
    def _(self, node: ast.List | ast.Tuple, /) -> _ResolvedAssignmentTarget:
        return [
            self._resolve_assignment_target(element_node)
            for element_node in node.elts
        ]

    @_resolve_assignment_target.register(ast.Name)
    def _(self, node: ast.Name, /) -> _ResolvedAssignmentTarget:
        return ResolvedAssignmentTargetSplitPath(
            self._namespace.local_path, LocalObjectPath(node.id)
        )

    @_resolve_assignment_target.register(ast.NamedExpr)
    def _(self, node: ast.NamedExpr, /) -> _ResolvedAssignmentTarget:
        return self._resolve_assignment_target(node.value)

    @_resolve_assignment_target.register(ast.Subscript)
    def _(self, node: ast.Subscript, /) -> _ResolvedAssignmentTarget:
        value_namespace = self._lookup_node(node.value)
        if value_namespace is None:
            return None
        if not (
            value_namespace.kind is ObjectKind.ROUTINE_CALL
            and value_namespace.module_path == BUILTINS_MODULE_PATH
            and value_namespace.local_path == GLOBALS_LOCAL_OBJECT_PATH
        ):
            return None
        try:
            slice_value = self._evaluate_node(node.slice)
        except EVALUATION_EXCEPTIONS:
            return None
        assert isinstance(slice_value, str), ast.unparse(node)
        return ResolvedAssignmentTargetSplitPath(
            LocalObjectPath(), LocalObjectPath(slice_value)
        )

    def _resolve_module_path(self, module_path: ModulePath, /) -> Namespace:
        return resolve_module_path(
            module_path,
            module_file_paths=self._module_file_paths,
            module_paths=self._module_paths,
        )

    def _resolve_node(
        self,
        node: ast.expr,
        /,
        *,
        local_path: LocalObjectPath,
        module_path: ModulePath,
    ) -> Namespace:
        return resolve_node(
            node,
            self._namespace,
            *self._parent_namespaces,
            local_path=local_path,
            module_path=module_path,
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
    module_file_paths: Mapping[ModulePath, Path | None],
    module_paths: Sequence[ModulePath],
) -> Namespace:
    root_component, *rest_components = module_path.components
    root_module_path = ModulePath(root_component)
    result = _load_module_path_namespace(
        root_module_path,
        module_file_paths=module_file_paths,
        module_paths=module_paths,
    )
    for component in rest_components:
        try:
            result = result.get_namespace_by_name(component)
        except KeyError:
            submodule_path = result.module_path.join(component)
            result = _load_module_path_namespace(
                submodule_path,
                module_file_paths=module_file_paths,
                module_paths=module_paths,
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
    target: _ResolvedAssignmentTarget, /
) -> Iterable[ResolvedAssignmentTargetSplitPath]:
    if target is None:
        return
    queue: list[_ResolvedAssignmentTarget] = (
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


BUILTINS_MODULE_NAMESPACE.set_object_by_path(
    LocalObjectPath.from_object_name(builtins.getattr.__qualname__),
    builtins.getattr,
)
BUILTINS_MODULE_NAMESPACE.set_object_by_path(
    LocalObjectPath.from_object_name(builtins.hasattr.__qualname__),
    builtins.hasattr,
)
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
    module_file_paths: Mapping[ModulePath, Path | None],
    module_paths: Sequence[ModulePath],
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
            module_file_paths=module_file_paths,
            module_paths=(*module_paths, module_path),
        )
        try:
            namespace_parser.visit(module_node)
        except Exception:
            del MODULE_NAMESPACES[module_path]
            raise
    return result
