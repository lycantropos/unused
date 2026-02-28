import argparse
import shlex
import subprocess
import sys
import traceback
from abc import ABC, abstractmethod
from functools import reduce
from importlib.machinery import EXTENSION_SUFFIXES, SOURCE_SUFFIXES
from itertools import chain
from pathlib import Path
from typing import ClassVar, Final

from typing_extensions import Self, override

from unused._core.valuespace import BaseValuespace

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

import unused


class Parameter(ABC):
    @property
    @abstractmethod
    def attribute_name(self, /) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self, /) -> str:
        raise NotImplementedError


class Argument(Parameter):
    @property
    @override
    def attribute_name(self, /) -> str:
        return self._value

    @property
    @override
    def name(self, /) -> str:
        return self._value

    _value: str
    __slots__ = ('_value',)

    def __new__(cls, value: str, /) -> Self:
        assert isinstance(value, str), value
        assert value.isidentifier(), value
        self = super().__new__(cls)
        self._value = value
        return self


class Option(Parameter):
    @property
    @override
    def attribute_name(self, /) -> str:
        return self._value

    @property
    @override
    def name(self, /) -> str:
        return '--' + self._value.replace('_', '-')

    _value: str
    __slots__ = ('_value',)

    def __new__(cls, value: str, /) -> Self:
        assert isinstance(value, str), value
        assert value.isidentifier(), value
        self = super().__new__(cls)
        self._value = value
        return self


class ArgumentValuespace(BaseValuespace[Argument]):
    @classmethod
    @override
    def value_cls(cls, /) -> type[Argument]:
        return Argument

    MODULE_PATHS: ClassVar = Argument('module_paths')


class OptionValuespace(BaseValuespace[Option]):
    @classmethod
    @override
    def value_cls(cls, /) -> type[Option]:
        return Option

    PYTHON_PATH: ClassVar = Option('python_path')
    ROOT_PATH: ClassVar = Option('root_path')
    VERSION: ClassVar = Option('version')


def main() -> None:
    parser = argparse.ArgumentParser(unused.__name__)
    parser.add_argument(
        OptionValuespace.VERSION.name,
        action='version',
        version=unused.__version__,
    )
    parser.add_argument(OptionValuespace.ROOT_PATH.name, default='.')
    parser.add_argument(
        OptionValuespace.PYTHON_PATH.name,
        default=None,
        help='path to `Python` executable.',
    )
    parser.add_argument(
        ArgumentValuespace.MODULE_PATHS.name,
        help='`Python` module directory/file path to process.',
        metavar='MODULE_PATH',
        nargs=argparse.ZERO_OR_MORE,
    )
    args = parser.parse_args()
    python_path_string = getattr(
        args, OptionValuespace.PYTHON_PATH.attribute_name
    )
    if python_path_string is not None:
        python_path = Path(python_path_string).resolve(strict=True)
        if python_path != Path(sys.executable).resolve(strict=True):
            raise SystemExit(
                subprocess.call(
                    [shlex.quote(python_path.as_posix()), *sys.orig_argv[1:]],
                    stderr=sys.stderr,
                    stdout=sys.stdout,
                )
            )
    root_path = Path(
        getattr(args, OptionValuespace.ROOT_PATH.attribute_name)
    ).resolve(strict=True)
    path_strings = getattr(
        args, ArgumentValuespace.MODULE_PATHS.attribute_name
    )
    if len(path_strings) == 0:
        paths = [root_path]
    else:
        unchecked_paths = dict.fromkeys(
            (
                path
                if (path := Path(path_string)).is_absolute()
                else root_path.joinpath(path)
            )
            for path_string in path_strings
        ).keys()
        if (
            len(
                path_validation_errors := [
                    validation_error
                    for path in unchecked_paths
                    if (
                        (
                            validation_error
                            := _to_module_file_path_validation_error(path)
                        )
                        is not None
                    )
                ]
            )
            > 0
        ):
            if len(path_validation_errors) == 1:
                raise path_validation_errors[0]
            raise ExceptionGroup(
                (
                    f'{len(path_validation_errors)} '
                    f'out of {len(unchecked_paths)} are invalid.'
                ),
                path_validation_errors,
            )
        paths = [path.resolve(strict=True) for path in unchecked_paths]

    import ast

    from unused._core.object_path import LocalObjectPath, ModulePath
    from unused._core.scope_parser import (
        load_module_file_paths,
        resolve_module_path,
    )

    module_file_paths = load_module_file_paths(root_path)
    function_definition_nodes: dict[
        tuple[ModulePath, LocalObjectPath],
        ast.AsyncFunctionDef | ast.FunctionDef,
    ] = {}

    stderr, stdout = sys.stderr, sys.stdout
    for module_file_path in chain.from_iterable(
        (
            chain(*[path.rglob('*' + suffix) for suffix in _MODULE_SUFFIXES])
            if path.is_dir()
            else [path]
        )
        for path in paths
    ):
        try:
            module_path = ModulePath(
                *module_file_path.relative_to(root_path).parent.parts,
                *(
                    (module_name,)
                    if (
                        (
                            module_name := reduce(
                                str.removesuffix,
                                _MODULE_SUFFIXES,
                                module_file_path.name,
                            )
                        )
                        != '__init__'
                    )
                    else ()
                ),
            )
        except ValueError as error:
            stderr.write(
                'Failed parsing module path of '
                f'{module_file_path.as_posix()!r}:\n'
            )
            stderr.writelines(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            )
            stderr.write('\n')
            stderr.flush()
            continue
        try:
            resolve_module_path(
                module_path,
                function_definition_nodes=function_definition_nodes,
                module_file_paths=module_file_paths,
            )
        except Exception as error:
            stderr.write(f'Failed loading {module_path.to_module_name()!r}:\n')
            stderr.writelines(
                traceback.format_exception(
                    type(error), error, error.__traceback__
                )
            )
            stderr.flush()
            continue
        stdout.write(str(module_file_path))
        stdout.write('\n')
        stdout.flush()


_MODULE_SUFFIXES: Final[tuple[str, ...]] = (
    *SOURCE_SUFFIXES,
    *EXTENSION_SUFFIXES,
)


def _to_module_file_path_validation_error(value: Path, /) -> Exception | None:
    try:
        value.resolve(strict=True)
    except OSError as error:
        return error
    if not (value.is_dir() or value.name.endswith(_MODULE_SUFFIXES)):
        return ValueError(
            'a directory or a single module '
            f'(with suffix {", ".join(map(repr, _MODULE_SUFFIXES))}) '
            'is expected'
        )
    return None


main()
