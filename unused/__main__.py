import argparse
import sys
import traceback
from enum import Enum, unique
from importlib.machinery import EXTENSION_SUFFIXES, SOURCE_SUFFIXES
from itertools import chain
from pathlib import Path
from typing import Final

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

import unused


@unique
class ArgumentName(str, Enum):
    ROOT_PATH = 'root_path'
    PATHS = 'paths'


def main() -> None:
    parser = argparse.ArgumentParser(unused.__name__)
    parser.add_argument('--' + ArgumentName.ROOT_PATH, default='.')
    parser.add_argument(
        ArgumentName.PATHS, metavar='PATH', nargs=argparse.ZERO_OR_MORE
    )
    args = parser.parse_args()
    root_path = Path(getattr(args, ArgumentName.ROOT_PATH)).resolve(
        strict=True
    )
    path_strings = getattr(args, ArgumentName.PATHS)
    if len(path_strings) == 0:
        paths = [root_path]
    else:
        unchecked_paths = dict.fromkeys(
            Path(path_string) for path_string in path_strings
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

    from ._core.namespace_parser import load_module_path_namespace
    from ._core.object_path import ModulePath

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
                    ()
                    if module_file_path.stem == '__init__'
                    else (module_file_path.stem,)
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
            load_module_path_namespace(module_path, module_paths=())
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
