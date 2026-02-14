import shlex
import subprocess
import sys
from pathlib import Path

import unused

python_executable_path = Path(sys.executable)
package_directory_path = Path(unused.__file__).parent


def test_run() -> None:
    completed_process = subprocess.run(
        [
            shlex.quote(python_executable_path.as_posix()),
            '-m',
            unused.__name__,
            unused.__name__,
        ],
        capture_output=True,
        cwd=package_directory_path.parent.as_posix(),
        text=True,
    )

    assert completed_process.returncode == 0
    assert completed_process.stderr == ''
    assert len(completed_process.stdout.strip()) > 0
    assert [
        module_file_path
        for module_file_path_string in completed_process.stdout.splitlines()
        if not (
            (module_file_path := Path(module_file_path_string)).is_absolute()
            and module_file_path.is_file()
            and module_file_path.is_relative_to(package_directory_path)
        )
    ] == []
