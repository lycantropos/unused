from __future__ import annotations

import enum
from typing import Final, Literal


class Missing(enum.Enum):
    _VALUE = enum.auto()


MISSING: Final[Literal[Missing._VALUE]] = Missing._VALUE  # noqa: SLF001
