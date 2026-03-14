from __future__ import annotations

import enum
from typing import Final


class Missing(enum.Enum):
    _VALUE = object()


MISSING: Final = Missing._VALUE  # noqa: SLF001
