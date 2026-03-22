from __future__ import annotations

import enum
from typing import Final


class Missing(enum.Enum):
    _VALUE = object()

    def __repr__(self, /) -> str:
        result = 'MISSING'
        assert globals()[result] is self
        return result


MISSING: Final = Missing._VALUE  # noqa: SLF001
