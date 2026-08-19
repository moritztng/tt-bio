"""``enum.StrEnum`` backport.

Added in Python 3.11; tt-bio supports 3.10, which is the deployed runtime. The
upstream atomworks package uses it for three enums.
"""

from enum import Enum


class StrEnum(str, Enum):
    """Minimal stand-in for :class:`enum.StrEnum`."""

    def __str__(self) -> str:
        return str(self.value)


__all__ = ["StrEnum"]
