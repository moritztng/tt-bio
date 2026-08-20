"""Minimal stand-in for ``foundry.utils.ddp``.

Only :class:`RankedLogger` is reachable from the featurization path, and upstream
pulls lightning, lightning_utilities and omegaconf in for it plus a set of
training-time accelerator helpers. tt-bio runs the host featurizer in a single
process, so rank is always zero and the adapter is a passthrough.
"""

import logging
from typing import Any


def get_current_rank() -> int:
    """Rank of the current process. The vendored featurizer is single-process."""
    return 0


def is_rank_zero() -> bool:
    return True


class RankedLogger(logging.LoggerAdapter):
    """Passthrough logger adapter matching the upstream constructor signature."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(logging.getLogger(name), extra or {})
        self.rank_zero_only = rank_zero_only

    def process(self, msg, kwargs):
        return msg, kwargs


__all__ = ["RankedLogger", "get_current_rank", "is_rank_zero"]
