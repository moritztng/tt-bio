"""Minimal stand-in for ``foundry/__init__.py``.

Upstream this module reads a ``.env`` file, optionally installs beartype and
jaxtyping import hooks, and probes for cuEquivariance on CUDA. None of that
applies to tt-bio: there is no CUDA path, and the import hooks would pull two
dependencies in for nothing. Only the two module-level flags are actually read by
the vendored code, so those are all that is kept.
"""

import logging

logger = logging.getLogger("tt_bio._vendor.foundry")

SHOULD_USE_CUEQUIVARIANCE = False
DISABLE_CHECKPOINTING = False

# Upstream reads these from a .env file. NAN_CHECK defaults to True there, and the
# vendored foundry.utils.torch swaps assert_no_nans for a no-op when it is False.
should_check_nans = True
should_debug = False
should_typecheck = False

__all__ = [
    "SHOULD_USE_CUEQUIVARIANCE",
    "DISABLE_CHECKPOINTING",
    "logger",
    "should_check_nans",
    "should_debug",
    "should_typecheck",
]
