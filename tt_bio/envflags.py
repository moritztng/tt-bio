"""One reader for the ``TT_BIO_*`` gates.

The gates are set by hand on the command line, by the release gate and by fleet
scripts, so they have to answer the same way everywhere. Before this module they
did not: twelve sites read ``!= "0"``, eight read ``== "1"``, two read
``not in ("0", "")`` and one read ``not in ("1", "true", "True")`` inverted. So
``TT_BIO_SDPA_DIV_K=true`` turned that gate on while ``TT_BIO_TRIATT_MASK_Q_SPLIT=true``
turned that one off, and an empty value meant on for some gates and off for others.

Stdlib only, no ttnn: the CLI imports this at module scope.
"""

from __future__ import annotations

import os

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean gate from the environment. Unset or empty means the default.

    An unrecognised value raises rather than guessing: ``TT_BIO_X=tru`` silently
    taking the opposite branch through a whole release gate is worse than a crash
    on the first line.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().casefold()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean; use one of "
        f"{sorted(_TRUE)} / {sorted(_FALSE)}"
    )


def env_int(name: str, default: int) -> int:
    """Read an integer knob from the environment. Unset or empty means the default.

    Same contract as ``env_flag``: an unparseable value raises rather than falling back to the
    default, because a silently-ignored ``TT_BIO_X=1o24`` reads as "the lever did nothing".
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip(), 10)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
