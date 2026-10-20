"""Which SFPU sigmoid the pair gates compute: metal's accurate one, or its fast lookup-table one.

The accurate sigmoid is `1 / (1 + exp(-x))` with an exp and a Newton reciprocal, ~1000 cycles a
tile on Blackhole; the fast one is metal's SFPLUT piecewise-linear fit, about the cost of a relu
(bc4-bet-fusedtri: 0.0944 vs 0.0223 ms per 96 tiles a core at 224 tokens). `SIGMOID_FAST` switches
every in-kernel gate sigmoid on BindCraft 2's gradient path (the gated channel move and its VJP, the
gate VJP, triangle attention's gated pack, the trimul gin / T2 kernels) and the standalone ones.
Off by default: it is a precision change and is armed only where designs grade it.
"""

from __future__ import annotations

from .envflags import env_flag

SIGMOID_FAST = env_flag("TT_BIO_SIGMOID_FAST", False)


def defines() -> list[tuple[str, str]]:
    """The compute-kernel define that selects the fast sigmoid (kernels/common/gate_sigmoid.hpp).
    Read per build, so every program cache that bakes it must carry `SIGMOID_FAST` in its key."""
    return [("TT_BIO_SIG_FAST", "1")] if SIGMOID_FAST else []
