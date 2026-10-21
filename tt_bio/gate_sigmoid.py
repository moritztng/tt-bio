"""Which SFPU sigmoid the trimul gates compute: metal's accurate one, or its fast lookup-table one.

The accurate sigmoid is `1 / (1 + exp(-x))` with an exp and a Newton reciprocal, ~1000 cycles a
tile on Blackhole; the fast one is metal's SFPLUT piecewise-linear fit, about the cost of a relu
(bc4-bet-fusedtri: 0.0944 vs 0.0223 ms per 96 tiles a core at 224 tokens). `SIGMOID_FAST` switches
the triangle multiplication's gates, forward and backward together: the gated channel move and its
VJP (reblock_permute_gated, _bw), the gated in-projection and the output tails (trimul_tail's
SIGPOLY kernels). Gates written as a ttnn multiply with a fused sigmoid keep the accurate one in
both directions (gate_bw's `fast` is for probes), so no gate is differentiated as a different
function than its forward computed. Off by default: a precision change, armed only where designs
grade it.
"""

from __future__ import annotations

from .envflags import env_flag

SIGMOID_FAST = env_flag("TT_BIO_SIGMOID_FAST", False)


def defines() -> list[tuple[str, str]]:
    """The compute-kernel define that selects the fast sigmoid (kernels/common/gate_sigmoid.hpp).
    Read per build, so every program cache that bakes it must carry `SIGMOID_FAST` in its key."""
    return [("TT_BIO_SIG_FAST", "1")] if SIGMOID_FAST else []
