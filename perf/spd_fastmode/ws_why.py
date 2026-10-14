"""Why `transition_ws` declines: one xmodel_ab fold with Transition._weight_stationary wrapped to tally, per call,
which of its gates refused (lever, L1 destination, dtype, Wormhole width, eligible, DRAM refusal) and whether
the kernel ran.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ws_why.py --why-out why.json <xmodel_ab.py args>

The p150a Boltz-2 --fast census (xdif, 059c9a698) shows the pair transition on the row-blocked swiglu
(8712 linears for 264 calls) and Boltz-2's Transition phase at 2.41 s on p150a vs 1.41 s on a Galaxy chip.
"""
import collections, json, sys
from pathlib import Path

why_out = Path(sys.argv[sys.argv.index("--why-out") + 1])
del sys.argv[sys.argv.index("--why-out"):sys.argv.index("--why-out") + 2]
tree = Path(sys.argv[sys.argv.index("--tree") + 1])
sys.path.insert(0, str(tree))
sys.path.insert(0, str(tree / "perf" / "of3_4xpd"))

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

tally = collections.Counter()
orig = T.Transition._weight_stationary


def gates(self, x, memory_config):
    if not T.lever("transition_ws"):
        return "lever transition_ws off"
    if len(x.shape) != 4:
        return "not 4-D"
    if x.dtype != ttnn.bfloat16:
        return f"x dtype {x.dtype}"
    if T._dest_guard.tail_exposed(self.compute_kernel_config):
        return "dest_guard tail exposed"
    if T.ops.taping():
        return "taping"
    if not (memory_config is None or memory_config.buffer_type == ttnn.BufferType.DRAM or T.lever("ws_l1dest")):
        return "L1 destination (ws_l1dest off)"
    if (self.dtype or T._dtype()) != ttnn.bfloat16:
        return f"module dtype {self.dtype or T._dtype()}"
    if not (self._hidden_b8 or x.shape[-1] <= 128 or T.lever("ws_l1acc") or not T.is_wormhole()):
        return "Wormhole c > 128 without ws_l1acc"
    if tuple(x.padded_shape) in T.WS_DRAM_REFUSED:
        return "DRAM refused earlier"
    return None


def wrapped(self, x, memory_config, *a, **k):
    shape = "x".join(str(int(d)) for d in x.shape)
    reason = gates(self, x, memory_config)
    out = orig(self, x, memory_config, *a, **k)
    if reason is None:
        reason = "ran" if out is not None else "declined in ws_swiglu (eligible / L1 budget / alloc refusal)"
    tally[f"{shape} c={int(x.shape[-1])} hidden_b8={self._hidden_b8} | {reason}"] += 1
    return out


T.Transition._weight_stationary = wrapped
import xmodel_ab  # noqa: E402
try:
    xmodel_ab.main()
finally:
    why_out.parent.mkdir(parents=True, exist_ok=True)
    why_out.write_text(json.dumps(dict(tally.most_common()), indent=1) + "\n")
    print(json.dumps(dict(tally.most_common()), indent=1))
