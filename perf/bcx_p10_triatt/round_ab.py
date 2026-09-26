#!/usr/bin/env python3
"""bcx-p10-triatt: the triangle-attention route A/B'd at the ROUND, reach stamped.

`perf/bcx_round/run_round.py` unchanged, with two things around it, the same shape
`bcx-p10-bfp8`'s round_ab used:

1. The arm alternates at the round boundary inside ONE process on ONE card, so a drift in
   the box lands on both arms equally. Both levers are live reads, so the flip needs no
   rebuild of 54 blocks and reaches a trunk that loads lazily later.
2. REACH IS STAMPED, per round and in the final artifact. `TRIATT_FUSED_HIFI_STATS` splits
   into served / declined / too_short / taped, `TRIATT_TAPED_SDPA_STATS` the same for the
   taped route, and `FP32_SOFTMAX_STATS["calls"]` counts the materialised path. An A/B
   whose two arms agree is only a null result if the moving arm shows serves; if `taped`
   moves instead, the lever never reached and the seconds mean nothing. That is the trap
   `bcx-p10-bfp8` lost a leg to, and it is what the `fused` arm here turned out to be.

The arms, set with `TRIATT_AB_ARMS` (comma separated, default `mat,fused`):

    mat     both levers off -- `_fp32_softmax_attention` on every call, what ships today
    fused   `TT_BIO_TRIATT_FUSED_HIFI` -- the persistent-mask kernel. MEASURED INERT under
            a tape: `triatt_sdpa.sdpa` declines on `ops.taping()` because `generic_op` has
            no tape entry, so a gradient round never reaches it
    taped   `TT_BIO_TRIATT_TAPED_SDPA` -- the stock fused SDPA verb, which IS taped, so
            `taped_ttnn._v_sdpa` puts `autograd.triangle_attention`'s chunked-recompute
            backward behind it instead of differentiating the materialised scores
    agtri   the same route with `TT_BIO_SDPA_OWN_FORWARD` on, so the forward is
            `autograd.triangle_attention`'s own chunked one -- one score block per chunk,
            each row reduced in a SINGLE pass under HiFi4 + fp32 destination accumulation --
            instead of the kernel's online bf16 softmax. Same backward as `taped`, so the
            pair prices the FORWARD's speed against its reduction order and nothing else.

Nothing here changes what the campaign computes: the arm picks which kernel serves a
softmax, and `--rounds` only stops collection after N full rounds.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                     # noqa: E402
import run_round as R                 # noqa: E402
from tt_bio import taped_ttnn as TT  # noqa: E402
from tt_bio import tenstorrent as tn  # noqa: E402

ARMS = tuple(a for a in os.environ.get("TRIATT_AB_ARMS", "mat,fused").split(",") if a)


def _reach() -> dict:
    """Every counter that says which route the triangle attentions actually took."""
    return {"fused": dict(tn.TRIATT_FUSED_HIFI_STATS),
            "taped": dict(tn.TRIATT_TAPED_SDPA_STATS),
            "vsdpa": dict(TT.SDPA_OWN_FORWARD_STATS),
            "fp32_softmax_calls": tn.FP32_SOFTMAX_STATS["calls"],
            "picks": {str(k): v for k, v in tn.TRIATT_FUSED_HIFI_PICKS.items()}}


def _arm(name: str) -> None:
    tn._TRIATT_FUSED_HIFI = (name == "fused")
    os.environ["TT_BIO_TRIATT_TAPED_SDPA"] = "1" if name in ("taped", "agtri") else "0"
    os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1" if name == "agtri" else "0"


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        # The parent stamps round_start (or raises the stop) and is the only place that
        # knows the round number, so set the arm first and tag the event it appends.
        arm = self.arms[self.entries % len(self.arms)]
        _arm(arm)
        before = _reach()
        # The PREVIOUS round now knows what it consumed. Written before the parent appends
        # a new round_start, so the last round_start in the list is still the previous one.
        for e in reversed(M.EVENTS):
            if e.get("kind") == "round_start":
                e["reach_after"] = before
                break
        super().on_sequence_gradients_enter()
        if M.EVENTS and M.EVENTS[-1].get("kind") == "round_start":
            M.EVENTS[-1]["arm"] = arm
            M.EVENTS[-1]["reach_before"] = before


def main():
    M.Meter = ArmMeter
    R.M.Meter = ArmMeter

    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = list(ARMS)
        stamp["arm_order"] = f"round 1 {ARMS[0]}, rotating"
        stamp["reach_end"] = _reach()
        stamp["TRIATT_FUSED_HIFI_flag_end"] = bool(tn._TRIATT_FUSED_HIFI)
        stamp["TRIATT_TAPED_SDPA_flag_end"] = tn._triatt_taped_sdpa_on()
        stamp["SDPA_OWN_FORWARD_flag_end"] = TT._sdpa_own_forward()
        stamp["TRIATT_FUSED_HIFI_MIN_S"] = tn._TRIATT_FUSED_HIFI_MIN_S
        stamp["triatt_sdpa_rejects"] = {str(k): v for k, v in tn._triatt_sdpa.REJECTS.items()}
        stamp["triatt_sdpa_stats"] = list(tn._triatt_sdpa.STATS)
        stamp["sdpa_picks"] = {str(k): v for k, v in getattr(tn, "SDPA_PICKS", {}).items()}
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump

    R.main()


if __name__ == "__main__":
    main()
