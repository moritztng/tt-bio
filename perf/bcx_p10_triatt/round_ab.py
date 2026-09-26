#!/usr/bin/env python3
"""bcx-p10-triatt: the fused triangle-attention SDPA A/B'd at the ROUND, reach stamped.

`perf/bcx_round/run_round.py` unchanged, with two things around it, the same shape
`bcx-p10-bfp8`'s round_ab used:

1. The arm alternates at the round boundary inside ONE process on ONE card, so a drift in
   the box lands on both arms equally. `_TRIATT_FUSED_HIFI` is the process-wide flag every
   `AF2PairBlock` reads PER CALL (`AF2PairBlock.fused_hifi` is None on this path), so the
   flip needs no rebuild of 54 blocks and reaches a trunk that loads lazily later.
2. REACH IS STAMPED, per round and in the final artifact. `TRIATT_FUSED_HIFI_STATS` splits
   into served / declined / too_short / taped, and `FP32_SOFTMAX_STATS["calls"]` counts the
   materialised path. An A/B whose two arms agree is only a null result if the fused arm
   shows serves; if `taped` moves instead, the lever never reached and the seconds mean
   nothing. That is the trap `bcx-p10-bfp8` lost a leg to.

Nothing here changes what the campaign computes: the arm picks which kernel serves a
softmax, and `--rounds` only stops collection after N full rounds.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                     # noqa: E402
import run_round as R                 # noqa: E402
from tt_bio import tenstorrent as tn  # noqa: E402

ARMS = ("mat", "fused")


def _reach() -> dict:
    """Every counter that says which route the triangle attentions actually took."""
    return {"fused": dict(tn.TRIATT_FUSED_HIFI_STATS),
            "fp32_softmax_calls": tn.FP32_SOFTMAX_STATS["calls"],
            "picks": {str(k): v for k, v in tn.TRIATT_FUSED_HIFI_PICKS.items()}}


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        # The parent stamps round_start (or raises the stop) and is the only place that
        # knows the round number, so set the arm first and tag the event it appends.
        arm = self.arms[self.entries % len(self.arms)]
        tn._TRIATT_FUSED_HIFI = (arm == "fused")
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
        stamp["arm_order"] = "round 1 mat, alternating"
        stamp["reach_end"] = _reach()
        stamp["TRIATT_FUSED_HIFI_flag_end"] = bool(tn._TRIATT_FUSED_HIFI)
        stamp["TRIATT_FUSED_HIFI_MIN_S"] = tn._TRIATT_FUSED_HIFI_MIN_S
        stamp["triatt_sdpa_rejects"] = {str(k): v for k, v in tn._triatt_sdpa.REJECTS.items()}
        stamp["triatt_sdpa_stats"] = list(tn._triatt_sdpa.STATS)
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump

    R.main()


if __name__ == "__main__":
    main()
