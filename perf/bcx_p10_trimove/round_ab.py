#!/usr/bin/env python3
"""bcx-p10-trimove: the taped channel move A/B'd at the ROUND, on top of the hifi arm.

Leg 4. `perf/bcx_round/run_round.py` unchanged, arm set at the round boundary inside ONE
process on ONE card so a drift in the box lands on both arms equally. The structure is
`bcx-p10-tapegen`'s `round_ab.py`; what differs is the lever.

Both arms are the composed `hifi` round the brief names `(1,1,hifi)`: `tri_att_sdpa_hifi`
taped and `TT_BIO_TRIATT_DIVIDING_K=1`. The ONLY difference between them is this row's flag.

    off   `reblock_permute.set_taped_channel_move(False)` -- `eligible`/`eligible_back`
          refuse the tape, so `_channel_move` runs `ttnn.permute` and `_channel_move_back`
          runs its two transposes, exactly as the shipped hifi arm does
    on    `set_taped_channel_move(True)` -- the tape itself is allowed the kernel. Every
          direct (untaped) caller is refused as before

Why this exists when `bcx-p10-tapegen` already measured a round: tapegen reached the same
two kernels by a DIFFERENT mechanism (`ops.fused_kernel` tape entries via
`TT_BIO_TAPED_KERNELS`) and measured `hifi` 8.271 -> `both` 8.339 s. This row's route is
`taped_ok` on the eligibility gate and it also converts the OUTPUT move, which tapegen's
did not. A per-op census extrapolated it to +0.213 s a round. That extrapolation is what
this harness exists to confirm or refuse at the round.

REACH IS STAMPED per round. `reblock_permute.STATS` is [served, declined] for the forward
channel move and `STATS_BACK` the same for the output move. An A/B whose arms agree is only
a null if the `on` arm's served count actually moved.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                         # noqa: E402
import run_round as R                     # noqa: E402
from tt_bio import tenstorrent as tn      # noqa: E402
from tt_bio import reblock_permute as rp  # noqa: E402

ARMS = tuple(a for a in os.environ.get("TRIMOVE_AB_ARMS", "off,on").split(",") if a)


def _reach() -> dict:
    """Every counter that says which route the round actually took."""
    return {"reblock": list(rp.STATS), "reblock_back": list(rp.STATS_BACK),
            "reblock_gated": list(rp.STATS_GATED),
            "taped_move_flag": bool(rp.TAPED_MOVE),
            "fused": dict(tn.TRIATT_FUSED_HIFI_STATS),
            "fp32_softmax_calls": tn.FP32_SOFTMAX_STATS["calls"]}


def _arm(name: str) -> None:
    # Both arms are the hifi round; only the channel-move flag moves.
    os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi"
    tn._TRIATT_FUSED_HIFI = True
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1"
    rp.set_taped_channel_move(name == "on")


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        arm = self.arms[self.entries % len(self.arms)]
        _arm(arm)
        before = _reach()
        # The PREVIOUS round now knows what it consumed. Written before the parent appends a
        # new round_start, so the last round_start in the list is still the previous one.
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
        stamp["arm_order"] = f"round 1 {ARMS[0]}, rotating at the round boundary"
        stamp["lever"] = "TT_BIO_TAPED_CHANNEL_MOVE / reblock_permute.set_taped_channel_move"
        stamp["reach_end"] = _reach()
        stamp["reblock_rejects"] = {str(k): v for k, v in rp.REJECTS.items()}
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump

    R.main()


if __name__ == "__main__":
    main()
