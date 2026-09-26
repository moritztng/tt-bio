#!/usr/bin/env python3
"""bcx-p10-rne leg 4: the residual's folded cast A/B'd at the ROUND, reach stamped.

`perf/bcx_round/run_round.py` unchanged, with the two things around it that
`perf/bcx_p10_triatt/round_ab.py` and `perf/bcx_p10_bfp8/round_ab.py` established:

1. The arm alternates at the round boundary inside ONE process on ONE card. The lever is a
   class attribute read at every `_residual` call, so the flip costs no rebuild and a drift in
   the box lands on both arms equally. That matters more here than on any lever this campaign
   has measured: the fold is ~0.350 s of a 9.5 s device column, about 1.032x, and a between-
   process A/B cannot see 3 % through this box's load.
2. REACH IS STAMPED per round. A bit-identical arm reads exactly like a lever that never ran,
   so `RESIDUAL_STATS` counts every `_residual` call and splits it by the branch it took. If
   `fold` does not move, the seconds mean nothing.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-rne \
    RNE_AB_ARMS=off,fold perf/bcx_p10_rne/arm.sh rne_ab 18
"""
from __future__ import annotations

import collections
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                     # noqa: E402
import run_round as R                 # noqa: E402
from tt_bio.af2 import AF2PairBlock   # noqa: E402

ARMS = tuple(a for a in os.environ.get("RNE_AB_ARMS", "off,fold").split(",") if a)

RESIDUAL_STATS = collections.Counter()
_shipped_residual = AF2PairBlock._residual


def _counted_residual(self, x, update):
    """One extra Python frame per residual, on both arms, and it is the only way an arm that
    computes the same numbers can be told from an arm that never ran."""
    if update is not None:
        RESIDUAL_STATS["calls"] += 1
        RESIDUAL_STATS["fold" if self.rne_fold_cast else "shipped"] += 1
    return _shipped_residual(self, x, update)


AF2PairBlock._residual = _counted_residual


def _reach() -> dict:
    return dict(RESIDUAL_STATS)


def _arm(name: str) -> None:
    AF2PairBlock.rne_fold_cast = (name == "fold")


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        arm = self.arms[self.entries % len(self.arms)]
        _arm(arm)
        before = _reach()
        # The previous round now knows what it consumed; written before the parent appends a
        # new round_start, so the last one in the list is still the previous round's.
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
        stamp["rne_fold_cast_flag_end"] = bool(AF2PairBlock.rne_fold_cast)
        stamp["rne_residual_flag"] = bool(AF2PairBlock.rne_residual)
        stamp["rne_wide_dram_flag"] = bool(AF2PairBlock.rne_wide_dram)
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump
    R.main()


if __name__ == "__main__":
    main()
