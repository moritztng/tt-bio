#!/usr/bin/env python3
"""bcx-p10-bfp8: the bfp8 arm A/B'd at the ROUND, on the shipped `tt_bio.bindcraft2` path.

`perf/bcx_round/run_round.py` unchanged, with two things added around it:

1. The arm alternates at the round boundary, inside ONE process on ONE card. `Meter` is where
   a round starts, so the flip goes there and the round_start event carries the arm it ran.
   The order rotates: round 1 is bf16, round 2 is b8, and so on, so a drift in the box lands on
   both arms equally.
2. `_triatt_dtype` is counted by what it returned. An arm that never returns bfloat8_b is
   inert at this token axis and its timing means nothing, which is the trap the n=128 grade
   fell into. The count is in the stamp, so the artifact says on its face whether the lever
   reached.

Nothing here changes what the campaign computes: the arm is a storage format inside triangle
attention, and `--rounds` only stops collection after N full rounds.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                     # noqa: E402
import run_round as R                 # noqa: E402
from tt_bio import autograd as ag     # noqa: E402
from tt_bio import tenstorrent as tn  # noqa: E402

ARMS = ("bf16", "b8")
DTYPE_CALLS: collections.Counter = collections.Counter()


def _count_triatt_dtype():
    real = tn._triatt_dtype

    def counted():
        d = real()
        DTYPE_CALLS[str(d).rsplit(".", 1)[-1]] += 1
        return d
    tn._triatt_dtype = counted


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        # The parent stamps round_start (or raises the stop) and is the only place that knows
        # the round number, so set the arm first and tag the event the parent just appended.
        arm = self.arms[self.entries % len(self.arms)]
        tn._TRIATT_B8 = (arm == "b8")
        before = dict(DTYPE_CALLS)
        super().on_sequence_gradients_enter()
        if M.EVENTS and M.EVENTS[-1].get("kind") == "round_start":
            M.EVENTS[-1]["arm"] = arm
            M.EVENTS[-1]["triatt_dtype_calls_before"] = before


def main():
    _count_triatt_dtype()
    M.Meter = ArmMeter
    R.M.Meter = ArmMeter

    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = list(ARMS)
        stamp["arm_order"] = "round 1 bf16, alternating"
        stamp["triatt_dtype_calls"] = dict(DTYPE_CALLS)
        stamp["DEVICE_ZEROS"] = bool(ag.DEVICE_ZEROS)
        stamp["TRIATT_B8_flag_end"] = bool(tn._TRIATT_B8)
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump

    R.main()


if __name__ == "__main__":
    main()
