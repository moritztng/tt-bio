#!/usr/bin/env python3
"""bcx-p10-trimul: the dX route A/B'd at the ROUND, on the shipped `tt_bio.bindcraft2` path.

`perf/bcx_round/run_round.py` unchanged, with two things added around it, both borrowed from
`perf/bcx_p10_bfp8/round_ab.py` so the two rows' round numbers are comparable:

1. The arm alternates at the round boundary, inside ONE process on ONE card. `Meter` is where a
   round starts, so the flip goes there and the round_start event carries the arm it ran. The
   order rotates, so a drift in the box lands on both arms equally.
2. `autograd.DGRAD_2D_STATS` is carried per round. An arm that never serves a product through
   `minimal_matmul` is inert and its timing means nothing, so the counter goes in the stamp and
   on every round_start: the artifact says on its face whether the lever reached.

Nothing here changes what the campaign computes. The arm is which kernel serves the dX of a
product against a 2-D weight; `--rounds` only stops collection after N full rounds.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                     # noqa: E402
import run_round as R                 # noqa: E402
from tt_bio import autograd as ag     # noqa: E402

ARMS = ("off", "on")


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        arm = self.arms[self.entries % len(self.arms)]
        ag.DGRAD_2D_MINIMAL = (arm == "on")
        before = dict(ag.DGRAD_2D_STATS)
        super().on_sequence_gradients_enter()
        if M.EVENTS and M.EVENTS[-1].get("kind") == "round_start":
            M.EVENTS[-1]["arm"] = arm
            M.EVENTS[-1]["dgrad_2d_before"] = before


def main():
    ag.DGRAD_2D_SHAPE_CENSUS = True
    M.Meter = ArmMeter
    R.M.Meter = ArmMeter
    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = list(ARMS)
        stamp["arm_order"] = "round 1 off, alternating"
        stamp["dgrad_2d_stats"] = dict(ag.DGRAD_2D_STATS)
        stamp["dgrad_2d_shapes"] = dict(ag.DGRAD_2D_SHAPES)
        stamp["DEVICE_ZEROS"] = bool(ag.DEVICE_ZEROS)
        stamp["DGRAD_2D_MINIMAL_end"] = bool(ag.DGRAD_2D_MINIMAL)
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump
    R.main()


if __name__ == "__main__":
    main()
