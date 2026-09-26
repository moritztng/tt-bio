#!/usr/bin/env python3
"""bcx-p10-trilay: the taped chunk width A/B'd at the ROUND, on the shipped `tt_bio.bindcraft2` path.

`perf/bcx_round/run_round.py` unchanged, with the arm alternating at the round boundary inside
ONE process on ONE card, the shape `bcx-p10-trimul`'s and `bcx-p10-bfp8`'s round A/Bs use, so
the three rows' round numbers are comparable. The arm is the width of the triangle
multiplication's channel loop while a tape is open, nothing else; it is a partition of an
independent-channel sum, so both arms compute the same function.

The reach counter is the width itself. `_trimul_chunk_size` is wrapped here rather than in
`tt_bio`, so a round that claims the arm fired has to show which widths it served and the
production path carries no census state.
"""
from __future__ import annotations

import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                         # noqa: E402
import run_round as R                     # noqa: E402
from tt_bio import tenstorrent as T       # noqa: E402

ARMS = ("off", "on")
WIDTHS: collections.Counter = collections.Counter()


def _install_width_counter():
    real = T._trimul_chunk_size

    def w(seq_len, hidden, batch=1):
        c = real(seq_len, hidden, batch)
        WIDTHS["%dx%d->c%d" % (int(seq_len), int(hidden), int(c))] += 1
        return c
    T._trimul_chunk_size = w


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        arm = self.arms[self.entries % len(self.arms)]
        T.set_trimul_taped_full_chunk(arm == "on")
        before = dict(WIDTHS)
        super().on_sequence_gradients_enter()
        if M.EVENTS and M.EVENTS[-1].get("kind") == "round_start":
            M.EVENTS[-1]["arm"] = arm
            M.EVENTS[-1]["widths_before"] = before


def main():
    _install_width_counter()
    M.Meter = ArmMeter
    R.M.Meter = ArmMeter
    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = list(ARMS)
        stamp["arm_order"] = "round 1 off, alternating"
        stamp["trimul_widths"] = dict(WIDTHS)
        stamp["TRIMUL_TAPED_FULL_CHUNK_end"] = bool(T._TRIMUL_TAPED_FULL_CHUNK)
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump
    R.main()


if __name__ == "__main__":
    main()
