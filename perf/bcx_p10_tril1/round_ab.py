#!/usr/bin/env python3
"""bcx-p10-tril1: the taped trimul's L1 residency A/B'd at the ROUND, alternating per round.

`perf/bcx_round/run_round.py` unchanged, with `TT_BIO_TRIMUL_TAPED_L1` flipped at the round
boundary inside ONE process on ONE card, so both arms see the same box, the same JAX cache and
the same clock. Moving a buffer between L1 and DRAM does not change the arithmetic, so the two
arms compute the same function; `budget.py clash` is where that is checked against a float
reference, and `bits` below is the cheap regression signal beside it.

The reach counter is the residency decision itself, taken from the engine's own census
(`TRIMUL_TAPED_L1_STATS`) rather than re-derived here: an `on` round whose `l1` count is 0 refused
the budget and measured nothing, and that has to be visible on the round rather than assumed.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                         # noqa: E402
import run_round as R                     # noqa: E402
from tt_bio import tenstorrent as T       # noqa: E402

ARMS = ("off", "on")


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and its reach recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    #: The round that is currently open, so its reach can be stamped when the next one starts.
    #: `Meter` has an enter hook and no exit hook, and `on_sequence_gradients_enter` raises
    #: `StopAfterRounds` on the last one, so both the normal boundary and the stop have to close
    #: the round that came before them. A round with `l1: 0` on the `on` arm refused the budget
    #: and measured nothing.
    open_round = None

    def _close(self):
        if self.open_round is not None:
            self.open_round["taped_l1"] = dict(T.TRIMUL_TAPED_L1_STATS)
            self.open_round = None

    def on_sequence_gradients_enter(self):
        self._close()
        arm = self.arms[self.entries % len(self.arms)]
        T.set_trimul_taped_l1(arm == "on")
        T.TRIMUL_TAPED_L1_STATS.update({"l1": 0, "dram": 0, "clash": 0})
        try:
            super().on_sequence_gradients_enter()
        finally:
            if M.EVENTS and M.EVENTS[-1].get("kind") == "round_start":
                M.EVENTS[-1]["arm"] = arm
                self.open_round = M.EVENTS[-1]


def main():
    M.Meter = ArmMeter
    R.M.Meter = ArmMeter
    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = list(ARMS)
        stamp["arm_order"] = "round 1 off, alternating"
        stamp["TRIMUL_TAPED_L1_end"] = bool(T._TRIMUL_TAPED_L1)
        stamp["TRIMUL_TAPED_L1_SHARE"] = T._TRIMUL_TAPED_L1_SHARE
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump
    R.main()


if __name__ == "__main__":
    main()
