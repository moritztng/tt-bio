#!/usr/bin/env python3
"""bcx-p10-devgap: the shared mask-bias cache A/B'd at the ROUND, reach stamped.

`perf/bcx_round/run_round.py` unchanged, with the arm set at the round boundary inside ONE
process on ONE card, so a drift in the box lands on both arms equally. The shape is
`perf/bcx_p10_tapegen/round_ab.py`'s; the arm is different.

    off     `AF2EvoformerBlock.mask_bias_cache = None`, which is the shipped default. Every
            block rebuilds `1e9 * (msa_mask - 1)` in both layouts, 52 times a forward and 52
            more on the checkpoint recompute.
    on      one `_MaskBiasCache` shared by the whole trunk. The mask does not change between
            blocks, so the rebuild is redundant; the cache is what removes it.

REACH IS STAMPED, per round and in the artifact. `af2.MASK_BIAS_CACHE_STATS` counts `build`
against `hit`, so an `on` round with 0 hits is visibly a round where the lever never reached,
not a null result. The first round also runs with `verify=1`: every hit rebuilds the pair and
compares it to the cached one with `torch.equal`, which is the bit-exactness the lever rests
on, measured rather than argued. Verification is off for every timed round after it, because
it costs two device-to-host copies a hit.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

import meter as M                      # noqa: E402
import run_round as R                  # noqa: E402
from tt_bio import af2 as A            # noqa: E402

ARMS = tuple(a for a in os.environ.get("DEVGAP_AB_ARMS", "off,on").split(",") if a)
#: Rounds whose `on` arm also rebuilds and compares. Round 1 carries the jit compile and the
#: lazy trunk load and is never timed, so it is the free place to pay for the check.
VERIFY_ROUNDS = int(os.environ.get("DEVGAP_AB_VERIFY_ROUNDS", "1"))


def _reach() -> dict:
    return dict(A.MASK_BIAS_CACHE_STATS)


def _arm(name: str, verify: bool) -> None:
    A.set_mask_bias_cache(name == "on", verify=verify)


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        # The parent stamps round_start (or raises the stop) and is the only place that knows
        # the round number, so set the arm first and tag the event it appends.
        arm = self.arms[self.entries % len(self.arms)]
        verify = self.entries < VERIFY_ROUNDS * len(self.arms)
        _arm(arm, verify)
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
            M.EVENTS[-1]["verify"] = verify
            M.EVENTS[-1]["reach_before"] = before


def main():
    M.Meter = ArmMeter
    R.M.Meter = ArmMeter

    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = list(ARMS)
        stamp["arm_order"] = f"round 1 {ARMS[0]}, rotating"
        stamp["verify_rounds"] = VERIFY_ROUNDS
        stamp["reach_end"] = _reach()
        stamp["mask_bias_cache_end"] = A.AF2EvoformerBlock.mask_bias_cache is not None
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump

    R.main()


if __name__ == "__main__":
    main()
