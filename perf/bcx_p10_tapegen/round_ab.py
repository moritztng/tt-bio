#!/usr/bin/env python3
"""bcx-p10-tapegen: the `generic_op` tape entries A/B'd at the ROUND, reach stamped.

`perf/bcx_round/run_round.py` unchanged, with the arm set at the round boundary inside ONE
process on ONE card, so a drift in the box lands on every arm equally. The shape is
`bcx-p10-triatt`'s and `bcx-p10-bfp8`'s; the arms are different.

    mat     nothing installed -- `TT_BIO_TAPED_KERNELS` empty, so every fused kernel declines
            under the tape exactly as origin/main does. `_fp32_softmax_attention` serves the
            triangle attentions and `_channel_move` takes the stock permute
    rp      `reblock_permute` and `reblock_permute_back` taped. A permutation's VJP is the
            inverse permutation, so this leg is bit-exact both ways and is the mechanism's
            proof: if the round is not bit-identical to `mat`, something other than the
            permutation moved
    hifi    `tri_att_sdpa_hifi` taped, `TT_BIO_TRIATT_FUSED_HIFI=1` and
            `TT_BIO_TRIATT_DIVIDING_K=1`. The persistent-mask kernel serves the forward and
            `autograd.triangle_attention` carries the backward. BOTH gates have to be open:
            the entry makes the arm reachable, dividing-k makes 288 servable, and opening one
            without the other measures the other
    both    rp + hifi

REACH IS STAMPED, per round and in the artifact. `taped_ttnn.KERNEL_STATS` counts served and
declined per entry, `TRIATT_FUSED_HIFI_STATS` splits served / declined / too_short / taped,
`reblock_permute.STATS` counts the channel moves and `FP32_SOFTMAX_STATS["calls"]` counts the
materialised path. An A/B whose arms agree is only a null result if the moving arm shows
serves; if `taped` moves instead, the lever never reached and the seconds mean nothing.
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
from tt_bio import taped_ttnn as TT    # noqa: E402
from tt_bio import tenstorrent as tn   # noqa: E402
from tt_bio import reblock_permute as rp  # noqa: E402

ARMS = tuple(a for a in os.environ.get("TAPEGEN_AB_ARMS", "mat,hifi").split(",") if a)

_RP = "reblock_permute,reblock_permute_back"


def _reach() -> dict:
    """Every counter that says which route the round actually took."""
    return {"entries": {k: list(v) for k, v in TT.KERNEL_STATS.items()},
            "fused": dict(tn.TRIATT_FUSED_HIFI_STATS),
            "triatt_sdpa": list(tn._triatt_sdpa.STATS),
            "reblock": list(rp.STATS),
            "reblock_back": list(rp.STATS_BACK),
            "fp32_softmax_calls": tn.FP32_SOFTMAX_STATS["calls"],
            "picks": {str(k): v for k, v in tn.TRIATT_FUSED_HIFI_PICKS.items()}}


def _arm(name: str) -> None:
    want = []
    if name in ("rp", "both"):
        want.append(_RP)
    if name in ("hifi", "both"):
        want.append("tri_att_sdpa_hifi")
    os.environ["TT_BIO_TAPED_KERNELS"] = ",".join(want)
    tn._TRIATT_FUSED_HIFI = name in ("hifi", "both")
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1" if name in ("hifi", "both") else "0"


class ArmMeter(M.Meter):
    """`Meter`, with the arm set at the round boundary and recorded on the round."""

    def __init__(self, rounds, arms=ARMS):
        super().__init__(rounds)
        self.arms = list(arms)

    def on_sequence_gradients_enter(self):
        # The parent stamps round_start (or raises the stop) and is the only place that knows
        # the round number, so set the arm first and tag the event it appends.
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
        stamp["arm_order"] = f"round 1 {ARMS[0]}, rotating"
        stamp["reach_end"] = _reach()
        stamp["registry"] = sorted(TT.KERNELS)
        stamp["TAPED_KERNELS_end"] = os.environ.get("TT_BIO_TAPED_KERNELS", "")
        stamp["TRIATT_FUSED_HIFI_flag_end"] = bool(tn._TRIATT_FUSED_HIFI)
        stamp["TRIATT_DIVIDING_K_end"] = tn._triatt_hifi_dividing_k()
        stamp["triatt_sdpa_rejects"] = {str(k): v for k, v in tn._triatt_sdpa.REJECTS.items()}
        stamp["triatt_sdpa_stats"] = list(tn._triatt_sdpa.STATS)
        stamp["reblock_rejects"] = {str(k): v for k, v in rp.REJECTS.items()}
        return real_dump(path, stamp)
    M.dump = dump
    R.M.dump = dump

    R.main()


if __name__ == "__main__":
    main()
