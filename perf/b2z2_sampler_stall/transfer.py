#!/usr/bin/env python3
"""What the measured sampler stall split does to b2z2-final-ceiling's 1.977x top rung.

`b2z2-final-ceiling`'s `closing.py` builds its optimistic rung by taking the Pairformer block's
movement-free multiplier -- 0.41, i.e. 14.900 ms of a 36.344 ms BH block survives if every CB
stall is deleted -- and applying it verbatim to the diffusion sampler:

    samp_opt = sampler_s * 0.41 + host_in_sampler_s * 0.59

That is the transfer this row exists to test. It rests on `DEVICE COMPUTE CB WAIT FRONT` being
blank in every committed step capture, so nobody could say what the sampler's own multiplier is.
This script substitutes the measured one and re-runs the same three rungs, changing nothing else.

The sampler's multiplier is built on the step's PRODUCTION bracket, not on the profiled span. The
profiler inflates the gaps between programs and leaves the kernels alone -- on BH the profiled
capture sums to 22.025 ms of device kernel time against the fold's own 22.015 ms while its span is
45.50 ms against a 26.400 ms production wall -- so the stall FRACTIONS are read off the profiled
capture and applied to the production wall.

Usage: transfer.py --split split_step_wh_c2.json [--out transfer.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- committed primaries, every one sourced -------------------------------------------------
STEP_WALL_MS = 26.400       # b2z2-sampler-ceiling-map, BH production step wall
STEP_KERNEL_MS = 22.015     # of which device kernel time; the rest is exposed dispatch
STEP_TRISC1_MS = 15.211     # math-thread residency in that step (BH)
STEPS = 200

MOVE_FREE_MULT = 0.41       # b2z-kernel-cycle-census: 14.900 / 36.344 ms, BH Pairformer block
HOST_IN_TRUNK_S = 0.355     # b2z2-final-ceiling closing.py
HOST_IN_SAMPLER_S = 0.647   # b2z-host-residual-kill, qb2 card 2
LAUNCH_HELD_STALE = 0.32    # the fraction redteam's conservative rung holds fixed
SPLIT = {"fold": 18.594, "trunk": 11.3209, "sampler": 5.2547, "rest": 2.0184}
PUBLISHED_CELL_S = 20.079


def held(phase_s, mult, host_s):
    return phase_s * mult + host_s * (1.0 - mult)


def rungs(samp_mult_opt, samp_mult_cons):
    f, tr, sa, re_ = SPLIT["fold"], SPLIT["trunk"], SPLIT["sampler"], SPLIT["rest"]
    trunk_floor = held(tr, MOVE_FREE_MULT, HOST_IN_TRUNK_S)
    out = {
        "trunk_only": trunk_floor + sa + re_,
        "trunk_sampler_cons": trunk_floor + held(sa, samp_mult_cons, HOST_IN_SAMPLER_S) + re_,
        "trunk_sampler_opt": trunk_floor + held(sa, samp_mult_opt, HOST_IN_SAMPLER_S) + re_,
    }
    return {k: {"fold_s": v, "x": f / v} for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = json.loads(a.split.read_text())

    f_in = d["trisc1_pct"]["wait_in_ms"] / 100.0
    f_out = d["trisc1_pct"]["wait_out_ms"] / 100.0
    f_cmp = d["trisc1_pct"]["compute_ms"] / 100.0

    wait_ms = (f_in + f_out) * STEP_TRISC1_MS
    step_free_ms = STEP_WALL_MS - wait_ms
    samp_mult = step_free_ms / STEP_WALL_MS
    samp_cons_mult = LAUNCH_HELD_STALE + (1 - LAUNCH_HELD_STALE) * samp_mult

    pub = rungs(MOVE_FREE_MULT, LAUNCH_HELD_STALE + (1 - LAUNCH_HELD_STALE) * MOVE_FREE_MULT)
    mea = rungs(samp_mult, samp_cons_mult)

    res = {
        "wh_fractions_of_trisc1": {"wait_in": f_in, "wait_out": f_out, "compute": f_cmp},
        "bh_step_wait_ms": wait_ms, "bh_step_movement_free_ms": step_free_ms,
        "sampler_movement_free_mult_measured": samp_mult,
        "sampler_movement_free_mult_transferred": MOVE_FREE_MULT,
        "sampler_s_measured_floor": held(SPLIT["sampler"], samp_mult, HOST_IN_SAMPLER_S),
        "sampler_s_transferred_floor": held(SPLIT["sampler"], MOVE_FREE_MULT, HOST_IN_SAMPLER_S),
        "rungs_as_published": pub, "rungs_with_measured_sampler": mea,
    }
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))

    P = print
    P(f"\nWH measured fractions of the step's math-thread residency:")
    P(f"  input-tile wait {100*f_in:5.1f} %   output-room wait {100*f_out:5.1f} %"
      f"   compute {100*f_cmp:5.1f} %")
    P(f"\nApplied to the BH production step (wall {STEP_WALL_MS} ms = kernel {STEP_KERNEL_MS}"
      f" + dispatch {STEP_WALL_MS-STEP_KERNEL_MS:.3f}, TRISC1 resident {STEP_TRISC1_MS} ms):")
    P(f"  CB stall in the step        {wait_ms:7.3f} ms  "
      f"{100*wait_ms/STEP_WALL_MS:5.1f} % of the step wall")
    P(f"  movement-free step          {step_free_ms:7.3f} ms   multiplier "
      f"{STEP_WALL_MS/step_free_ms:.4f}x   (survivor fraction {samp_mult:.4f})")
    P(f"  the trunk's transferred multiplier would be {1/MOVE_FREE_MULT:.4f}x "
      f"(survivor fraction {MOVE_FREE_MULT})")
    P(f"  sampler at its own floor    {res['sampler_s_measured_floor']:.4f} s   "
      f"transferred {res['sampler_s_transferred_floor']:.4f} s   "
      f"difference {res['sampler_s_measured_floor']-res['sampler_s_transferred_floor']:+.4f} s")
    P(f"\nb2z2-final-ceiling's rungs, quoted against its own 18.594 s levered arm:")
    P(f"  {'rung':26s} {'as published':>14s} {'with measured':>14s}")
    for k in pub:
        P(f"  {k:26s} {pub[k]['x']:13.4f}x {mea[k]['x']:13.4f}x"
          f"   ({100*(mea[k]['x']/pub[k]['x']-1):+.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
