#!/usr/bin/env python3
"""The arm table the state doc quotes, generated from the result files so no figure is copied
by hand. Two growth readings per arm: the PRE-REGISTERED fit over k = 2..20, and a second one
restricted to rungs that clear 10x the fp32 differencing floor, because a fit that includes
floor-limited rungs reads a steeper exponent than the divergence has. The pre-registered one
is the bar; the second is disclosed beside it, not instead of it."""
import json
import sys

import numpy as np


def fit(rows, keep):
    live = [r for r in rows if keep(r) and r["rel_d"] > 0.0]
    if len(live) < 2:
        return None, [], 0.0
    x = np.log([r["k"] for r in live]); y = np.log([r["rel_d"] for r in live])
    s, i = np.polyfit(x, y, 1)
    r2 = 1.0 - (y - (s * x + i)).var() / y.var() if y.var() > 0 else 1.0
    return float(s), [r["k"] for r in live], float(r2)


print(f"{'arm':10s} {'warm':>5s} {'acc':>4s} {'rho':>6s} {'share20':>8s} | "
      f"{'d1_ours':>10s} {'rel_d2':>10s} {'rel_d20':>10s} {'med20':>9s} {'floor20':>9s} "
      f"{'d20/floor':>10s} | {'exp k2-20':>9s} {'r2':>5s} | {'exp >10xfl':>10s} {'from k':>6s}")
for p in sys.argv[1:]:
    d = json.load(open(p))
    rows = d["per_step"]
    e1, _, r1 = fit(rows, lambda r: r["k"] >= 2)
    e2, ks, _ = fit(rows, lambda r: r["k"] >= 2 and r["rel_over_floor"] >= 10.0)
    print(f"{d['arm']:10s} {d['warmup_no_steps']:5d} {d['accumulate_grad_batches']:4d} "
          f"{d['rho']:6.3f} {d['feedback_share_of_drive_norm']['20']:8.4f} | "
          f"{d['d1']['ours_norm']:10.3e} {rows[1]['rel_d']:10.3e} {rows[-1]['rel_d']:10.3e} "
          f"{rows[-1]['median_per_tensor']:9.3e} {rows[-1]['fp32_differencing_floor']:9.2e} "
          f"{rows[-1]['rel_over_floor']:10.3e} | {e1:+9.3f} {r1:5.3f} | "
          f"{(f'{e2:+10.3f}' if e2 is not None else 'n/a'.rjust(10))} "
          f"{(min(ks) if ks else 0):6d}")
