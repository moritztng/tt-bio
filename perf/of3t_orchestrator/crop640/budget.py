#!/usr/bin/env python3
"""Size the crop-640 and crop-768 backward deficit from of3t-l1's measured ladder.

Re-derivable rather than transcribed: every measured byte below is read from of3t-l1's committed
artifacts, and the three estimates are computed here. The point of the file is that it can be
re-run when a rung moves, so the budget cannot go stale the way a table does.

The control is the important part. The 640 run REACHED 34.215 GB and still failed, so the true
requirement is at least that -- any model predicting less is refuted before it is used. All three
estimates predict 45.6-51.4 GB, which is what makes the bracket quotable rather than a guess.
"""
import json, math, pathlib, sys

GB = 1 << 30
CARD_GB = 34.22
FLOOR_640_GB = 34.215          # where it died; a lower bound on the requirement
BND_640_GB = 5.070             # of3t-l1's K13: the 48 retained (s, z) block boundaries at 640

root = pathlib.Path(__file__).resolve().parents[2]
peaks = {}
for n in (128, 256, 384):
    p = root / f"of3t_l1/out/r3_{n}.json"
    if not p.is_file():
        sys.exit(f"missing {p} -- run from a tree that carries of3t-l1's artifacts")
    peaks[n] = json.loads(p.read_text())["backward"]["dram_peak_b"] / GB

# The retained boundaries are 48 (s, z) pairs. z is N^2 * c_z and dominates s at N * c_s, so the
# set scales as N^2 and K13's 640 figure carries to any other crop.
bnd = lambda n: BND_640_GB * (n / 640) ** 2
rest384 = peaks[384] - bnd(384)
decomp = lambda n: rest384 * (n / 384) ** 2 + bnd(n)

# peak = a + b*N^2 over the three completed rungs
xs, ys = list(peaks), [peaks[n] for n in peaks]
s01 = sum(x * x for x in xs); s11 = sum(x ** 4 for x in xs)
t0 = sum(ys); t1 = sum(y * x * x for y, x in zip(ys, xs))
det = len(xs) * s11 - s01 * s01
a = (t0 * s11 - t1 * s01) / det
b = (len(xs) * t1 - s01 * t0) / det
quad = lambda n: a + b * n * n

expo = math.log(peaks[384] / peaks[256]) / math.log(384 / 256)
power = lambda n: peaks[384] * (n / 384) ** expo

est = {"decomposition": decomp, "quadratic": quad, "power_law": power}
refuted = [k for k, f in est.items() if f(640) < FLOOR_640_GB]
if refuted:
    sys.exit(f"REFUTED by the observational floor at 640: {refuted} predict under "
             f"{FLOOR_640_GB} GB, which the run already reached and still failed")

out = {"card_GB": CARD_GB, "measured_backward_GB": peaks, "floor_640_GB": FLOOR_640_GB,
       "all_estimates_clear_the_floor": True, "estimates_GB": {}, "levers": {}}
for n in (640, 768):
    out["estimates_GB"][n] = {k: round(f(n), 2) for k, f in est.items()}
    need = decomp(n) - CARD_GB
    spill = bnd(n)
    out["levers"][n] = {
        "requirement_GB": round(decomp(n), 2),
        "deficit_GB": round(need, 2),
        "boundary_spill_saves_GB": round(spill, 3),
        "spill_share_of_deficit_pct": round(100 * spill / need),
        "sufficient_alone": bool(spill >= need),
        "other_term_must_fall_pct": round(100 * (need - spill) / (decomp(n) - bnd(n))),
    }
json.dump(out, sys.stdout, indent=1)
print()
