#!/usr/bin/env python3
"""Validate abag_xm_frontier_analysis math against KNOWN slab numbers.

For the 12 frozen targets, the selection script (p1) fixed per-target s50:
  fails all 0; 9ma0=1, 9q6z=1, 9j4c=7, 9uoi=7, 9m8l=48, 9ldx=47.
Therefore, from slab labels via the analysis functions:
  oracle@50 (thr 0.23) must equal exactly 6/12 = 0.5
  oracle@1 must equal 111/600 = 0.185
  per-target oracle@1 must equal s50/50.
Any deviation = a bug in the analysis code path (dockq extraction or
hypergeometric), caught before real frontier data exists.
"""
import importlib.util, statistics, sys
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/abag-xm-seeds-vs-samples-oracle-frontier-p2")
spec = importlib.util.spec_from_file_location("ana", WT / "scripts" / "abag_xm_frontier_analysis.py")
ana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana)

S50 = {"9q6y": 0, "9tmp": 0, "9gei": 0, "9fte": 0, "9wpm": 0, "9qrv": 0,
       "9ma0": 1, "9q6z": 1, "9j4c": 7, "9uoi": 7, "9m8l": 48, "9ldx": 47}
SLAB = Path.home() / "abag_xm" / "tier_a" / "labels"

per = {}
missing = []
for t in ana.TARGETS:
    p = SLAB / f"opendde_abag_{t}.json"
    if p.exists():
        dq, d = ana.dockq_list(p)
        s = sum(1 for x in dq if x is not None and x >= 0.23)
        assert s == S50[t], f"{t}: recomputed s50={s} != frozen {S50[t]}"
        per[t] = dq
    else:
        # qb1 unreachable (since 2026-07-30 ~02:30 UTC) — substitute the frozen
        # count (exact at thr 0.23, see ana.FROZEN_S50); mark as not file-verified.
        missing.append(t)
        per[t] = [1.0] * S50[t] + [0.0] * (50 - S50[t])
if missing:
    print(f"NOTE: {len(missing)} slab labels unreachable (qb1 down): {missing} — "
          f"their frozen s50 counts substituted (exact at thr 0.23); "
          f"{12 - len(missing)}/12 targets file-verified")
else:
    print("per-target s50 recomputation: exact match on all 12")

o50 = ana.mean_oracle(per, [50], 0.23)[50][0]
o1 = ana.mean_oracle(per, [1], 0.23)[1][0]
print(f"oracle@50 = {o50:.4f} (expect 0.5000)")
print(f"oracle@1  = {o1:.4f} (expect {111/600:.4f})")
assert abs(o50 - 0.5) < 1e-9, o50
assert abs(o1 - 111 / 600) < 1e-9, o1

_, per_t_vals = ana.mean_oracle(per, [1], 0.23)[1]
for t, v in zip(ana.TARGETS, per_t_vals):
    assert abs(v - S50[t] / 50) < 1e-9, (t, v)
print("per-target oracle@1 == s50/50: exact on all 12")

# hypergeometric spot checks: 9m8l S=48/50 -> oracle@2 = 1 - C(2,2)/C(50,2)
import math
exp = 1 - math.comb(2, 2) / math.comb(50, 2)
_, vals2 = ana.mean_oracle(per, [2], 0.23)[2]
got = vals2[ana.TARGETS.index("9m8l")]
assert abs(got - exp) < 1e-9, (got, exp)
print(f"oracle@2 (9m8l) = {got:.4f} == analytic {exp:.4f}")

# Arm-B-style seed-block math on a synthetic: w=3 blocks w/ success, k=2 ->
# 1 - C(17,2)/C(20,2) = 1 - 136/190
fake = {t: {j: ([0.5] if j < 3 else [0.0], None) for j in range(20)} for t in ana.TARGETS}
v = ana.arm_b_seed_oracle(fake, 0.23)[20][0]
exp = 1 - math.comb(17, 2) / math.comb(20, 2)
assert abs(v - exp) < 1e-9, (v, exp)
print(f"arm-B seed-block oracle k=2 synthetic = {v:.4f} == analytic {exp:.4f}")

# bootstrap equivalence CI on identical arms must be (0, 0, 0)
pt, lo, hi = ana.equivalence(per, per, 0.23, 50)
assert pt == 0.0 and lo == 0.0 and hi == 0.0, (pt, lo, hi)
print("bootstrap equivalence on identical arms = (0, 0, 0): exact")

print("ALL ANALYSIS-MATH CHECKS PASS")
