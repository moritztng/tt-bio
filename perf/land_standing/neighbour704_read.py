#!/usr/bin/env python3
"""Is 832's dividing-k divergence a new risk, or the route difference main already ships?

At 704 the fused HiFi route is the shipped default and the materialised fp32 chain is the
fall-back. At 832 that is exactly reversed by a k-ladder gap, and TT_BIO_TRIATT_DIVIDING_K
closes the gap. So the fused-vs-materialised trunk divergence measured HERE, at a length whose
default is already the fused route and which the release gate already folds, is the yardstick
832's 2.152e-01 / 8.724e-02 has been missing.

Pre-registered before the legs ran:
  within 3x of 832's figures  -> the lever's effect is the ordinary route difference we ship
  more than 3x smaller        -> 832 is special, and the hold stands
"""
import json
import math
import sys

# Measured 2026-09-26 on qb2 card 3 at 832 tokens, perf/land_standing/out/trunk_832/.
AT_832 = {0: 2.152e-01, 1: 8.724e-02}
NAME = {0: "s  [1,n,384]", 1: "z  [1,n,n,128]"}
FACTOR = 3.0


def legs(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def rel_l2(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b))
    return num / den if den else float("nan")


fused_a, mat_b, fused_c = (legs(p) for p in sys.argv[1:4])

print("firing, counted per leg rather than argued:")
for tag, recs in (("fusedA (shipped default)", fused_a),
                  ("matB   (-openfold3.trunk)", mat_b),
                  ("fusedC (control)", fused_c)):
    r = recs[-1]
    print(f"  {tag:26s} stats={r.get('hifi_stats')} picks={r.get('hifi_picks')}")

n = min(len(fused_a), len(mat_b), len(fused_c))
print(f"\ntrunk calls captured: A={len(fused_a)} B={len(mat_b)} C={len(fused_c)}, comparing {n}")

worst_ctrl, worst_eff = 0.0, {}
for call in range(n):
    A = {t["i"]: t for t in fused_a[call]["tensors"] if "probe" in t}
    B = {t["i"]: t for t in mat_b[call]["tensors"] if "probe" in t}
    C = {t["i"]: t for t in fused_c[call]["tensors"] if "probe" in t}
    print(f"\ncall {call + 1}:")
    print(f"  {'tensor':16s} {'control A vs C':>16s} {'route A vs B':>16s} "
          f"{'same at 832':>13s} {'ratio 704/832':>14s}")
    for i in sorted(A):
        ctrl = rel_l2(A[i]["probe"], C[i]["probe"])
        eff = rel_l2(B[i]["probe"], A[i]["probe"])
        worst_ctrl = max(worst_ctrl, ctrl)
        worst_eff[i] = max(worst_eff.get(i, 0.0), eff)
        ref = AT_832.get(i)
        ratio = f"{eff / ref:12.2f}x" if ref else " n/a"
        print(f"  {NAME.get(i, str(i)):16s} {ctrl:16.3e} {eff:16.3e} "
              f"{(f'{ref:.3e}' if ref else 'n/a'):>13s} {ratio:>14s}")

print(f"\ncontrol (two shipped-default runs) max rel L2 = {worst_ctrl:.3e}")
if worst_ctrl == 0.0:
    print("  the trunk is bit-deterministic at 704 too, so this instrument has no floor to clear")

print("\nverdict against the pre-registered falsifier:")
for i in sorted(worst_eff):
    ref = AT_832.get(i)
    if ref is None:
        continue
    r = worst_eff[i] / ref
    if r >= 1.0 / FACTOR:
        word = f"within {FACTOR:.0f}x -> the same class of difference main already ships"
    else:
        word = f"more than {FACTOR:.0f}x smaller -> 832 is special, the hold stands"
    print(f"  {NAME.get(i, i):16s} 704={worst_eff[i]:.3e}  832={ref:.3e}  "
          f"ratio {r:.2f}x  {word}")
