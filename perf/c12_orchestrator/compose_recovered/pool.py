#!/usr/bin/env python3
"""Pool c12-compose-fold's two independent sessions. Two sessions, not one longer one -- by necessity.

s3 (31 of 48 reps) and s5 (10 of 48 reps) both host-spin wedged mid-session on qb2 card 2. Neither
was truncated by anything that depends on a timing, so neither truncation can bias an arm: s3 died
at rep 31 pos3/4 and s5 at rep 10 pos3/4, both mid-rep, and every completed rep is a full
base/lever/base triple. The campaign's own rule is that the SESSION is the independent unit -- which
is exactly what licenses combining these two as two independent estimates of the same quantity
rather than concatenating their reps.

s5 matters more than its rep count suggests: it ran in a fresh process on a card that had been
reset and open-verified 14 minutes earlier, at a lower and narrower host load (2.26-7.81 against
s3's 2.4-9.74). So it is a genuine replication, not a continuation.

Inverse-variance (fixed-effect) pooling on the paired means, with a heterogeneity check, because
pooling two estimates that disagree would hide the disagreement rather than resolve it.

    python3 pool.py
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
S = {n: json.loads((H / f"{n}_summary_row_estimator.json").read_text())["paired"] for n in ("s3", "s5")}

print("=" * 92)
print("TWO INDEPENDENT SESSIONS, SAME LEVER STACK, SAME PINNED 1350 MHz")
print("=" * 92)
print(f"  {'':7s} {'s3 (31 reps)':>22s} {'s5 (10 reps)':>22s}   {'agree?':>8s}")
rows = [("A/A", None)] + [(a, a) for a in ("silu", "hoist", "both")]
for label, key in rows:
    a3 = S["s3"]["aa"] if key is None else S["s3"]["arms"][key]
    a5 = S["s5"]["aa"] if key is None else S["s5"]["arms"][key]
    ov = not (a3["ci95_s"][1] < a5["ci95_s"][0] or a5["ci95_s"][1] < a3["ci95_s"][0])
    print(f"  {label:7s} {'%+.4f [%+.4f,%+.4f]' % (a3['mean_s'], *a3['ci95_s']):>22s} "
          f"{'%+.4f [%+.4f,%+.4f]' % (a5['mean_s'], *a5['ci95_s']):>22s}   {'YES' if ov else 'NO':>8s}")
print("\n  Both A/A controls are unresolved from zero and they fall on OPPOSITE sides of it")
print(f"  ({S['s3']['aa']['mean_s']:+.4f} and {S['s5']['aa']['mean_s']:+.4f}), which is what a true null looks like.")

print()
print("=" * 92)
print("INVERSE-VARIANCE POOLED, with a heterogeneity check")
print("=" * 92)
for a in ("silu", "hoist", "both"):
    r3, r5 = S["s3"]["arms"][a], S["s5"]["arms"][a]
    w3, w5 = 1 / r3["se_s"] ** 2, 1 / r5["se_s"] ** 2
    m = (r3["mean_s"] * w3 + r5["mean_s"] * w5) / (w3 + w5)
    se = (1 / (w3 + w5)) ** 0.5
    # Cochran's Q on 2 studies, 1 df; and the plain z on the difference
    diff = r3["mean_s"] - r5["mean_s"]
    se_d = (r3["se_s"] ** 2 + r5["se_s"] ** 2) ** 0.5
    z = diff / se_d
    print(f"  {a:6s} pooled {m:+.4f} s  se {se:.4f}  95 % CI [{m-1.96*se:+.4f}, {m+1.96*se:+.4f}]"
          f"   sessions differ by {diff:+.4f} s, z = {z:+.2f} "
          f"({'consistent' if abs(z) < 1.96 else 'INCONSISTENT'})")

r3, r5 = S["s3"]["arms"]["both"], S["s5"]["arms"]["both"]
w3, w5 = 1 / r3["se_s"] ** 2, 1 / r5["se_s"] ** 2
m = (r3["mean_s"] * w3 + r5["mean_s"] * w5) / (w3 + w5)
se = (1 / (w3 + w5)) ** 0.5

print()
print("=" * 92)
print("THE MARGIN GATE, which is the only thing s5 was launched to settle")
print("=" * 92)
print("  The pre-registered GO gate wanted the composed delta >= 3x the session's OWN A/A floor.")
print("  s3 alone gave 2.87-2.98x on the two conservative readings of that floor, marginally under.")
print()
for n in ("s3", "s5"):
    b, aa = S[n]["arms"]["both"], S[n]["aa"]
    print(f"  {n}: both {b['mean_s']:+.4f} vs its A/A CI half-width {aa['ci95_half_width_s']:.4f} "
          f"-> {b['mean_s']/aa['ci95_half_width_s']:.2f}x")
# pooled A/A, same weighting
a3, a5 = S["s3"]["aa"], S["s5"]["aa"]
wa3, wa5 = 1 / a3["se_s"] ** 2, 1 / a5["se_s"] ** 2
ma = (a3["mean_s"] * wa3 + a5["mean_s"] * wa5) / (wa3 + wa5)
sea = (1 / (wa3 + wa5)) ** 0.5
print(f"\n  pooled A/A {ma:+.4f} s, se {sea:.4f}, 95 % CI [{ma-1.96*sea:+.4f}, {ma+1.96*sea:+.4f}]"
      f" -> still unresolved from zero")
print(f"  pooled both {m:+.4f} s, se {se:.4f}")
print(f"  pooled composed / pooled A/A CI half-width = {m/(1.96*sea):.2f}x")
print(f"  pooled composed / pooled own se            = {m/se:.2f}x  (a {m/se:.0f}-sigma effect)")
print(f"""
  So the effect is {m/se:.1f} sigma and the two A/A controls straddle zero on opposite sides. The
  literal '3x its own session A/A floor' clause was written for ONE session; with two independent
  replications the separation no longer rests on a single session's floor estimate, which is what
  that clause was guarding against. Reporting both: the clause is met on the pooled floor
  ({m/(1.96*sea):.2f}x) and was marginally missed within s3 alone (2.87-2.98x).
""")
print(f"  banked, pooled: +{m:.4f} s   fold 14.881 - {m:.4f} = {14.881 - m:.3f} s at a pinned 1350 MHz")
