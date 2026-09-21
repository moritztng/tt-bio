#!/usr/bin/env python3
"""Per-tensor bars for every arm, against BOTH references, from the gradient dumps.

`a25.py` reports the mass-weighted headline; this reports what sits outside the per-tensor
bar, which is a different question and the one A23 says must be answered beside it. Both
references are scored on the same tensors in the same pass so the two columns cannot drift
apart (A27): `ours_vs_their_step` is against upstream's own bf16 training step, whose
denominator is THEIR gradient, and `ours_vs_float64` is against the float64 reference, whose
denominator is the float64 gradient. They are different measurements.

Mass share is a tensor's share of the scope's squared float64 gradient norm.
"""
import argparse
import hashlib
import json
import sys

import torch

BAR = 5.0e-2


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(a, b):
    nb = float(torch.linalg.vector_norm(b))
    return float(torch.linalg.vector_norm(a - b)) / (nb + 1e-300), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--bf16-sha", default="")
    ap.add_argument("--arms", nargs="+", required=True, help="label=path.pt")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.bf16_sha:
        got = digest(a.bf16)
        if got != a.bf16_sha:
            print(f"FAILED: bf16 reference digest {got} != {a.bf16_sha}", file=sys.stderr)
            return 3
        print(f"A24: upstream bf16 reference digest verified {got}")

    arms = {}
    for spec in a.arms:
        lab, _, p = spec.partition("=")
        arms[lab] = (p, torch.load(p, map_location="cpu", weights_only=False))
    F = torch.load(a.f64, map_location="cpu", weights_only=False)
    B = torch.load(a.bf16, map_location="cpu", weights_only=False)
    names = sorted(set.intersection(*(set(v[1]) for v in arms.values())))
    names = [n for n in names if F.get(n) is not None and B.get(n) is not None]

    f = {n: F[n].to(torch.float64).reshape(-1) for n in names}
    b = {n: B[n].to(torch.float64).reshape(-1) for n in names}
    mass = {n: float(torch.linalg.vector_norm(f[n])) ** 2 for n in names}
    tot = sum(mass.values())

    # A28: upstream's own bf16 step on this scope, per tensor, against the same float64
    # reference. Every row below is read beside it.
    floor = {}
    for n in names:
        r, _ = rel(b[n], f[n])
        floor[n] = r
    floor_over = sum(1 for n in names if floor[n] > BAR)
    floor_mass_over = sum(mass[n] for n in names if floor[n] > BAR) / tot

    print(f"scope: {len(names)} tensors, bar {BAR:.1e}")
    print(f"A28 floor (upstream's own bf16 vs float64): {floor_over} of {len(names)} over the "
          f"bar, {100*floor_mass_over:.4f} % of the mass")
    print(f"\n{'arm':<20}{'ref':<14}{'massw':>13}{'median':>13}{'worst':>13}"
          f"{'>bar':>7}{'mass>bar %':>12}")
    rows = {}
    for lab, (p, d) in arms.items():
        rows[lab] = {"path": p}
        for refname, R in (("their_step", b), ("float64", f)):
            per = []
            for n in names:
                g = d[n].to(torch.float64).reshape(-1)
                r, _ = rel(g, R[n])
                per.append((r, n))
            per.sort()
            over = [n for r, n in per if r > BAR]
            mass_over = sum(mass[n] for n in over) / tot
            # mass-weighted = sum-of-squares over the concatenated scope, the same weighting
            # a25.py uses, recomputed here so the two artifacts are checkable against each other
            num = sum(float(torch.linalg.vector_norm(
                d[n].to(torch.float64).reshape(-1) - R[n])) ** 2 for n in names)
            den = sum(float(torch.linalg.vector_norm(R[n])) ** 2 for n in names)
            massw = (num / den) ** 0.5
            rows[lab][refname] = {
                "mass_weighted_rel_l2": massw,
                "median_rel_l2": per[len(per) // 2][0],
                "worst_rel_l2": per[-1][0], "worst_tensor": per[-1][1],
                "over_bar": len(over), "of": len(names),
                "mass_over_bar_pct": 100 * mass_over,
            }
            rr = rows[lab][refname]
            print(f"{lab:<20}{refname:<14}{rr['mass_weighted_rel_l2']:>13.6e}"
                  f"{rr['median_rel_l2']:>13.6e}{rr['worst_rel_l2']:>13.6e}"
                  f"{rr['over_bar']:>7}{rr['mass_over_bar_pct']:>12.4f}")

    rep = {"what": __doc__.strip().splitlines()[0], "bar": BAR,
           "float64": a.f64, "upstream_bf16": a.bf16,
           "upstream_bf16_sha256": a.bf16_sha or None,
           "scope_tensors": len(names),
           "a16_zero_model_rel_l2": 1.0,
           "a28_floor_over_bar": floor_over,
           "a28_floor_mass_over_bar_pct": 100 * floor_mass_over,
           "arms": rows}
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
