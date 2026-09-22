#!/usr/bin/env python3
"""A25 in one pass: three distances and the cosine BETWEEN THE TWO ERROR VECTORS, per arm.

Two distances from one reference do not order each other, and two distances from two
references do not say whether the errors are the same error. The statistic that does is

    cos( ours - float64 , theirs - float64 )

computed over the concatenated scope, which is the same mass weighting `agreement.py` uses
(every figure here is sum-of-squares over the concatenated tensors, so a tensor's weight is
its share of the scope's squared gradient norm). Near 0 means two independent roundings of
comparable size and there is no shared mechanism left to find; near 1 means one mechanism
that both stacks would have to share; negative means they lean apart.

Every input is verified by digest before it is loaded (A24).
"""
import argparse
import hashlib
import json
import sys

import torch


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def flat(d, names):
    return torch.cat([d[n].to(torch.float64).reshape(-1) for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--bf16-sha", default="")
    ap.add_argument("--arms", nargs="+", required=True, help="label=path.pt")
    ap.add_argument("--sections", default="")
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
    names = sorted(set.intersection(*(set(v[1]) for v in arms.values())))

    f64_full = torch.load(a.f64, map_location="cpu", weights_only=False)
    bf16_full = torch.load(a.bf16, map_location="cpu", weights_only=False)
    names = [n for n in names if f64_full.get(n) is not None and bf16_full.get(n) is not None]
    F = flat(f64_full, names)
    B = flat(bf16_full, names)
    model_sq = sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
                   for v in f64_full.values() if v is not None)
    scope_pct = 100.0 * float(F.norm()) ** 2 / model_sq

    eb = B - F                      # upstream's own bf16 error
    nB, nF, neb = float(B.norm()), float(F.norm()), float(eb.norm())

    def cosv(x, y):
        nx, ny = float(x.norm()), float(y.norm())
        return float(torch.dot(x, y) / (nx * ny)) if nx and ny else float("nan")

    base = {
        "scope_tensors": len(names),
        "scope_pct_of_model_squared_gradient_norm": scope_pct,
        "model_squared_gradient_norm": model_sq,
        "theirs_from_float64": neb / nF,
        "theirs_norm_ratio_vs_float64": nB / nF,
        "theirs_cos_vs_float64": cosv(B, F),
        "a16_zero_model_vs_their_step": 1.0,
        "perfect_fix_threshold_vs_their_step": (neb / nF) / (nB / nF),
    }
    print(f"scope: {len(names)} tensors, {scope_pct:.4f} % of the model's squared gradient norm")
    print(f"theirs from float64 {base['theirs_from_float64']:.6e}   "
          f"perfect-fix threshold vs their step {base['perfect_fix_threshold_vs_their_step']:.6e}")
    print(f"\n{'arm':<26}{'v their step':>14}{'v float64':>13}{'r':>9}{'cos v them':>12}"
          f"{'ERRCOS':>9}{'x thresh':>10}")
    rows = {}
    for lab, (p, d) in arms.items():
        A = flat(d, names)
        ea = A - F
        rows[lab] = {
            "path": p,
            "ours_from_their_step": float((A - B).norm()) / nB,
            "ours_from_float64": float(ea.norm()) / nF,
            "theirs_from_float64": base["theirs_from_float64"],
            "norm_ratio_vs_their_step": float(A.norm()) / nB,
            "cos_vs_their_step": cosv(A, B),
            "cos_between_error_vectors": cosv(ea, eb),
            "our_error_over_theirs": float(ea.norm()) / neb,
            "x_perfect_fix_threshold": (float((A - B).norm()) / nB)
                                       / base["perfect_fix_threshold_vs_their_step"],
        }
        r = rows[lab]
        print(f"{lab:<26}{r['ours_from_their_step']:>14.6e}{r['ours_from_float64']:>13.6e}"
              f"{r['norm_ratio_vs_their_step']:>9.4f}{r['cos_vs_their_step']:>12.6f}"
              f"{r['cos_between_error_vectors']:>+9.4f}{r['x_perfect_fix_threshold']:>10.4f}")

    rep = {"what": __doc__.strip().splitlines()[0], "float64": a.f64,
           "upstream_bf16": a.bf16, "upstream_bf16_sha256": a.bf16_sha or None,
           **base, "arms": rows}
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
