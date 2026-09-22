#!/usr/bin/env python3
"""The of3t-wirefix arm table and every rung of every arm.

    perf/of3t_traj20/wirefix_report.py > perf/of3t_traj20/WIREFIX_RUNGS.txt

`d_20` is reported but it is not the bar. The bar is the growth law over k = 2..20, so every
rung is here and the endpoint is one of them.
"""
import json
import os
import sys

ARMS = ["fixed5", "fixed", "ulp", "ulp4", "aa", "miswire5", "scaled", "avg", "miswire"]
HERE = os.path.dirname(os.path.abspath(__file__))

WHAT = {
    "fixed5": "the result: all five divergences closed, betas from the shipped signature",
    "fixed":  "the four of3t-traj20 found, with betas left at AdamW's class default",
    "ulp":    "rounding floor on the fixed5 path: ours vs ours, one side perturbed 2**-24",
    "ulp4":   "the same floor on the fixed path",
    "aa":     "A/A, upstream against upstream, a path this row did not touch",
    "miswire5": "instrument-can-fail: fixed5 with D11's off-by-one restored, our side only",
    "scaled": "of3t-traj20's shipped arm, before any fix",
    "avg":    "of3t-traj20's three-fix arm, closed in the harness rather than the source",
    "miswire": "of3t-traj20's mis-wire arm, on the pre-fix path",
}


def betas_of(arm, d):
    """Per entry, never from a comment above the table. The of3t-traj20 arms predate
    `shipped_defaults` and ran against the PRE-FIX recipe, so their construction is stated
    from that row's record and marked, rather than inherited from this row's key lookup."""
    if "shipped_defaults" not in d:
        return "(0.9, 0.999)", "0.01" if arm == "scaled" else "0.0", "None"
    if arm == "aa":
        # Both sides are upstream, so the betas are upstream's on both and this
        # row's shipped defaults never enter the arm.
        return "both upstream", "--", "--"
    s = d["shipped_defaults"]
    b = str(tuple(s["betas"])) if arm in ("fixed5", "ulp", "miswire5") else "(0.9, 0.999)"
    return b, str(s.get("weight_decay")), str(s.get("plateau_until"))


def main():
    rows = []
    for a in ARMS:
        p = os.path.join(HERE, "traj20_%s.json" % a)
        if os.path.exists(p):
            rows.append((a, json.load(open(p))))
    print("=" * 112)
    print("of3t-wirefix -- PROTOCOL 7 re-run with the wiring fixes on.")
    print("4,147 of 4,147 declared parameters, 368,293,788 elements, warmup 20, "
          "4 samples/step, rho 0.005, seed 20260920.")
    print("d_1 is reported as the two NORMS, not as their ratio: a reference d_1 of exactly 0")
    print("makes the ratio 1e+30 by construction, which is an artifact of the denominator.")
    print("=" * 112)
    hdr = ("%-9s %-13s %5s %8s | %10s %10s %11s %11s %10s %10s | %8s %6s"
           % ("arm", "betas", "wd", "plateau", "|d_1|ours", "|d_1|ref", "d_2", "d_20",
              "med20", "d20/floor", "exp2-20", "r2"))
    print(hdr)
    print("-" * len(hdr))
    for a, d in rows:
        b, wd, pl = betas_of(a, d)
        g = d["growth_k2_20"]
        e = g["exponent"]
        r20 = d["per_step"][19]
        print("%-9s %-13s %5s %8s | %10.3e %10.3e %11.4e %11.4e %10.3e %10.3e | %8s %6s"
              % (a, b, wd, pl,
                 d["d1"]["ours_norm"], d["d1"]["theirs_norm"],
                 d["per_step"][1]["rel_d"], r20["rel_d"],
                 r20["median_per_tensor"], r20["rel_over_floor"],
                 ("%+.3f" % e) if e is not None else "--",
                 ("%.3f" % g["r2"]) if e is not None else "--"))
    for a, d in rows:
        print()
        print("--- %s: every rung, k = 1..20 %s" % (a, "-" * max(0, 62 - len(a))))
        print("    %s" % WHAT.get(a, ""))
        print("%3s %12s %11s %10s %11s %11s %11s %9s %8s  %s"
              % ("k", "d_k", "fp32 floor", "d_k/floor", "median", "p90", "worst",
                 "bit-ident", "zero-ref", "worst tensor"))
        for r in d["per_step"]:
            tot = r["tensors_scored"] + r["tensors_zero_ref"]
            print("%3d %12.6e %11.3e %10.3e %11.3e %11.3e %11.3e %4d/%-4d %8d  %s"
                  % (r["k"], r["rel_d"], r["fp32_differencing_floor"], r["rel_over_floor"],
                     r["median_per_tensor"], r["p90_per_tensor"], r["worst_per_tensor"],
                     r["tensors_bit_identical"], tot, r["tensors_zero_ref"],
                     r["worst_tensor"] or "-"))
        fb = d["feedback_share_of_drive_norm"]
        print("    feedback share of the drive norm: %s"
              % ", ".join("k=%s %.4f" % (k, v)
                          for k, v in sorted(fb.items(), key=lambda x: int(x[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
