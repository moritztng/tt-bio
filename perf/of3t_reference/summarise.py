#!/usr/bin/env python3
"""Reduce a reference bundle to the numbers the state doc and the equivalence row quote.

Reads trajectory.json (written by gpu_reference.py --mode bundle) and reports:
  - the LR actually used at each step, against the independent lr_reference.json table;
  - |d_k| = ||w_k - w_0||_2 and its growth in k, which PROTOCOL 7 makes the trajectory's bar;
  - the clip coefficient their grad_manager applied, per step;
  - the gradient-presence pattern: how many parameters had NO gradient, per step;
  - the loss terms that actually fired, and the ones that did not, which is a coverage claim and
    is reported as NOT COVERED rather than omitted.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True, type=Path)
    ap.add_argument("--lr-reference", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    t = json.loads(args.trajectory.read_text())
    steps = t["steps"]
    lr_ref = json.loads(args.lr_reference.read_text())["lr"] if args.lr_reference else None

    rows = []
    for s in steps:
        k = s["step"]
        lr_used = s.get("lr_used")
        if lr_used is None:
            # Older runs recorded the post-scheduler-step value, which is the NEXT step's LR.
            prev = next((x for x in steps if x["step"] == k - 1), None)
            lr_used = 0.0 if k == 1 else prev["lr"]
        rows.append({
            "step": k,
            "pdb_id": s["pdb_id"],
            "n_tokens": s["n_tokens"],
            "lr_used": lr_used,
            "lr_expected": (float(lr_ref[str(k - 1)]) if lr_ref and str(k - 1) in lr_ref else None),
            "loss": s["loss_terms"].get("loss"),
            "unclipped_grad_norm": s["unclipped_grad_norm"],
            "clip_coef": s["clip_coef"],
            "n_grad_absent": s["n_grad_absent"],
            "n_disabled": s["n_disabled"],
            "delta_global_l2": s["delta_global_l2"],
        })

    lr_ok = all(r["lr_expected"] is None or r["lr_used"] == r["lr_expected"] for r in rows)

    # Growth law. PROTOCOL 7: linear or sub-linear in k passes; super-linear is a different
    # update rule. Fit log|d_k| against log k over the steps where |d_k| > 0.
    pts = [(r["step"], r["delta_global_l2"]) for r in rows if r["delta_global_l2"] > 0]
    exponent = None
    if len(pts) >= 3:
        import math

        n = len(pts)
        xs = [math.log(k) for k, _ in pts]
        ys = [math.log(v) for _, v in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        exponent = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None

    fired, silent = {}, []
    for s in steps:
        for name, value in s["loss_terms"].items():
            if value not in (0.0, None):
                fired[name] = fired.get(name, 0) + 1
    all_terms = sorted({k for s in steps for k in s["loss_terms"]})
    silent = [k for k in all_terms if k not in fired]

    summary = {
        "n_steps": len(rows),
        "lr_matches_reference_table": lr_ok,
        "first_nonzero_delta_step": next((r["step"] for r in rows if r["delta_global_l2"] > 0), None),
        "delta_growth_exponent_in_k": exponent,
        "clip_engaged_steps": [r["step"] for r in rows if r["clip_coef"] < 1.0],
        "max_unclipped_grad_norm": max(r["unclipped_grad_norm"] for r in rows),
        "steps_with_absent_gradients": [r["step"] for r in rows if r["n_grad_absent"] > 0],
        "loss_terms_fired": fired,
        "loss_terms_never_nonzero": silent,
        "rows": rows,
    }
    body = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.write_text(body)

    print(f"{'k':>3} {'pdb':>6} {'tok':>4} {'lr_used':>11} {'lr_exp':>11} "
          f"{'loss':>10} {'|g|':>10} {'clip':>7} {'absent':>6} {'|d_k|':>12}")
    for r in rows:
        print(f"{r['step']:>3} {str(r['pdb_id'][0]):>6} {r['n_tokens']:>4} "
              f"{r['lr_used']:>11.4g} {(r['lr_expected'] if r['lr_expected'] is not None else float('nan')):>11.4g} "
              f"{r['loss']:>10.5g} {r['unclipped_grad_norm']:>10.5g} {r['clip_coef']:>7.4g} "
              f"{r['n_grad_absent']:>6} {r['delta_global_l2']:>12.6g}")
    print()
    print(f"lr matches the independent reference table: {lr_ok}")
    print(f"first step with a non-zero weight delta:    {summary['first_nonzero_delta_step']}")
    print(f"growth exponent of |d_k| in k:              {exponent:.4g}" if exponent else "")
    print(f"steps where clipping engaged:               {summary['clip_engaged_steps'] or 'none'}")
    print(f"max unclipped per-sample grad norm:         {summary['max_unclipped_grad_norm']:.6g}")
    print(f"steps with any absent gradient:             {summary['steps_with_absent_gradients'] or 'none'}")
    print(f"loss terms that never fired:                {silent or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
