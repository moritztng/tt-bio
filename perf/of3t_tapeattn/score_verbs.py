#!/usr/bin/env python3
"""Deliverable 2: which tape verb carries the per-block factor, by call census and by each
verb's own float64-differenced backward error.

Two tables per width. `injection` is the sum over the 48 blocks of a verb's own absolute
backward error, `||dev_bw - f64_bw||` on the SAME operands -- an absolute error, not a share of a
moving denominator, because a share moves when its denominator collapses and a difference of
absolute errors is what locates a carrier. `growth` is that sum at 384 over the same sum at 64.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load(p):
    return json.load(open(p))


def table(side, ref="own"):
    per = defaultdict(lambda: {"n": 0, "abs_err": 0.0, "ref_norm": 0.0, "dev_norm": 0.0,
                               "worst_rel": 0.0, "worst_at": None, "cos_min": 2.0})
    for n in side["nodes"]:
        for p in n["parents"]:
            e = p.get(ref)
            if not e or e.get("abs_err") is None:
                continue
            # Key by the verb AND its output shape with the padded width folded out, so the
            # two matmuls in the chain -- q@k^T and probs@v -- do not aggregate into one row,
            # and the key still joins across the two widths.
            N = n["out_shape"][2] if len(n["out_shape"]) > 2 else None
            shp = ",".join("N" if d == N else str(d) for d in n["out_shape"])
            k = f"{n['verb']}[{shp}].{p['i']}"
            d = per[k]
            d["n"] += 1
            d["abs_err"] = math.hypot(d["abs_err"], e["abs_err"])
            d["ref_norm"] = math.hypot(d["ref_norm"], e["ref_norm"])
            d["dev_norm"] = math.hypot(d["dev_norm"], e["dev_norm"])
            if e.get("rel_l2") is not None and e["rel_l2"] > d["worst_rel"]:
                d["worst_rel"] = e["rel_l2"]
                d["worst_at"] = {"sm_ord": n["sm_ord"], "shape": p["shape"]}
            if e.get("cos") is not None:
                d["cos_min"] = min(d["cos_min"], e["cos"])
    for k, d in per.items():
        d["rel_l2_aggregate"] = (d["abs_err"] / d["ref_norm"]) if d["ref_norm"] else None
    return dict(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", action="append", required=True, metavar="W=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sides = {}
    for s in a.side:
        w, _, p = s.partition("=")
        sides[w] = load(p)

    out = {"what": __doc__.strip().splitlines()[0], "widths": {}}
    for w, side in sides.items():
        out["widths"][w] = {
            "census_backward": {k: v for k, v in side["backward"].items()
                                if k.split(" ", 1)[1].startswith("[1, 16,")},
            "captured_nodes": side["captured_nodes"],
            "capture_errors": side["capture_errors"],
            "injection_own": table(side, "own"),
            "injection_true": table(side, "true"),
        }
    ws = sorted(sides, key=lambda x: int(x))
    if len(ws) == 2:
        lo, hi = ws
        for ref in ("own", "true"):
            g = {}
            A = out["widths"][lo][f"injection_{ref}"]
            B = out["widths"][hi][f"injection_{ref}"]
            tot_a = math.sqrt(sum(v["abs_err"] ** 2 for v in A.values()))
            tot_b = math.sqrt(sum(v["abs_err"] ** 2 for v in B.values()))
            for k in sorted(set(A) | set(B)):
                a_, b_ = A.get(k), B.get(k)
                g[k] = {
                    f"abs_err_{lo}": a_["abs_err"] if a_ else None,
                    f"abs_err_{hi}": b_["abs_err"] if b_ else None,
                    "growth": (b_["abs_err"] / a_["abs_err"])
                              if a_ and b_ and a_["abs_err"] else None,
                    f"share_{lo}": (a_["abs_err"] ** 2 / tot_a ** 2) if a_ and tot_a else None,
                    f"share_{hi}": (b_["abs_err"] ** 2 / tot_b ** 2) if b_ and tot_b else None,
                    f"rel_l2_{lo}": a_["rel_l2_aggregate"] if a_ else None,
                    f"rel_l2_{hi}": b_["rel_l2_aggregate"] if b_ else None,
                }
            out[f"growth_{ref}"] = {"total_abs_err": {lo: tot_a, hi: tot_b,
                                                      "growth": tot_b / tot_a if tot_a else None},
                                    "by_verb": g}
    json.dump(out, open(a.out, "w"), indent=1)

    for ref in ("own", "true"):
        key = f"growth_{ref}"
        if key not in out:
            continue
        print(f"\n=== {ref} reference: per-verb injected absolute error, {lo} -> {hi} ===")
        print(f"{'verb[operand]':>28} {'abs@'+lo:>12} {'abs@'+hi:>12} {'growth':>8} "
              f"{'share@'+hi:>9} {'rel@'+lo:>10} {'rel@'+hi:>10}")
        rows = sorted(out[key]["by_verb"].items(),
                      key=lambda kv: -(kv[1][f"abs_err_{hi}"] or 0))
        for k, v in rows:
            def f(x, w=12, p=4):
                return ("%*.*e" % (w, p, x)) if isinstance(x, float) else " " * (w - 1) + "-"
            print(f"{k:>28} {f(v['abs_err_'+lo])} {f(v['abs_err_'+hi])} "
                  f"{(('%8.3f' % v['growth']) if v['growth'] else '       -')} "
                  f"{(('%9.5f' % v['share_'+hi]) if v['share_'+hi] is not None else '        -')} "
                  f"{f(v['rel_l2_'+lo],10)} {f(v['rel_l2_'+hi],10)}")
        t = out[key]["total_abs_err"]
        print(f"{'TOTAL':>28} {t[lo]:12.4e} {t[hi]:12.4e} {t['growth']:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
