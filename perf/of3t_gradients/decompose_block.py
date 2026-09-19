#!/usr/bin/env python3
"""Group a block-scope instrument A run by sub-module, and compare two arms of it.

`of3t-orchestrator` decomposed the first block-0 run this way and got a mechanism-shaped result:
every group that touches the pair track or attention fails while `single_transition`, the one
with neither, passes. That grouping is worth being a script rather than a one-off, because the
question it raises -- which sub-module carries the failure -- is the only question that
distinguishes a flag from a wiring defect.

A14: a tensor whose reference norm is ~0 is excluded, because a relative L2 against a zero
denominator is a property of the denominator. It is reported by name rather than dropped.
A15: every figure carries the share of the squared gradient norm its set holds, read from
`reach_by_norm.json`, so a median is never quoted as if it were a statement about the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

OUT = "perf/of3t_gradients"
BAR, MEDIAN_BAR = 5.0e-02, 2.0e-02
ZERO_REF = 1e-15


def group_of(key: str) -> str:
    k = key.split("pair_stack.", 1)[-1]
    for g in ("tri_att_end", "tri_att_start", "tri_mul_in", "tri_mul_out", "pair_transition",
              "attn_pair_bias", "single_transition"):
        if k.startswith(g) or key.startswith(g):
            return g
    return "other"


def load(tag):
    with open(os.path.join(OUT, f"instrument_a_bundle_{tag}.json")) as f:
        return json.load(f)


def summarise(rep):
    import numpy as np
    rows = rep["per_parameter"]
    excluded = [r for r in rows if r["ref_norm"] <= ZERO_REF or r["ref_norm"] < 1e-12]
    kept = [r for r in rows if r not in excluded]
    groups = {}
    for r in kept:
        groups.setdefault(group_of(r["key"]), []).append(r)
    out = {}
    for g, rs in groups.items():
        v = sorted(x["rel_l2"] for x in rs)
        w = max(rs, key=lambda x: x["rel_l2"])
        out[g] = {"n": len(rs), "median": float(np.median(v)), "worst": w["rel_l2"],
                  "worst_tensor": w["their_tensor"],
                  "over_bar": sum(1 for x in v if x > BAR)}
    allv = sorted(x["rel_l2"] for x in kept)
    # A15 inside the block. "8 of 51 over the bar" and "the over-bar tensors hold 0.3 % of this
    # block's gradient" are different claims, and only the second one says whether the miss
    # matters. It is the same rule the campaign applies at model scope, applied one level down,
    # and it is the honest answer to a relative-L2 bar on a tensor whose reference gradient is
    # four to eight orders below its neighbours': a bf16 forward cannot make that ratio small,
    # and no porting fix will.
    blk_sq = sum(r["ref_norm"] ** 2 for r in kept) or 1.0
    over = [r for r in kept if r["rel_l2"] > BAR]
    for g, rs in groups.items():
        out[g]["norm_share_of_block"] = sum(r["ref_norm"] ** 2 for r in rs) / blk_sq
        out[g]["over_bar_norm_share_of_block"] = sum(
            r["ref_norm"] ** 2 for r in rs if r["rel_l2"] > BAR) / blk_sq
    return {"groups": dict(sorted(out.items(), key=lambda x: -x[1]["median"])),
            "over_bar_norm_share_of_block": sum(r["ref_norm"] ** 2 for r in over) / blk_sq,
            "block_squared_norm_of_compared_set": blk_sq,
            "n_compared": len(kept), "n_excluded_zero_ref": len(excluded),
            "excluded": [{"tensor": r["their_tensor"], "ref_norm": r["ref_norm"]}
                         for r in excluded],
            "median": float(np.median(allv)) if allv else None,
            "worst": allv[-1] if allv else None,
            "over_bar": sum(1 for x in allv if x > BAR),
            "median_inside_bar": bool(allv and float(np.median(allv)) <= MEDIAN_BAR)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, metavar="LABEL=TAG")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    a.out = a.out or os.path.join(OUT, f"decompose_block{a.block}.json")
    reach = json.load(open(os.path.join(OUT, "reach_by_norm.json")))
    share = reach["per_pairformer_block"][str(a.block)]["compared_norm_share"]
    rep = {"instrument": "block-scope instrument A, grouped by sub-module",
           "block": a.block,
           "bars": {"per_tensor": BAR, "median": MEDIAN_BAR},
           "a14_zero_reference_rule": f"tensors with ref_norm below {ZERO_REF} are excluded and "
                                      f"named; a relative L2 on a zero denominator measures the "
                                      f"denominator",
           "a15_norm_share_of_compared_set": share,
           "a15_note": f"every median below is over a set holding {100*share:.2f} % of the "
                       f"reference gradient's squared norm",
           "arms": {}}
    for spec in a.arms:
        label, tag = spec.split("=", 1)
        r = load(tag)
        s = summarise(r)
        s["tag"] = tag
        s["config"] = r.get("shipped_config")
        s["forward_z_masked"] = r["forward_rel"]["z_masked"]
        s["forward_s_masked"] = r["forward_rel"]["s_masked"]
        s["crop"] = r.get("crop") or r.get("probe", {}).get("tokens")
        rep["arms"][label] = s

    json.dump(rep, open(a.out, "w"), indent=1)
    for label, s in rep["arms"].items():
        print(f"\n=== {label}  ({s['tag']})  median {s['median']:.4f}  "
              f"over bar {s['over_bar']}/{s['n_compared']}  "
              f"fwd z {s['forward_z_masked']:.3e}")
        print(f"   over-bar tensors hold {100*s['over_bar_norm_share_of_block']:.3f} % of this "
              f"block's compared gradient norm^2")
        for g, d in s["groups"].items():
            print(f"   {g:18s} n={d['n']:3d}  median {d['median']:.4f}  worst {d['worst']:.4f}"
                  f"  {d['over_bar']}/{d['n']} over  "
                  f"block-norm {100*d['norm_share_of_block']:5.2f} %  "
                  f"over-bar-norm {100*d['over_bar_norm_share_of_block']:5.2f} %")
        if s["excluded"]:
            print(f"   excluded (A14): " + ", ".join(
                f"{x['tensor'].split('.',3)[-1]} ref_norm {x['ref_norm']:.2e}"
                for x in s["excluded"]))
    print(f"\ncompared set holds {100*share:.2f} % of the squared gradient norm (A15)")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
