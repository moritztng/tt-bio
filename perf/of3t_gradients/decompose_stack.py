#!/usr/bin/env python3
"""Decompose a STACK-scope instrument A run by sub-module and by depth.

`decompose_block.py` answers "which sub-module carries the failure" inside one block. Over 48
blocks there is a second question it cannot ask, and it is the one that separates a bf16 floor
from a wiring defect: does the disagreement GROW with depth? A per-block error that composes
would rise monotonically through the stack; a per-block error that is local would not. Neither
shape is visible from a single block, and the campaign has been reading single blocks.

A14: a tensor whose reference gradient norm is below the declared floor is excluded from the
statistics and reported by name and count. A relative L2 on a near-zero denominator is a
statement about the denominator -- one tensor in the block-0 run produced 1.14e13 that way.
A15: every figure carries the share of the squared gradient norm its set holds, computed from
the reference's own entries rather than from a tensor count.
"""
from __future__ import annotations

import argparse
import json
import os
import re

OUT = "perf/of3t_gradients"
BAR, MEDIAN_BAR = 5.0e-02, 2.0e-02
#: A14's denominator floor, declared rather than discovered. Below this a reference gradient is
#: numerically zero and the ratio it sits under is noise over noise.
REF_NORM_FLOOR = 1e-12
GROUPS = ("tri_att_end", "tri_att_start", "tri_mul_in", "tri_mul_out", "pair_transition",
          "attn_pair_bias", "single_transition")


def group_of(key: str) -> str:
    k = key.split("pair_stack.", 1)[-1]
    for g in GROUPS:
        if k.startswith(g) or key.split(".", 2)[-1].startswith(g):
            return g
    return "other"


def block_of(name: str) -> int:
    m = re.search(r"\.blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def stats(rows, total_sq):
    import numpy as np
    if not rows:
        return None
    v = sorted(r["rel_l2"] for r in rows)
    sq = sum(r["ref_norm"] ** 2 for r in rows)
    over = [r for r in rows if r["rel_l2"] > BAR]
    return {"n": len(rows), "median": float(np.median(v)), "worst": v[-1], "best": v[0],
            "worst_tensor": max(rows, key=lambda r: r["rel_l2"])["their_tensor"],
            "over_bar": len(over),
            "share_of_compared_sq_norm_pct": 100.0 * sq / total_sq if total_sq else None,
            "over_bar_share_of_group_sq_pct":
                100.0 * sum(r["ref_norm"] ** 2 for r in over) / sq if sq else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = json.load(open(os.path.join(OUT, f"instrument_a_bundle_{a.tag}.json")))
    rows = rep["per_parameter"]

    excluded = [r for r in rows if r["ref_norm"] < REF_NORM_FLOOR]
    kept = [r for r in rows if r["ref_norm"] >= REF_NORM_FLOOR]
    total_sq = sum(r["ref_norm"] ** 2 for r in kept)

    out = {
        "source": f"instrument_a_bundle_{a.tag}.json",
        "scope": rep.get("scope_note"),
        "a14_denominator_floor": {
            "floor_on_reference_norm": REF_NORM_FLOOR,
            "excluded": len(excluded), "kept": len(kept),
            "excluded_tensors": [r["their_tensor"] for r in excluded][:40],
            "why": "a relative L2 whose denominator is numerically zero measures the "
                   "denominator; the block-0 run read 1.14e13 on one such tensor",
        },
        "overall": stats(kept, total_sq),
        "by_group": {g: stats([r for r in kept if group_of(r["key"]) == g], total_sq)
                     for g in GROUPS},
        "by_block": {},
        "depth_trend": None,
    }
    blocks = sorted({block_of(r["their_tensor"]) for r in kept})
    for b in blocks:
        out["by_block"][str(b)] = stats([r for r in kept if block_of(r["their_tensor"]) == b],
                                        total_sq)

    # Does the disagreement grow with depth? Fitted on the per-block median, which is the only
    # statistic robust to the near-zero family concentrating in particular blocks. Reported as
    # a slope with its endpoints, never as a single correlation: a flat trend and a rising one
    # are different findings and a coefficient alone cannot be read back into either.
    import numpy as np
    xs = np.array([b for b in blocks if out["by_block"][str(b)]], dtype=float)
    ys = np.array([out["by_block"][str(int(b))]["median"] for b in xs], dtype=float)
    if len(xs) > 2:
        sl, ic = np.polyfit(xs, ys, 1)
        out["depth_trend"] = {
            "fit": "median_rel_l2 = slope * block_index + intercept",
            "slope": float(sl), "intercept": float(ic),
            "first_block": {"index": int(xs[0]), "median": float(ys[0])},
            "last_block": {"index": int(xs[-1]), "median": float(ys[-1])},
            "ratio_last_over_first": float(ys[-1] / ys[0]) if ys[0] else None,
            "reading": (
                "GROWS with forward depth -- a per-block error that composes forward"
                if ys[-1] > 1.5 * ys[0] else
                "FALLS with forward depth, i.e. GROWS with BACKWARD depth -- block 47's "
                "gradient is set mostly by the captured cotangent, which is theirs exactly, "
                "while block 0's has to come back through all 47 of our blocks. This is the "
                "signature of a backward driven at a point the forward has already moved."
                if ys[0] > 1.5 * ys[-1] else
                "flat in depth -- neither the forward nor the backward direction accumulates "
                "over the stack"),
        }

    path = a.out or os.path.join(OUT, f"decompose_{a.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    o = out["overall"]
    print(f"kept {o['n']}, A14-excluded {len(excluded)}")
    print(f"overall median {o['median']:.4f} worst {o['worst']:.4f} "
          f"over bar {o['over_bar']}/{o['n']}")
    for g in GROUPS:
        st = out["by_group"][g]
        if st:
            print(f"  {g:18s} n={st['n']:4d} median {st['median']:.4f} "
                  f"worst {st['worst']:.4f} over {st['over_bar']}/{st['n']} "
                  f"sq% {st['share_of_compared_sq_norm_pct']:.2f}")
    if out["depth_trend"]:
        d = out["depth_trend"]
        print(f"  depth: block {d['first_block']['index']} {d['first_block']['median']:.4f} -> "
              f"block {d['last_block']['index']} {d['last_block']['median']:.4f}, "
              f"slope {d['slope']:.3e} -- {d['reading']}")
    print("->", path)


if __name__ == "__main__":
    main()
