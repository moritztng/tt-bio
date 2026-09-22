#!/usr/bin/env python3
"""of3t-twoside step 4: upstream-injected-bf16's per-block cotangent error, beside ours.

of3t-cotcoh measured OUR cotangent error at 0.1100 after one block on this frame and there has
never been an upstream counterpart to it. This produces one, at the same two leaf families, the
same 56-real-row masking and the same walk-back index, from two `perf/of3t_cotcoh/refcot.py`
passes that differ ONLY in `--policy`.

WHY THE PAIR AND NOT of3t-cotcoh's OWN f64 ARM. This row's step-1 control showed
`cot_model_n384.pt` is not the cotangent that drove `grads_f64_043.pt`'s trunk (D242), so no
reading on this frame may be taken against the model bundle. Both passes here share that one
cotangent, so their difference is bf16 against float64 inside ONE frame and survives D242
untouched. It is also why of3t-cotcoh's own premise -- "the cotangent entering block 47's
backward IS the reference's own float64 cotangent, exact by construction" -- does not hold, and
that is reported rather than used.

Depth is `47 - b`: block 47 is the first block the backward traverses, so depth 1 is "after one
block", which is the index the 0.1100 is quoted at.

    curvescore.py --f64 cot_curve_f64.pt --bf16 cot_curve_bf16.pt --out CURVE.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

FAM = {"A": "attn_pair_bias.layer_norm_a",
       "B": "pair_stack.pair_transition.layer_norm"}


def spearman(xs, ys):
    """Rank correlation, the statistic of3t-cotcoh's grading is written against."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True, type=Path)
    ap.add_argument("--bf16", required=True, type=Path)
    ap.add_argument("--ours-after-one-block", type=float, default=0.1100,
                    help="of3t-cotcoh's published reading for OUR arm at depth 1, quoted for "
                         "comparison and not recomputed here")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    ref = torch.load(a.f64, map_location="cpu", weights_only=False)
    arm = torch.load(a.bf16, map_location="cpu", weights_only=False)
    if ref["real_rows"] != arm["real_rows"]:
        raise SystemExit(f"STOP: real_rows differ, {ref['real_rows']} against {arm['real_rows']}")

    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": os.uname().nodename, "row": "of3t-twoside", "defect": "D241",
        "device_involved": False,
        "real_rows": ref["real_rows"], "families": FAM,
        "reference_named": "upstream 0.4.3 float64 on the INJECTED boundary and cotangent "
                           "(policy f64), not the model bundle -- see D242",
        "arm_named": "upstream 0.4.3 bf16 autocast on the SAME injected pair (policy bf16auto)",
        "inputs": {"f64": str(a.f64), "bf16": str(a.bf16)},
        "per_family": {},
        "ours_after_one_block_of3t_cotcoh": a.ours_after_one_block,
    }

    for fam in FAM:
        rows = []
        for b in range(48):
            k = f"{fam}:{b}"
            if k not in ref["sites"] or k not in arm["sites"]:
                continue
            g0 = ref["sites"][k]["g"].to(torch.float64)
            g1 = arm["sites"][k]["g"].to(torch.float64)
            n0 = float(torch.linalg.vector_norm(g0))
            d = float(torch.linalg.vector_norm(g1 - g0))
            rows.append({"block": b, "depth_blocks_traversed": 47 - b,
                         "rel_l2": (d / n0) if n0 > 0 else None,
                         "ref_norm": n0,
                         "norm_ratio": (float(torch.linalg.vector_norm(g1)) / n0)
                                       if n0 > 0 else None})
        rows.sort(key=lambda r: r["depth_blocks_traversed"])
        vals = [(r["depth_blocks_traversed"], r["rel_l2"]) for r in rows
                if r["rel_l2"] is not None]
        d1 = next((v for k, v in vals if k == 1), None)
        mx = max((v for _, v in vals), default=None)
        out["per_family"][fam] = {
            "site": FAM[fam], "n_blocks": len(rows),
            "after_one_block_depth1_block47": d1,
            "max_over_depth": mx,
            "at_depth_48_block0": next((v for k, v in vals if k == 48), None),
            "spearman_rel_l2_against_depth": spearman([k for k, _ in vals],
                                                      [v for _, v in vals]),
            "theirs_over_ours_at_depth_1": (d1 / a.ours_after_one_block) if d1 else None,
            "curve": rows,
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for fam, v in out["per_family"].items():
        print(f"family {fam} ({v['site']}): depth1 {v['after_one_block_depth1_block47']!r} "
              f"max {v['max_over_depth']!r} depth48 {v['at_depth_48_block0']!r} "
              f"spearman {v['spearman_rel_l2_against_depth']!r}")
        print(f"   theirs/ours at depth 1 = {v['theirs_over_ours_at_depth_1']!r} "
              f"(ours {a.ours_after_one_block} from of3t-cotcoh)")
    print("wrote " + str(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
