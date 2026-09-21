#!/usr/bin/env python3
"""The 24 diffusion-transformer instances of the leaf, pre-repair against the RENORM arm.

AMENDMENT 2's table is `perf/of3t_adaln/device_gradient_real043_pertensor.json`, which records
no arm and no flag and so predates `TT_BIO_SOFTMAX_BW_RENORM` entirely. This puts the same 24
rows from the renorm arm beside it, with each block's share of the LEAF's error mass, so
"did the repair move the 99.5 % concentration" is a number.

error mass of a block = ||ours - ref||^2 = (rel_l2 * ref_norm)^2, which is what the campaign
ranks by and is independent of the reference norm.
"""
from __future__ import annotations

import argparse
import json
import re

LEAF = "attention_pair_bias.layer_norm_a.layer_norm_s.weight"
BLK = re.compile(r"diffusion_transformer\.blocks\.(\d+)\.")


def rows_of(path, key="per_tensor"):
    d = json.load(open(path))
    rows = d[key] if key in d else d
    out = {}
    for r in rows:
        t = r["tensor"]
        if not t.endswith(LEAF):
            continue
        m = BLK.search(t)
        if not m:
            continue
        out[int(m.group(1))] = r
    return out, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pre, dpre = rows_of(a.pre)
    post, dpost = rows_of(a.post)

    def emass(r):
        return (r["rel_l2"] * r["ref_norm"]) ** 2

    tp, tq = sum(emass(r) for r in pre.values()), sum(emass(r) for r in post.values())
    healthy = lambda r: r["cos"] >= 0.95 and 0.5 <= r["norm_ratio"] <= 2.0

    tbl = {}
    for b in sorted(set(pre) | set(post)):
        p, q = pre.get(b), post.get(b)
        tbl[b] = {
            "pre": None if not p else {
                "rel_l2": p["rel_l2"], "norm_ratio": p["norm_ratio"], "cos": p["cos"],
                "ref_norm": p["ref_norm"], "device_norm": p["device_norm"],
                "share_of_leaf_error_mass": emass(p) / tp, "healthy": healthy(p)},
            "post_renorm": None if not q else {
                "rel_l2": q["rel_l2"], "norm_ratio": q["norm_ratio"], "cos": q["cos"],
                "ref_norm": q["ref_norm"], "device_norm": q["device_norm"],
                "share_of_leaf_error_mass": emass(q) / tq, "healthy": healthy(q)},
        }
        if p and q:
            tbl[b]["rel_post_over_pre"] = q["rel_l2"] / p["rel_l2"] if p["rel_l2"] else None
            tbl[b]["error_mass_post_over_pre"] = emass(q) / emass(p) if emass(p) else None

    FOUR = [5, 7, 8, 12]
    ANTI = [0, 3, 6]
    rep = {
        "what": __doc__.strip().splitlines()[0],
        "leaf": LEAF, "pre": a.pre, "post": a.post,
        "pre_arm": "no arm and no flag recorded -- predates TT_BIO_SOFTMAX_BW_RENORM",
        "post_arm": "--softmax-bw-renorm, --structs all",
        "pre_compared": dpre.get("compared"), "post_compared": dpost.get("compared"),
        "leaf_error_mass_total": {"pre": tp, "post_renorm": tq,
                                  "post_over_pre": (tq / tp) if tp else None},
        "the_four_5_7_8_12_share_of_the_leaf": {
            "pre": sum(emass(pre[b]) for b in FOUR if b in pre) / tp,
            "post_renorm": sum(emass(post[b]) for b in FOUR if b in post) / tq},
        "healthy_blocks": {
            "pre": sorted(b for b, r in pre.items() if healthy(r)),
            "post_renorm": sorted(b for b, r in post.items() if healthy(r))},
        "anti_correlated_0_3_6_cos": {
            "pre": {b: pre[b]["cos"] for b in ANTI if b in pre},
            "post_renorm": {b: post[b]["cos"] for b in ANTI if b in post}},
        "by_block": tbl,
    }
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)

    print(f"{'blk':>3} | {'pre r':>7} {'pre cos':>8} {'pre em%':>8} | "
          f"{'post r':>7} {'post cos':>8} {'post em%':>8} | {'rel x':>7}")
    for b, v in tbl.items():
        p, q = v["pre"], v["post_renorm"]
        print(f"{b:>3} | {p['norm_ratio']:7.3f} {p['cos']:+8.3f} "
              f"{p['share_of_leaf_error_mass']*100:8.4f} | "
              f"{q['norm_ratio']:7.3f} {q['cos']:+8.3f} "
              f"{q['share_of_leaf_error_mass']*100:8.4f} | "
              f"{v.get('rel_post_over_pre') or float('nan'):7.3f}")
    print(json.dumps({k: rep[k] for k in ("leaf_error_mass_total",
                                          "the_four_5_7_8_12_share_of_the_leaf",
                                          "healthy_blocks",
                                          "anti_correlated_0_3_6_cos")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
