#!/usr/bin/env python3
"""Where the frame-matched trunk's error mass is, so the campaign has an aim and not just a factor.

`model_scope.py` writes one row per tensor into its sidecar. This groups those rows three ways --
by block, by leaf module and by parameter kind -- always by SHARE OF THE ERROR MASS and never by
relative error alone, because a share moves when its denominator collapses and a leaf's rel_l2
says nothing about whether fixing it would move the headline (memory: a share moves when the
denominator collapses; effort spent is not evidence of share).

Every group carries what the same group reads in UPSTREAM's own bf16 step, so a group we are
merely as bad as upstream at is visibly not ours to fix.

  attrib.py --sidecar <dir> --out <report>.json
"""
from __future__ import annotations

import argparse
import json
import re
import socket
from collections import defaultdict
from pathlib import Path

SEC = "pairformer_stack."


def load(sidecar, label):
    rows = json.loads((Path(sidecar) / f"{label}.json").read_text())
    return [r for r in rows if r["param"].startswith(SEC)]


def group(rows, key):
    g = defaultdict(lambda: {"n": 0, "ref_sq": 0.0, "err_sq": 0.0, "worst_rel": 0.0,
                             "worst": None})
    for r in rows:
        k = key(r["param"])
        rn, dn, rel = r.get("ref_norm"), r.get("diff_norm"), r.get("rel_l2")
        if rn is None or dn is None:
            continue
        d = g[k]
        d["n"] += 1
        d["ref_sq"] += rn ** 2
        d["err_sq"] += dn ** 2          # the scorer's own ||ours - ref||, not rebuilt from rel_l2
        if rel is not None and rel > d["worst_rel"]:
            d["worst_rel"], d["worst"] = rel, r["param"]
    return g


def table(ours, floor, key, top):
    go, gf = group(ours, key), group(floor, key)
    tot_err = sum(v["err_sq"] for v in go.values())
    tot_ref = sum(v["ref_sq"] for v in go.values())
    out = []
    for k, v in go.items():
        f = gf.get(k, {"err_sq": 0.0, "ref_sq": v["ref_sq"]})
        out.append({
            "group": k, "n": v["n"],
            "pct_of_trunk_error_mass": 100.0 * v["err_sq"] / tot_err if tot_err else None,
            "pct_of_trunk_reference_mass": 100.0 * v["ref_sq"] / tot_ref if tot_ref else None,
            "rel_l2": (v["err_sq"] / v["ref_sq"]) ** 0.5 if v["ref_sq"] else None,
            "upstreams_own_rel_l2_here": ((f["err_sq"] / f["ref_sq"]) ** 0.5
                                          if f.get("ref_sq") else None),
            "over_upstream": (((v["err_sq"] / v["ref_sq"]) ** 0.5)
                              / ((f["err_sq"] / f["ref_sq"]) ** 0.5)
                              if v["ref_sq"] and f.get("ref_sq") and f["err_sq"] else None),
            "worst_tensor": v["worst"], "worst_rel_l2": v["worst_rel"],
        })
    out.sort(key=lambda r: -(r["pct_of_trunk_error_mass"] or 0))
    return out[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    ours = load(a.sidecar, "renorm_vs_UPSTREAM_BF16")
    floor = load(a.sidecar, "UPSTREAM_BF16_vs_FLOAT64")
    f64 = load(a.sidecar, "renorm_vs_FLOAT64")

    def blk(n):
        m = re.match(r"pairformer_stack\.blocks\.(\d+)\.", n)
        return f"block {int(m.group(1)):02d}" if m else "other"

    def leaf(n):
        m = re.match(r"pairformer_stack\.blocks\.\d+\.(.*)\.[^.]+$", n)
        return m.group(1) if m else "other"

    def kind(n):
        return n.rsplit(".", 1)[-1]

    def sub(n):
        m = re.match(r"pairformer_stack\.blocks\.\d+\.([^.]+)", n)
        return m.group(1) if m else "other"

    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(),
        "sidecar": a.sidecar,
        "n_trunk_tensors": len(ours),
        "denominator": "share of the TRUNK SECTION's error mass against upstream's own bf16 "
                       "step, which is the quantity the failing clause is built from",
        "by_subtree": table(ours, floor, sub, 12),
        "by_leaf_module": table(ours, floor, leaf, a.top),
        "by_parameter_kind": table(ours, floor, kind, a.top),
        "by_block": table(ours, floor, blk, a.top),
        "against_float64": {
            "by_subtree": table(f64, floor, sub, 12),
        },
    }
    txt = json.dumps(out, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt)
    for name in ("by_subtree", "by_leaf_module", "by_parameter_kind", "by_block"):
        print(f"--- {name}")
        for r in out[name]:
            print(f"  {r['group']:<44s} {r['pct_of_trunk_error_mass']:7.3f} %% err  "
                  f"{r['pct_of_trunk_reference_mass']:7.3f} %% ref  rel {r['rel_l2']:.6f}  "
                  f"upstream {r['upstreams_own_rel_l2_here'] or float('nan'):.6f}  "
                  f"x{r['over_upstream'] or float('nan'):.3f}  n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
