#!/usr/bin/env python3
"""of3t-trunkceiling: the A/A determinism floor, and the denominator floor applied.

Two jobs, both of which have to happen before a ceiling is readable.

**A/A.** The same arm, same lever set, same host, same card, same boundary, run twice. Scored
through `of3t_trunkg043/score.py` -- the same definition of mass-weighted rel L2 every row
above this one uses (A23), imported and not reimplemented. Whatever this reads is the floor
under every difference this row reports: a lever whose effect is inside it bought nothing.

**The denominator floor.** A23's rel L2 divides by the reference tensor's own norm, and at crop
384 the reference has tensors down to ref_norm 2.6e-19, which is how a 1.05e+14 per-tensor
figure gets into an artifact (A14). The mass-weighted headline is immune because it weights by
ref_sq, but a per-tensor WORST CASE is not, so every worst case here is reported twice: over
all tensors, and over the set whose reference norm clears the floor, with the count below it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "of3t_trunkg043"))
from score import score, by_leaf                                             # noqa: E402

PRE = "pairformer_stack.blocks."


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return {k: v for k, v in g.items() if k.startswith(PRE) and v is not None}


def floored(s, floor):
    """The same statistics restricted to tensors whose REFERENCE norm clears the floor."""
    rows = [x for x in s["_rows"] if x["ref_norm"] >= floor]
    below = [x for x in s["_rows"] if x["ref_norm"] < floor]
    if not rows:
        return {"kept": 0, "below_floor": len(below)}
    tot = sum(x["ref_sq"] for x in rows)
    mw = sum((x["ref_sq"] / tot) * x["rel_l2"] ** 2 for x in rows) ** 0.5 if tot else 0.0
    worst = max(rows, key=lambda x: x["rel_l2"])
    return {"kept": len(rows), "below_floor": len(below),
            "mass_below_floor": sum(x["ref_sq"] for x in below) / (
                sum(x["ref_sq"] for x in s["_rows"]) or 1.0),
            "mass_weighted_rel_l2": mw,
            "worst_by_rel": {k: worst[k] for k in
                             ("tensor", "rel_l2", "norm_ratio", "cos", "ref_norm", "our_norm")},
            "tensors_below_floor": [x["tensor"] for x in below][:16]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm A")
    ap.add_argument("--b", required=True, help="arm A repeated -- same config, same card")
    ap.add_argument("--floor", type=float, default=1e-12,
                    help="denominator floor on the reference tensor norm (A14)")
    ap.add_argument("--label", default="A/A")
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    keys = sorted(set(A) & set(B))
    n_eq = sum(1 for k in keys if torch.equal(A[k].to(torch.float64), B[k].to(torch.float64)))
    worst_abs, worst_t = 0.0, None
    for k in keys:
        d = float((A[k].to(torch.float64) - B[k].to(torch.float64)).abs().max())
        if d > worst_abs:
            worst_abs, worst_t = d, k

    # B is the reference here: an A/A has no privileged side, and naming one keeps the
    # denominator the same shape as every other reading in this row.
    s = score(A, B, keys)
    out = {
        "what": f"{a.label} determinism floor for the of3t-trunkceiling arm, and the "
                f"denominator floor A14 requires, both taken BEFORE any arm is read as a "
                f"difference.",
        "host": a.host,
        "scored_on": os.uname().nodename,
        "arms": {"a": {"path": a.a, "sha256": sha256_file(a.a)},
                 "b": {"path": a.b, "sha256": sha256_file(a.b)}},
        "scope": {"in_a": len(A), "in_b": len(B), "compared": len(keys),
                  "only_in_a": sorted(set(A) - set(B))[:8],
                  "only_in_b": sorted(set(B) - set(A))[:8]},
        "bit_identity": {"compared": len(keys), "bit_identical": n_eq,
                         "all_bit_identical": n_eq == len(keys),
                         "max_absdiff": worst_abs, "worst_tensor": worst_t},
        "FLOOR": {"mass_weighted_rel_l2": s["mass_weighted_rel_l2"],
                  "median_rel_l2_over_tensors": s["median_rel_l2_over_tensors"],
                  "mass_weighted_norm_ratio": s["mass_weighted_norm_ratio"],
                  "mass_weighted_cos": s["mass_weighted_cos"],
                  "reference_squared_norm": s["reference_squared_norm"],
                  "worst_by_rel": s["worst_by_rel"],
                  "worst_by_error_mass": s["worst_by_error_mass"]},
        "denominator_floor": {"floor_on_ref_norm": a.floor, **floored(s, a.floor)},
        "by_leaf_error_mass": by_leaf(s["_rows"]),
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("AAFLOOR " + json.dumps({
        "mass_weighted_rel_l2": out["FLOOR"]["mass_weighted_rel_l2"],
        "bit_identical": f"{n_eq}/{len(keys)}",
        "max_absdiff": worst_abs,
        "worst_by_rel": s["worst_by_rel"]["tensor"],
        "worst_rel": s["worst_by_rel"]["rel_l2"],
        "floored_worst_rel": out["denominator_floor"].get("worst_by_rel", {}).get("rel_l2"),
        "below_floor": out["denominator_floor"]["below_floor"],
        "out": a.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
