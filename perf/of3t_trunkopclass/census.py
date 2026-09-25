#!/usr/bin/env python3
"""of3t-trunkopclass: per-OP-CLASS absolute gradient error mass over the whole 48-block trunk.

The other axis of `of3t-trunkblocks`, which splits the same error by block. Same pinned float64
reference, same denominator as the graded artifact, same statistic as
`perf/of3t_wholemodel/agreement.py`:

    mass_weighted_rel_l2 = sqrt( sum_k ||arm_k - ref_k||^2 / sum_k ||ref_k||^2 )

ABSOLUTE FIRST. A share moves when its denominator collapses, so every class carries its raw
squared error mass beside its share, and its share of the REFERENCE mass beside that -- a class
holding 40 % of the error while holding 40 % of the reference is not enriched at all.

The class map is `perf/of3t_trunkopclass/PREDICTION.md`'s, fixed before this ran. Every leaf is
asserted into exactly one class; an unassigned leaf is a hard failure, not a footnote.
"""
from __future__ import annotations

import argparse
import json
import math
import os

PRE = "pairformer_stack.blocks."

# The four-class axis the brief names, plus the five-family split it folds.
CLASS_OF_PREFIX = [
    ("pair_stack.tri_mul_in.",        "TRI_MUL", "tri_mul_in"),
    ("pair_stack.tri_mul_out.",       "TRI_MUL", "tri_mul_out"),
    ("pair_stack.tri_att_start.",     "TRI_ATT", "tri_att_start"),
    ("pair_stack.tri_att_end.",       "TRI_ATT", "tri_att_end"),
    ("pair_stack.pair_transition.",   "TRANS",   "pair_transition"),
    ("single_transition.",            "TRANS",   "single_transition"),
    ("attn_pair_bias.",               "APB",     "attn_pair_bias"),
]

# Published figures this instrument must reproduce before any class number is read.
CONTROLS = {
    "ours_vs_f64":      0.8354121633458239,
    "ours_vs_bf16auto": 1.029395337772341,
    "bf16auto_vs_f64":  0.37393839211303687,
}
CONTROL_TOL = 1e-12
NORM_FLOOR = 1e-12          # PREDICTION.md: per-tensor relative stats only


def classify(leaf):
    for pfx, cls, fam in CLASS_OF_PREFIX:
        if leaf.startswith(pfx):
            return cls, fam
    raise SystemExit(f"UNASSIGNED LEAF {leaf!r} -- the class map must cover every leaf")


def load(path, unwrap="grads"):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and unwrap in d and isinstance(d[unwrap], dict):
        d = d[unwrap]
    out = {}
    for k, v in d.items():
        if v is None or not hasattr(v, "shape"):
            continue
        kk = k if k.startswith(PRE) else (PRE + k if k.split(".")[0].isdigit() else k)
        if kk.startswith(PRE):
            out[kk] = v
    return out


def rows_for(ref, arm):
    import torch
    rows = []
    for k, r in ref.items():
        a = arm.get(k)
        if a is None:
            continue
        r = r.to(torch.float64).reshape(-1)
        a = a.to(torch.float64).reshape(-1)
        leaf = ".".join(k.split(".")[3:])
        cls, fam = classify(leaf)
        rows.append({"param": k, "leaf": leaf, "cls": cls, "fam": fam,
                     "block": int(k.split(".")[2]),
                     "ref_norm": float(r.norm()), "arm_norm": float(a.norm()),
                     "diff_norm": float((a - r).norm()), "dot": float((a * r).sum())})
    return rows


def stat(rows, label, total_ref_sq, total_err_sq):
    sq = sum(r["ref_norm"] ** 2 for r in rows)
    d2 = sum(r["diff_norm"] ** 2 for r in rows)
    a2 = sum(r["arm_norm"] ** 2 for r in rows)
    dot = sum(r["dot"] for r in rows)
    ok = [r for r in rows if r["ref_norm"] >= NORM_FLOOR]
    rels = sorted(r["diff_norm"] / (r["ref_norm"] + 1e-30) for r in ok)
    worst = max(rows, key=lambda r: r["diff_norm"]) if rows else None
    worst_rel = max(ok, key=lambda r: r["diff_norm"] / (r["ref_norm"] + 1e-30)) if ok else None
    return {
        "set": label, "n_tensors": len(rows),
        "abs_err_sq": d2, "abs_err": math.sqrt(d2),
        "ref_sq": sq, "ref_norm": math.sqrt(sq),
        "share_of_error_mass": (d2 / total_err_sq) if total_err_sq else None,
        "share_of_reference_mass": (sq / total_ref_sq) if total_ref_sq else None,
        "enrichment": ((d2 / total_err_sq) / (sq / total_ref_sq))
                      if total_err_sq and total_ref_sq and sq else None,
        "mass_weighted_rel_l2": math.sqrt(d2 / sq) if sq else None,
        "mass_weighted_norm_ratio": math.sqrt(a2 / sq) if sq else None,
        "mass_weighted_cos": (dot / math.sqrt(a2 * sq)) if a2 > 0 and sq > 0 else None,
        "median_rel_l2_over_tensors": (rels[len(rels) // 2] if rels else None),
        "max_rel_l2_over_tensors": (rels[-1] if rels else None),
        "n_below_norm_floor": len(rows) - len(ok),
        "n_over_per_tensor_bar_5e-2": sum(1 for x in rels if x > 0.05),
        "worst_tensor_by_absolute_error": worst and worst["param"],
        "worst_tensor_absolute_error": worst and worst["diff_norm"],
        "worst_tensor_by_relative_error": worst_rel and worst_rel["param"],
        "worst_tensor_relative_error": worst_rel and
            worst_rel["diff_norm"] / (worst_rel["ref_norm"] + 1e-30),
    }


def group(rows, key, label_prefix):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    tref = sum(r["ref_norm"] ** 2 for r in rows)
    terr = sum(r["diff_norm"] ** 2 for r in rows)
    return {k: stat(v, f"{label_prefix}:{k}", tref, terr) for k, v in sorted(g.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16auto", required=True)
    ap.add_argument("--bf16full", default="")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--controls", action="store_true",
                    help="require the three published trunk figures back to 1e-12")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    refs = {"f64": load(a.f64), "bf16auto": load(a.bf16auto)}
    if a.bf16full:
        refs["bf16full"] = load(a.bf16full)
    arms = {}
    for spec in a.arm:
        nm, _, p = spec.partition("=")
        arms[nm] = load(p)

    R = {"what": __doc__.strip().splitlines()[0],
         "host": os.uname().nodename,
         "inputs": {"f64": a.f64, "bf16auto": a.bf16auto, "bf16full": a.bf16full,
                    "arms": a.arm},
         "norm_floor_for_relative_stats": NORM_FLOOR,
         "class_map": {fam: cls for _, cls, fam in CLASS_OF_PREFIX},
         "n_tensors": {k: len(v) for k, v in list(refs.items()) + list(arms.items())},
         "overall": {}, "by_class": {}, "by_family": {}, "by_leaf": {}, "by_block": {}}

    todo = []
    for nm in arms:
        for rn in refs:
            todo.append((f"{nm}_vs_{rn}", refs[rn], arms[nm]))
    for rn in ("bf16auto", "bf16full"):
        if rn in refs:
            todo.append((f"{rn}_vs_f64", refs["f64"], refs[rn]))

    for label, ref, arm in todo:
        rows = rows_for(ref, arm)
        tref = sum(r["ref_norm"] ** 2 for r in rows)
        terr = sum(r["diff_norm"] ** 2 for r in rows)
        R["overall"][label] = stat(rows, label, tref, terr)
        R["by_class"][label] = group(rows, lambda r: r["cls"], label)
        R["by_family"][label] = group(rows, lambda r: r["fam"], label)
        R["by_leaf"][label] = group(rows, lambda r: r["leaf"], label)
        R["by_block"][label] = group(rows, lambda r: r["block"], label)
        o = R["overall"][label]
        print(f"{label:28s} n={o['n_tensors']:5d} rel={o['mass_weighted_rel_l2']!r} "
              f"ratio={o['mass_weighted_norm_ratio']:.6f} cos={o['mass_weighted_cos']:.6f}",
              flush=True)

    # --- control: the instrument must reproduce the published trunk figures ----------------
    C, bad = {}, []
    first = next(iter(arms))
    for k, want in CONTROLS.items():
        nm = k.replace("ours", first)
        got = R["overall"].get(nm, {}).get("mass_weighted_rel_l2")
        rd = (abs(got - want) / want) if got else None
        C[k] = {"published": want, "this_instrument": got, "rel_difference": rd}
        print(f"CONTROL {k:18s} want {want!r} got {got!r} rel={rd}")
        if rd is None or rd > CONTROL_TOL:
            bad.append(k)
    R["controls"] = C
    R["controls_pass"] = not bad
    if a.controls and bad:
        json.dump(R, open(a.out, "w"), indent=2)
        raise SystemExit(f"CONTROL FAILED for {bad} -- no class number may be read")

    json.dump(R, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
