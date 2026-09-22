#!/usr/bin/env python3
"""of3t-trunkopclass: score the class-scoped ablation arms, with the A/A floor FIRST.

The A/A floor is printed before any arm's reading, because three of the four class differences
may sit inside it and a difference inside the floor bought nothing. It is a bit-identity check
between two runs of the same arm on the same card, plus the worst per-tensor relative move.

Every arm is then scored against the same pinned float64 reference and against upstream's own
bf16 autocast step, over the same 2,736 tensors and the same denominator, and reported as the
FRACTION OF ITS OWN CEILING it reached -- the ceiling being what that class would buy if its
error went to exactly zero, which `ceiling.py` computes from the census.
"""
from __future__ import annotations

import argparse
import json
import math
import os

PRE = "pairformer_stack.blocks."
NORM_FLOOR = 1e-12


def load(path, unwrap="grads"):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, dict) and unwrap in d and isinstance(d[unwrap], dict):
        d = d[unwrap]
    return {k: v for k, v in d.items() if v is not None and hasattr(v, "shape")}


def aa(pa, pb):
    """The determinism floor: two runs of the same arm, same card, same config."""
    import torch
    a, b = load(pa), load(pb)
    common = sorted(set(a) & set(b))
    moved, worst, worst_t = 0, 0.0, None
    bit = True
    for k in common:
        x = a[k].to(torch.float64).reshape(-1)
        y = b[k].to(torch.float64).reshape(-1)
        if not torch.equal(a[k], b[k]):
            bit = False
            moved += 1
            n = float(x.norm())
            r = float((y - x).norm()) / (n + 1e-30)
            if r > worst:
                worst, worst_t = r, k
    return {"a": pa, "b": pb, "n_a": len(a), "n_b": len(b), "n_common": len(common),
            "n_only_in_a": len(set(a) - set(b)), "n_only_in_b": len(set(b) - set(a)),
            "n_tensors_moved": moved, "bit_identical": bit,
            "worst_rel_move": worst, "worst_tensor": worst_t}


def score(ref, arm):
    import torch
    d2 = sq = a2 = dot = 0.0
    rels, worst, worst_t, worst_abs, worst_abs_t = [], 0.0, None, 0.0, None
    below = 0
    for k, r in ref.items():
        v = arm.get(k)
        if v is None:
            continue
        r = r.to(torch.float64).reshape(-1)
        v = v.to(torch.float64).reshape(-1)
        rn = float(r.norm())
        dn = float((v - r).norm())
        d2 += dn * dn
        sq += rn * rn
        a2 += float(v.norm()) ** 2
        dot += float((v * r).sum())
        if dn > worst_abs:
            worst_abs, worst_abs_t = dn, k
        if rn < NORM_FLOOR:
            below += 1
            continue
        x = dn / (rn + 1e-30)
        rels.append(x)
        if x > worst:
            worst, worst_t = x, k
    rels.sort()
    return {"n": sum(1 for k in ref if k in arm),
            "mass_weighted_rel_l2": math.sqrt(d2 / sq) if sq else None,
            "mass_weighted_norm_ratio": math.sqrt(a2 / sq) if sq else None,
            "mass_weighted_cos": dot / math.sqrt(a2 * sq) if a2 and sq else None,
            "abs_err_sq": d2, "ref_sq": sq,
            "median_rel_l2_over_tensors": rels[len(rels) // 2] if rels else None,
            "worst_rel_l2_over_tensors": worst, "worst_tensor_by_relative_error": worst_t,
            "worst_tensor_by_absolute_error": worst_abs_t,
            "worst_absolute_error": worst_abs,
            "n_below_norm_floor": below,
            "n_over_per_tensor_bar_5e-2": sum(1 for x in rels if x > 0.05)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16auto", required=True)
    ap.add_argument("--aa", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--base", required=True, help="NAME=PATH of the shipped arm")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--ceiling", default="", help="CEILING_*.json, to price each arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R = {"what": __doc__.strip().splitlines()[0], "host": os.uname().nodename,
         "inputs": vars(a)}

    # ---- the floor, FIRST -------------------------------------------------------------
    R["aa_floor"] = aa(*a.aa)
    print("A/A FLOOR  bit_identical=%s  tensors moved %d of %d  worst rel move %.3e  (%s)"
          % (R["aa_floor"]["bit_identical"], R["aa_floor"]["n_tensors_moved"],
             R["aa_floor"]["n_common"], R["aa_floor"]["worst_rel_move"],
             R["aa_floor"]["worst_tensor"]))
    print()

    refs = {"f64": load(a.f64), "bf16auto": load(a.bf16auto)}
    bn, _, bp = a.base.partition("=")
    specs = [(bn, bp)] + [s.partition("=")[::2] for s in a.arm]

    ceil = json.load(open(a.ceiling)) if a.ceiling else None
    R["arms"] = {}
    base_rel = None
    print("%-10s %16s %16s %10s %9s %9s %s" % (
        "arm", "vs bf16auto", "vs float64", "removed", "ceiling", "of ceil", "worst tensor (rel)"))
    for nm, path in specs:
        arm = load(path)
        e = {rn: score(refs[rn], arm) for rn in refs}
        rel = e["bf16auto"]["mass_weighted_rel_l2"]
        if base_rel is None:
            base_rel = rel
        removed = 1.0 - (rel ** 2) / (base_rel ** 2)
        cl = None
        if ceil and nm in ceil.get("ceilings", {}):
            cl = ceil["ceilings"][nm]["trunk_in_frame"]
        # the ceiling's own removed-fraction at c64 scope has to be recomputed from this
        # scope's census, so `of_ceiling` is only printed when a ceiling was supplied
        of_ceil = None
        if cl is not None and base_rel:
            cmax = 1.0 - (cl ** 2) / (base_rel ** 2)
            of_ceil = removed / cmax if cmax else None
        e["mass_weighted_rel_l2_vs_bf16auto"] = rel
        e["fraction_of_base_error_mass_removed"] = removed
        e["ceiling_trunk_in_frame"] = cl
        e["fraction_of_ceiling_reached"] = of_ceil
        R["arms"][nm] = e
        print("%-10s %16.10f %16.10f %9.4f%% %9s %9s %s" % (
            nm, rel, e["f64"]["mass_weighted_rel_l2"], 100 * removed,
            ("%.6f" % cl) if cl is not None else "-",
            ("%.2f%%" % (100 * of_ceil)) if of_ceil is not None else "-",
            e["bf16auto"]["worst_tensor_by_relative_error"]))

    json.dump(R, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
