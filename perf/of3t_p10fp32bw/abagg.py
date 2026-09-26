#!/usr/bin/env python3
"""Mass-weighted aggregation of one step's per-parameter gradient A/B.

The pre-registered clause is a MASS-WEIGHTED rel_l2 per section, so the worst-tensor line the
harness prints cannot be compared to it: a 74x move on a tensor carrying 1e-9 of the gradient
mass changes the clause by nothing. This reads the per-parameter table out whole and composes
it the way `model_scope.py` composes a section -- sqrt(sum ||delta||^2 / sum ||ref||^2), with
arm A's own norms as the denominator, because the step's fixture has no upstream reference.

  abagg.py ARM.json [FLOOR.json]

With a second table it prints the SAME aggregation for the floor arm beside it. Reading the
first without the second is the mistake this file exists to stop: the step is not reproducible
rep to rep, so a move that is not larger than the ON/ON floor is not the flag's.
"""
import json, math, sys
from collections import defaultdict


def group(name):
    p = name.split(".")
    if p[0] == "trunk":
        return "trunk." + (p[1] if len(p) > 1 else "?")
    if p[0] == "diffusion":
        return "diffusion." + (p[1] if len(p) > 1 else "?")
    return p[0]


def agg(rows):
    out = defaultdict(lambda: {"n": 0, "d2": 0.0, "r2": 0.0, "worst": 0.0, "worst_n": None,
                               "numel": 0})
    for r in rows:
        g = out[group(r["name"])]
        g["n"] += 1
        g["numel"] += r["numel"]
        g["d2"] += r["delta_l2"] ** 2
        g["r2"] += r["norm_a"] ** 2
        if r["rel_l2"] > g["worst"]:
            g["worst"], g["worst_n"] = r["rel_l2"], r["name"]
    for g in out.values():
        g["mass_weighted_rel_l2"] = math.sqrt(g["d2"] / g["r2"]) if g["r2"] else None
    return out


def whole(rows):
    d2 = sum(r["delta_l2"] ** 2 for r in rows)
    r2 = sum(r["norm_a"] ** 2 for r in rows)
    return math.sqrt(d2 / r2) if r2 else None


def main():
    arm = json.load(open(sys.argv[1]))["rows"]
    flo = json.load(open(sys.argv[2]))["rows"] if len(sys.argv) > 2 else None
    a, f = agg(arm), (agg(flo) if flo else {})
    print(f"{'group':34s} {'n':>5s} {'arm':>12s} {'floor':>12s} {'arm/floor':>10s}"
          f" {'share of arm d2':>16s}")
    tot = sum(g["d2"] for g in a.values())
    for k in sorted(a, key=lambda k: -a[k]["d2"]):
        g = a[k]
        fg = f.get(k)
        fv = fg["mass_weighted_rel_l2"] if fg else None
        ratio = (g["mass_weighted_rel_l2"] / fv) if (fv not in (None, 0.0)) else None
        print(f"{k:34s} {g['n']:5d} {g['mass_weighted_rel_l2']:12.6e} "
              f"{(fv if fv is not None else float('nan')):12.6e} "
              f"{(ratio if ratio is not None else float('nan')):10.2f} "
              f"{100 * g['d2'] / tot:15.3f}%")
    print()
    print(f"WHOLE STEP mass-weighted rel_l2  arm={whole(arm):.6e}"
          + (f"  floor={whole(flo):.6e}  arm/floor={whole(arm)/whole(flo):.2f}" if flo else ""))
    print(f"parameters compared: arm {len(arm)}" + (f", floor {len(flo)}" if flo else ""))


if __name__ == "__main__":
    main()
