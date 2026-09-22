#!/usr/bin/env python3
"""Where the residual lives, by leaf op, with no card run.

A worst-tensor list names the tail, not the location. The scope headline is
`sqrt(sum diff_norm^2) / sqrt(sum ref_norm^2)` over the 547 compared tensors, so each
tensor's contribution to the numerator is `diff_norm^2` and the question "what is the
2.04x" is exactly "which leaves own `sum diff_norm^2` after the softmax is bounded, over
and above what upstream's own bf16 step already spends there".

Prints, per leaf class, the bound arm's squared error, upstream's own bf16 squared error at
the same tensors (the floor), and the EXCESS -- which is the thing that has to go away.
"""
import argparse
import json
import re
import sys


def leaf(param: str) -> str:
    p = param
    p = re.sub(r"\.blocks\.\d+\.", ".blocks.N.", p)
    p = re.sub(r"\.layers\.\d+\.", ".layers.N.", p)
    return p


def path_of(param: str) -> str:
    if "conditioned_transition" in param:
        return "transition"
    if "attention_pair_bias" in param or ".mha." in param:
        return "attention"
    return "other"


def load(path):
    d = json.load(open(path))
    rows = d["per_tensor"] if isinstance(d, dict) else d
    return {r["param"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="per-tensor sidecar of the arm under test")
    ap.add_argument("--floor", required=True, help="per-tensor sidecar of upstream bf16 vs float64")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=18)
    a = ap.parse_args()

    arm, flo = load(a.arm), load(a.floor)
    common = [p for p in arm if p in flo]
    missing = [p for p in arm if p not in flo]
    if not common:
        print("FAILED: the two sidecars share no parameter", file=sys.stderr)
        return 3

    tot_ref = sum(arm[p]["ref_norm"] ** 2 for p in common)
    tot_arm = sum(arm[p]["diff_norm"] ** 2 for p in common)
    tot_flo = sum(flo[p]["diff_norm"] ** 2 for p in common)
    head = (tot_arm / tot_ref) ** 0.5
    floor_head = (tot_flo / tot_ref) ** 0.5

    groups = {}
    for p in common:
        k = (path_of(p), leaf(p))
        g = groups.setdefault(k, {"n": 0, "mass": 0.0, "arm_sq": 0.0, "flo_sq": 0.0,
                                  "ref_sq": 0.0, "worst": (0.0, "")})
        g["n"] += 1
        g["mass"] += arm[p]["pct_of_model_mass"]
        g["arm_sq"] += arm[p]["diff_norm"] ** 2
        g["flo_sq"] += flo[p]["diff_norm"] ** 2
        g["ref_sq"] += arm[p]["ref_norm"] ** 2
        if arm[p]["diff_norm"] ** 2 > g["worst"][0]:
            g["worst"] = (arm[p]["diff_norm"] ** 2, p)

    rows = []
    for (pth, lf), g in groups.items():
        exc = g["arm_sq"] - g["flo_sq"]
        rows.append({
            "path": pth, "leaf": lf, "n": g["n"],
            "pct_of_model_mass": round(g["mass"], 4),
            "arm_sq": g["arm_sq"], "floor_sq": g["flo_sq"],
            "excess_sq": exc,
            "share_of_total_excess_pct": None,
            "arm_rel_in_group": (g["arm_sq"] / g["ref_sq"]) ** 0.5 if g["ref_sq"] else None,
            "floor_rel_in_group": (g["flo_sq"] / g["ref_sq"]) ** 0.5 if g["ref_sq"] else None,
            "worst_tensor_in_group": g["worst"][1],
        })
    tot_exc = sum(r["excess_sq"] for r in rows)
    for r in rows:
        r["share_of_total_excess_pct"] = round(100.0 * r["excess_sq"] / tot_exc, 4) if tot_exc else None
    rows.sort(key=lambda r: -r["excess_sq"])

    by_path = {}
    for r in rows:
        b = by_path.setdefault(r["path"], {"n": 0, "mass": 0.0, "arm_sq": 0.0,
                                           "floor_sq": 0.0, "excess_sq": 0.0})
        b["n"] += r["n"]
        b["mass"] += r["pct_of_model_mass"]
        b["arm_sq"] += r["arm_sq"]
        b["floor_sq"] += r["floor_sq"]
        b["excess_sq"] += r["excess_sq"]
    for b in by_path.values():
        b["mass"] = round(b["mass"], 4)
        b["share_of_total_excess_pct"] = round(100.0 * b["excess_sq"] / tot_exc, 4) if tot_exc else None

    rep = {
        "what": __doc__.strip().splitlines()[0],
        "arm_sidecar": a.arm, "floor_sidecar": a.floor, "label": a.label,
        "n_compared": len(common),
        "n_in_arm_not_in_floor": len(missing),
        "scope_headline_recomputed": head,
        "scope_floor_recomputed": floor_head,
        "ratio_headline_over_floor": head / floor_head if floor_head else None,
        "total_excess_sq": tot_exc,
        "by_path": by_path,
        "by_leaf": rows,
    }
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"{a.label}: headline {head:.6e}  floor {floor_head:.6e}  "
          f"ratio {head/floor_head:.4f}  over {len(common)} tensors")
    print(f"{'path':<11}{'n':>4} {'mass%':>9} {'excess%':>9}  leaf")
    for pth, b in sorted(by_path.items(), key=lambda kv: -kv[1]["excess_sq"]):
        print(f"{pth:<11}{b['n']:>4} {b['mass']:>9.4f} {b['share_of_total_excess_pct']:>9.4f}  "
              f"(whole path)")
    print()
    for r in rows[:a.top]:
        print(f"{r['path']:<11}{r['n']:>4} {r['pct_of_model_mass']:>9.4f} "
              f"{r['share_of_total_excess_pct']:>9.4f}  {r['leaf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
