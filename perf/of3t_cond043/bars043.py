#!/usr/bin/env python3
"""Score an upstream floor arm against the float64 reference of ITS OWN capture, per leaf family.

One boundary per invocation, because that is the only thing a ratio may be taken inside. The
report carries its own package, capture, thread count and float64 reference in a `provenance`
block, so an entry lifted out of this file still says what it was measured on. `--compare` embeds
a previously written report under `beside`; it is placed NEXT TO these numbers, never divided
into them.

Statistics per set, never per set without its mass (A23): instance count, error mass
(sum ||arm - f64||^2), reference mass (sum ||f64||^2), and the mass-weighted rel
sqrt(error_mass / reference_mass), plus the per-instance table for every named leaf.

Controls:
  f32 arm       the instrument floor. What the harness cannot tell apart.
  A16           gradient replaced by zeros. Must read exactly 1.0.
  A14           the reference norm every rel divides by, with the count below 1e-12.
  BREAK         each instance scored against a DIFFERENT instance's reference, rolled by one
                within each (leaf, shape) group. It completes and returns a number: a control
                that raises before it acts has tested nothing.
  A/A           two runs of the same policy differenced against each other.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

A14_FLOOR = 1e-12
SITE = re.compile(r"^(.*)\.blocks\.(\d+)\.(.*)$")


def split(k):
    m = SITE.match(k)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (k.rsplit(".", 1)[0], -1, k)


def row(m, r):
    m = m.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    nr = float(torch.linalg.vector_norm(r))
    err = float(torch.linalg.vector_norm(m - r))
    return {"rel": err / nr if nr else float("inf"), "err_sq": err * err,
            "ref_norm": nr, "ref_sq": nr * nr}


def summarise(rows, scope_rows=None):
    em = sum(r["err_sq"] for r in rows)
    rm = sum(r["ref_sq"] for r in rows)
    out = {"n": len(rows), "error_mass": em, "reference_mass": rm,
           "mass_weighted_rel": float(np.sqrt(em / rm)) if rm else None,
           "median_rel": float(np.median([r["rel"] for r in rows])) if rows else None,
           "min_ref_norm": min((r["ref_norm"] for r in rows), default=None),
           "a14_dropped": sum(1 for r in rows if r["ref_norm"] < A14_FLOOR)}
    if scope_rows is not None:
        sem = sum(r["err_sq"] for r in scope_rows)
        srm = sum(r["ref_sq"] for r in scope_rows)
        out["share_of_scope_error_mass"] = em / sem if sem else None
        out["mass_share_of_scope"] = rm / srm if srm else None
    return out


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    g = d["grads"] if isinstance(d, dict) and "grads" in d else d
    return {k: v for k, v in g.items() if v is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-f64", required=True, help="sub_boundary.pt of THIS capture")
    ap.add_argument("--arm", required=True, action="append", metavar="NAME=PATH")
    ap.add_argument("--leaf", required=True, action="append",
                    help="leaf suffix, matched with endswith")
    ap.add_argument("--break-arm", default=None, help="which arm the BREAK control rolls")
    ap.add_argument("--aa", nargs=2, default=None, metavar=("A", "B"),
                    help="two arm names to difference against each other")
    ap.add_argument("--package", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--omp", required=True)
    ap.add_argument("--compare", default=None, help="an earlier report to place BESIDE this one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gref = {k: v for k, v in torch.load(a.ref_f64, map_location="cpu",
                                        weights_only=False)["grad_f64"].items() if v is not None}
    arms = {}
    for spec in a.arm:
        n, _, p = spec.partition("=")
        arms[n] = load(p)
    keys = sorted(set(gref).intersection(*[set(g) for g in arms.values()]))

    rep = {"what": __doc__.strip().splitlines()[0],
           "provenance": {"package": a.package, "version": a.version, "capture": a.capture,
                          "float64_reference": a.ref_f64, "OMP_NUM_THREADS": a.omp,
                          "torch": torch.__version__,
                          "scope": "diffusion_module, every parameter both the reference and "
                                   "every arm carry a gradient for"},
           "leaves": a.leaf, "tensors_compared": len(keys),
           "tensors_in_reference": len(gref),
           "A14_floor_on_the_reference_norm": A14_FLOOR,
           "arms": {}}

    scope_rows_by_arm = {}
    for n, g in arms.items():
        rows = {k: row(g[k], gref[k]) for k in keys}
        scope_rows_by_arm[n] = rows
        e = {"scope": summarise(list(rows.values()))}
        by_leaf = {}
        for k in keys:
            by_leaf.setdefault(split(k)[2], []).append(rows[k])
        e["by_leaf"] = {lf: summarise(v, list(rows.values()))
                        for lf, v in sorted(by_leaf.items())}
        e["named_leaves"] = {}
        for lf in a.leaf:
            sel = [k for k in keys if k.endswith(lf)]
            d = summarise([rows[k] for k in sel], list(rows.values()))
            bysite = {}
            for k in sel:
                bysite.setdefault(split(k)[0], []).append(rows[k])
            d["by_site"] = {s: summarise(v) for s, v in sorted(bysite.items())}
            d["per_instance"] = {}
            for k in sorted(sel, key=lambda x: (split(x)[0], split(x)[1])):
                s, b, _ = split(k)
                d["per_instance"][f"{s}.{b:02d}" if b >= 0 else k] = {
                    "tensor": k, "ref_norm": rows[k]["ref_norm"],
                    "error_mass": rows[k]["err_sq"], "rel": rows[k]["rel"]}
            e["named_leaves"][lf] = d
        worst = max(keys, key=lambda k: rows[k]["rel"])
        e["worst_rel"], e["worst_tensor"] = rows[worst]["rel"], worst
        rep["arms"][n] = e

    # ---- A16: the zero baseline, measured -------------------------------------------------
    zrows = [row(torch.zeros_like(gref[k]), gref[k]) for k in keys]
    rep["A16_zero_baseline"] = {
        "scope": summarise(zrows),
        "exactly_one": bool(all(abs(r["rel"] - 1.0) < 1e-15 for r in zrows)),
        "per_named_leaf": {lf: summarise([row(torch.zeros_like(gref[k]), gref[k])
                                          for k in keys if k.endswith(lf)])
                           for lf in a.leaf}}

    # ---- A14: the denominator audit -------------------------------------------------------
    rn = {k: float(torch.linalg.vector_norm(gref[k].reshape(-1).to(torch.float64)))
          for k in keys}
    rep["A14"] = {"floor": A14_FLOOR,
                  "tensors_below_the_floor": sum(1 for v in rn.values() if v < A14_FLOOR),
                  "scope_min_ref_norm": min(rn.values()),
                  "scope_min_tensor": min(rn, key=rn.get),
                  "per_named_leaf": {
                      lf: {"min_ref_norm": min(rn[k] for k in keys if k.endswith(lf)),
                           "median_ref_norm": float(np.median(
                               [rn[k] for k in keys if k.endswith(lf)]))}
                      for lf in a.leaf}}

    # ---- BREAK: same arm, wrong reference, rolled within (leaf, shape) ---------------------
    if a.break_arm:
        g = arms[a.break_arm]
        groups = {}
        for k in keys:
            groups.setdefault((split(k)[2], tuple(gref[k].shape)), []).append(k)
        rolled, brows, bleaf = 0, [], {lf: [] for lf in a.leaf}
        for grp in groups.values():
            grp = sorted(grp)
            if len(grp) < 2:
                continue
            for i, k in enumerate(grp):
                r = row(g[k], gref[grp[(i + 1) % len(grp)]])
                brows.append(r)
                rolled += 1
                for lf in a.leaf:
                    if k.endswith(lf):
                        bleaf[lf].append(r)
        rep["BREAK_control"] = {
            "what": f"arm {a.break_arm}, each instance scored against the NEXT instance's "
                    f"reference within its (leaf, shape) group. Completes and returns a number.",
            "tensors_rolled": rolled,
            "scope": summarise(brows),
            "per_named_leaf": {lf: summarise(v) for lf, v in bleaf.items()},
            "moved": {n: {"arm_scope": rep["arms"][a.break_arm]["scope"]["mass_weighted_rel"],
                          "break_scope": summarise(brows)["mass_weighted_rel"]}
                      for n in [a.break_arm]}}

    # ---- A/A ------------------------------------------------------------------------------
    if a.aa:
        x, y = a.aa
        gx, gy = arms[x], arms[y]
        md = max(float((gx[k].double() - gy[k].double()).abs().max()) for k in keys)
        rep["A_over_A"] = {
            "arms": [x, y], "max_abs_difference": md, "bit_identical": md == 0.0,
            "scope_rel_x": rep["arms"][x]["scope"]["mass_weighted_rel"],
            "scope_rel_y": rep["arms"][y]["scope"]["mass_weighted_rel"],
            "what": "two runs of the same policy at the same OMP_NUM_THREADS. A float64 CPU "
                    "reduction is a function of its thread count, so this is what pins it."}

    if a.compare:
        rep["beside"] = json.load(open(a.compare))
        rep["beside_note"] = ("a DIFFERENT capture, package and float64 reference. Placed beside "
                              "these numbers for reading. No ratio crosses this boundary: A27 "
                              "permits a ratio only inside one capture.")

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)

    p = rep["provenance"]
    print(f"\n=== openfold3 {p['version']} at {p['package']}, capture {p['capture']}, "
          f"OMP={p['OMP_NUM_THREADS']}, {rep['tensors_compared']} tensors ===")
    for n, e in rep["arms"].items():
        s = e["scope"]
        print(f"\narm {n:16s} scope rel {s['mass_weighted_rel']:.6e}  "
              f"err mass {s['error_mass']:.6e}  ref mass {s['reference_mass']:.6e}  n={s['n']}")
        for lf, d in e["named_leaves"].items():
            print(f"  {lf}")
            print(f"    n={d['n']:3d}  rel {d['mass_weighted_rel']:.6e}  "
                  f"err mass {d['error_mass']:.6e}  ref mass {d['reference_mass']:.6e}  "
                  f"err share {d['share_of_scope_error_mass']*100:.3f} %  "
                  f"mass share {d['mass_share_of_scope']*100:.3f} %")
            for s_, sd in d["by_site"].items():
                print(f"      {s_:34s} n={sd['n']:2d}  rel {sd['mass_weighted_rel']:.6e}  "
                      f"ref mass {sd['reference_mass']:.6e}")
    if "BREAK_control" in rep:
        print(f"\nBREAK  scope rel {rep['BREAK_control']['scope']['mass_weighted_rel']:.6e} "
              f"over {rep['BREAK_control']['tensors_rolled']} rolled tensors")
    if "A_over_A" in rep:
        print(f"A/A    max |diff| {rep['A_over_A']['max_abs_difference']:.6e}")
    print(f"A16    scope rel {rep['A16_zero_baseline']['scope']['mass_weighted_rel']:.6f} "
          f"exactly_one={rep['A16_zero_baseline']['exactly_one']}")
    print(f"A14    {rep['A14']['tensors_below_the_floor']} below {A14_FLOOR}, "
          f"scope min ref norm {rep['A14']['scope_min_ref_norm']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
