#!/usr/bin/env python3
"""Score trunk parameter gradients against ONE reference, by mass (A23) and never by count.

Every arm handed in is scored against the same reference over the same tensor set, so the arms
order each other and not just the reference. Three things this reports that a plain per-tensor
table does not:

  * the MASS-WEIGHTED headline -- rel_l2 over the CONCATENATION of the compared tensors, with the
    norm ratio and the error cosine beside it. ||concat(x)||^2 = sum ||x_i||^2 and
    <concat(a),concat(b)> = sum <a_i,b_i>, so the concatenated figures come out exact from four
    sums per tensor without holding 2496 of them.
  * how much of the compared reference MASS sits outside the per-tensor bar, beside how many
    tensors do. `of3t-auxgrad` had 89 of 176 tensors over the bar holding 1.17e-07 of the mass;
    the count and the mass said opposite things and the mass decided.
  * where the ERROR mass is. A tensor whose reference gradient is near zero and whose device
    gradient is not contributes unbounded diff_sq and no ref_sq, so it can carry a mass-weighted
    headline on its own. The numerator is decomposed by tensor and by leaf op, and the headline
    is given both over everything scored and over the A14-surviving population.

A26-SCOPE: against a float64 reference only one side carries error, so the reachable bar is the
threshold itself. No sqrt(2) is quoted here and none may rescue a tensor.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

PER_TENSOR_BAR = 5.0e-02
MASS_BAR = 2.0e-02
A14_FLOOR_RATIO = 1e-8

_BLK = re.compile(r"^pairformer_stack\.blocks\.(\d+)\.(.*)$")


def _split(name):
    m = _BLK.match(name)
    if not m:
        return None, name
    blk, rest = int(m.group(1)), m.group(2)
    if rest.startswith("pair_stack."):
        rest = rest[len("pair_stack."):]
    return blk, rest


def _leaf_of(name):
    _, rest = _split(name)
    return rest.split(".")[0]


def _row(name, ours, ref):
    o, r = ours.flatten().double(), ref.flatten().double()
    d = o - r
    on, rn = float(o.norm()), float(r.norm())
    blk, _ = _split(name)
    return {"name": name, "block": blk, "leaf": _leaf_of(name),
            "rel_l2": float(d.norm() / (rn + 1e-300)),
            "norm_ratio": (on / rn) if rn else None,
            "cos": (float(torch.dot(o, r)) / (on * rn)) if (on and rn) else None,
            "ref_norm": rn, "our_norm": on,
            "ref_sq": float((r ** 2).sum()), "our_sq": float((o ** 2).sum()),
            "diff_sq": float((d ** 2).sum()), "dot": float(torch.dot(o, r))}


def _mass(rows):
    if not rows:
        return None
    sd = sum(r["diff_sq"] for r in rows)
    sr = sum(r["ref_sq"] for r in rows)
    so = sum(r["our_sq"] for r in rows)
    dot = sum(r["dot"] for r in rows)
    return {"rel_l2": math.sqrt(sd) / (math.sqrt(sr) + 1e-300),
            "norm_ratio": math.sqrt(so) / (math.sqrt(sr) + 1e-300),
            "cos": dot / (math.sqrt(so) * math.sqrt(sr) + 1e-300),
            "n_tensors": len(rows), "ref_squared_norm": sr, "error_squared_norm": sd}


def _group(rows, key, total_ref_sq, total_diff_sq):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    stats = []
    for k, rs in sorted(out.items(), key=lambda kv: str(kv[0])):
        v = sorted(x["rel_l2"] for x in rs)
        m = _mass(rs)
        stats.append({key: k, "n": len(rs),
                      "median_rel_l2": v[len(v) // 2], "worst_rel_l2": v[-1],
                      "mass_weighted_rel_l2": m["rel_l2"],
                      "share_of_compared_ref_mass": m["ref_squared_norm"] / (total_ref_sq or 1.0),
                      "share_of_error_mass": m["error_squared_norm"] / (total_diff_sq or 1.0),
                      "n_over_bar": sum(1 for x in rs if x["rel_l2"] > PER_TENSOR_BAR),
                      "ref_mass_over_bar": sum(x["ref_sq"] for x in rs
                                               if x["rel_l2"] > PER_TENSOR_BAR)
                      / (m["ref_squared_norm"] or 1.0)})
    stats.sort(key=lambda d: -d["share_of_error_mass"])
    return stats


def score(ref, arm, keys, label):
    rows = [_row(k, arm[k], ref[k]) for k in keys]
    all_mass = _mass(rows)
    norms = sorted(r["ref_norm"] for r in rows) or [1.0]
    med_ref = norms[len(norms) // 2]
    floor = A14_FLOOR_RATIO * med_ref
    kept = [r for r in rows if r["ref_norm"] >= floor]
    a14 = [r for r in rows if r["ref_norm"] < floor]
    kept_mass = _mass(kept)
    tot_ref_sq = all_mass["ref_squared_norm"]
    tot_diff_sq = all_mass["error_squared_norm"]
    over = [r for r in rows if r["rel_l2"] > PER_TENSOR_BAR]
    rel_sorted = sorted(r["rel_l2"] for r in rows)
    by_err = sorted(rows, key=lambda r: -r["diff_sq"])
    worst = max(rows, key=lambda r: r["rel_l2"])
    return {
        "arm": label,
        "n_scored": len(rows),
        "mass_weighted": all_mass,
        "mass_weighted_a14_survivors": kept_mass,
        "verdict_mass_bar": "PASS" if all_mass["rel_l2"] <= MASS_BAR else "FAIL",
        "median_rel_l2_over_tensors": rel_sorted[len(rel_sorted) // 2],
        "per_tensor_bar": PER_TENSOR_BAR, "mass_bar": MASS_BAR,
        "n_over_per_tensor_bar": len(over),
        "ref_mass_outside_per_tensor_bar":
            sum(r["ref_sq"] for r in over) / (tot_ref_sq or 1.0),
        "ref_mass_inside_per_tensor_bar":
            sum(r["ref_sq"] for r in rows if r["rel_l2"] <= PER_TENSOR_BAR) / (tot_ref_sq or 1.0),
        "a14": {"floor": floor, "median_ref_norm": med_ref, "n_excluded": len(a14),
                "excluded_share_of_ref_mass":
                    sum(r["ref_sq"] for r in a14) / (tot_ref_sq or 1.0),
                "excluded_share_of_error_mass":
                    sum(r["diff_sq"] for r in a14) / (tot_diff_sq or 1.0),
                "excluded_worst": [{"name": r["name"], "rel_l2": r["rel_l2"],
                                    "ref_norm": r["ref_norm"], "our_norm": r["our_norm"]}
                                   for r in sorted(a14, key=lambda x: -x["diff_sq"])[:8]]},
        "worst_tensor": worst,
        "error_mass_top20": [{"name": r["name"], "leaf": r["leaf"], "block": r["block"],
                              "share_of_error_mass": r["diff_sq"] / (tot_diff_sq or 1.0),
                              "share_of_ref_mass": r["ref_sq"] / (tot_ref_sq or 1.0),
                              "rel_l2": r["rel_l2"], "norm_ratio": r["norm_ratio"],
                              "cos": r["cos"]} for r in by_err[:20]],
        "by_leaf": _group(rows, "leaf", tot_ref_sq, tot_diff_sq),
        "by_block": _group(rows, "block", tot_ref_sq, tot_diff_sq),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-label", default="")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--zero-arm", default="", metavar="NAME",
                    help="A16, measured: an arm whose every gradient is zero")
    ap.add_argument("--self-arm", default="", metavar="NAME",
                    help="the instrument floor: the reference file loaded a second time from "
                         "disk and scored against itself")
    ap.add_argument("--intersect", action="store_true",
                    help="score every arm over the intersection of all arms' reached keys, so "
                         "the arms share a denominator even when a lever changes what the "
                         "bijection can place")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()

    ref = torch.load(a.reference, map_location="cpu", weights_only=False)
    ref = {k: v.double() for k, v in ref.items() if v is not None}
    full_ref_sq = sum(float((v ** 2).sum()) for v in ref.values())

    arms = {}
    for spec in a.arm:
        name, _, path = spec.partition("=")
        d = torch.load(path, map_location="cpu", weights_only=False)
        arms[name] = {"path": path, "grads": {k: v.double() for k, v in d.items()
                                              if v is not None}}
        print(f"[{time.perf_counter()-t0:.0f}s] loaded {name}: {len(arms[name]['grads'])} "
              f"tensors from {path}", flush=True)

    shared = None
    if a.intersect and arms:
        shared = set(ref)
        for v in arms.values():
            shared &= set(v["grads"])
        shared = {k for k in shared if tuple(ref[k].shape) == tuple(
            next(iter(arms.values()))["grads"][k].shape)}

    out = {"what": __doc__.strip().splitlines()[0],
           "reference": {"file": a.reference, "label": a.reference_label,
                         "n_tensors": len(ref), "squared_norm": full_ref_sq,
                         "global_norm": full_ref_sq ** 0.5},
           "bars": {"per_tensor": PER_TENSOR_BAR, "mass_weighted": MASS_BAR,
                    "a26_scope": "against a float64 reference only one side carries error, so "
                                 "the reachable bar is the threshold itself; sqrt(2) does not "
                                 "apply and is not quoted"},
           "scoring_set": ("intersection of every arm's reached keys" if a.intersect
                           else "each arm's own reached keys"),
           "note": a.note, "arms": {}}

    for name, v in arms.items():
        keys = sorted((shared if shared is not None
                       else {k for k in v["grads"] if k in ref}))
        keys = [k for k in keys if tuple(v["grads"][k].shape) == tuple(ref[k].shape)]
        r = score(ref, v["grads"], keys, name)
        cmp_sq = r["mass_weighted"]["ref_squared_norm"]
        r["source"] = v["path"]
        r["reach"] = {
            "rule": "A20: an unplaceable tensor is UNREACHED, not absent from the denominator",
            "compared_tensors": len(keys), "reference_tensors": len(ref),
            "squared_norm_compared": cmp_sq, "squared_norm_full_reference": full_ref_sq,
            "reach_over_full_reference_mass": cmp_sq / (full_ref_sq or 1.0),
            "unreached": sorted(
                ({"tensor": k, "ref_norm": float(ref[k].norm()),
                  "share_of_full_reference_mass": float((ref[k] ** 2).sum()) / (full_ref_sq or 1.0)}
                 for k in set(ref) - set(keys)),
                key=lambda d: -d["share_of_full_reference_mass"])[:12],
            "unreached_share_of_full_reference_mass": 1.0 - cmp_sq / (full_ref_sq or 1.0)}
        out["arms"][name] = r
        m = r["mass_weighted"]
        print(f"  {name:22s} mass-weighted {m['rel_l2']:.6e}  r {m['norm_ratio']:.6f}  "
              f"cos {m['cos']:.6f}  over-bar {r['n_over_per_tensor_bar']}/{r['n_scored']} "
              f"holding {r['ref_mass_outside_per_tensor_bar']*100:.4f} % of the compared mass  "
              f"reach {r['reach']['reach_over_full_reference_mass']*100:.3f} %", flush=True)

    if a.zero_arm:
        keys = sorted(shared) if shared is not None else sorted(ref)
        zero = {k: torch.zeros_like(ref[k]) for k in keys}
        r = score(ref, zero, keys, a.zero_arm)
        r["source"] = "A16, measured: every gradient replaced by zeros"
        out["arms"][a.zero_arm] = r
        print(f"  {a.zero_arm:22s} mass-weighted {r['mass_weighted']['rel_l2']:.6e} "
              f"(A16 expects exactly 1.0)", flush=True)

    if a.self_arm:
        keys = sorted(shared) if shared is not None else sorted(ref)
        again = torch.load(a.reference, map_location="cpu", weights_only=False)
        again = {k: again[k].double() for k in keys}
        r = score(ref, again, keys, a.self_arm)
        r["source"] = "the instrument floor: the reference re-read from disk, scored on itself"
        out["arms"][a.self_arm] = r
        print(f"  {a.self_arm:22s} mass-weighted {r['mass_weighted']['rel_l2']:.6e} "
              f"(the instrument floor; anything but 0.0 is the scorer's own arithmetic)",
              flush=True)

    out["seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
