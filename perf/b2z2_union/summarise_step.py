#!/usr/bin/env python3
"""Score an `ab_union.py --mode timing` session three ways, because the box was not quiet.

The global-median ratio every earlier row quotes assumes the base arm is stationary across the
session. On a contended host it is not: this session's base folds drift with a co-tenant's load,
and a global median charges that drift to whichever arm happened to sit in the noisy stretch.

So three estimators, printed side by side, and the row quotes the one it can defend:

  global    median(base over the whole session) / median(arm)        — what the parent quoted
  paired    per fold, the mean of the base folds ADJACENT to it IN ITS OWN REP, then the median
            of those per-fold ratios. A co-tenant that arrives inside one rep moves the arm and
            its two neighbours together and largely cancels.
  quiet     global, recomputed over only the folds whose loadavg_before[0] is at or below the
            session median — the subset the box was quietest for.

Every ratio is reported for the fold, for the PairformerLayer wall and for the 200-step sampler
wall, because a step lever can move the step and be invisible on the fold.

    summarise_step.py <timing json> [--out <json>]
"""
import argparse, json, statistics as st
from pathlib import Path

KEYS = (("fold_s", "fold"), ("block_s", "block"), ("step_s", "step"))


def ratios_global(warm, key, subset=None):
    rows = subset if subset is not None else warm
    base = [r[key] for r in rows if r["arm"] == "base"]
    if not base:
        return {}
    b = st.median(base)
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        v = [r[key] for r in rows if r["arm"] == arm]
        if v:
            out[arm] = {"n": len(v), "median": round(st.median(v), 4),
                        "ratio": round(b / st.median(v), 5)}
    return out


def ratios_paired(warm, key, order):
    """Each arm fold against the base folds adjacent to it in its own rep."""
    base_pos = [p for p, a in enumerate(order) if a == "base"]
    by_rep = {}
    for r in warm:
        by_rep.setdefault(r["rep"], {})[r["pos"]] = r
    per_arm = {}
    for rep, folds in by_rep.items():
        for pos, r in folds.items():
            if r["arm"] == "base":
                continue
            near = [p for p in base_pos if p in folds]
            if not near:
                continue
            d = min(abs(p - pos) for p in near)
            nb = [folds[p][key] for p in near if abs(p - pos) == d]
            per_arm.setdefault(r["arm"], []).append(st.mean(nb) / r[key])
    return {a: {"n": len(v), "ratio": round(st.median(v), 5),
                "min": round(min(v), 5), "max": round(max(v), 5)}
            for a, v in sorted(per_arm.items())}


def aa_floor(warm, key):
    pos = sorted({r["pos"] for r in warm if r["arm"] == "base"})
    p0 = [r[key] for r in warm if r["arm"] == "base" and r["pos"] == pos[0]]
    fl = {}
    for p in pos[1:]:
        v = [r[key] for r in warm if r["arm"] == "base" and r["pos"] == p]
        if v and p0:
            fl[p] = round(max(st.median(p0) / st.median(v), st.median(v) / st.median(p0)), 5)
    return (max(fl.values()) if fl else None), fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = json.loads(a.json.read_text())
    warm = [r for r in d["runs"] if not r.get("cold")]
    order = d["env"]["protocol"]["order"].split(",")
    loads = sorted(r["loadavg_before"][0] for r in warm)
    med_load = st.median(loads)
    quiet = [r for r in warm if r["loadavg_before"][0] <= med_load]

    rep = {"commit": d["env"]["commit"], "arch": d["env"]["arch"], "grid": d["env"]["grid"],
           "n_warm": len(warm), "reps_completed": max(r["rep"] for r in warm) + 1,
           "order": order,
           "loadavg_1m": {"min": min(loads), "median": med_load, "max": max(loads)},
           "n_quiet": len(quiet),
           "foreign_pids": sorted({p for r in warm for p in r["occupancy"].get("pids", [])}),
           "sha_by_arm": {arm: sorted({r["sha256"] for r in warm if r["arm"] == arm})
                          for arm in sorted({r["arm"] for r in warm})},
           "step_n_all_200": all(r["step_n"] == 200 for r in warm),
           "block_n_all_280": all(r["block_n"] == 280 for r in warm)}
    for key, name in KEYS:
        fl, per = aa_floor(warm, key)
        rep[name] = {"aa_floor": fl, "aa_floor_per_position": per,
                     "global": ratios_global(warm, key),
                     "paired": ratios_paired(warm, key, order),
                     "quiet": ratios_global(warm, key, quiet)}
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))

    print("commit {}  {}  {}x{}  warm={} reps={}".format(
        rep["commit"][:8], rep["arch"], *rep["grid"], rep["n_warm"], rep["reps_completed"]))
    print("loadavg 1m  min {:.2f}  median {:.2f}  max {:.2f}   step_n==200 everywhere: {}".format(
        *rep["loadavg_1m"].values(), rep["step_n_all_200"]))
    for key, name in KEYS:
        s = rep[name]
        print("\n=== {}   A/A floor {}   (positions {})".format(
            name.upper(), s["aa_floor"], s["aa_floor_per_position"]))
        print("  {:8s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
            "arm", "n", "global", "paired", "quiet"))
        for arm in sorted(s["global"]):
            g = s["global"][arm]["ratio"]
            p = s["paired"].get(arm, {}).get("ratio", float("nan"))
            q = s["quiet"].get(arm, {}).get("ratio", float("nan"))
            print("  {:8s} {:8d} {:10.5f} {:10.5f} {:10.5f}  median {:.3f}s".format(
                arm, s["global"][arm]["n"], g, p, q, s["global"][arm]["median"]))

    # Composition, on the paired estimator, which is the one the contention argues for.
    pf = rep["fold"]["paired"]
    combos = [("UNION x LN", ("UNION", "LN"), "U_LN"),
              ("UNION x LAY", ("UNION", "LAY"), "U_LAY"),
              ("UNION x LN x LAY", ("UNION", "LN", "LAY"), "STACK"),
              ("U_LN x LAY", ("U_LN", "LAY"), "STACK")]
    print("\n=== COMPOSITION on the fold, paired estimator")
    for name, parts, whole in combos:
        if whole in pf and all(p in pf for p in parts):
            prod = 1.0
            for p in parts:
                prod *= pf[p]["ratio"]
            m = pf[whole]["ratio"]
            print("  {:20s} product {:.5f}  measured {:.5f}  discount {:+.2f} %".format(
                name, prod, m, 100 * (prod / m - 1)))


if __name__ == "__main__":
    raise SystemExit(main())
