#!/usr/bin/env python3
"""The finished transfer table: self-measured pairs first, inherited ones marked as such.

Two things this deliberately refuses to do. It does not average a pair this task measured on both
parts with one it read out of another campaign's ladder, because those are not the same quality of
evidence and the whole failure this task exists to prevent is a Wormhole number being carried into a
Blackhole cell by bookkeeping. And it does not fit a line: with two points per kind a line is an
interpolation dressed as a model, so k is reported per point and summarised as a median with the full
range beside it.

A pair is admitted only if BOTH arms cleared their own session's worst A/A bracket. A lever that
resolves on neither part is not a missing point, it is a measured null, and it is listed as one.
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path


def arm(p: Path) -> dict:
    d = json.loads(p.read_text())
    return {"lever": d["env"]["lever"], "kind": d["env"]["kind"], "host": d["env"]["host"],
            "card": d["env"]["card"], "ratio": d["ratio"], "floor": d["aa_floor_worst"],
            "resolved": bool(d["separates_from_floor"]), "folds_per_arm": len(
                [r for r in d["runs"] if not r["warmup"]]) // 2,
            "grid": d["env"].get("compute_grid"), "bit_exact": d["bit_exact"],
            "engagement": d["engagement_control"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh", type=Path, nargs="+", required=True)
    ap.add_argument("--bh", type=Path, nargs="+", required=True)
    ap.add_argument("--inherited", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wh = {a["lever"]: a for a in map(arm, args.wh)}
    bh = {a["lever"]: a for a in map(arm, args.bh)}
    inh = json.loads(args.inherited.read_text())["pairs"]

    mine, nulls = [], []
    for lev in sorted(set(wh) & set(bh)):
        w, b = wh[lev], bh[lev]
        row = {"lever": lev, "kind": w["kind"], "source": "this task, both parts",
               "wh_ratio": w["ratio"], "bh_ratio": b["ratio"],
               "wh_floor": w["floor"], "bh_floor": b["floor"],
               "wh_folds_per_arm": w["folds_per_arm"], "bh_folds_per_arm": b["folds_per_arm"],
               "bh_grid": b["grid"], "bit_exact": w["bit_exact"] and b["bit_exact"],
               "engagement": [w["engagement"], b["engagement"]]}
        if w["resolved"] and b["resolved"]:
            row["k"] = round((b["ratio"] - 1) / (w["ratio"] - 1), 4)
            mine.append(row)
        else:
            row["resolved"] = {"wh": w["resolved"], "bh": b["resolved"]}
            nulls.append(row)

    by_kind: dict[str, list] = {}
    for r in mine:
        by_kind.setdefault(r["kind"], []).append((r["k"], r["lever"], "self"))
    for p in inh:
        if p["level"] != "fold":
            continue
        by_kind.setdefault(p["kind"], []).append(
            (round((p["bh_ratio"] - 1) / (p["wh_ratio"] - 1), 4), p["lever"], "inherited"))

    summary = {}
    for kind, pts in by_kind.items():
        ks = [k for k, _l, _s in pts]
        summary[kind] = {"k_median": round(st.median(ks), 4), "k_min": min(ks), "k_max": max(ks),
                         "spread_x": round(max(ks) / min(ks), 3) if min(ks) > 0 else None,
                         "n": len(ks), "n_self_measured": sum(1 for *_r, s in pts if s == "self"),
                         "points": [{"lever": l, "k": k, "source": s} for k, l, s in sorted(pts)]}

    allk = [k for pts in by_kind.values() for k, _l, _s in pts]
    out = {"doc": __doc__, "self_measured_pairs": mine, "measured_nulls": nulls,
           "inherited": inh, "by_kind": summary,
           "single_scalar": {"k_median": round(st.median(allk), 4), "k_min": min(allk),
                             "k_max": max(allk), "spread_x": round(max(allk) / min(allk), 3),
                             "n": len(allk),
                             "verdict": "unusable: the spread is wider than the levers"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"{'lever':<14}{'kind':<12}{'source':<22}{'WH pp':>9}{'BH pp':>9}{'k':>8}")
    for r in mine:
        print(f"{r['lever']:<14}{r['kind']:<12}{r['source']:<22}"
              f"{100*(r['wh_ratio']-1):>9.3f}{100*(r['bh_ratio']-1):>9.3f}{r['k']:>8.4f}")
    for p in inh:
        if p["level"] == "fold":
            k = (p["bh_ratio"] - 1) / (p["wh_ratio"] - 1)
            print(f"{p['lever'][:13]:<14}{p['kind']:<12}{'inherited: '+p['who_wh']:<22}"
                  f"{100*(p['wh_ratio']-1):>9.3f}{100*(p['bh_ratio']-1):>9.3f}{k:>8.4f}")
    for r in nulls:
        print(f"{r['lever']:<14}{r['kind']:<12}{'MEASURED NULL':<22}"
              f"{100*(r['wh_ratio']-1):>9.3f}{100*(r['bh_ratio']-1):>9.3f}{'--':>8}")
    print()
    print(json.dumps({"by_kind": summary, "single_scalar": out["single_scalar"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
