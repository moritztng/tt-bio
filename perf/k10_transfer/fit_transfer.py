#!/usr/bin/env python3
"""Fit the WH -> BH transfer from paired lever ratios, and state its spread.

Levers are small, so the useful quantity is not the ratio but the MARGIN it carries, r - 1: a lever
worth 4 % and one worth 0.4 % are a factor of ten apart in margin and 3.6 percentage points apart in
ratio. The transfer is therefore fitted on margins,

    margin_BH = k * margin_WH

with k estimated per point as the margin ratio, and reported as a median with the full range, not a
least-squares line: four points cannot support a line, and the spread IS the deliverable.

A point is admitted only if BOTH arms cleared their own session's A/A floor. A lever inside the
floor has no measured margin on that part, so it constrains nothing and is listed as unresolved
rather than folded into the fit with a noise-sized margin.
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path


def load(p: Path) -> dict:
    d = json.loads(p.read_text())
    return d


def rec(d: dict) -> dict:
    return {"lever": d["env"]["lever"], "kind": d["env"]["kind"],
            "host": d["env"]["host"], "card": d["env"]["card"],
            "ratio": d["ratio"], "floor": d.get("aa_floor_worst"),
            "resolved": bool(d.get("separates_from_floor")),
            "engagement": d.get("engagement_control"),
            "bit_exact": d.get("bit_exact"),
            "median_fold_s": d["median_fold_s"],
            "commit": d["env"]["commit"][:12], "width": d["env"].get("width")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wh", type=Path, nargs="+", required=True)
    ap.add_argument("--bh", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wh = {r["lever"]: r for r in (rec(load(p)) for p in args.wh)}
    bh = {r["lever"]: r for r in (rec(load(p)) for p in args.bh)}

    points, unresolved = [], []
    for lev in sorted(set(wh) | set(bh)):
        w, b = wh.get(lev), bh.get(lev)
        row = {"lever": lev, "kind": (w or b)["kind"],
               "wh": w, "bh": b, "paired": bool(w and b)}
        if w and b and w["resolved"] and b["resolved"]:
            mw, mb = w["ratio"] - 1.0, b["ratio"] - 1.0
            row["margin_wh_pp"] = round(100 * mw, 3)
            row["margin_bh_pp"] = round(100 * mb, 3)
            row["k"] = round(mb / mw, 3) if mw else None
            points.append(row)
        else:
            why = []
            if not w:
                why.append("no WH arm")
            elif not w["resolved"]:
                why.append(f"WH {w['ratio']:.5f} inside its floor {w['floor']:.5f}")
            if not b:
                why.append("no BH arm")
            elif not b["resolved"]:
                why.append(f"BH {b['ratio']:.5f} inside its floor {b['floor']:.5f}")
            row["unresolved_because"] = "; ".join(why)
            unresolved.append(row)

    out = {"doc": __doc__, "points": points, "unresolved": unresolved,
           "n_paired_resolved": len(points)}
    ks = [p["k"] for p in points if p["k"] is not None]
    if ks:
        out["transfer"] = {
            "k_median": round(st.median(ks), 3),
            "k_min": round(min(ks), 3), "k_max": round(max(ks), 3),
            "k_spread_factor": round(max(ks) / min(ks), 2) if min(ks) > 0 else None,
            "n": len(ks),
        }
        by_kind: dict[str, list] = {}
        for p in points:
            by_kind.setdefault(p["kind"], []).append(p["k"])
        out["by_kind"] = {k: {"k_median": round(st.median(v), 3),
                              "k_min": round(min(v), 3), "k_max": round(max(v), 3),
                              "n": len(v), "levers": [p["lever"] for p in points
                                                      if p["kind"] == k]}
                          for k, v in by_kind.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "doc"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
