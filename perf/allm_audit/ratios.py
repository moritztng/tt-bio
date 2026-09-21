#!/usr/bin/env python3
"""The deliverable, computed from the result files rather than typed into the state doc.

One transfer ratio per model: the old arm's figure over the new arm's, both measured on qb2
card 1 at a DURING-sampled 1350 MHz through the same instrument. Folds carry seconds per fold
and designs carry seconds per design, and this script never converts between the two units --
a design is a different unit with different batching
(design-row-throughput-denies-gpu-its-batch).

The A/A floor printed beside each ratio is that arm's own within-session spread, so a ratio
smaller than its floor is a spread rather than an effect.

    ratios.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FOLDS = [("ESMFold2", "esm"), ("OpenDDE", "odd"), ("OpenFold3", "of3")]
DESIGNS = [("BoltzGen", "bg"), ("RFdiffusion3", "rfd3")]


def fold_arm(out: Path, tag: str):
    p = out / f"{tag}_s1.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    s = d.get("summary")
    if not s:
        return None
    return {"s": s["median_fold_s"], "floor_pct": s["aa_floor_pct"],
            "floor_s": s["aa_floor_s"], "n": s["n"],
            "clk_mean": s["aiclk_mean_over_timed"], "clk_min": s["aiclk_min_over_timed"],
            "digest": "/".join(sorted(s["digests"])), "metric": sorted(s["plddts"]),
            "clean": s["clean_session"], "cotenanted": s["cotenanted_folds"],
            "tree": d["tree"], "started": d["started_utc"]}


def design_arm(out: Path, tag: str):
    p = out / f"{tag}_s1.jsonl"
    if not p.is_file() or not p.read_text().strip():
        return None
    rec = json.loads([x for x in p.read_text().splitlines() if x.strip()][-1])
    clk = out / f"{tag}_s1_clk.json"
    c = json.loads(clk.read_text()) if clk.is_file() else {}
    return {"s": rec["s_per_design"], "floor_pct": rec["spread_pct"],
            "floor_s": round(rec["chunk_s_max"] - rec["chunk_s_min"], 3)
            if "chunk_s_max" in rec else
            round(rec["s_per_design_max"] - rec["s_per_design_min"], 3),
            "n": rec.get("n_warm"), "clk_mean": c.get("aiclk_mean"), "clk_min": c.get("aiclk_min"),
            # a design has no CIF digest, so the comparable output signature is the per-design
            # atom counts. "-" on both arms made every design pair read "identical" for free.
            "digest": "atoms " + ",".join(str(x) for x in (rec.get("atoms") or [])),
            "metric": rec.get("atoms"), "clean": rec.get("output_ok"),
            "cotenanted": len(c.get("foreign_at_start") or []),
            "reasserts": c.get("clock_reasserts")}


def _span(xs):
    return "%d-%d" % (min(xs), max(xs)) if xs else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/home/ttuser/allm_audit/out"))
    a = ap.parse_args()
    rows = []
    for name, tag in FOLDS:
        rows.append((name, "s/fold", fold_arm(a.out, f"{tag}_old"), fold_arm(a.out, f"{tag}_new")))
    for name, tag in DESIGNS:
        rows.append((name, "s/design",
                     design_arm(a.out, f"{tag}_old"), design_arm(a.out, f"{tag}_new")))

    print(f"{'model':<14} {'unit':<9} {'old':>10} {'new':>10} {'ratio':>9}  "
          f"{'A/A old':>8} {'A/A new':>8}  clock  digest/output")
    for name, unit, old, new in rows:
        if not (old and new):
            have = "old only" if old else ("new only" if new else "neither arm")
            print(f"{name:<14} {unit:<9} {'-':>10} {'-':>10} {'pending':>9}  ({have})")
            continue
        ratio = old["s"] / new["s"]
        eff = abs(old["s"] - new["s"])
        floor = max(old["floor_s"], new["floor_s"])
        clk = "OK" if all(x == 1350 for x in (old["clk_min"], new["clk_min"])) else "BAD"
        if old["digest"] == new["digest"]:
            dig = "identical" if unit == "s/fold" else "atoms identical design for design"
        elif unit == "s/design":
            dig = "atoms differ: old %s, new %s" % (_span(old["metric"]), _span(new["metric"]))
        else:
            dig = f"{old['digest']}->{new['digest']}"
        print(f"{name:<14} {unit:<9} {old['s']:>10.3f} {new['s']:>10.3f} {ratio:>8.4f}x  "
              f"{old['floor_pct']:>7.2f}% {new['floor_pct']:>7.2f}%  {clk:<5}  {dig}")
        print(f"{'':<14} effect {eff:.3f} {unit.split('/')[0]}, "
              f"{eff / floor if floor else float('inf'):.1f}x the larger floor; "
              f"n={old['n']}/{new['n']}, clk_mean {old['clk_mean']}/{new['clk_mean']}, "
              f"clean {old['clean']}/{new['clean']}, cotenanted {old['cotenanted']}/"
              f"{new['cotenanted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
