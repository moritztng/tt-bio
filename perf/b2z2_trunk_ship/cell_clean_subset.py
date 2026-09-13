#!/usr/bin/env python3
"""Recompute a cell_ab.py run over the reps the box was actually quiet for.

benchlock admits a run below loadavg 2.0 and then stops looking, so a foreign fold that starts
mid-run lands inside the sample without flagging it. That happened to the guarded 512 aa cell on
2026-09-13: reps 0-2 ran at loadavg 2.3-3.8, which is this fold plus the box, and rep 3 ran at
6.3-8.7 with folds 3.3 s longer in both arms.

The cut is on the CAUSE, not on the outcome: a rep is dropped if any of its four folds read a
one-minute loadavg above --maxload, a threshold fixed at benchlock admission ceiling + 3.0, which
is one concurrent fold of headroom over the quiet band. It does not look at the timings. Ratios
are recomputed exactly as cell_ab.py computes them, over the surviving reps.

    cell_clean_subset.py --in <cell.json> --out <clean.json> [--maxload 5.0]
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path

ORDER = ["off", "on", "on", "off"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--maxload", type=float, default=5.0)
    a = ap.parse_args()
    d = json.loads(a.src.read_text())
    timed = [r for r in d["runs"] if not r["warmup"]]
    reps: dict = {}
    for r in timed:
        reps.setdefault(r["rep"], []).append(r)

    kept, dropped = [], []
    for i in sorted(reps):
        g = reps[i]
        peak = max(r["loadavg1"] for r in g)
        (dropped if peak > a.maxload else kept).append(
            {"rep": i, "peak_loadavg1": peak,
             "folds": [(r["arm"], r["fold_s"], r["loadavg1"]) for r in g]})

    live = [reps[k["rep"]] for k in kept]
    ab, aa = [], []
    for g in live:
        assert [x["arm"] for x in g] == ORDER, [x["arm"] for x in g]
        off, on = [g[0]["fold_s"], g[3]["fold_s"]], [g[1]["fold_s"], g[2]["fold_s"]]
        ab += [round(o / n, 5) for o, n in zip(off, on)]
        aa += [round(max(off) / min(off), 5), round(max(on) / min(on), 5)]
    folds = [r for g in live for r in g]
    med = {arm: st.median([r["fold_s"] for r in folds if r["arm"] == arm]) for arm in ("off", "on")}
    shas = {arm: sorted({r["cif_sha256"] for r in folds if r["arm"] == arm}) for arm in ("off", "on")}
    served = {arm: {k: [sum(r[k][j] for r in folds if r["arm"] == arm) for j in (0, 1)]
                    for k in ("qkvg", "qkvgb", "gout")} for arm in ("off", "on")}
    out = {
        "source": str(a.src), "source_env": d["env"], "maxload": a.maxload,
        "reps_kept": kept, "reps_dropped": dropped,
        "folds_kept": len(folds),
        "median_fold_s": med,
        "ratio_median": round(med["off"] / med["on"], 5),
        "ab_paired_ratios": ab,
        "ab_paired_median": round(st.median(ab), 5),
        "ab_paired_all_positive": all(x > 1 for x in ab),
        "aa_ratios": aa, "aa_floor_max": max(aa) if aa else None,
        "plddt": {arm: sorted({r["plddt"] for r in folds if r["arm"] == arm}) for arm in ("off", "on")},
        "cif_sha256": shas,
        "bit_exact": len(shas["off"]) == 1 and shas["off"] == shas["on"],
        "served": served,
    }
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in (
        "folds_kept", "median_fold_s", "ratio_median", "ab_paired_median", "ab_paired_ratios",
        "ab_paired_all_positive", "aa_floor_max", "bit_exact", "served")}, indent=1))
    for r in dropped:
        print("dropped rep%d: peak loadavg1 %s %s" % (r["rep"], r["peak_loadavg1"], r["folds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
