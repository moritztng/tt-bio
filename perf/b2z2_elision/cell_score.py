#!/usr/bin/env python3
"""Score an ABBA cell run: the paired A/B, and the A/A floor of that same estimator.

A ratio of medians and a floor taken from a spread are two different statistics, and quoting one
against the other is how a lever gets called real or unreal for the wrong reason. The ABBA order
gives both from one run: within a rep the two `off` folds bracket the two `on` folds, so
`off_first` against `off_last` is an A/A pair drawn the same way the A/B pairs are.

Derivation only, no measurement. Reads the runs the harness wrote and writes `derived` back.
"""
import json
import statistics as st
import sys
from pathlib import Path

PUBLISHED_S = 20.113          # site/data/perf-512aa.json, boltz2 p150a, as of 84da2a49


def pairs(v):
    return [round(a - b, 4) for a, b in zip(v[::2], v[1::2])]


def main(path: Path) -> int:
    d = json.loads(path.read_text())
    timed = [r for r in d["runs"] if not r["warmup"]]
    rep = {}
    for r in timed:
        rep.setdefault(r["rep"], []).append(r)

    arms = {}
    for a in ("off", "on"):
        v = sorted(r["fold_s"] for r in timed if r["arm"] == a)
        arms[a] = {"n": len(v), "median_s": st.median(v), "mean_s": round(st.mean(v), 4),
                   "min_s": v[0], "max_s": v[-1],
                   "spread_pct": round(100 * (v[-1] - v[0]) / v[0], 2)}

    ab, aa = [], []
    for i in sorted(rep):
        r = rep[i]
        assert [x["arm"] for x in r] == ["off", "on", "on", "off"], [x["arm"] for x in r]
        off, on = [r[0]["fold_s"], r[3]["fold_s"]], [r[1]["fold_s"], r[2]["fold_s"]]
        ab += [round(o / n, 5) for o, n in zip(off, on)]
        aa += [round(max(off) / min(off), 5), round(max(on) / min(on), 5)]

    ctl = arms["off"]["median_s"]
    d["derived"] = {
        "doc": __doc__,
        "arms": arms,
        "ratio_median": round(arms["off"]["median_s"] / arms["on"]["median_s"], 5),
        "ab_paired_ratios": ab,
        "ab_paired_median": round(st.median(ab), 5),
        "ab_paired_all_positive": all(x > 1 for x in ab),
        "aa_paired_ratios": aa,
        "aa_paired_max": max(aa),
        "aa_paired_median": round(st.median(aa), 5),
        "published_cell_s": PUBLISHED_S,
        "control_vs_published_s": round(ctl - PUBLISHED_S, 4),
        "lever_s": round(ctl - arms["on"]["median_s"], 4),
        "plddt": sorted({r["plddt"] for r in timed}),
        "cif_sha256": sorted({r["cif_sha256"] for r in timed}),
        "gather_stats": sorted({tuple(r["gather_stats"]) for r in timed}),
    }
    path.write_text(json.dumps(d, indent=1))
    print(json.dumps(d["derived"], indent=1, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
