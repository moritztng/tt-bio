#!/usr/bin/env python3
"""Fold the null-lever width ladder into the number that sets how many agents this box can host.

Throughput is the wrong statistic for this question. Folds per hour says how much work a box does;
it does not say whether a 1.005x lever measured on one of its chips is real. The statistic that does
is the NULL RATIO: both arms of the null lever are the shipped default, so its true value is 1.0 and
whatever a chip reads instead is the noise a paired A/B carries at that width.

Reported per width: the worst null ratio any chip read, the median, and the smallest lever margin
that clears the worst chip. A width whose worst null read is 1.02 cannot screen a 1.01 lever on
every chip, however many folds per hour it turns out.
"""
from __future__ import annotations

import argparse, json, re, statistics as st
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    widths: dict[int, list] = {}
    for p in sorted(args.dir.glob("w*_c*.json")):
        m = re.match(r"w(\d+)_c(\d+)\.json", p.name)
        d = json.loads(p.read_text())
        if "ratio" not in d:
            continue
        widths.setdefault(int(m.group(1)), []).append({
            "card": m.group(2), "null_ratio": d["ratio"],
            "aa_floor_worst": d.get("aa_floor_worst"),
            "median_fold_s": st.median(d["median_fold_s"].values()),
            "fold_s": [r["fold_s"] for r in d["runs"] if not r["warmup"]],
            "omp": d["env"].get("omp_num_threads"),
            "digest": sorted({r["cif_sha256"][:16] for r in d["runs"]}),
        })

    rows = []
    for w in sorted(widths):
        cs = widths[w]
        dev = [abs(c["null_ratio"] - 1.0) for c in cs]
        folds = [f for c in cs for f in c["fold_s"]]
        floors = [c["aa_floor_worst"] for c in cs if c["aa_floor_worst"]]
        rows.append({
            "width": w, "n_chips_reporting": len(cs),
            "null_ratio_worst": round(1 + max(dev), 5),
            "null_ratio_median": round(1 + st.median(dev), 5),
            "aa_floor_worst": round(max(floors), 5) if floors else None,
            "median_fold_s": round(st.median(folds), 3),
            "min_fold_s": round(min(folds), 3), "max_fold_s": round(max(folds), 3),
            "fold_spread_x": round(max(folds) / min(folds), 4),
            "cv_pct": round(100 * st.pstdev(folds) / st.mean(folds), 3),
            "smallest_resolvable_margin_pp": round(100 * max(dev), 3),
            "omp_num_threads": sorted({c["omp"] for c in cs}),
            "digests": sorted({d for c in cs for d in c["digest"]}),
        })
    out = {"doc": __doc__, "widths": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
