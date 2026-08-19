"""The FreeBindCraft wall split that upstream already paid for, recomputed from its own artifacts.

FreeBindCraft ships the accepted-design statistics of a real PDL1 run in
`performance_data/pdl1_miniprotein/`, and its README states the end-to-end wall clock and the
trajectory count for that same run. Each accepted design carries a `DesignTime` field, which is
`mpnn_end_time` in bindcraft.py: everything from the start of one MPNN sequence's block to its end,
so the two AF2 validation predictions (complex and binder-alone, 2 models each), the relax and the
interface scoring. Summing it over the accepted designs and subtracting from the published total
separates the per-design validation stage from everything else, and everything else is dominated by
the gradient hallucination loop.

This is a cross-check, not a substitute for our own measurement: it is upstream's hardware,
upstream's clock, and the remainder still contains the MPNN designs that were rejected before the
full scoring ran. It exists so the number in the feasibility verdict is reproducible from files
anyone can hash, instead of a figure quoted from a README.

    python perf/freebindcraft/published_split.py --repo /path/to/FreeBindCraft
"""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import statistics

# From performance_data/pdl1_miniprotein/README.md, the head-to-head PDL1 run on a single
# B200-class GPU. Both numbers are upstream's, for the same run the CSVs come from.
PUBLISHED = {
    "freebindcraft": {"csv": "pdl1_final_design_stats_freebindcraft.csv", "wall_h": 12.25, "trajectories": 91},
    "pyrosetta": {"csv": "pdl1_final_design_stats_pyrosetta.csv", "wall_h": 33.19, "trajectories": 144},
}

_HMS = re.compile(r"(\d+) hours, (\d+) minutes, (\d+) seconds")


def _seconds(text):
    h, m, s = map(int, _HMS.findall(text)[0])
    return h * 3600 + m * 60 + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout of github.com/cytokineking/FreeBindCraft")
    ap.add_argument("--report", help="write the split to this JSON path")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.repo) / "performance_data" / "pdl1_miniprotein"
    out = {"source": "FreeBindCraft performance_data/pdl1_miniprotein", "arms": {}}

    for arm, meta in PUBLISHED.items():
        path = data_dir / meta["csv"]
        rows = list(csv.DictReader(path.open()))
        times = [_seconds(r["DesignTime"]) for r in rows]
        validation_h = sum(times) / 3600.0
        rest_h = meta["wall_h"] - validation_h
        out["arms"][arm] = {
            "csv": meta["csv"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "accepted_designs": len(rows),
            "trajectories": meta["trajectories"],
            "published_wall_h": meta["wall_h"],
            "mean_design_time_s": round(statistics.mean(times), 1),
            "median_design_time_s": round(statistics.median(times), 1),
            "validation_stage_h": round(validation_h, 2),
            "validation_stage_pct": round(100 * validation_h / meta["wall_h"], 1),
            "remainder_h": round(rest_h, 2),
            "remainder_pct": round(100 * rest_h / meta["wall_h"], 1),
            "remainder_s_per_trajectory": round(rest_h * 3600 / meta["trajectories"]),
            "min_per_accepted_design": round(meta["wall_h"] * 60 / len(rows), 2),
        }
        a = out["arms"][arm]
        print(
            f"{arm:14s} {a['accepted_designs']} accepted / {a['trajectories']} trajectories / "
            f"{a['published_wall_h']} h -> per-design validation+relax+scoring "
            f"{a['validation_stage_h']} h ({a['validation_stage_pct']}%), everything else "
            f"{a['remainder_h']} h ({a['remainder_pct']}%, {a['remainder_s_per_trajectory']} s/trajectory)"
        )

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
