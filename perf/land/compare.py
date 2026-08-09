#!/usr/bin/env python3
"""Score the landing-stack arms against BASE: ratio, bit-exactness, confidence move.

    python3 perf/land/compare.py --model protenix-v2 --size 298 --md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
ORDER = ["L00", "L0", "L1F", "L2", "L2F", "L3", "L3F", "L4", "L4F"]


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", default="298")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()

    recs = {}
    for tag in ORDER:
        js = OUT / f"{tag}_{args.model}_{args.size}.json"
        if js.exists():
            recs[tag] = json.loads(js.read_text())
    if "L0" not in recs:
        return print("no L0 baseline yet") or 1

    base = recs["L0"]
    base_c = np.load(OUT / f"L0_{args.model}_{args.size}" / "coords.npy")
    rows = []
    for tag, r in recs.items():
        c = np.load(OUT / f"{tag}_{args.model}_{args.size}" / "coords.npy")
        exact = c.shape == base_c.shape and np.array_equal(c, base_c)
        rows.append({
            "arm": tag, "commit": r["commit"][:8], "fused": r["fused"],
            "warm_median_s": r["warm_median_s"],
            "ratio": round(base["warm_median_s"] / r["warm_median_s"], 4),
            "ms_per_fold": round((base["warm_median_s"] - r["warm_median_s"]) * 1000, 1),
            "exact": exact,
            "max_abs_delta_A": 0.0 if exact else round(float(np.abs(c - base_c).max()), 4),
            "rmsd_vs_base_A": 0.0 if exact else round(_kabsch_rmsd(c, base_c), 4),
            "plddt": r["confidence"].get("plddt"),
            "intra_run_max_abs_delta_A": r["intra_run_max_abs_delta_A"],
        })

    if not args.md:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"| arm | commit | fused | warm median (s) | vs L0 | ms/fold | bit-exact | "
          f"max abs Δ (Å) | RMSD vs L0 (Å) | pLDDT |")
    print("|---|---|---|---:|---:|---:|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['arm']} | {r['commit']} | {r['fused']} | {r['warm_median_s']} | "
              f"{r['ratio']}x | {r['ms_per_fold']} | {'yes' if r['exact'] else 'no'} | "
              f"{r['max_abs_delta_A']} | {r['rmsd_vs_base_A']} | {r['plddt']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
