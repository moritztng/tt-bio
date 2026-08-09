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
# Session-1 tags, then their session-2 repeats (suffix b) and the opendde arms (suffix o).
# A tag is a label, not a file: fold_arm.py keys its JSON on (tag, model, size), so the same
# arm name is reused across models and only the sessions need distinguishing.
_ARMS = ["L00", "L0", "L1F", "L2", "L2F", "L3", "L3F", "L4", "L4F"]
ORDER = _ARMS + [t + s for s in ("b", "c", "o") for t in _ARMS]


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
    ap.add_argument("--base", default=None,
                    help="baseline tag; defaults to L0, or L0o when that is all there is")
    args = ap.parse_args()

    recs = {}
    for tag in ORDER:
        js = OUT / f"{tag}_{args.model}_{args.size}.json"
        if js.exists():
            recs[tag] = json.loads(js.read_text())
    btag = args.base or ("L0" if "L0" in recs else "L0o")
    if btag not in recs:
        return print(f"no {btag} baseline yet") or 1

    base = recs[btag]
    base_c = np.load(OUT / f"{btag}_{args.model}_{args.size}" / "coords.npy")
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
    print(f"| arm | commit | fused | warm median (s) | vs base | ms/fold | bit-exact | "
          f"max abs Δ (Å) | RMSD vs base (Å) | pLDDT |")
    print("|---|---|---|---:|---:|---:|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['arm']} | {r['commit']} | {r['fused']} | {r['warm_median_s']} | "
              f"{r['ratio']}x | {r['ms_per_fold']} | {'yes' if r['exact'] else 'no'} | "
              f"{r['max_abs_delta_A']} | {r['rmsd_vs_base_A']} | {r['plddt']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
