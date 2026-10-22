"""Paired multi-seed grade of RFD4 design arms: per seed, the designed chain's CA RMSD of each arm
against the base arm at the same seed, beside the base's own seed floor (seed i vs seed i+1).

    python perf/rfd4/bfp8/grade_seeds.py --base <dir> --arm <dir> [--arm <dir> ...] --out <json>

Each dir holds design.py's `*_s<seed>_r<i>_generated.cif`; the warm design (largest r) of each seed
is graded. Superposition is on the target chain's CA, as perf/rfd4/e2e/grade.py does (the target is
held by coordinate conditioning), then the designed chain is compared without a further fit.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
from grade import compare, designed_and_target, load  # noqa: E402


def by_seed(d):
    out = {}
    for p in sorted(Path(d).glob("*_generated.cif")):
        m = re.search(r"_s(\d+)_r(\d+)_generated", p.name)
        if m:
            s, r = int(m.group(1)), int(m.group(2))
            if s not in out or r > out[s][0]:
                out[s] = (r, p)
    return {s: p for s, (r, p) in out.items()}


def pair(a_path, b_path):
    a, b = load(str(a_path)), load(str(b_path))
    c = compare(a, b, *designed_and_target(a))
    return {"rmsd_a": c["designed_ca_rmsd_a"], "seq_id": c["sequence_identity"],
            "target_a": c["target_ca_rmsd_a"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = by_seed(a.base)
    seeds = sorted(base)
    floor = [pair(base[s], base[t])["rmsd_a"] for s, t in zip(seeds, seeds[1:])]
    res = {"base": a.base, "seeds": seeds, "seed_floor_a": floor,
           "seed_floor_median_a": float(np.median(floor)) if floor else None, "arms": {}}
    for d in a.arm:
        arm = by_seed(d)
        per = {s: pair(base[s], arm[s]) for s in seeds if s in arm}
        v = [x["rmsd_a"] for x in per.values()]
        res["arms"][d] = {"per_seed": per, "n": len(v), "median_a": float(np.median(v)) if v else None,
                          "max_a": max(v) if v else None,
                          "median_seq_id": float(np.median([x["seq_id"] for x in per.values()])) if v else None,
                          "identical": all(x == 0.0 for x in v) if v else None}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
