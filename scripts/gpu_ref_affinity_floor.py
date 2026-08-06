#!/usr/bin/env python3
"""Compute the GPU-reference SELF-FLOOR (R) for a Boltz-2 affinity leg.

The committed CPU reference fixtures score device-vs-reference (X) against the
CPU reference's self-floor R. When the reference is regenerated on a GPU (vast.ai
RTX-class, --no_kernels so the torch-einsum triangle kernel matches the CPU
reference, only the execution device differs), the reference's own self-floor R
can change. This script reports that GPU-reference R for the four affinity
metrics (affinity_pred_value, affinity_probability_binary, ligand-pose RMSD,
1-pocket-lDDT) using the SAME pairwise / noise_floor_verdict core as
scripts/boltz2_affinity_parity.py, so the GPU-R and CPU-R are directly
comparable. It needs no device side -- R is a property of the reference seeds
alone.

Each --ref-dir is a converted harness-layout seed dir:
  <dir>/affinity_<tid>.json   (scalar affinity outputs)
  <dir>/structures/<tid>.cif  (best-sample structure)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boltz2_affinity_parity import AFFINITY_KEYS, _pose_metrics  # noqa: E402
from pharma_parity import noise_floor_verdict  # noqa: E402


def _extract(ref_dir: Path, target_id: str) -> dict:
    cand = list(ref_dir.rglob(f"affinity_{target_id}.json"))
    if not cand:
        raise FileNotFoundError(f"no affinity_{target_id}.json under {ref_dir}")
    d = json.loads(cand[0].read_text())
    return {k: float(d[k]) for k in AFFINITY_KEYS if k in d}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dirs", nargs="+", required=True)
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.ref_dirs]
    vals = [_extract(d, args.target_id) for d in dirs]
    print(f"### GPU-reference self-floor (R): {args.target_id}  ({len(dirs)} seeds)\n")
    print("Per-seed affinity_pred_value / affinity_probability_binary:")
    print("| seed | affinity_pred_value | affinity_probability_binary |")
    print("|---|---|---|")
    for i, v in enumerate(vals):
        print(f"| {i} | {v.get('affinity_pred_value', float('nan')):.4f} "
              f"| {v.get('affinity_probability_binary', float('nan')):.4f} |")
    print()

    report = {"target": args.target_id, "n_seeds": len(dirs), "metrics": {}}
    print("| metric | R (self-floor mean) | R std | R min | R max | n |")
    print("|---|---|---|---|---|---|")
    for key in AFFINITY_KEYS:
        r = [v[key] for v in vals if key in v]
        if not r:
            continue
        ref_floor = [abs(a - b) for a, b in itertools.combinations(r, 2)]
        v = noise_floor_verdict(ref_floor, ref_floor, ref_floor, key)
        report["metrics"][key] = {"R": v["ref_floor"]}
        print(f"| {key} | {v['ref_floor']['mean']:.4f} | {v['ref_floor']['std']:.4f} "
              f"| {v['ref_floor']['min']:.4f} | {v['ref_floor']['max']:.4f} "
              f"| {v['ref_floor']['n']} |")

    pose_keys = ("ligand_rmsd", "1-pocket_lddt")
    pose_labels = {"ligand_rmsd": "ligand-pose RMSD (Å)", "1-pocket_lddt": "1-pocket-lDDT"}
    pose_rf = {k: [] for k in pose_keys}
    for da, db in itertools.combinations(dirs, 2):
        m = _pose_metrics(da, db, args.target_id)
        if m:
            for k in pose_keys:
                pose_rf[k].append(m[k])
    print()
    print("| metric | R (self-floor mean) | R std | R min | R max | n |")
    print("|---|---|---|---|---|---|")
    for k in pose_keys:
        if not pose_rf[k]:
            print(f"| {pose_labels[k]} | (no matched CIF pairs) | - | - | - | 0 |")
            continue
        v = noise_floor_verdict(pose_rf[k], pose_rf[k], pose_rf[k], k)
        report["metrics"][k] = {"R": v["ref_floor"]}
        print(f"| {pose_labels[k]} | {v['ref_floor']['mean']:.3f} | {v['ref_floor']['std']:.3f} "
              f"| {v['ref_floor']['min']:.3f} | {v['ref_floor']['max']:.3f} "
              f"| {v['ref_floor']['n']} |")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
