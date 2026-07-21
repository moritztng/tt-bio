#!/usr/bin/env python3
"""GPU-vs-CPU reference pocket-lDDT / ligand-RMSD agreement (committed CIFs only).

Question: does switching the primary comparison from the committed CPU bf16
reference to the committed GPU bf16 reference widen the pocket-lDDT floor enough
to flip the affinity pocket-lDDT GAP to a PASS?

Method: treat the 5 GPU reference seeds as side A and the 5 CPU reference seeds
as side B and run them through ``boltz2_affinity_parity._pose_metrics`` and the
shared ``pharma_parity.noise_floor_verdict`` core. X = GPU-vs-CPU reference
distance, R = CPU self-floor, D = GPU self-floor. Both references are the pinned
official boltz 2.2.1, ``bf16-mixed`` (pytorch-lightning AMP), ``--no_kernels``
(torch-einsum triangle path) — only the execution device differs (x86 CPU vs
NVIDIA RTX 3090). So X is a pure backend-divergence distance, not a port defect.

If X GAPs the same way device-vs-CPU does, the pocket-lDDT residual is a
bf16-backend-divergence property (CPU bf16 vs CUDA bf16 vs ttnn bf16 each land
in a slightly different narrow pocket basin), and the GPU reference cannot flip
the verdict: device-vs-GPU X ~= device-vs-CPU X ~= GPU-vs-CPU X, all >> the
~0.005-0.012 GPU self-floor.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boltz2_affinity_parity import _pose_metrics  # noqa: E402
from pharma_parity import noise_floor_verdict  # noqa: E402

FIX = Path("docs/pharma-benchmark-data/ref-fixtures/boltz2")
TARGETS = {
    "affinity_fkg": "fkbp12",
    "affinity_dhfr": "dhfr",
    "affinity_tryp": "tryp",
}
POSE_KEYS = ("ligand_rmsd", "1-pocket_lddt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/pharma-benchmark-data/boltz2-affinity-gpu-vs-cpu-pocket.json")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    report = {"mode": "gpu-vs-cpu-reference-pocket", "targets": {}}
    for tid, label in TARGETS.items():
        sub = tid
        gpu = [FIX / sub / "nomsa_200step_5affsample_3recycle_bf16_mwcorr_gpu" / f"seed{i}" for i in range(args.n_seeds)]
        cpu = [FIX / sub / "nomsa_200step_5affsample_3recycle_bf16_mwcorr" / f"seed{i}" for i in range(args.n_seeds)]
        cross = {k: [] for k in POSE_KEYS}
        rf = {k: [] for k in POSE_KEYS}   # CPU self-floor (R)
        df = {k: [] for k in POSE_KEYS}   # GPU self-floor (D)
        for da, db in itertools.product(gpu, cpu):       # A=gpu, B=cpu
            m = _pose_metrics(da, db, tid)
            if m:
                for k in POSE_KEYS: cross[k].append(m[k])
        for da, db in itertools.combinations(cpu, 2):
            m = _pose_metrics(da, db, tid)
            if m:
                for k in POSE_KEYS: rf[k].append(m[k])
        for da, db in itertools.combinations(gpu, 2):
            m = _pose_metrics(da, db, tid)
            if m:
                for k in POSE_KEYS: df[k].append(m[k])
        print(f"\n=== {label} ({tid})  GPU-vs-CPU reference, {args.n_seeds}+{args.n_seeds} seeds ===")
        print("| metric | GPU-vs-CPU (X) | CPU self-floor (R) | GPU self-floor (D) | X/floor | within floor |")
        print("|---|---|---|---|---|---|")
        tgt = {}
        for k in POSE_KEYS:
            if not cross[k]:
                continue
            v = noise_floor_verdict(cross[k], rf[k], df[k], k)
            tgt[k] = v
            print(f"| {k} | {v['cross']['mean']:.4f}+/-{v['cross']['std']:.4f} "
                  f"| {v['ref_floor']['mean']:.4f} | {v['dev_floor']['mean']:.4f} "
                  f"| {v['cross_over_floor']:.2f} | {'yes' if v['within_noise_floor'] else 'NO'} |")
        report["targets"][label] = tgt
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
