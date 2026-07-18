#!/usr/bin/env python3
"""Boltz-2 binding-affinity implementation parity: tt-bio device vs the official Boltz-2 reference.

The structure legs of docs/pharma-benchmark.md compare predicted *coordinates*
(Kabsch CA-RMSD) across seeds. Affinity prediction instead emits a scalar
(``affinity_pred_value`` = log10(IC50) in uM, MW-corrected ensemble mean over the
``--diffusion_samples_affinity`` samples and the two affinity heads; plus
``affinity_probability_binary``). A scalar has no alignment step, so the
distance is the absolute delta |device - reference|, and the same R/D/X
noise-floor framework the rest of the benchmark uses applies directly:

  R = |ref(seed i) - ref(seed j)|   across reference-seed pairs   (ref self-floor)
  D = |dev(seed i) - dev(seed j)|   across device-seed pairs      (dev self-floor)
  X = |dev(seed i) - ref(seed j)|   across all dev x ref pairs    (the parity question)

Parity holds when X sits within max(R, D): the device-vs-reference delta is
indistinguishable from the run-to-run diffusion sampling spread each
implementation already exhibits with itself. Reported as a distribution
(mean/std/min/max/n), never one number, via the shared statistical core
(`pharma_parity.noise_floor_verdict`).

Both sides run Boltz-2 affinity mode on the SAME input (a real protein-ligand
complex, msa: empty so single-sequence, no network). The reference is the
official `boltz` package (torch + pytorch-lightning, CPU); the device is the
ttnn port via `tt-bio predict --model boltz2 --affinity_mw_correction`. Both
hardcode affinity recycling_steps=5 and use --recycling_steps 3 for the
upstream structure, so the inputs and model settings are identical.

Reference output layout (official boltz):  <out>/boltz_results_<id>/predictions/<id>/affinity_<id>.json
Device output layout (tt-bio):              <out>/boltz_results_<id>/results.json  (list, one entry per target)

Usage:
  python3 scripts/boltz2_affinity_parity.py \
      --ref-dirs /path/to/ref_seed0 /path/to/ref_seed1 /path/to/ref_seed2 \
      --dev-dirs /path/to/dev_seed0 /path/to/dev_seed1 /path/to/dev_seed2 \
      --target-id affinity_fkg [--out /tmp/affinity_parity.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pharma_parity import noise_floor_verdict, summarize  # noqa: E402


AFFINITY_KEYS = ["affinity_pred_value", "affinity_probability_binary"]


def _find_ref_affinity(ref_dir: Path, target_id: str) -> Path:
    """Official boltz writes <out>/boltz_results_<id>/predictions/<id>/affinity_<id>.json."""
    cand = list(ref_dir.rglob(f"affinity_{target_id}.json"))
    if cand:
        return cand[0]
    cand = list(ref_dir.rglob("affinity_*.json"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"no affinity_*.json under reference dir {ref_dir}")


def _load_device_results(dev_dir: Path):
    """tt-bio writes results.json (a list with one entry per target)."""
    cand = list(dev_dir.rglob("results.json"))
    if not cand:
        raise FileNotFoundError(f"no results.json under device dir {dev_dir}")
    return json.loads(cand[0].read_text())


def _extract_ref(ref_dir: Path, target_id: str) -> dict:
    d = json.loads(_find_ref_affinity(ref_dir, target_id).read_text())
    return {k: float(d[k]) for k in AFFINITY_KEYS if k in d}


def _extract_dev(dev_dir: Path, target_id: str) -> dict:
    rows = _load_device_results(dev_dir)
    if isinstance(rows, dict):
        rows = [rows]
    row = None
    for r in rows:
        if r.get("id") == target_id or str(r.get("id", "")).endswith(target_id):
            row = r
            break
    if row is None and rows:
        row = rows[0]
    if row is None:
        raise FileNotFoundError(f"no results entry in {dev_dir}")
    return {k: float(row[k]) for k in AFFINITY_KEYS if k in row}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dirs", nargs="+", required=True,
                    help="official-boltz reference output dirs, one per seed")
    ap.add_argument("--dev-dirs", nargs="+", required=True,
                    help="tt-bio device output dirs, one per seed")
    ap.add_argument("--target-id", default="affinity_fkg",
                    help="target id (the yaml stem / record id)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ref_vals = [_extract_ref(Path(d), args.target_id) for d in args.ref_dirs]
    dev_vals = [_extract_dev(Path(d), args.target_id) for d in args.dev_dirs]

    print(f"### Boltz-2 binding-affinity parity: {args.target_id}\n")
    print(f"reference seeds: {len(ref_vals)}   device seeds: {len(dev_vals)}\n")
    print("Per-seed affinity_pred_value (log10 IC50 uM, MW-corrected) / affinity_probability_binary:")
    print("| side | seed | affinity_pred_value | affinity_probability_binary |")
    print("|---|---|---|---|")
    for i, v in enumerate(ref_vals):
        print(f"| ref | {i} | {v.get('affinity_pred_value', float('nan')):.4f} "
              f"| {v.get('affinity_probability_binary', float('nan')):.4f} |")
    for i, v in enumerate(dev_vals):
        print(f"| dev | {i} | {v.get('affinity_pred_value', float('nan')):.4f} "
              f"| {v.get('affinity_probability_binary', float('nan')):.4f} |")
    print()

    report = {"mode": "affinity", "target": args.target_id, "metrics": {}}
    print("| metric | dev-vs-ref (X) | ref-floor (R) | dev-floor (D) | X/floor | within floor |")
    print("|---|---|---|---|---|---|")
    for key in AFFINITY_KEYS:
        r = [v[key] for v in ref_vals if key in v]
        d = [v[key] for v in dev_vals if key in v]
        if not r or not d:
            print(f"| {key} | - | - | - | - | - |")
            continue
        cross = [abs(di - ri) for di, ri in itertools.product(d, r)]
        ref_floor = [abs(a - b) for a, b in itertools.combinations(r, 2)]
        dev_floor = [abs(a - b) for a, b in itertools.combinations(d, 2)]
        v = noise_floor_verdict(cross, ref_floor, dev_floor, key)
        report["metrics"][key] = v
        print(f"| {key} | {v['cross']['mean']:.4f}+/-{v['cross']['std']:.4f} "
              f"| {v['ref_floor']['mean']:.4f} "
              f"| {v['dev_floor']['mean']:.4f} "
              f"| {v['cross_over_floor']:.2f} "
              f"| {'yes' if v['within_noise_floor'] else 'NO'} |")

    print("\nInterpretation: affinity_pred_value is a scalar (log10 IC50), so the")
    print("parity distance is |device - reference|. X within max(R, D) means the")
    print("device-vs-reference affinity delta is no larger than the run-to-run")
    print("diffusion-sampling spread each implementation already shows with itself.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
