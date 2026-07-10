#!/usr/bin/env python3
"""Prototype: consensus/medoid selection for Protenix-v2 multi-sample folds.

Protenix-v2's confidence head barely discriminates between diffusion samples on
hard, shallow-MSA targets (pTM ~0.715-0.726 across 5 samples on 7ROA) and can
*anti-rank* — delivering a 3.87 A sample as "best" while a 2.34 A sample was in
the ensemble (docs/protenix-accuracy-investigation.md). This is Protenix-v2's own
confidence-head weakness, reproduced in the official upstream reference
(docs/protenix-v2-reference-rootcause.md) — not a tt-bio port bug, and NOT present
in Boltz-2 / ESMFold2 (whose confidence heads rank fine).

When the confidence signal is unreliable but the ensemble clusters near the right
answer, a classic robust trick is to ignore the score and pick the *most typical*
structure: the medoid, i.e. the sample with the lowest mean pairwise CA-RMSD to the
others. This script evaluates whether medoid selection beats confidence selection
against experimental ground truth, with ZERO change to the diffusion model.

It reuses tests/test_structure.py verbatim for all RMSD math:
  - compute_rmsd(name, m)   -> ground-truth CA-RMSD/TM of sample m (chain-matched)
  - get_ca_atoms(cif)       -> per-chain CA coords, for the pairwise medoid matrix
  - _kabsch_deviations(P,Q) -> the Kabsch superposition primitive (NOT re-derived)

Usage (from a dir that holds boltz_results_<name>/structures/*.cif):
    python scripts/protenix_medoid_selection.py <name> --ground-truth <gt.cif>

Samples are the confidence-RANKED cifs tt-bio already wrote: <name>.cif is
best-confidence (model 0), <name>_model_{r}.cif the rest. So "best-conf" below is
exactly the structure the current pipeline hands the user.
"""
import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def _load_structure_harness():
    """Import tests/test_structure.py by path (tests/ is not an installed package)."""
    repo = Path(__file__).resolve().parent.parent
    path = repo / "tests" / "test_structure.py"
    spec = importlib.util.spec_from_file_location("tt_bio_test_structure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ca_vector(cif: str, ts) -> np.ndarray:
    """Flatten a prediction's CA coords into one (n_ca, 3) array in a fixed order
    (sorted chain, then sorted residue position). Every sample of the SAME fold
    shares chain ids and residue positions, so this ordering aligns them for a
    pairwise superposition — no chain matching needed within an ensemble."""
    chains = ts.get_ca_atoms(cif)
    rows = []
    for cid in sorted(chains):
        for pos in sorted(chains[cid]):
            rows.append(chains[cid][pos])
    return np.asarray(rows, dtype=float)


def _pairwise_rmsd(vecs, ts) -> np.ndarray:
    """Symmetric matrix of Kabsch CA-RMSD between every pair of samples."""
    n = len(vecs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dev = ts._kabsch_deviations(vecs[i], vecs[j])
            M[i, j] = M[j, i] = float(np.sqrt((dev ** 2).mean()))
    return M


def analyse(name: str, work_dir: Path, gt: Path, tie_tol: float = 0.5):
    """Return per-target medoid-vs-best-conf verdict. All ground-truth RMSD/TM come
    from tests/test_structure.compute_rmsd (chain-matched, Kabsch), evaluated in a
    staging dir laid out exactly as that harness expects."""
    ts = _load_structure_harness()

    struct = work_dir / f"boltz_results_{name}" / "structures"
    sample_files = [struct / f"{name}.cif"] + sorted(
        struct.glob(f"{name}_model_*.cif"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    n = len(sample_files)
    if n < 2:
        raise SystemExit(f"need >=2 samples, found {n} in {struct}")

    # compute_rmsd(name, m) reads boltz_results_<name>/ and
    # examples/ground_truth_structures/<name>.cif relative to cwd. Stage both so we
    # reuse it verbatim instead of re-deriving chain-matched ground-truth RMSD.
    stage = work_dir / "_medoid_stage"
    (stage / "examples" / "ground_truth_structures").mkdir(parents=True, exist_ok=True)
    dst = stage / f"boltz_results_{name}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(work_dir / f"boltz_results_{name}", dst)
    shutil.copy(gt, stage / "examples" / "ground_truth_structures" / f"{name}.cif")

    gt_rmsd, gt_tm = [], []
    cwd = os.getcwd()
    os.chdir(stage)
    try:
        for m in range(n):
            r, t = ts.compute_rmsd(name, m)
            gt_rmsd.append(r)
            gt_tm.append(t)
    finally:
        os.chdir(cwd)

    # Pairwise consensus matrix (index m == sample written as model m == rank m).
    vecs = [_ca_vector(str(f), ts) for f in sample_files]
    sizes = {len(v) for v in vecs}
    if len(sizes) != 1:
        raise SystemExit(f"samples disagree on CA count: {sizes}")
    M = _pairwise_rmsd(vecs, ts)
    mean_pair = M.sum(1) / (n - 1)

    medoid = int(np.argmin(mean_pair))
    best_conf = 0                       # model 0 is tt-bio's confidence pick
    oracle = int(np.argmin(gt_rmsd))

    # Optional pTM tie-breaker: if two samples are within tie_tol A of the medoid's
    # mean-pair distance, prefer the higher-confidence one. Ranks are by confidence,
    # so lower rank index == higher confidence -> pick the lowest such index.
    close = [m for m in range(n) if mean_pair[m] - mean_pair[medoid] <= tie_tol]
    medoid_tb = min(close)              # lowest rank == highest confidence among near-ties

    print(f"\n{'='*66}\nTARGET {name}  ({n} samples)\n{'='*66}")
    print(f"{'rank':>4}{'gt_rmsd':>9}{'gt_tm':>7}{'mean_pair':>11}   tag")
    for m in range(n):
        tags = []
        if m == best_conf:
            tags.append("best-conf")
        if m == medoid:
            tags.append("MEDOID")
        if m == medoid_tb and medoid_tb != medoid:
            tags.append("medoid+tiebreak")
        if m == oracle:
            tags.append("oracle")
        print(f"{m:>4}{gt_rmsd[m]:>9.3f}{gt_tm[m]:>7.3f}{mean_pair[m]:>11.3f}   {', '.join(tags)}")

    print(f"\n  best-conf (delivered today): {gt_rmsd[best_conf]:.3f} A  TM {gt_tm[best_conf]:.3f}")
    print(f"  medoid (consensus)         : {gt_rmsd[medoid]:.3f} A  TM {gt_tm[medoid]:.3f}")
    print(f"  medoid + pTM tiebreak      : {gt_rmsd[medoid_tb]:.3f} A  TM {gt_tm[medoid_tb]:.3f}")
    print(f"  oracle (unattainable)      : {gt_rmsd[oracle]:.3f} A  TM {gt_tm[oracle]:.3f}")
    delta = gt_rmsd[best_conf] - gt_rmsd[medoid]
    verdict = "medoid BEATS best-conf" if delta > 1e-3 else (
        "medoid TIES best-conf" if abs(delta) <= 1e-3 else "medoid LOSES to best-conf")
    print(f"  --> {verdict} by {delta:+.3f} A")
    return {"name": name, "n": n, "best_conf": gt_rmsd[best_conf], "medoid": gt_rmsd[medoid],
            "medoid_tb": gt_rmsd[medoid_tb], "oracle": gt_rmsd[oracle], "verdict": verdict}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="target stem, e.g. prot (reads boltz_results_<name>/)")
    ap.add_argument("--work-dir", type=Path, default=Path.cwd(),
                    help="dir containing boltz_results_<name>/ (default: cwd)")
    ap.add_argument("--ground-truth", type=Path, required=True, help="experimental .cif")
    ap.add_argument("--tie-tol", type=float, default=0.5,
                    help="mean-pair-RMSD window (A) within which pTM breaks the medoid tie")
    args = ap.parse_args()
    analyse(args.name, args.work_dir, args.ground_truth, args.tie_tol)
