#!/usr/bin/env python3
"""Standing accuracy release gate — the on-hardware accuracy leg of RELEASING.md.

For every shipped structure model (Boltz-2, ESMFold2, ESMFold2-fast, Protenix-v2)
this folds one easy, foldable target end-to-end on the real device with production
sampling and then applies two independent gates to the result:

  1. PARSE   — the written mmCIF must load under a strict ``Bio.PDB.MMCIFParser``.
               Biopython is stricter about required ``_atom_site`` columns than the
               geometry parser below, so it is the right tool to catch writer/format
               regressions (e.g. the missing-occupancy bug fixed in 17aeab9e).
  2. RMSD/TM — the confidence-selected structure (best-of-N, exactly what a user
               receives) must land within a per-model ground-truth CA-RMSD / TM-score
               floor of the experimental structure. Reuses the Kabsch + TM + best-of-N
               harness in ``tests/test_structure.py`` (do not re-derive it here).

Self-consistency (seed-vs-reference RMSD) is NOT sufficient — it passes even when the
fold is wrong (see docs/protenix-accuracy-investigation.md). A tag must clear a real
ground-truth floor for every model.

    # gate all four models on card 1
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> \
        python scripts/release_gate.py
    # one model
    python scripts/release_gate.py --model protenix-v2

Exit code 0 iff every requested model PASSES both gates; 1 otherwise. Runs on the
device serially (one card context per predict); no CPU shortcut for the fold.
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The default foldable gate target: examples/prot.yaml == PDB 7ROA, a 117-residue
# monomer that Boltz-2 folds to 1.55 A — proof the target is easy, so a large RMSD
# is a real model/port problem, not a hard target.
DATA = REPO_ROOT / "examples" / "prot.yaml"
GROUND_TRUTH = REPO_ROOT / "examples" / "ground_truth_structures" / "prot.cif"
NAME = DATA.stem  # "prot" -> results land in boltz_results_prot/

# Production sampling. n_step=10 (the old self-consistency harness) undersamples and
# fails a correct model (docs/protenix-accuracy-investigation.md); 200 steps / 5
# samples is the floor for a real accuracy read.
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 5
SEED = 0

# Per-model ground-truth floors on 7ROA, of the confidence-selected structure.
# Anchored to the measured on-hardware baselines (docs/protenix-accuracy-investigation.md)
# with margin for TT diffusion's seed-to-seed stochasticity — deliberately generous
# floors that catch a regression or a gross fold failure, NOT tight targets. Tighten as
# a model's baseline distribution is nailed down; never set below what a correct fold hits.
#   measured best-conf: Boltz-2 1.55 A / TM 0.93 | ESMFold2 2.28 / 0.83 | Protenix-v2 3.87 / 0.71
MODELS = {
    "boltz2":        {"max_rmsd": 3.0, "min_tm": 0.75},
    "esmfold2":      {"max_rmsd": 4.0, "min_tm": 0.65},
    "esmfold2-fast": {"max_rmsd": 4.5, "min_tm": 0.60},
    "protenix-v2":   {"max_rmsd": 6.0, "min_tm": 0.50},
}


def _load_structure_harness():
    """Import tests/test_structure.py by path (tests/ is not an installed package)."""
    path = REPO_ROOT / "tests" / "test_structure.py"
    spec = importlib.util.spec_from_file_location("tt_bio_test_structure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _results_cifs() -> list[Path]:
    d = REPO_ROOT / f"boltz_results_{NAME}" / "structures"
    return sorted(d.glob(f"{NAME}*.cif")) if d.exists() else []


def _parse_gate(cifs: list[Path]) -> None:
    """Strict Bio.PDB.MMCIFParser parse of every written sample. Raises on a bad file."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    if not cifs:
        raise FileNotFoundError("predict wrote no CIF output")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PDBConstructionWarning)  # promote writer sloppiness to a failure
        for cif in cifs:
            structure = MMCIFParser(QUIET=True).get_structure(NAME, str(cif))
            n_atoms = sum(1 for _ in structure.get_atoms())
            if n_atoms == 0:
                raise ValueError(f"{cif.name}: parsed but contains 0 atoms")


def run_model(model: str, harness, keep: bool) -> dict:
    """Fold, parse, and ground-truth-score one model. Returns a result row."""
    out = REPO_ROOT / f"boltz_results_{NAME}"
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this predict crashes

    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(DATA),
        "--model", model,
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--seed", str(SEED),
        "--use_msa_server",
        "--out_dir", str(REPO_ROOT),
    ]
    print(f"\n{'='*70}\n[{model}] folding {DATA.name} "
          f"({SAMPLING_STEPS} steps, {DIFFUSION_SAMPLES} samples)\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "rmsd": None, "tm": None,
           "parse": False, "gate": False, "error": None}
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        row["error"] = f"predict exited {proc.returncode}"
        return row

    cifs = _results_cifs()
    try:
        _parse_gate(cifs)
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    # Ground-truth RMSD/TM of the confidence-selected structure (harness reads
    # boltz_results_<NAME>/ and examples/ground_truth_structures/ relative to REPO_ROOT).
    try:
        rmsd, tm = harness.evaluate(NAME)  # no thresholds -> returns numbers, never raises
    except Exception as e:
        row["error"] = f"RMSD eval failed: {e}"
        return row
    row["rmsd"], row["tm"] = rmsd, tm

    th = MODELS[model]
    row["gate"] = (rmsd <= th["max_rmsd"]) and (tm >= th["min_tm"])

    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), action="append",
                    help="Gate only this model (repeatable). Default: all four.")
    ap.add_argument("--keep", action="store_true", help="Keep boltz_results_ output dirs for inspection.")
    args = ap.parse_args()

    if not DATA.exists():
        sys.exit(f"missing gate target {DATA}")
    if not GROUND_TRUTH.exists():
        sys.exit(f"missing ground truth {GROUND_TRUTH}")
    harness = _load_structure_harness()

    models = args.model or list(MODELS)
    rows = [run_model(m, harness, args.keep) for m in models]

    print(f"\n{'#'*78}\nRELEASE GATE — {DATA.name} ({NAME}), "
          f"{SAMPLING_STEPS} steps / {DIFFUSION_SAMPLES} samples, seed {SEED}\n{'#'*78}")
    print(f"{'model':<15}{'RMSD (A)':>10}{'TM':>8}{'floor':>16}{'wall':>9}  result")
    all_pass = True
    for r in rows:
        th = MODELS[r["model"]]
        floor = f"<={th['max_rmsd']}/>={th['min_tm']}"
        rmsd = f"{r['rmsd']:.3f}" if r["rmsd"] is not None else "  -  "
        tm = f"{r['tm']:.3f}" if r["tm"] is not None else "  -  "
        wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
        verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
        all_pass &= r["gate"]
        print(f"{r['model']:<15}{rmsd:>10}{tm:>8}{floor:>16}{wall:>9}  {verdict}")
    print(f"{'#'*78}")
    print("GATE PASS — all models cleared parse + ground-truth floor" if all_pass
          else "GATE FAIL — a model missed parse or the ground-truth floor (see above)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
