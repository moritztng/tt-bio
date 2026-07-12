"""Ca-RMSD of the OpenDDE e2e smoke fold (scripts/opendde_e2e_smoke.py) vs ground truth,
reusing tests/test_structure.py's get_ca_atoms/_kabsch_deviations (not re-derived).

This is NOT scripts/release_gate.py's production gate (10 cycles/200 steps/5 samples,
confidence-selected, written-CIF harness) -- OpenDDE has no CIF writer or confidence
output wired yet. It is a raw single-sample, reduced-setting (2 cycles/10 steps) direct
coordinate comparison: the first real accuracy signal for the port, honestly scoped.

Run after scripts/opendde_e2e_smoke.py has written /tmp/opendde_e2e_coords.pt:
  PYTHONPATH=<worktree> /home/ttuser/tt-bio-dev/env/bin/python3 scripts/opendde_e2e_rmsd.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from tt_bio.data import const
from tt_bio.protenix_data import RESTYPE_ORDER

REPO_ROOT = Path(__file__).resolve().parent.parent
SEQ = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
       "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")

spec = importlib.util.spec_from_file_location("tt_bio_test_structure", REPO_ROOT / "tests" / "test_structure.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

_LETTER_TO_RES = {v: k for k, v in const.prot_token_to_letter.items()}


def predicted_ca_coords(coords, seq):
    """coords: (N_atom,3) in protein_atom_features' atom order (const.ref_atoms[res],
    +OXT on the last residue). Returns (n_res,3) CA coordinates."""
    ca = []
    off = 0
    n_res = len(seq)
    for i, letter in enumerate(seq):
        aa = RESTYPE_ORDER.index(letter)
        res = _LETTER_TO_RES[RESTYPE_ORDER[aa]]
        atoms = list(const.ref_atoms[res])
        if i == n_res - 1:
            atoms = atoms + ["OXT"]
        ca.append(off + atoms.index("CA"))
        off += len(atoms)
    return coords[np.array(ca)]


def main():
    coords_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opendde_e2e_coords.pt"
    coords = torch.load(coords_path).numpy()[0]  # (N_atom,3)
    pred_ca = predicted_ca_coords(coords, SEQ)

    truth_chains = ts.get_ca_atoms(str(REPO_ROOT / "examples" / "ground_truth_structures" / "prot.cif"))
    (chain_id, truth_by_pos), = truth_chains.items()
    truth_positions = sorted(truth_by_pos)
    n = min(len(pred_ca), len(truth_positions))
    truth_ca = np.array([truth_by_pos[p] for p in truth_positions[:n]])
    pred_ca = pred_ca[:n]

    dev = ts._kabsch_deviations(pred_ca, truth_ca)
    rmsd = float(np.sqrt((dev ** 2).mean()))
    tm = ts._tm_score(dev, n)
    print(f"OpenDDE e2e smoke fold ({coords_path}, 1 sample, no confidence selection)")
    print(f"  Ca-RMSD vs ground truth (7ROA, {n} residues): {rmsd:.3f} A")
    print(f"  TM-score: {tm:.3f}")
    print("(Compare Protenix-v2's production gate floor: max_rmsd=6.0 A, min_tm=0.50 -- "
          "NOT an apples-to-apples comparison: this run is unselected -- no confidence-based best-of-N.)")


if __name__ == "__main__":
    main()
