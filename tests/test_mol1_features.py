"""tt-bio's Protenix-family featurizer equals Mol-1's own on every golden job. CPU only.

The goldens (tests/data/mol1_io/features/<mode>/<job>.npz) are Mol-1 e6e64866's `InferenceDataset.process_one`
output for the jobs in tests/data/mol1_io/*.json, made by scripts/mol1_io/feature_goldens.py. Each job is staged
the way `tt-bio predict jobs.json` stages it and run through the worker's featurizer. Integer features compare
exactly (tolerance 0) and so do the float ones, since both sides compute them the same way.

Two things are outside the equality:
* ref_pos. Upstream rotates every reference conformer at random, and takes the conformer from its pickled RDKit
  table where tt-bio uses the CCD ideal coordinates, so the coordinates differ by more than a rigid motion
  (state/mol1/DECISIONS.md D-io-4).
* the keys tt-bio does not build, listed in NOT_BUILT with what reads them upstream.
Jobs whose alignments live in the upstream clone ($MOL1_UPSTREAM) are skipped where it is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "mol1_io"))
import compare_features as cf  # noqa: E402

#: upstream keys tt-bio does not build, and why that is not a parity gap at inference
NOT_BUILT = {
    "bond_mask", "mol_id", "mol_atom_index", "entity_mol_id", "resolution",       # training / permutation only
    "pae_rep_atom_mask", "plddt_m_rep_atom_mask", "modified_res_mask",           # loss masks
    "frame_atom_index", "is_dna", "is_rna", "is_protein",                        # not read by mol/model
    "template_atom_mask", "template_atom_positions",                             # template embedder reads derived keys
    "msa_mask", "deletion_matrix",                                               # assembly intermediates
    "prot_pair_num_alignments", "prot_unpair_num_alignments", "prot_paired_num_alignments",
    "prot_unpaired_num_alignments", "rna_pair_num_alignments", "rna_unpair_num_alignments",
    "rna_unpaired_num_alignments",
    # read only by the confidence summary (mol.py: token_has_frame, atom_is_polymer), which the output writer computes
    "has_frame", "is_ligand",
}
SKIP = {"ref_pos"}
JOBS = cf.jobs()
CASES = [(m, p.stem) for m in cf.MODES for p in sorted((cf.GOLD / m).glob("*.npz"))]


@pytest.mark.parametrize("mode,job", CASES, ids=[f"{m}-{j}" for m, j in CASES])
def test_features_equal_upstream(tmp_path, mode, job):
    if job not in JOBS:
        pytest.skip(f"{job}: its alignments are in the upstream clone, set MOL1_UPSTREAM")
    gold = dict(np.load(cf.GOLD / mode / f"{job}.npz"))
    res = cf.diff(cf.tt_features(JOBS[job], mode, tmp_path), gold)
    bad = {k: v for k, v in res.items() if v and k not in SKIP and not (v == "missing in tt-bio" and k in NOT_BUILT)
           and v != "tt-bio only"}
    assert not bad, bad


def test_the_goldens_are_not_trivially_equal():
    """Negative control: a one-entry change to a golden is caught."""
    gold = dict(np.load(cf.GOLD / "no_msa" / "glycan_bonds.npz"))
    other = {k: v.copy() for k, v in gold.items()}
    other["restype"][0] = 1 - other["restype"][0]
    assert cf.diff(other, gold)["restype"]
