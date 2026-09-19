"""tt_bio.antibody_rmsd is the ABodyBuilder3 reproduction's measuring instrument.

The real validation is external and lives in ``perf/abb3/verify_instrument.py``: it recomputes
all 1,236 released per-structure evaluations from the released structures and lands within
0.00843 A of every one. That needs a 440 MB Zenodo download, so what is pinned here is the part
of the instrument that silently corrupts every per-region number if it drifts.

Two conventions carry the whole result:

* "Backbone" is N, CA, C, CB. Upstream slices ``positions[:, :, :4]`` of an atom37 tensor, whose
  order is N, CA, C, CB, O, so the carbonyl O is absent and CB is present. Scoring the true
  backbone instead moves CDR-H3 by ~0.07 A against a 0.20 A accuracy bar.
* Residues are paired through their index into the region list, per chain, not through their
  residue number. Their own variants number the light chain from 501 while the ABodyBuilder2
  baseline restarts it at 1, and their PDB fixer drops residues that have no resolved atoms, so
  number-matching pairs the wrong residues or none at all.
"""
from __future__ import annotations

import pytest
import torch

from tt_bio.antibody_rmsd import (
    BACKBONE_ATOMS,
    TRUE_BACKBONE_ATOMS,
    gather,
    per_region_rmsd,
    residue_indices,
)

REGIONS = ["fwh1", "cdrh1", "fwh2", "fwl1", "cdrl1", "fwl2"]  # 3 heavy, 3 light


def _atoms(keys, offset=0.0):
    return {
        k: {name: (i + offset, j + offset, 0.0) for j, name in enumerate(BACKBONE_ATOMS)}
        for i, k in enumerate(keys)
    }


def test_backbone_is_n_ca_c_cb_not_the_carbonyl():
    assert BACKBONE_ATOMS == ("N", "CA", "C", "CB")
    assert TRUE_BACKBONE_ATOMS == ("N", "CA", "C", "O")


def test_indices_follow_residue_numbers_when_a_residue_was_dropped():
    """Heavy 1..n_h and light 501.., with the first heavy residue missing."""
    keys = [("H", 2, " "), ("H", 3, " "), ("L", 501, " "), ("L", 502, " "), ("L", 503, " ")]
    assert residue_indices(keys, REGIONS) == {
        ("H", 2, " "): 1, ("H", 3, " "): 2,
        ("L", 501, " "): 3, ("L", 502, " "): 4, ("L", 503, " "): 5,
    }


def test_indices_follow_position_when_the_chain_is_complete():
    """The ABodyBuilder2 baseline numbers both chains from 1 and uses insertion codes."""
    keys = [("H", 1, " "), ("H", 1, "A"), ("H", 3, " "), ("L", 1, " "), ("L", 2, " "), ("L", 5, " ")]
    assert residue_indices(keys, REGIONS) == {
        ("H", 1, " "): 0, ("H", 1, "A"): 1, ("H", 3, " "): 2,
        ("L", 1, " "): 3, ("L", 2, " "): 4, ("L", 5, " "): 5,
    }


def test_pairs_across_two_different_light_chain_numberings():
    """Truth numbered 501.. against a prediction that restarts the light chain at 1."""
    truth = [("H", 1, " "), ("H", 2, " "), ("H", 3, " "),
             ("L", 501, " "), ("L", 502, " "), ("L", 503, " ")]
    pred = [("H", 1, " "), ("H", 2, " "), ("H", 3, " "),
            ("L", 1, " "), ("L", 2, " "), ("L", 7, " ")]
    chains, regions, t_xyz, p_xyz = gather(_atoms(truth), _atoms(pred), REGIONS)
    assert len(chains) == 6 * len(BACKBONE_ATOMS)
    assert regions[:4] == ["fwh1"] * 4
    # identical coordinates residue-for-residue, so every region scores zero
    assert torch.allclose(t_xyz, p_xyz)


def test_each_chain_is_superposed_independently():
    """A rigid move of one chain alone must not leak into the other chain's RMSD."""
    keys = [("H", 1, " "), ("H", 2, " "), ("H", 3, " "),
            ("L", 501, " "), ("L", 502, " "), ("L", 503, " ")]
    truth = _atoms(keys)
    pred = {k: dict(v) for k, v in truth.items()}
    for k in [("L", 501, " "), ("L", 502, " "), ("L", 503, " ")]:
        pred[k] = {n: (x + 100.0, y, z) for n, (x, y, z) in pred[k].items()}
    scored = per_region_rmsd(*gather(truth, pred, REGIONS))
    assert scored["rmsd_H"] == pytest.approx(0.0, abs=1e-9)
    assert scored["rmsd_L"] == pytest.approx(0.0, abs=1e-9)


def test_regions_are_reported_under_their_own_chain_transform():
    keys = [("H", 1, " "), ("H", 2, " "), ("H", 3, " "),
            ("L", 501, " "), ("L", 502, " "), ("L", 503, " ")]
    truth = _atoms(keys)
    pred = {k: dict(v) for k, v in truth.items()}
    pred[("H", 2, " ")] = {n: (x, y + 1.0, z) for n, (x, y, z) in pred[("H", 2, " ")].items()}
    scored = per_region_rmsd(*gather(truth, pred, REGIONS))
    assert set(scored) == {"rmsd_H", "rmsd_L", "rmsd_fwh", "rmsd_cdrh1", "rmsd_fwl", "rmsd_cdrl1"}
    assert scored["rmsd_cdrh1"] > 0.0  # the moved residue
    assert scored["rmsd_L"] == pytest.approx(0.0, abs=1e-9)


def test_only_atoms_present_in_both_structures_are_scored():
    """Their PDB fixer adds OXT and hydrogens to one side; those must not be scored."""
    keys = [("H", 1, " "), ("H", 2, " "), ("H", 3, " "),
            ("L", 501, " "), ("L", 502, " "), ("L", 503, " ")]
    truth = _atoms(keys)
    truth[("H", 1, " ")]["OXT"] = (99.0, 99.0, 99.0)
    chains, _, t_xyz, _ = gather(truth, _atoms(keys), REGIONS)
    assert len(chains) == 6 * len(BACKBONE_ATOMS)
    assert float(t_xyz.abs().max()) < 99.0
