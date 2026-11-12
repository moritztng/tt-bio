"""OpenDDE's input features against upstream OpenDDE 1.2.0's own featurizer (no device, no weights).

tests/data/opendde_featurizer/<case>.npz holds what upstream's inference featurizer emits for each
public input in cases.py (make_fixtures.py regenerates them). tt-bio's build_complex_features +
build_structural_token_features must reproduce every token, atom, template and structural-token
tensor exactly: no template, a protein + DNA + RNA complex, an N-glycan, modified residues, a
SMILES ligand, a modified nucleotide and a head-to-tail cyclic peptide.

Reference conformers come from OpenDDE's own CCD pickle (the opendde-ccd artifact), so that check
runs only where the table is already cached.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio.opendde_data import build_structural_token_features
from tt_bio.protenix_data import build_complex_features

DATA = Path(__file__).parent / "data" / "opendde_featurizer"
_spec = importlib.util.spec_from_file_location("opendde_featurizer_cases", DATA / "cases.py")
cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cases)

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.boltz/mols")),
                                reason="needs bundled CCD mol library (~/.boltz/mols)")

ONE_HOT = ("restype", "ref_element", "ref_atom_name_chars")


def _feats(name, table=None):
    f = build_complex_features(**cases.CASES[name][1], conformer_table=table)
    f.update(build_structural_token_features(f))
    return f


@pytest.mark.parametrize("name", list(cases.CASES))
def test_features_match_upstream(name):
    up = np.load(DATA / f"{name}.npz")
    f = _feats(name)
    for k in up.files:
        if k == "ref_pos":
            continue
        ours = f[k].argmax(-1) if k in ONE_HOT else f[k]
        ours = ours.double().numpy()
        assert ours.shape == up[k].shape, f"{name} {k}: {ours.shape} vs upstream {up[k].shape}"
        diff = np.argwhere(ours != up[k])
        assert diff.size == 0, f"{name} {k}: {len(diff)} entries differ, first at {diff[:3].tolist()}"


def test_empty_template_slots_match_upstream():
    """Upstream pads a chain with no template to slot 0 = gap (31) and slots 1-3 = index 0."""
    aatype = _feats("no_template")["template_aatype"]
    assert (aatype[0] == 31).all() and (aatype[1:] == 0).all()


def test_glycan_and_modified_residues_have_no_chain_neighbours():
    """A per-atom token (a sugar, a modified residue) is no residue's prev/next, as upstream."""
    f = _feats("modified_residue")
    atomized = (f["subtoken_role_id"] == 0).nonzero().flatten()
    assert len(atomized) > 0
    assert (f["prev_parent_residue_idx"][atomized] == -1).all()
    assert (f["next_parent_residue_idx"][atomized] == -1).all()
    linked = set(f["prev_parent_residue_idx"].tolist()) | set(f["next_parent_residue_idx"].tolist())
    assert not linked & set(f["parent_residue_idx"][atomized].tolist())


def test_cyclic_peptide_closing_bond_is_a_residue_edge():
    f = _feats("cyclic_peptide")
    n = int(f["parent_residue_idx"].max())
    assert f["polymer_backbone_bonds"].tolist() == [[0, n]]
    first = f["parent_residue_idx"] == 0
    assert (f["next_parent_residue_idx"][first] == n).all()
    assert "polymer_backbone_bonds" not in _feats("no_template")


def _opendde_table():
    from tt_bio import weights
    from tt_bio.protenix_data import load_conformer_table
    p = weights.cached_path("opendde-ccd")
    if p is None or not (p / "ccd_table.npz").exists():
        pytest.skip("opendde-ccd table not cached (tt-bio fetch opendde)")
    return load_conformer_table(p / "ccd_table.npz")


@pytest.mark.parametrize("name", list(cases.CASES))
def test_reference_conformers_match_upstream(name):
    """Every CCD component's reference conformer is upstream's, up to the rigid motion both sides
    apply: each residue's intra-residue distances agree. A SMILES ligand is embedded by RDKit on
    both sides with different settings, so it is left out."""
    up = np.load(DATA / f"{name}.npz")
    f = _feats(name, _opendde_table())
    ours, ref, uid = f["ref_pos"].double(), torch.from_numpy(up["ref_pos"]).double(), up["ref_space_uid"]
    chains = cases.CASES[name][1]["chains"]
    for u in np.unique(uid):
        m = torch.from_numpy(uid == u)
        tok = int(f["atom_to_token_idx"][m][0])
        seq, _, mol_type = chains[int(f["asym_id"][tok])]
        if mol_type == "ligand" and not seq.startswith("CCD_"):
            continue
        a, b = ours[m], ref[m]
        err = (torch.cdist(a, a) - torch.cdist(b, b)).abs().max().item()
        assert err < 1e-3, f"{name}: residue uid {u} conformer off by {err:.3f} A"
