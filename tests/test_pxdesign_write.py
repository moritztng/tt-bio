"""A designed binder CIF must survive a STRICT mmCIF parse, not just render in a viewer.

The release-gate leg found this the hard way: the writer emitted only the `label_*` `_atom_site`
columns, which Mol* and PyMOL are perfectly happy with and Biopython rejects with
`KeyError: '_atom_site.auth_asym_id'`. A user hits that after the design run they waited for, and
every eyeball check passes because the file looks right on screen.

So the property is pinned here rather than only in the gate: the gate's version of this check needs
a card and a checkpoint, and a defect that only a device leg can catch is a defect that reaches a
release candidate. These arms need neither -- the coordinates are synthetic and the strictness is
Bio.PDB with `PDBConstructionWarning` promoted to an error, exactly what
`scripts/release_gate.py::_parse_gate` uses.

The frame arm is the other half. `write_design_cifs` fits the generated distogram-representative
atoms of the conditioned tokens onto the coordinates the featurizer conditioned on, and reports
that fit as `fit_rmsd`. Feeding it an EXACT rigid transform of the conditioning coordinates means a
correct implementation must recover the frame to ~0, so a Kabsch that returns the inverse rotation
(the `kabsch-inverse-rotation-swap-phantom-rmsd` shape) fails here instead of quietly writing every
binder into the wrong place.

Device-free: no ttnn, no checkpoint, no forward pass.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tt_bio.pxdesign.inputs import design_inputs_from_yaml
from tt_bio.pxdesign.write import BINDER_RESTYPE, write_design_cifs

FIXTURE = Path(__file__).parent / "fixtures" / "pxdesign" / "PDL1.yaml"

# Bio.PDB reads chain and residue identity from the auth_* columns and hard-requires occupancy;
# the label_* set alone parses in a viewer and raises in Biopython.
REQUIRED_COLUMNS = ("_atom_site.auth_asym_id", "_atom_site.auth_seq_id",
                    "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
                    "_atom_site.label_atom_id", "_atom_site.label_comp_id",
                    "_atom_site.Cartn_x")


@pytest.fixture(scope="module")
def feats():
    return design_inputs_from_yaml(FIXTURE)


def _synthetic_coords(feats, angle=0.7, shift=(13.0, -7.0, 4.0), seed=0):
    """Target atoms at an EXACT rigid transform of the conditioning coords; binder scattered."""
    n_atom = int(feats["ref_pos"].shape[0])
    disto = feats["distogram_rep_atom_mask"].bool()
    c, s = float(np.cos(angle)), float(np.sin(angle))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    tr = torch.tensor(shift, dtype=torch.float64)

    coords = torch.zeros(1, n_atom, 3)
    coords[0][disto] = (feats["condition"]["coord"].double() @ rot.T + tr).float()
    a2t = feats["atom_to_token_idx"].long()
    binder = (feats["restype"].argmax(-1) == BINDER_RESTYPE)[a2t]
    torch.manual_seed(seed)
    coords[0][binder] = (torch.randn(int(binder.sum()), 3).double() * 8 + tr).float()
    return coords


def test_cif_parses_under_the_gates_own_strictness(feats, tmp_path):
    """The exact parse `_parse_gate` runs: warnings promoted to errors, atoms must be non-zero."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    rows = write_design_cifs(_synthetic_coords(feats), feats, tmp_path, stem="PDL1")
    assert len(rows) == 1
    cif = Path(rows[0]["cif"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", PDBConstructionWarning)
        structure = MMCIFParser(QUIET=True).get_structure("pxdesign", str(cif))

    atoms = list(structure.get_atoms())
    residues = list(structure.get_residues())
    assert len(atoms) == rows[0]["binder_atoms"] > 0
    assert [c.id for c in structure.get_chains()] == ["A"]
    # PXDesign generates a backbone with no sequence, which is exactly GLY's atom set.
    assert {r.get_resname() for r in residues} == {"GLY"}
    assert {a.get_id() for a in residues[0]} == {"N", "CA", "C", "O"}


def test_cif_carries_the_columns_biopython_requires(feats, tmp_path):
    """Named individually, so a future trim of the column list says which one it broke."""
    rows = write_design_cifs(_synthetic_coords(feats), feats, tmp_path, stem="PDL1")
    text = Path(rows[0]["cif"]).read_text()
    missing = [c for c in REQUIRED_COLUMNS if c not in text]
    assert not missing, f"_atom_site columns absent: {missing}"
    # Every data row must carry one field per declared column, or a parser silently mis-assigns.
    header = [l for l in text.splitlines() if l.startswith("_atom_site.")]
    for line in (l for l in text.splitlines() if l.startswith("ATOM ")):
        assert len(line.split()) == len(header), f"{len(line.split())} fields vs {len(header)} columns"


def test_binder_is_placed_in_the_targets_frame(feats, tmp_path):
    """An exact rigid transform in must come back out as ~0 fit RMSD."""
    rows = write_design_cifs(_synthetic_coords(feats), feats, tmp_path, stem="PDL1")
    assert rows[0]["fit_rmsd"] < 1e-4, rows[0]["fit_rmsd"]
    assert rows[0]["conditioned_tokens"] > 0
    assert rows[0]["binder_residues"] > 0


def test_refuses_a_feature_dict_with_no_conditioning(feats, tmp_path):
    """Without `condition` there is no frame to fit, and writing the binder at the diffusion
    origin would look plausible and be wrong. It has to raise instead."""
    stripped = {k: v for k, v in feats.items() if k != "condition"}
    with pytest.raises(ValueError, match="condition"):
        write_design_cifs(_synthetic_coords(feats), stripped, tmp_path, stem="PDL1")
