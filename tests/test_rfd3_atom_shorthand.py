"""The BKBN atom shorthand resolves; TIP says why it does not.

`select_fixed_atoms: {"A5": "BKBN"}` is the documented way to fix a motif residue's
backbone and let its sidechain be redesigned. It used to raise "BKBN shorthand for ligand
atoms" on a protein residue, naming the wrong cause: the mask helper is shared by the
protein and ligand paths and its message was not.
"""
import numpy as np
import pytest

from tt_bio.rfd3.featurize import _atom_selection_mask
from tt_bio.rfd3.input import AtomSelection, _parse_atom_spec

RESIDUE = ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"]
LIGAND = ["C1", "C2", "N9", "O2"]


def test_bkbn_is_the_four_backbone_atoms():
    mask = _atom_selection_mask(_parse_atom_spec("BKBN"), RESIDUE)
    assert [n for n, m in zip(RESIDUE, mask) if m] == ["N", "CA", "C", "O"]


def test_bkbn_matches_the_explicit_atom_list():
    assert np.array_equal(_atom_selection_mask(_parse_atom_spec("BKBN"), RESIDUE),
                          _atom_selection_mask(_parse_atom_spec("N,CA,C,O"), RESIDUE))


def test_bkbn_on_a_residue_with_no_backbone_says_so():
    with pytest.raises(NotImplementedError, match="does not carry"):
        _atom_selection_mask(_parse_atom_spec("BKBN"), LIGAND)


def test_tip_says_what_it_cannot_resolve():
    with pytest.raises(NotImplementedError, match="residue-dependent"):
        _atom_selection_mask(_parse_atom_spec("TIP"), RESIDUE)


def test_all_and_empty_are_unchanged():
    assert _atom_selection_mask(AtomSelection.all_(), RESIDUE).all()
    assert not _atom_selection_mask(AtomSelection.none_(), RESIDUE).any()
