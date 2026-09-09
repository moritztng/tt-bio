"""Each 3Di token lands on the residue foldseek computed it for.

Foldseek reports 3Di only for the residues a structure resolves. Reconciling that string
to the sequence by length alone shifted every token after an unresolved loop, which on a
117-residue structure missing three residues put the wrong token on 82 of them while the
run reported success. `_map_3di` matches the structure's own sequence instead.
"""
import pytest

from tt_bio.saprot import _map_3di

AA = "ACDEFGHIKLMNPQRSTVWY"
TDI = "abcdefghijklmnopqrst"


def test_exact_match_is_the_string_itself():
    assert _map_3di(AA, AA, TDI, "x") == TDI


def test_an_unresolved_stretch_is_masked_and_the_rest_keeps_its_own_token():
    # The structure resolves everything except positions 5..9 of the sequence.
    struct_aa = AA[:5] + AA[10:]
    struct_3di = TDI[:5] + TDI[10:]
    got = _map_3di(AA, struct_aa, struct_3di, "x")
    assert len(got) == len(AA)
    assert got[:5] == TDI[:5]
    assert got[5:10] == "#####"
    assert got[10:] == TDI[10:]


def test_the_old_length_rule_would_have_shifted_it():
    """The negative control: padding to length puts the tail on the wrong residues."""
    struct_3di = TDI[:5] + TDI[10:]
    by_length = struct_3di + "#" * (len(AA) - len(struct_3di))
    assert by_length[10:] != TDI[10:]


def test_a_leading_gap_shifts_nothing():
    got = _map_3di(AA, AA[3:], TDI[3:], "x")
    assert got == "###" + TDI[3:]


def test_a_structure_of_another_sequence_is_refused():
    with pytest.raises(ValueError, match="not of this sequence"):
        _map_3di(AA[::-1], AA, TDI, "somefile.pdb")


def test_a_sequence_shorter_than_the_structure_is_refused():
    with pytest.raises(ValueError, match="not of this sequence"):
        _map_3di(AA[:10], AA, TDI, "somefile.pdb")


def test_the_refusal_names_the_file_and_says_what_to_do():
    with pytest.raises(ValueError) as e:
        _map_3di(AA[:10], AA, TDI, "somefile.pdb")
    msg = str(e.value)
    assert "somefile.pdb" in msg and "--structure" in msg
