"""Regression: a chain id longer than one character killed every PDB write.

Job 050e3d84fb7150adb28382f6bbc1f623 (esmfold2-fast, three FASTAs with bare ``>sample_NN``
headers, ``output_format: pdb``) folded all three chains on device, ran 68 diffusion steps and
confidence, then died in the writer with ``Some chain IDs exceed 1 character``. The chain id is
the FASTA header, the PDB chain column is one character wide, and nothing between the two
noticed until the whole fold had been paid for.

The fix lives in ``tt_bio/data/pdb.py`` and every PDB writer goes through it. Host-only: no
device, no checkpoints.
"""
import io

import numpy as np
import pytest

from tt_bio._vendor.esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexMetadata,
)
from tt_bio.data.pdb import CHAIN_ALPHABET, chain_id_at, write_atom_array
from tt_bio.main import _write_structure


def _complex(chain_names):
    """One ALA per chain, three atoms each, pLDDT rising with the chain index."""
    n = len(chain_names)
    coords = np.arange(n * 3 * 3, dtype=np.float32).reshape(n * 3, 3) * 0.5
    return MolecularComplex(
        id="test",
        sequence=["ALA"] * n,
        atom_positions=coords,
        atom_elements=np.array(["N", "C", "C"] * n),
        token_to_atoms=np.array([[3 * i, 3 * i + 3] for i in range(n)]),
        chain_id=np.arange(n),
        plddt=np.array([0.5 + 0.01 * i for i in range(n)], dtype=np.float32),
        metadata=MolecularComplexMetadata(
            entity_lookup={0: "1"},
            chain_lookup=dict(enumerate(chain_names)),
        ),
    )


def _atom_lines(path):
    return [ln for ln in path.read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]


def _chain_column(path):
    """Column 22 of every ATOM record, in order of first appearance."""
    return list(dict.fromkeys(ln[21] for ln in _atom_lines(path)))


def _remarks(path):
    return [ln for ln in path.read_text().splitlines() if ln.startswith("REMARK")]


def test_bare_fasta_header_chain_ids_are_relabelled_and_remarked(tmp_path):
    """The reported failure: `sample_03`/`sample_05` now write, as A and B."""
    out = tmp_path / "t.pdb"
    _write_structure(_complex(["sample_03", "sample_05"]), out, "pdb")

    assert _chain_column(out) == ["A", "B"]
    assert _remarks(out) == [
        "REMARK 999 NAMES RELABELLED IN THIS PDB, WITH THEIR ORIGINALS",
        "REMARK 999 CHAIN A IS sample_03",
        "REMARK 999 CHAIN B IS sample_05",
    ]


def test_the_remark_block_comes_before_the_coordinates(tmp_path):
    """A REMARK after the first ATOM record is not a PDB a parser will read."""
    out = tmp_path / "t.pdb"
    _write_structure(_complex(["sample_03", "sample_05"]), out, "pdb")

    lines = out.read_text().splitlines()
    first_atom = next(i for i, ln in enumerate(lines) if ln.startswith(("ATOM", "HETATM")))
    assert all(i < first_atom for i, ln in enumerate(lines) if ln.startswith("REMARK"))


def test_single_character_chain_ids_are_left_exactly_as_they_were(tmp_path):
    """The 528 jobs a day that already worked must not move."""
    out = tmp_path / "t.pdb"
    _write_structure(_complex(["A", "B"]), out, "pdb")

    assert _chain_column(out) == ["A", "B"]
    assert _remarks(out) == []


def test_the_cif_path_still_writes_the_names_the_user_submitted(tmp_path):
    """mmCIF has no width limit, so relabelling there would be a loss, not a fix."""
    cx = _complex(["sample_03", "sample_05"])
    out = tmp_path / "t.cif"
    _write_structure(cx, out, "cif")

    assert out.read_text() == cx.to_mmcif()          # the cif path is untouched code
    assert "sample_03" in out.read_text()
    assert "sample_05" in out.read_text()


def test_plddt_still_reaches_the_b_factor_column_through_the_relabelling(tmp_path):
    """Guards the 2026-08-16 fix (extra_fields=[b_factor, occupancy]) against this rewrite."""
    out = tmp_path / "t.pdb"
    _write_structure(_complex(["sample_03", "sample_05"]), out, "pdb")

    b = [float(ln[60:66]) for ln in _atom_lines(out)]
    assert b == [50.0, 50.0, 50.0, 51.0, 51.0, 51.0]


def test_thirty_chains_get_thirty_distinct_one_character_ids(tmp_path):
    """Past 26 the old ``_chain_label`` scheme returns "AA", which is not a legal PDB chain
    id at all -- biotite refuses it and the columnar writers shift every field after it. The
    62-symbol alphabet is the only thing the format actually holds."""
    out = tmp_path / "t.pdb"
    _write_structure(_complex([f"chain_{i}" for i in range(30)]), out, "pdb")

    ids = _chain_column(out)
    assert ids == list(CHAIN_ALPHABET[:30])
    assert len(set(ids)) == 30
    assert all(len(i) == 1 for i in ids)
    assert "REMARK 999 CHAIN a IS chain_26" in _remarks(out)


def test_a_63rd_chain_is_refused_by_name_of_the_format_not_by_biotite():
    """62 is the hard ceiling. The message has to send the user to mmCIF."""
    assert chain_id_at(61) == "9"
    with pytest.raises(ValueError, match="output_format: cif"):
        chain_id_at(62)


def _atom_array(chain_ids, res_names):
    import biotite.structure as struc

    arr = struc.AtomArray(len(chain_ids))
    arr.coord = np.arange(len(chain_ids) * 3, dtype="float32").reshape(-1, 3)
    arr.add_annotation("b_factor", float)
    arr.b_factor[:] = 42.0
    for i, (cid, rname) in enumerate(zip(chain_ids, res_names)):
        arr.chain_id[i] = cid
        arr.res_id[i] = 1
        arr.res_name[i] = rname
        arr.atom_name[i] = "CA"
        arr.element[i] = "C"
    return arr


def test_a_five_character_ccd_code_is_truncated_and_remarked(tmp_path):
    """1512 of the 45227 components in the ccd cache have five-character codes, and the PDB
    resName column is three wide. Same class as the chain id, one column over."""
    out = tmp_path / "t.pdb"
    write_atom_array(_atom_array(["A", "A"], ["ALA", "A1A06"]), out)

    assert [ln[17:20] for ln in _atom_lines(out)] == ["ALA", "A1A"]
    assert "REMARK 999 RESIDUE A1A IS A1A06" in _remarks(out)


def test_write_atom_array_does_not_mutate_the_caller_s_array(tmp_path):
    arr = _atom_array(["sample_03"], ["A1A06"])
    write_atom_array(arr, tmp_path / "t.pdb")

    assert arr.chain_id[0] == "sample_03"[:4]   # biotite's chain_id annotation is U4
    assert arr.res_name[0] == "A1A06"


def _boltz_structure(chain_name):
    """The smallest thing ``tt_bio.data.write.to_pdb`` will write: one protein chain, one
    ALA, three atoms."""
    from tt_bio.data import const
    from tt_bio.data.types import AtomV2, Bond, Chain, Connection, Interface, Residue, Structure

    atoms = np.array([(n, (float(i), 0.0, 0.0), True, 0.0, 0.0)
                      for i, n in enumerate(("N", "CA", "C"))], dtype=AtomV2)
    residues = np.array([("ALA", 0, 0, 0, 3, 1, 1, True, True)], dtype=Residue)
    chains = np.array([(chain_name, const.chain_type_ids["PROTEIN"], 0, 0, 0, 0, 3, 0, 1, 0)],
                      dtype=Chain)
    return Structure(
        atoms=atoms,
        bonds=np.array([], dtype=Bond),
        residues=residues,
        chains=chains,
        connections=np.array([], dtype=Connection),
        interfaces=np.array([], dtype=Interface),
        mask=np.array([True]),
    )


@pytest.mark.parametrize("name,expected", [("A", "A"), ("sampl", "A")])
def test_the_boltz_writer_keeps_its_columns_aligned_for_any_chain_name(name, expected):
    """``{chain_tag:>1}`` pads but never truncates, so a long name here shifted every column
    after it and produced a silently wrong file rather than an error. The chain name arrives
    already clipped to 5 characters by the Chain dtype, which is still 5 too many."""
    from tt_bio.data.write import to_pdb

    lines = [ln for ln in to_pdb(_boltz_structure(name), None, True).splitlines()
             if ln.startswith("ATOM")]

    assert len(lines) == 3
    for ln in lines:
        assert ln[21] == expected
        assert ln[22:26] == "   1"                  # resSeq, unshifted
        assert ln[17:20] == "ALA"
        assert float(ln[30:38]) in (0.0, 1.0, 2.0)  # x, unshifted


def test_the_boltz_writer_remarks_a_chain_it_had_to_rename():
    from tt_bio.data.write import to_pdb

    text = to_pdb(_boltz_structure("sampl"), None, True)
    assert "REMARK 999 CHAIN A IS sampl" in text
    assert to_pdb(_boltz_structure("A"), None, True).count("REMARK") == 0
