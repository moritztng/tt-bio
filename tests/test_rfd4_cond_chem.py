"""RFD4's non-protein condition families: ligands, nucleic acids, glycans, NA secondary structure.

Chemistry is correctness, not precision. A design that lands within the Angstrom bar with the
wrong element, the wrong bond order or the wrong formal charge is wrong, so the checks here are
exact-match and have no tolerance to relax.

What is checked without a device and without the reference env: the 147-class vocabulary covers
the glycan tokens with usable atom37 layouts, and the chemistry grader
(`perf/rfd4/cond_chem/chem_check.py`) catches each of the three ways chemistry can be lost.
The full-design grades per family, which need a card, are in `state/rfd4/FAMILIES-CHEM.md`.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _chem_check():
    path = REPO / "perf" / "rfd4" / "cond_chem" / "chem_check.py"
    spec = importlib.util.spec_from_file_location("rfd4_chem_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: A two-component CIF: one ALA residue and one ligand whose chemistry is the point. The
#: ligand is a sulfonium/carboxylate pair, the shape that makes SAM a real chemistry test:
#: a parser that drops leaving hydrogens or infers charge from bond degree gets these wrong.
LIGAND_CIF = """\
data_fixture
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
LIG S1 C1 SING N
LIG C1 C2 DOUB N
LIG C2 O1 SING N
LIG C2 O2 DOUB N
#
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.id
_atom_site.occupancy
_atom_site.pdbx_formal_charge
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM   N CA  ALA A 1 1 1 ALA A CA  1 1.0 ?  0.0 0.0 0.0 1
HETATM S S1  LIG B 2 . 2 LIG B S1  2 1.0 +1 1.0 0.0 0.0 1
HETATM C C1  LIG B 2 . 2 LIG B C1  3 1.0 ?  2.0 0.0 0.0 1
HETATM C C2  LIG B 2 . 2 LIG B C2  4 1.0 ?  3.0 0.0 0.0 1
HETATM O O1  LIG B 2 . 2 LIG B O1  5 1.0 -1 4.0 0.0 0.0 1
HETATM O O2  LIG B 2 . 2 LIG B O2  6 1.0 ?  3.0 1.0 0.0 1
#
"""


@pytest.fixture
def ligand_cif(tmp_path):
    path = tmp_path / "input.cif"
    path.write_text(LIGAND_CIF)
    return path


def _variant(tmp_path, name, old, new):
    path = tmp_path / name
    assert old in LIGAND_CIF
    path.write_text(LIGAND_CIF.replace(old, new, 1))
    return path


def test_identical_chemistry_passes_and_reads_both_charges(ligand_cif):
    """The grader has to see the charges before its failures mean anything."""
    res = _chem_check().check(ligand_cif, ligand_cif)
    assert res["verdict"] == "PASS", res["differences"]
    assert res["atoms_checked"] == 5 and res["bonds_checked"] == 4
    assert res["components"] == ["LIG"]
    charges = {row[2]: row[3] for row in res["charged_atoms"]}
    assert charges == {"S1": 1, "O1": -1}


def test_a_lost_formal_charge_fails(ligand_cif, tmp_path):
    """The covalent-glycan case: a leaving atom kept or dropped moves a formal charge, and
    nothing about the geometry says so."""
    got = _variant(tmp_path, "neutral.cif", "LIG B S1  2 1.0 +1", "LIG B S1  2 1.0 ?")
    res = _chem_check().check(ligand_cif, got)
    assert res["verdict"] == "FAIL"
    assert [d["key"][-1] for d in res["differences"]] == ["S1"]
    assert res["differences"][0]["input"] == ("S", 1)
    assert res["differences"][0]["generated"] == ("S", 0)


def test_a_wrong_bond_order_fails(ligand_cif, tmp_path):
    got = _variant(tmp_path, "single.cif", "LIG C2 O2 DOUB N", "LIG C2 O2 SING N")
    res = _chem_check().check(ligand_cif, got)
    assert res["verdict"] == "FAIL"
    assert res["differences"][0]["input"] == ("DOUB", "N")
    assert res["differences"][0]["generated"] == ("SING", "N")


def test_a_wrong_element_fails(ligand_cif, tmp_path):
    got = _variant(tmp_path, "oxygen.cif", "HETATM S S1  LIG", "HETATM O S1  LIG")
    res = _chem_check().check(ligand_cif, got)
    assert res["verdict"] == "FAIL"
    assert res["differences"][0]["input"][0] == "S"
    assert res["differences"][0]["generated"][0] == "O"


def test_protein_only_input_is_reported_empty_not_passing(tmp_path):
    """A family whose input carries no chemistry must not come out of the grader as a PASS;
    that is how a chemistry check silently stops checking anything."""
    only_protein = "\n".join(
        line for line in LIGAND_CIF.splitlines()
        if "LIG" not in line and "chem_comp_bond" not in line
    )
    path = tmp_path / "protein.cif"
    path.write_text(only_protein + "\n")
    assert _chem_check().check(path, path)["verdict"] == "EMPTY"


# ---- the 147-class vocabulary, where the non-protein residues live ----

GLYCANS = ("NAG", "BMA", "MAN", "FUC", "GLC", "GAL", "BGC", "SIA", "NDG", "XYP", "GLA", "A2G", "NGA")


def test_the_vocabulary_carries_all_thirteen_glycans():
    from tt_bio._vendor.rfd4_prep._cif.encoding import GLYCAN_NAMES, UNIFIED_ATOM37_ENCODING

    assert GLYCAN_NAMES == GLYCANS
    assert len(UNIFIED_ATOM37_ENCODING.token_to_idx) == 147
    idx = [UNIFIED_ATOM37_ENCODING.token_to_idx[name] for name in GLYCANS]
    assert idx == list(range(134, 147)), "glycan class indices are checkpoint order; they cannot move"


def test_every_glycan_has_a_usable_atom37_layout():
    """Thirteen tokens with empty layouts would still pass a class-count check."""
    from tt_bio._vendor.rfd4_prep._cif.encoding import UNIFIED_ATOM37_ENCODING

    for name in GLYCANS:
        atoms = [str(a) for a in UNIFIED_ATOM37_ENCODING.token_atoms[name] if a]
        # XYP is a pentose and is the smallest at 10; every other sugar here has a C6/O6.
        assert len(atoms) >= 10, f"{name} has only {len(atoms)} atom37 slots"
        assert len(set(atoms)) == len(atoms), f"{name} repeats an atom name"
        assert "C1" in atoms, f"{name} is missing its anomeric carbon"
        assert all(a[0] in "CON" for a in atoms), f"{name} has a non-sugar element: {atoms}"


def test_the_n_acetyl_glycans_carry_their_amide_nitrogen():
    """NAG/NDG/A2G/NGA are the N-linked sugars; the amide N is the atom the atomworks 3.0.4
    link-chemistry change moves from +1 to 0, so a layout without it cannot represent either."""
    from tt_bio._vendor.rfd4_prep._cif.encoding import UNIFIED_ATOM37_ENCODING

    for name in ("NAG", "NDG", "A2G", "NGA"):
        atoms = set(UNIFIED_ATOM37_ENCODING.token_atoms[name])
        assert {"N2", "C7", "C8", "O7"} <= atoms, f"{name} is missing its N-acetyl group"


# ---- the recycle schedule is per design task, so the CLI has to expose it ----

def test_the_cli_offers_every_shipped_recycle_schedule():
    """Upstream prescribes a different schedule per task: midpeak for small-molecule binders,
    front_loaded for DNA binders. A CLI that only runs thinking_medium cannot run either at
    its own task's recipe. The list is spelled out in main.py to keep `--help` free of torch,
    so it has to be checked against the real one."""
    from tt_bio.main import _RFD4_RECIPES
    from tt_bio.rfd4.sampler import RECYCLE_SCHEDULES

    assert sorted(_RFD4_RECIPES) == sorted(RECYCLE_SCHEDULES)


def test_every_recipe_is_sixty_steps():
    """`--nsteps` defaults to 60 for all of them; a schedule that did not sum to 60 would make
    the default silently wrong for that recipe (the sampler refuses the mismatch)."""
    from tt_bio.rfd4.sampler import RECYCLE_SCHEDULES

    for name, schedule in RECYCLE_SCHEDULES.items():
        assert sum(n for n, _ in schedule) == 60, name


def test_design_help_lists_the_recipes():
    from click.testing import CliRunner

    from tt_bio.main import cli

    out = CliRunner().invoke(cli, ["design", "--help"]).output
    assert "--recipe" in out
    for name in ("medium/midpeak", "medium/front_loaded", "thinking_medium"):
        assert name in out


# ---- two things that are NOT chemistry loss, and must not read as it ----

def test_an_expandable_segment_placeholder_is_not_a_component(tmp_path):
    """A spec's variable-length segment is written as one residue named `<min>-<max>` holding a
    single element-X atom, and the design replaces it with the residues it sampled. Counting that
    as a lost component makes every conditioned design fail."""
    cc = _chem_check()
    assert not cc.is_chemistry("150-150")
    assert not cc.is_chemistry("120-150")
    assert cc.is_chemistry("SAM") and cc.is_chemistry("NAG") and cc.is_chemistry("DA")
    assert not cc.is_chemistry("ALA")


def test_hydrogens_are_counted_not_compared(ligand_cif, tmp_path):
    """RFD4's atom37 vocabulary is heavy-atom only, so a design never carries the hydrogens an
    input had. That is the encoding, not a chemistry change -- but it is reported, because a
    check that silently drops atoms is how one stops checking."""
    cc = _chem_check()
    with_h = LIGAND_CIF.replace(
        "HETATM O O2  LIG B 2 . 2 LIG B O2  6 1.0 ?  3.0 1.0 0.0 1\n",
        "HETATM O O2  LIG B 2 . 2 LIG B O2  6 1.0 ?  3.0 1.0 0.0 1\n"
        "HETATM H HO1 LIG B 2 . 2 LIG B HO1 7 1.0 ?  4.5 0.5 0.0 1\n")
    src = tmp_path / "with_h.cif"
    src.write_text(with_h)

    res = cc.check(src, ligand_cif)
    assert res["verdict"] == "PASS", res["differences"]
    assert res["hydrogens_dropped"] == {"input": 1, "generated": 0}
    assert cc.check(src, ligand_cif, hydrogens=True)["verdict"] == "FAIL"
