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


# ---- the serving path: `tt-bio design --model rfd4 --annotated_cif`, no card ----

#: Built here, not shipped by upstream: 3AVE chain C, the 8-residue biantennary complex N-glycan
#: of IgG1 Fc (NAG-NAG-BMA-MAN-NAG-MAN-NAG-FUC), with 80 designed residues around it. None of
#: RFD4's own 391 benchmark inputs contains a sugar. Coordinates are from the PDB entry.
GLYCAN_CIF = REPO / "tests" / "fixtures" / "rfd4_glycan_binder.cif"


def _writer_probe():
    path = REPO / "perf" / "rfd4" / "cond_chem" / "writer_probe.py"
    spec = importlib.util.spec_from_file_location("rfd4_writer_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StandInEngine:
    """The engine boundary, without a card. Returns the right shapes and nothing meaningful."""

    def __init__(self, recipe):
        self.recipe = recipe

    def design(self, features, *, num_designs, nsteps, seed, **kw):
        return _writer_probe().stand_in_design(features)



def test_the_cli_designs_a_glycan_input_and_keeps_its_chemistry(tmp_path, monkeypatch):
    """End to end through `tt-bio design --model rfd4 --annotated_cif`, with the card replaced
    at the engine boundary: the CLI has to featurize a sugar-bearing input, write a CIF, and
    bring every glycan atom and bond through unchanged.

    The glycan families have no upstream recipe, so this is also the check that the CLI's
    `--recipe` reaches the engine.
    """
    from click.testing import CliRunner

    from tt_bio import rfd4
    from tt_bio.main import cli

    seen = {}

    def fake_get_engine(name="auto", **kwargs):
        seen.update(kwargs)
        return _StandInEngine(kwargs.get("recipe"))

    # `rfd4.run` imports `get_engine` from the engine module at call time, which is the name the
    # CLI reaches; the package attribute alone leaves the CLI opening a real card.
    from tt_bio.rfd4 import engine as _engine

    monkeypatch.setattr(_engine, "get_engine", fake_get_engine)
    monkeypatch.setattr(rfd4, "get_engine", fake_get_engine)
    out = tmp_path / "designs"
    result = CliRunner().invoke(cli, [
        "design", str(GLYCAN_CIF), "--model", "rfd4", "--annotated_cif",
        "--recipe", "medium/midpeak", "--seed", "0", "--out_dir", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert seen.get("recipe") == "medium/midpeak", "the CLI did not pass --recipe to the engine"
    assert "88 tokens" in result.output, result.output

    written = out / "rfd4_glycan_binder_generated.cif"
    assert written.exists(), sorted(p.name for p in out.iterdir())

    res = _chem_check().check(GLYCAN_CIF, written)
    assert res["verdict"] == "PASS", res["differences"]
    assert res["components"] == ["BMA", "FUC", "MAN", "NAG"]
    assert res["atoms_checked"] == 99 and res["bonds_checked"] == 101
    # atom37 is heavy-atom only, so the input's hydrogens are gone and that is the encoding.
    assert res["hydrogens_dropped"] == {"input": 93, "generated": 0}

# ---- the structural grader: which chain is the design? ----
#
# `perf/rfd4/e2e/grade.py` grades the protein families by taking the designed chain to be the
# shortest chain holding an atom named `CA`. Neither half of that survives these four families,
# and the two ways it breaks are the two tests below.


def _grade_chem():
    path = REPO / "perf" / "rfd4" / "cond_chem" / "grade_chem.py"
    spec = importlib.util.spec_from_file_location("rfd4_grade_chem", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain(chain_id, res_name, atom_names, res_id_start=1, one_residue=False):
    import biotite.structure as struc
    import numpy as np

    atoms = []
    for i, name in enumerate(atom_names):
        a = struc.Atom(
            [float(i), 0.0, 0.0],
            chain_id=chain_id,
            res_id=res_id_start if one_residue else res_id_start + i,
            res_name=res_name,
            atom_name=name,
            element=name[0],
            hetero=False,
        )
        atoms.append(a)
    return struc.array(atoms)


def _joined(*parts):
    import functools

    return functools.reduce(lambda a, b: a + b, parts)


def test_a_ligand_atom_named_ca_does_not_make_the_ligand_the_design():
    """SAM really does carry an atom named `CA`. Taking the shortest CA-bearing chain grades the
    27-atom ligand as the design and the 150-residue design as the target, which is a wrong
    number rather than a crash."""
    gc = _grade_chem()
    protein = _chain("A", "ALA", ["CA"] * 12)
    ligand = _chain("B", "SAM", ["CA", "N", "O", "S"], one_residue=True)
    designed, kind, anchors, kinds = gc.selections(_joined(protein, ligand))
    assert (designed, kind) == ("A", "protein")
    assert anchors == ["B"]
    assert kinds["B"] == "other"


def test_a_dna_condition_is_a_frame_even_though_it_has_no_ca():
    """A DNA/RNA chain has no `CA` at all, so a CA-only reading of a DNA-binder design finds no
    target to superpose on and dies on an empty concatenate."""
    gc = _grade_chem()
    dna_a = _chain("A", "DA", ["C1'"] * 9)
    dna_b = _chain("B", "DT", ["C1'"] * 10)
    protein = _chain("C", "GLY", ["CA"] * 14)
    designed, kind, anchors, kinds = gc.selections(_joined(dna_a, dna_b, protein))
    assert (designed, kind) == ("C", "protein")
    assert anchors == ["A", "B"]
    assert kinds["A"] == kinds["B"] == "nucleic"


def test_rna_design_has_no_condition_so_it_is_graded_on_its_own_frame():
    """Pseudoknot design conditions on secondary structure, not on a held structure: nothing is
    anchored, so the grade is a shape comparison and has to say so rather than invent a frame."""
    gc = _grade_chem()
    rna = _chain("A", "G", ["C1'"] * 20)
    sel = gc.selections(rna)
    designed, kind, anchors, _ = sel
    assert (designed, kind, anchors) == ("A", "nucleic", [])
    grade = gc.compare(rna, rna, *sel)
    assert grade["frame"].startswith("self")
    assert grade["rmsd_a"] == pytest.approx(0.0, abs=1e-6)


def test_grading_a_design_against_itself_is_zero():
    gc = _grade_chem()
    structure = _joined(_chain("A", "ALA", ["CA"] * 12), _chain("B", "SAM", ["N", "O"], one_residue=True))
    sel = gc.selections(structure)
    grade = gc.compare(structure, structure, *sel)
    assert grade["rmsd_a"] == pytest.approx(0.0, abs=1e-6)
    assert grade["anchor_rmsd_a"] == pytest.approx(0.0, abs=1e-6)
    assert grade["designed_residues"] == 12


# --- the designed half of the chemistry grade: what the model CHOSE, not what it held ---


def test_a_renumbered_design_still_grades_its_chemistry(ligand_cif, tmp_path):
    """An RNA design names its own bases, so residue 7 is a different component in the two
    files and the preservation check misses every key. Per component, the chemistry is still
    there to grade."""
    cc = _chem_check()
    moved = tmp_path / "moved.cif"
    moved.write_text(LIGAND_CIF.replace("LIG B 2 . 2", "LIG B 9 . 9"))
    assert cc.check(ligand_cif, moved)["verdict"] == "FAIL"
    res = cc.check_designed(moved, ligand_cif)
    assert res["verdict"] == "PASS", res["differences"]
    assert res["components"] == ["LIG"] and res["atoms_checked"] == 5


def test_a_wrong_charge_in_a_designed_component_still_fails(ligand_cif, tmp_path):
    res = _chem_check().check_designed(_variant(tmp_path, "bad.cif", "S S1  LIG B 2 . 2 LIG B S1  2 1.0 +1",
                                                "S S1  LIG B 2 . 2 LIG B S1  2 1.0 ?"), ligand_cif)
    assert res["verdict"] == "FAIL"
    assert any(d["key"] == ["LIG", "S1"] for d in res["differences"])


def test_a_component_that_disagrees_with_itself_fails(tmp_path):
    """Two instances of one component must carry the same chemistry, or collapsing them hides
    the difference instead of grading it."""
    second = ("HETATM N S1  LIG B 3 . 3 LIG B S1  7 1.0 +1 0.0 5.0 0.0 1\n"
              "HETATM C C1  LIG B 3 . 3 LIG B C1  8 1.0 ?  2.0 5.0 0.0 1\n")
    path = tmp_path / "two.cif"
    path.write_text(LIGAND_CIF.rstrip().rsplit("#", 1)[0] + second + "#\n")
    ref = tmp_path / "ref.cif"
    ref.write_text(LIGAND_CIF)
    res = _chem_check().check_designed(path, ref)
    assert res["verdict"] == "FAIL"
    assert any(d["key"] == ["LIG", "S1"] for d in res["differences"])


def test_a_diffused_ligand_can_be_dropped_as_a_frame():
    """The sm_binder recipe does not hold its ligand: upstream's own two seeds move SAM's 27
    heavy atoms 1.44 A internally, and both sit ~5 A from the input conformer. Fitting on it
    grades against a frame that does not exist, so `--frame self` drops the anchor."""
    gc = _grade_chem()
    protein = _chain("A", "ALA", ["CA"] * 12)
    ligand = _chain("B", "SAM", ["CA", "N", "O", "S"], one_residue=True)
    joined = _joined(protein, ligand)
    assert gc.selections(joined, frame="condition")[2] == ["B"]
    designed, kind, anchors, kinds = gc.selections(joined, frame="self")
    assert (designed, kind, anchors) == ("A", "protein", [])
    grade = gc.compare(joined, joined, designed, kind, anchors, kinds)
    assert grade["frame"].startswith("self")
    assert grade["anchor_rmsd_a"] is None


# ---- nucleic-acid secondary structure -------------------------------------------------
# The completeness audit (2026-10-25) found this the one RFD4 feature with no test at all:
# nothing built or checked na_ss_state / na_ss_pair. These cover the contract a spec writer
# meets -- the two condition classes a `C_NA_SS_STATE:` or `C_BP:` block resolves to, and the
# masks that decide whether the model sees a signal at all. What they do NOT cover is the
# featurization of a real base-paired structure into the [n_tokens, n_tokens] pair matrix;
# that needs a nucleic-acid input and belongs with the family tests.


def _conditions():
    from tt_bio._vendor.rfd4_prep._cif.conditions import CONDITIONS

    return CONDITIONS


def test_the_na_ss_aliases_a_spec_writes_resolve_to_the_right_conditions():
    """`C_NA_SS_STATE` and `C_BP` are what a condition spec names. An alias that resolved
    elsewhere would silently condition on something other than secondary structure."""
    c = _conditions()
    assert c.na_ss_state.aliases == ("C_NA_SS_STATE",)
    assert c.na_ss_state.n_body == 1
    assert c.basepair.aliases == ("C_BP",)
    assert c.basepair.n_body == 2
    assert c.basepair.is_symmetric, "a base pair i-j is the same pair as j-i"


def test_unspecified_is_the_only_state_that_carries_no_signal():
    """`na_ss_state_mask` is `annotation != UNSPECIFIED`, so UNSPECIFIED means 'say nothing'
    while LOOP is a real instruction: this position is explicitly NOT paired. Collapsing the
    two would turn every unannotated residue into an assertion that it is unpaired."""
    import numpy as np

    from tt_bio._vendor.rfd4_prep._cif.conditions import BasePairEnum

    state = np.array([BasePairEnum.UNSPECIFIED, BasePairEnum.PAIR, BasePairEnum.LOOP],
                     dtype=np.int16)
    mask = _conditions().na_ss_state.mask_from_annotation(state)
    assert list(mask) == [False, True, True]
    assert int(BasePairEnum.UNSPECIFIED) == 0, "the default has to be the zero value"


def test_an_unconditioned_structure_declares_no_pairs():
    """The 2-body default is sparse and empty, not a dense zero matrix: a structure nobody
    annotated must not read as 'every pair is explicitly absent'."""
    import numpy as np

    class _Fake:
        def array_length(self):
            return 7

    default = _conditions().basepair.default_annotation(_Fake())
    assert default.n_atoms == 7
    assert len(default.values) == 0
    assert len(default.pairs) == 0

    pairs = np.array([[0, 6], [1, 5]])
    ann = type(default)(n_atoms=7, pairs=pairs,
                        values=np.array([1, 1], dtype=default.values.dtype))
    mask = _conditions().basepair.mask_from_annotation(ann)
    assert list(mask.values) == [True, True]
    assert mask.pairs.tolist() == pairs.tolist(), "the mask marks the pairs that were given"
