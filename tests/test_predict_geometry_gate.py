"""The predict-leg geometry arm, on the chain kinds it has to tell apart.

The arm exists because a fold can clear a global RMSD/TM floor with a broken backbone and
atoms sitting on top of each other: the corrupted OpenDDE output that motivated it shipped 19
backbone gaps and 9.50% clashing atoms with a HIGHER plDDT than the fixed code. So a gate arm
that cannot fail is decoration, and these cases pin both directions.

The fixture is real coordinates from PDB 3HDD, an engrailed homeodomain bound to a DNA duplex,
trimmed to backbone atoms and 10 nt + 10 nt + 25 aa. It discriminates by construction: the two
DNA chains measure 0.8889 of steps in band, which clears the nucleic floor (0.85) but not the
protein one (0.90). Score a nucleic chain on the protein band and this fixture fails.
"""
import pathlib
import sys

import gemmi
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from release_gate import PREDICT_MAX_CLASH_FRAC, _load_geometry_harness, _predict_geometry

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "geom_protein_dna.cif"


@pytest.fixture(scope="module")
def geom():
    return _load_geometry_harness()


def _chains(geom, path=FIXTURE):
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.setup_entities()
    return st, {c["chain"]: c for c in geom.chain_geometry(st)[0]}


def test_nucleic_chains_are_measured_on_their_own_band(geom):
    """DNA is measured P-P, protein Ca-Ca, in the same structure."""
    _, by_name = _chains(geom)
    assert set(by_name) == {"A", "C", "D"}

    for name in ("C", "D"):
        assert by_name[name]["kind"] == "nucleic"
        assert by_name[name]["step_band"] == list(geom.PP_BAND)
    assert by_name["A"]["kind"] == "protein"
    assert by_name["A"]["step_band"] == list(geom.CA_CA_BAND)


def test_nucleic_chain_clears_its_floor_but_not_the_protein_one(geom):
    """The discriminating property: the wrong band would fail this fixture."""
    _, by_name = _chains(geom)
    dna = by_name["C"]["in_band_frac"]
    assert geom.PP_FAIL_FRAC <= dna < geom.CA_CA_FAIL_FRAC, (
        f"fixture no longer discriminates: DNA in-band {dna} must sit between the nucleic "
        f"floor {geom.PP_FAIL_FRAC} and the protein floor {geom.CA_CA_FAIL_FRAC}")
    assert _predict_geometry([FIXTURE], geom)["fails"] == []


def test_radius_of_gyration_is_not_applied_to_nucleic(geom):
    """A duplex is a rod. Rg is a globular-protein relation and says nothing about it."""
    _, by_name = _chains(geom)
    assert by_name["C"]["rg_ratio"] > geom.RG_BAND[1], "fixture duplex must be outside RG_BAND"
    assert _predict_geometry([FIXTURE], geom)["fails"] == []


def test_one_chain_holding_both_polymers_is_split_per_kind(geom):
    """RFD3's na-binder output merges the designed protein and a DNA strand into ONE chain.

    Classifying by the chain's first residue and measuring P-P across all of it scored 13.6%
    in band on a design whose two halves are each perfect.
    """
    st = gemmi.read_structure(str(FIXTURE))
    st.remove_alternative_conformations()
    st.setup_entities()
    merged = gemmi.Chain("X")
    for name in ("C", "A"):
        for res in st[0][name]:
            merged.add_residue(res)
    model = gemmi.Model("1")
    model.add_chain(merged)
    one = gemmi.Structure()
    one.add_model(model)

    by_name = {c["chain"]: c for c in geom.chain_geometry(one)[0]}
    assert set(by_name) == {"X(nucleic)", "X(protein)"}
    assert by_name["X(nucleic)"]["step_band"] == list(geom.PP_BAND)
    assert by_name["X(protein)"]["step_band"] == list(geom.CA_CA_BAND)


def test_reported_in_band_names_the_chain_closest_to_its_own_floor(geom):
    """Two floors, so the smallest raw fraction is not always the chain in trouble."""
    real = geom.chain_geometry

    def crafted(_st):
        chains = [{"chain": "A", "kind": "protein", "in_band_frac": 0.88, "breaks": 0},
                  {"chain": "C", "kind": "nucleic", "in_band_frac": 0.86, "breaks": 0}]
        return chains, ["chain A (protein): only 88.0% of consecutive backbone steps in "
                        "[3.6, 4.1] A (floor 90%)"], []

    geom.chain_geometry = crafted
    try:
        g = _predict_geometry([FIXTURE], geom)
    finally:
        geom.chain_geometry = real

    # 0.86 is the smaller number but it clears the 0.85 nucleic floor; 0.88 is the failure.
    assert g["in_band"] == 0.88
    assert g["in_band_chain"] == "A/protein"
    assert g["fails"], "the arm must still fail the leg, attribution is a reporting concern"


def test_a_clashing_structure_fails_the_arm(geom):
    """Negative control: the arm must reject what it exists to catch."""
    st = gemmi.read_structure(str(FIXTURE))
    st.remove_alternative_conformations()
    st.setup_entities()
    real = geom.clashes
    n_heavy = sum(1 for _ in st[0].all())
    over = int(PREDICT_MAX_CLASH_FRAC * n_heavy) + geom.CLASH_MAX_ABS + 1

    geom.clashes = lambda _st: (over, n_heavy, 1.1)
    try:
        g = _predict_geometry([FIXTURE], geom)
    finally:
        geom.clashes = real

    assert any("heavy-atom clashes" in f for f in g["fails"]), (
        f"{over} clashes in {n_heavy} atoms must fail the arm, got {g['fails']}")
