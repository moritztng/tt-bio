"""The release gate's ESMFold2 cocrystal leg, and the reason it needs its own verdict rule.

Every parity metric the esmfold2 harness reports is dominated by the L**2 protein block, so a
fold that dropped the ligand scores about the same as one that placed it. The gate's recorded
behaviour for esmfold2 is PASS-if-scored, which on a cocrystal target would sign off the exact
failure this leg exists to catch: the ligand missing and the job saying it worked.

These are the negative controls for that rule. A check nothing can break is not reading what it
claims to read, so each case below mutates one field of a real measured record (1FKG, 140
tokens, Wormhole card 1, 2026-09-09) and asserts the gap is reported.
"""
import ast
import copy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "full_parity_gate.py"

# One real target record, the shape scripts/esmfold2_e2e_parity.py writes for a cocrystal.
MEASURED = {
    "protein": "fkg_ligand", "target": "107aa+SB3", "L": 140, "n_seeds": 2,
    "n_ligand_chains": 1,
    "plddt_pcc": 0.99917, "distogram_rel_l2": 0.05026,
    "kabsch_rmsd": {"metric": "kabsch_rmsd", "within_noise_floor": True},
    "ligand": {"n_ligand_atoms": 33, "n_protein_atoms": 831,
               "ligand_rmsd_protein_frame": 0.8895,
               "ref": {"min_contact_A": 2.778, "n_contacts": 54},
               "device": {"min_contact_A": 2.821, "n_contacts": 52}},
}


def _gate_ns():
    """The ligand check and its thresholds, without running the gate's module-level setup."""
    want = {"_esmfold2_ligand_gap", "ESMFOLD2_LIGAND_MAX_RMSD_A",
            "ESMFOLD2_LIGAND_MAX_CONTACT_A", "ESMFOLD2_LIGAND_MIN_CONTACTS"}
    body = [n for n in ast.parse(GATE.read_text()).body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in want)]
    assert len(body) == len(want), f"the gate no longer defines all of {sorted(want)}"
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GATE), "exec"), ns)
    return ns


def _gap(mutate=lambda r: None):
    rec = copy.deepcopy(MEASURED)
    mutate(rec)
    return _gate_ns()["_esmfold2_ligand_gap"]([rec])


def test_the_measured_cocrystal_record_passes():
    assert _gap() == ""


def test_a_protein_only_record_is_not_asked_for_a_ligand():
    """The same rule runs over every esmfold2 leg, and three of the four fold no ligand."""
    assert _gap(lambda r: (r.update(n_ligand_chains=0, target="107aa"), r.pop("ligand"))) == ""


@pytest.mark.parametrize("case,mutate", [
    # The silent drop, in the two shapes it can take in a record.
    ("ligand block emptied", lambda r: r.update(ligand={})),
    ("ligand block absent", lambda r: r.pop("ligand")),
    ("zero ligand atoms", lambda r: r["ligand"].update(n_ligand_atoms=0)),
    # Placed, but not where the reference places it.
    ("ligand far off the reference pose", lambda r: r["ligand"].update(ligand_rmsd_protein_frame=9.9)),
    # Placed somewhere, but not on the protein.
    ("ligand in solvent", lambda r: r["ligand"]["device"].update(min_contact_A=30.0)),
    ("pocket barely touched", lambda r: r["ligand"]["device"].update(n_contacts=3)),
])
def test_each_way_the_ligand_can_go_wrong_is_reported(case, mutate):
    gap = _gap(mutate)
    assert gap, f"{case}: the gate called this a PASS"
    assert "107aa+SB3" in gap, f"{case}: the gap does not name the target ({gap})"


def test_the_leg_is_registered_and_names_its_own_target():
    """A leg missing from LEGS is a leg that never runs. Read as source: importing the gate
    opens no device but does pull in ttnn, which this test has no need of."""
    src = GATE.read_text()
    assert 'Leg("esmfold2-cocrystal"' in src, "the cocrystal leg is not in LEGS"
    rest = src.split('Leg("esmfold2-cocrystal"', 1)[1]
    leg = rest.split("\n    Leg(", 1)[0].split("\n]", 1)[0]
    assert "examples/fkg_ligand.yaml" in leg
    assert '"--proteins"' in leg, (
        "the leg must name its target through device_args; without it the harness folds the "
        "four protein-only defaults and the leg silently duplicates esmfold2-trpcage")
    assert 'committed_json="esmfold2-cocrystal.json"' in leg
