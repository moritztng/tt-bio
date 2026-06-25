"""Platform-side constraint gate: which models accept which constraint kind.

The fix makes the gate constraint-kind-aware. Pocket/contact "binding constraints"
need a constraint embedder (Boltz-2 only); covalent "bond" constraints only need the
token-bond graph, which Boltz-2 and Protenix-v2 both honour. The engine-side
counterpart lives in test_constraints.py.

CPU-only.
"""
import pytest

from tt_bio.platform.limits import _check_one, inspect, InputError

_POCKET = """sequences:
  - protein: {id: A, sequence: MKVLAAAAAA}
  - ligand: {id: B, ccd: SO4}
constraints:
  - pocket: {binder: B, contacts: [[A, 3]]}
"""
_BOND = """sequences:
  - protein: {id: A, sequence: MKVLAAAAAA}
  - ligand: {id: B, ccd: SO4}
constraints:
  - bond: {atom1: [A, 3, CA], atom2: [B, 1, S]}
"""
_DISULFIDE = """sequences:
  - protein: {id: A, sequence: MKCVLAACAA}
constraints:
  - bond: {atom1: [A, 3, SG], atom2: [A, 8, SG]}
"""


def _allowed(spec, model):
    try:
        _check_one(spec, where="S", model=model)
        return True
    except InputError:
        return False


def test_pocket_constraint_boltz2_only():
    assert _allowed(_POCKET, "boltz2")
    assert not _allowed(_POCKET, "protenix-v2")
    assert not _allowed(_POCKET, "esmfold2")


def test_bond_constraint_allowed_for_protenix():
    # The fix: covalent bond constraints are a token-bond feature Protenix honours.
    assert _allowed(_BOND, "boltz2")
    assert _allowed(_BOND, "protenix-v2")
    assert _allowed(_DISULFIDE, "protenix-v2")
    # ESMFold has no ligand/bond support — still blocked, even for a protein-only bond.
    assert not _allowed(_BOND, "esmfold2")
    assert not _allowed(_DISULFIDE, "esmfold2")


def test_constraint_kind_classification():
    assert inspect(_POCKET)["binding_constraints"] == 1
    assert inspect(_POCKET)["bond_constraints"] == 0
    assert inspect(_BOND)["bond_constraints"] == 1
    assert inspect(_BOND)["binding_constraints"] == 0
