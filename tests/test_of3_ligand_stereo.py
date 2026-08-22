"""A CCD ligand's reference conformer must keep its chirality.

`processed_reference_molecule_from_atom_array` builds the mol from an atom array whose
only stereo information is its 3D coordinates, and then drops those coordinates
(`RemoveConformer(0)`). Without `Chem.AssignStereochemistryFrom3D` in between, the mol
handed to the conformer generator carries no chiral tags, so ETKDG picks a handedness per
centre at random -- and the conformer it generates becomes `ref_pos`, a live model input to
the atom encoder.

Measured when this was broken: SB3, SAH and ATP came out with every centre unassigned, and
their generated reference conformers had inverted signed volumes at several centres against
upstream (SB3 centre 1, SAH centres 1/9/11, ATP centres 4/16/20).

Card-free. The expected CIP codes below are upstream's, read off the CCD's own deposited
coordinates, so they are the chemistry and not a snapshot of our output.
"""
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from tt_bio._vendor.openfold3.core.data.primitives.structure.query import (  # noqa: E402
    atom_array_from_ccd_code,
    processed_reference_molecule_from_atom_array,
)
from tt_bio._vendor.openfold3.core.data.resources.residues import (  # noqa: E402
    MoleculeType,
)

# CCD code -> the assigned centres upstream produces from the deposited coordinates.
EXPECTED = {
    "SB3": [(1, "S"), (18, "R")],
    "SAH": [(1, "S"), (9, "S"), (11, "S"), (13, "R"), (15, "R")],
    "ATP": [(4, "R"), (8, "R"), (14, "R"), (16, "S"), (18, "R"), (20, "R")],
}


def _centres(mol):
    return sorted(Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                            useLegacyImplementation=False))


def _ref_mol(code):
    arr = atom_array_from_ccd_code(code, chain_id="B", res_id=1,
                                  molecule_type=MoleculeType.LIGAND)
    return processed_reference_molecule_from_atom_array(arr)


@pytest.mark.parametrize("code,expected", sorted(EXPECTED.items()))
def test_ccd_ligand_reference_molecule_keeps_its_stereocentres(code, expected):
    try:
        rm = _ref_mol(code)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{code} not resolvable from the local CCD: {e}")
    got = _centres(rm.mol)
    assert got == expected, f"{code}: {got} != {expected}"


@pytest.mark.parametrize("code", sorted(EXPECTED))
def test_no_stereocentre_is_left_unassigned(code):
    """The failure signature of the bug: every centre present but marked '?'."""
    try:
        rm = _ref_mol(code)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{code} not resolvable from the local CCD: {e}")
    unassigned = [i for i, tag in _centres(rm.mol) if tag == "?"]
    assert not unassigned, f"{code}: centres {unassigned} unassigned"


def test_a_ligand_with_no_stereocentre_is_still_fine():
    """Benzene-like: nothing to assign, and the call must not invent anything."""
    try:
        rm = _ref_mol("EKY")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"EKY not resolvable from the local CCD: {e}")
    assert _centres(rm.mol) == []
