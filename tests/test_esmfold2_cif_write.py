"""Regression: MolecularComplex.to_mmcif() must emit a parseable mmCIF.

`tt-bio predict --model esmfold2 ... --override` writes its structure via
MolecularComplex.to_mmcif() (tt_bio/_vendor/esm/utils/structure/molecular_complex.py).
That writer added a b_factor annotation but never an occupancy one, so biotite
omitted the `_atom_site.occupancy` column entirely — which crashes
Bio.PDB.MMCIFParser (KeyError '_atom_site.occupancy') on read-back, including
in this repo's own compute_rmsd/get_ca_atoms harness (tests/test_structure.py).
No TT device is involved in writing or parsing mmCIF, so this is a plain CPU test.
"""
import warnings

import numpy as np

from tt_bio._vendor.esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexMetadata,
)


def _minimal_complex():
    # Two-residue, backbone-only single-chain complex — same shape produced by
    # build_molecular_complex_from_features for an ESMFold2 prediction.
    atom_positions = np.array([
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.9, 0.0, 0.0],
        [4.4, 0.0, 0.0], [5.9, 0.0, 0.0], [7.3, 0.0, 0.0],
    ], dtype=np.float32)
    return MolecularComplex(
        id="test",
        sequence=["ALA", "GLY"],
        atom_positions=atom_positions,
        atom_elements=np.array(["N", "C", "C", "N", "C", "C"]),
        token_to_atoms=np.array([[0, 3], [3, 6]]),
        chain_id=np.array([0, 0]),
        plddt=np.array([90.0, 92.0]),
        metadata=MolecularComplexMetadata(entity_lookup={0: "1"}, chain_lookup={0: "A"}),
    )


def test_to_mmcif_includes_occupancy_column():
    cif_text = _minimal_complex().to_mmcif()
    assert "_atom_site.occupancy" in cif_text


def test_to_mmcif_output_is_parseable_by_biopython(tmp_path):
    warnings.filterwarnings("ignore")
    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)

    out = tmp_path / "esmfold2_test.cif"
    out.write_text(_minimal_complex().to_mmcif())

    structure = MMCIFParser(QUIET=True).get_structure("s", str(out))
    n_atoms = sum(1 for _ in structure.get_atoms())
    assert n_atoms == 6


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_to_mmcif_includes_occupancy_column()
    with tempfile.TemporaryDirectory() as d:
        test_to_mmcif_output_is_parseable_by_biopython(Path(d))
    print("PASS")
