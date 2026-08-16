"""Regression: ``output_format: pdb`` on the ESMFold2 ids wrote every B-factor
as 0.00. The PDB path round-trips the mmCIF through biotite's ``get_structure``,
which drops the b_factor column unless it is named in ``extra_fields`` — the
per-residue pLDDT the CIF carries was silently discarded for anyone consuming
PDB. Host-only: no device, no checkpoints.
"""
import io

import numpy as np

from tt_bio._vendor.esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexMetadata,
)
from tt_bio.main import _write_structure


def _minimal_complex():
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
        plddt=np.array([0.5123, 0.9876], dtype=np.float32),
        metadata=MolecularComplexMetadata(entity_lookup={0: "1"}, chain_lookup={0: "A"}),
    )


def test_pdb_output_carries_the_plddt_b_factors(tmp_path):
    """The PDB's B-factor column must equal the CIF's pLDDT*100, per atom."""
    import biotite.structure.io.pdbx as pdbx

    cx = _minimal_complex()
    cif_path, pdb_path = tmp_path / "t.cif", tmp_path / "t.pdb"
    _write_structure(cx, cif_path, "cif")
    _write_structure(cx, pdb_path, "pdb")

    arr = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(cif_path.read_text())),
                             model=1, extra_fields=["b_factor"])
    cif_b = np.round(arr.b_factor, 2)
    pdb_b = np.array([float(line[60:66]) for line in pdb_path.read_text().splitlines()
                      if line.startswith(("ATOM", "HETATM"))])
    assert len(pdb_b) == 6
    assert np.array_equal(cif_b, pdb_b)
    assert set(pdb_b) == {51.23, 98.76}
