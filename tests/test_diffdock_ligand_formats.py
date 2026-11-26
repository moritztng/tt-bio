"""Every ligand input format upstream's `read_molecule` dispatches on.

Upstream reads a ligand from SMILES or from a 3D file, drops whatever conformer
the file carried, adds hydrogens and embeds one fresh ETKDG conformer. So the
format must not change the graph: the same molecule through .sdf, .mol2, .mol,
.pdb or .pdbqt has to give identical atom features, bonds and torsion mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem

from tt_bio.diffdock.ligand import (
    LIGAND_SUFFIXES,
    build_ligand_graph,
    generate_conformer,
    read_ligand,
    read_molecule,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diffdock"
INDEX = FIXTURES / "INDEX.json"

#: Small, rigid-ish, unambiguous: it survives the round trip through formats
#: that carry no bond orders (.pdb) without a perception difference.
BENZAMIDINE = "NC(=N)c1ccccc1"


def _reference_mol(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


def _write_formats(mol, tmp_path: Path) -> dict[str, Path]:
    out = {}
    sdf = tmp_path / "lig.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)
    out[".sdf"] = sdf

    mol_file = tmp_path / "lig.mol"
    mol_file.write_text(Chem.MolToMolBlock(mol))
    out[".mol"] = mol_file

    pdb = tmp_path / "lig.pdb"
    pdb.write_text(Chem.MolToPDBBlock(mol))
    out[".pdb"] = pdb

    # A .pdbqt is a PDB with extra trailing columns; upstream truncates every
    # line to 66 characters and reads it as PDB.
    pdbqt = tmp_path / "lig.pdbqt"
    pdbqt.write_text("".join(
        line.ljust(66) + "  0.000 A\n" if line.startswith(("ATOM", "HETATM")) else line + "\n"
        for line in Chem.MolToPDBBlock(mol).splitlines()))
    out[".pdbqt"] = pdbqt
    return out


def test_every_upstream_format_reads(tmp_path):
    """Each format reads, and gives the same graph as the SMILES path."""
    mol = _reference_mol(BENZAMIDINE)
    files = _write_formats(mol, tmp_path)
    assert set(files) <= set(LIGAND_SUFFIXES)

    def graph_from(path_or_smiles):
        m = read_ligand(str(path_or_smiles), seed=11)
        return build_ligand_graph(m, remove_hs=True)

    def bonds(graph):
        """Bonds as a set of (atom, atom, bond-type), order-independent.

        The edge list is RDKit's bond enumeration order, which depends on the
        reader: a .pdb round trip puts the ring-closing bond first. That order
        reaches nothing the model reads as an order (every consumer scatter-sums
        over edges), so the formats are compared on the bond set.
        """
        ei = graph["edge_index"].numpy()
        attr = graph["edge_attr"].numpy().argmax(1)
        return {(int(a), int(b), int(t)) for a, b, t in zip(ei[0], ei[1], attr)}

    def torsions(graph):
        """The rotatable bonds, as the set of atom sets each one moves."""
        return {frozenset(np.flatnonzero(row).tolist()) for row in graph["mask_rotate"]}

    reference = graph_from(BENZAMIDINE)
    for suffix, path in files.items():
        got = graph_from(path)
        assert got["x"].shape == reference["x"].shape, suffix
        assert np.array_equal(got["x"].numpy(), reference["x"].numpy()), suffix
        assert bonds(got) == bonds(reference), suffix
        assert torsions(got) == torsions(reference), suffix


def test_mol2_reads(tmp_path):
    """.mol2 has no RDKit writer, so it is written by hand and only read back."""
    pytest.importorskip("rdkit")
    mol2 = tmp_path / "lig.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\nwater\n 3 2 1 0 0\nSMALL\nNO_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "      1 O1      0.0000   0.0000   0.0000 O.3     1  HOH1        0.0000\n"
        "      2 H1      0.9572   0.0000   0.0000 H       1  HOH1        0.0000\n"
        "      3 H2     -0.2400   0.9266   0.0000 H       1  HOH1        0.0000\n"
        "@<TRIPOS>BOND\n     1    1    2 1\n     2    1    3 1\n")
    mol = read_molecule(mol2, sanitize=True)
    assert mol is not None
    assert mol.GetNumAtoms() == 3


def test_conformer_is_regenerated_not_taken_from_the_file(tmp_path):
    """A 3D file's own coordinates never reach the graph.

    Upstream calls `RemoveAllConformers()` before embedding, so a crystal pose
    in the input cannot leak into the model's starting geometry.
    """
    mol = _reference_mol(BENZAMIDINE)
    sdf = tmp_path / "lig.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        w.write(mol)
    original = mol.GetConformer().GetPositions()

    got = read_ligand(str(sdf), seed=3)
    assert got.GetNumConformers() == 1
    new = got.GetConformer().GetPositions()
    assert new.shape[0] >= original.shape[0]
    assert not np.allclose(new[: len(original)], original)


def test_unknown_suffix_is_refused(tmp_path):
    bad = tmp_path / "lig.xyz"
    bad.write_text("3\n\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
    with pytest.raises(ValueError, match="must be one of"):
        read_molecule(bad)


def test_unreadable_ligand_raises_not_returns_none():
    """A description that is neither SMILES nor a file is a loud failure."""
    with pytest.raises((ValueError, FileNotFoundError)):
        read_ligand("not-a-molecule-and-not-a-path")


def test_conformer_seed_is_reproducible():
    """tt-bio's seed makes the ETKDG conformer reproducible; upstream has none."""
    a = read_ligand(BENZAMIDINE, seed=5).GetConformer().GetPositions()
    b = read_ligand(BENZAMIDINE, seed=5).GetConformer().GetPositions()
    assert np.array_equal(a, b)


@pytest.mark.skipif(not INDEX.exists(), reason="DiffDock input fixtures are not present")
def test_fixture_ligands_all_read():
    """Every fixture ligand reads from its .sdf and builds a graph."""
    for entry in json.loads(INDEX.read_text())["complexes"]:
        mol = read_molecule(FIXTURES / f"{entry['id']}_ligand.sdf", sanitize=True)
        assert mol is not None, entry["id"]
        mol.RemoveAllConformers()
        mol = AddHs(mol)
        generate_conformer(mol, seed=42)
        graph = build_ligand_graph(mol, remove_hs=True)
        assert graph["x"].shape[0] == entry["lig_heavy_atoms"], entry["id"]
