"""The DiffDock-L ligand graph: atom features, bonds, conformer, torsion mask.

Reproduces `datasets.process_mols.read_molecule`, `lig_atom_featurizer`,
`get_lig_graph`, `generate_conformer` and `utils.torsion.get_transformation_mask`.
Upstream reaches the same code through torch_geometric; here the bond graph is a
plain edge index and the torsion mask is computed on it with networkx directly,
which is what `to_homogeneous()` hands networkx anyway at the point it is called
(the complex graph holds nothing but ligand atoms yet).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT

from .constants import allowable_features
from .receptor import safe_index

#: Upstream's bond vocabulary; anything else is folded into SINGLE.
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

#: The file suffixes upstream's `read_molecule` dispatches on.
LIGAND_SUFFIXES = (".sdf", ".mol2", ".mol", ".pdb", ".pdbqt")

_CHIRALITY_FALLBACK = {"CHI_SQUAREPLANAR", "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL"}


def read_molecule(path: str | Path, *, sanitize: bool = True, remove_hs: bool = False):
    """Read a ligand from a 3D file, by suffix, exactly as upstream does.

    `.mol` is read with the SDF reader: an MDL molfile is one SDF record, and
    upstream's suffix list omits it only because its own CLI never sees one.
    Returns None when RDKit cannot sanitize the molecule, so the caller can
    isolate the failure per complex.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    elif suffix in (".sdf", ".mol"):
        mol = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)[0]
    elif suffix == ".pdbqt":
        block = "".join(line[:66] + "\n" for line in path.read_text().splitlines())
        mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError(
            f"ligand file must be one of {', '.join(LIGAND_SUFFIXES)}, got {path.name}")
    if mol is None:
        return None
    try:
        if sanitize:
            Chem.SanitizeMol(mol)
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception:
        return None
    return mol


def generate_conformer(mol, *, seed: int | None = None) -> bool:
    """Embed one ETKDGv2 conformer in place. True if random coordinates were used.

    Upstream retries the standard embedding three times and only then falls back
    to random coordinates plus an MMFF cleanup. `seed` is tt-bio's addition: at
    the default (None) the embedding uses RDKit's own stream, as upstream does.
    """
    params = AllChem.ETKDGv2()
    if seed is not None:
        params.randomSeed = int(seed)
    for _ in range(3):
        if AllChem.EmbedMolecule(mol, params) != -1:
            return False
    params.useRandomCoords = True
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, confId=0)
    return True


def read_ligand(description: str, *, seed: int | None = None):
    """A ligand from a SMILES string or a 3D file, with a fresh conformer.

    Upstream's detection order: try SMILES first, fall back to the file reader.
    Either way any conformer in the input is dropped, hydrogens are added, and
    one conformer is embedded, so a crystal pose never leaks into the input.

    A path is still handed to the SMILES parser first, to keep that order, but RDKit's
    complaint about it is silenced: it is five lines of parse error about a filename that
    read perfectly well a moment later.
    """
    from pathlib import Path

    from rdkit import RDLogger

    looks_like_a_path = Path(description).expanduser().exists()
    if looks_like_a_path:
        RDLogger.DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromSmiles(description)
    finally:
        if looks_like_a_path:
            RDLogger.EnableLog("rdApp.error")
    if mol is None:
        mol = read_molecule(description, sanitize=True, remove_hs=False)
        if mol is None:
            raise ValueError(f"RDKit could not read the molecule {description}")
        mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    generate_conformer(mol, seed=seed)
    return mol


def atom_features(mol) -> torch.Tensor:
    """The 16 categorical ligand-atom features, in the checkpoint's order."""
    ring = mol.GetRingInfo()
    af = allowable_features
    rows = []
    for idx, atom in enumerate(mol.GetAtoms()):
        chiral = str(atom.GetChiralTag())
        if chiral in _CHIRALITY_FALLBACK:
            chiral = "CHI_OTHER"
        rows.append([
            safe_index(af["possible_atomic_num_list"], atom.GetAtomicNum()),
            af["possible_chirality_list"].index(chiral),
            safe_index(af["possible_degree_list"], atom.GetTotalDegree()),
            safe_index(af["possible_formal_charge_list"], atom.GetFormalCharge()),
            safe_index(af["possible_implicit_valence_list"], atom.GetImplicitValence()),
            safe_index(af["possible_numH_list"], atom.GetTotalNumHs()),
            safe_index(af["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
            safe_index(af["possible_hybridization_list"], str(atom.GetHybridization())),
            af["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
            safe_index(af["possible_numring_list"], ring.NumAtomRings(idx)),
            af["possible_is_in_ring3_list"].index(ring.IsAtomInRingOfSize(idx, 3)),
            af["possible_is_in_ring4_list"].index(ring.IsAtomInRingOfSize(idx, 4)),
            af["possible_is_in_ring5_list"].index(ring.IsAtomInRingOfSize(idx, 5)),
            af["possible_is_in_ring6_list"].index(ring.IsAtomInRingOfSize(idx, 6)),
            af["possible_is_in_ring7_list"].index(ring.IsAtomInRingOfSize(idx, 7)),
            af["possible_is_in_ring8_list"].index(ring.IsAtomInRingOfSize(idx, 8)),
        ])
    return torch.tensor(rows)


def transformation_mask(edge_index: torch.Tensor, n_atoms: int):
    """Which bonds are rotatable, and which atoms each rotation moves.

    A bond is rotatable when cutting it splits the molecule and the smaller
    side holds more than one atom; the side that moves is the one not holding
    the bond's first endpoint. Returns `(edge_mask, mask_rotate)` with
    `mask_rotate` of shape `(n_rotatable, n_atoms)`, upstream's layout.
    """
    edges = edge_index.T.numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    graph.add_edges_from(edges.tolist())

    to_rotate: list[list[int]] = []
    for i in range(0, edges.shape[0], 2):
        if edges[i, 0] != edges[i + 1, 1]:
            raise ValueError("ligand bond edges are not stored as forward/reverse pairs")
        cut = graph.copy()
        cut.remove_edge(*edges[i])
        if not nx.is_connected(cut):
            smaller = sorted(sorted(nx.connected_components(cut), key=len)[0])
            if len(smaller) > 1:
                if edges[i, 0] in smaller:
                    to_rotate += [[], smaller]
                else:
                    to_rotate += [smaller, []]
                continue
        to_rotate += [[], []]

    edge_mask = np.asarray([bool(l) for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((int(edge_mask.sum()), n_atoms), dtype=bool)
    idx = 0
    for i, moved in enumerate(to_rotate):
        if edge_mask[i]:
            mask_rotate[idx][np.asarray(moved, dtype=int)] = True
            idx += 1
    return edge_mask, mask_rotate


def build_ligand_graph(mol, *, remove_hs: bool = True) -> dict:
    """Atom features, the bond graph, the conformer and the torsion mask.

    `remove_hs` is the checkpoint's `remove_hs: true`: the graph the model sees
    is heavy-atom only. The molecule the caller keeps for writing poses still
    carries its hydrogens, as upstream's does.
    """
    graph_mol = Chem.RemoveHs(mol) if remove_hs else mol

    row, col, edge_type = [], [], []
    for bond in graph_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bt = bond.GetBondType()
        edge_type += 2 * [BOND_TYPES[bt]] if bt != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.nn.functional.one_hot(
        torch.tensor(edge_type, dtype=torch.long), num_classes=len(BOND_TYPES)
    ).to(torch.float)

    graph = {
        "x": atom_features(graph_mol),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "mol": graph_mol,
    }
    if graph_mol.GetNumConformers() > 0:
        graph["pos"] = torch.from_numpy(graph_mol.GetConformer().GetPositions()).float()

    edge_mask, mask_rotate = transformation_mask(edge_index, graph_mol.GetNumAtoms())
    graph["edge_mask"] = torch.tensor(edge_mask)
    graph["mask_rotate"] = mask_rotate
    return graph
