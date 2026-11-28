"""The DiffDock-L receptor graph: nodes, side-chain vectors and edges.

Reproduces `datasets.process_mols.moad_extract_receptor_structure` and
`new_extract_receptor_structure` without torch_cluster or ProDy. Two graphs come
out of the same parse: the score model's residue graph (`all_atoms=False`) and
the confidence model's all-atom graph (`all_atoms=True`), which the confidence
model builds for itself because its checkpoint sets
`use_original_model_cache: false`.
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import (
    aa_short2long,
    allowable_features,
    atom_order,
    chi,
)
from .structure import MAX_RESIDUE_ATOMS, Receptor

#: Upstream refuses a receptor above this many residues
#: (`new_extract_receptor_structure`, `datasets/process_mols.py:169`).
MAX_RECEPTOR_RESIDUES = 3000

_PERIODIC_TABLE = None


def _periodic_table():
    global _PERIODIC_TABLE
    if _PERIODIC_TABLE is None:
        from rdkit.Chem import GetPeriodicTable

        _PERIODIC_TABLE = GetPeriodicTable()
    return _PERIODIC_TABLE


def safe_index(values, element) -> int:
    """Index of `element` in `values`, or the trailing 'misc' slot."""
    try:
        return values.index(element)
    except ValueError:
        return len(values) - 1


# ---------------------------------------------------------------- chi angles

def _dihedral_indices_for(letter: str) -> np.ndarray:
    """The four atom slots of each of the four chi angles, NaN where undefined."""
    out = np.full((4, 4), np.nan)
    names = atom_order.get(letter)
    if names is None or letter not in chi:
        return out
    for k in range(1, 5):
        if k not in chi[letter]:
            continue
        try:
            out[k - 1] = [names.index(x) for x in chi[letter][k]]
        except ValueError:
            pass
    return out


_DIHEDRAL_INDICES = {letter: _dihedral_indices_for(letter) for letter in atom_order}


def dihedral_angles(points: np.ndarray) -> np.ndarray:
    """Dihedral angle in degrees on [0, 360) for `(n, 4, 3)` point quadruples."""
    b0 = points[:, 0] - points[:, 1]
    b1 = points[:, 1] - points[:, 2]
    b2 = points[:, 2] - points[:, 3]
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    m1 = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True))
    deg = np.degrees(np.arctan2(np.sum(m1 * n2, axis=1), np.sum(n1 * n2, axis=1)))
    deg[deg < 0] += 360
    return deg


def chi_angles(coords: np.ndarray, sequence: str) -> np.ndarray:
    """The four chi angles per residue, NaN where the atoms are missing."""
    n = coords.shape[0]
    indices = np.stack([_DIHEDRAL_INDICES.get(a, _DIHEDRAL_INDICES["X"]) for a in sequence])
    mask = np.isnan(indices)
    gather = np.where(mask, 0, indices).astype(int)
    picked = coords[np.arange(n)[:, None, None], gather, :]
    picked[mask] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return dihedral_angles(picked.reshape(-1, 4, 3)).reshape(n, 4)


def side_chain_vectors(coords: np.ndarray, sequence: str) -> torch.Tensor:
    """Upstream's 10-d per-residue geometry: chi/360, then N-CA and C-CA."""
    angles = chi_angles(coords, sequence)
    n_rel = coords[:, 0, :] - coords[:, 1, :]
    c_rel = coords[:, 2, :] - coords[:, 1, :]
    return torch.from_numpy(np.concatenate([angles / 360, n_rel, c_rel], axis=1)).float()


# ------------------------------------------------------------------- k-NN

#: Rows of the distance matrix computed at once. Only a memory knob: the atom
#: graph of a 3000-residue receptor is ~42,000 nodes, so the full matrix is not
#: an option.
_KNN_BLOCK = 256


def knn_edge_index(pos: torch.Tensor, k: int) -> torch.Tensor:
    """`torch_cluster.knn_graph(pos, k)` without torch_cluster.

    Returns `[2, n*k]` with row 0 the neighbour and row 1 the node, the layout
    torch_cluster produces for its default `flow='source_to_target'`, with the
    self edge excluded.

    The distance has to be the direct sum of squared differences, not
    `torch.cdist`. cdist expands to `|a|^2 + |b|^2 - 2ab`, which in float32
    costs enough precision to reorder two neighbours 3.2e-5 A apart: against
    torch_cluster that put 4 of 18,192 receptor edges and up to 159 of 303,872
    atom edges in the wrong place on the PoseBusters set. Ties that survive are
    broken by node index, which a stable argsort gives.
    """
    n = pos.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    neighbours = []
    for start in range(0, n, _KNN_BLOCK):
        block = pos[start:start + _KNN_BLOCK]
        d = ((block[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        rows = torch.arange(block.shape[0])
        d[rows, rows + start] = float("inf")
        neighbours.append(torch.argsort(d, dim=1, stable=True)[:, :k])
    order = torch.cat(neighbours)
    col = torch.arange(n).repeat_interleave(k)
    return torch.stack([order.reshape(-1), col])


# ------------------------------------------------------------ node features

def residue_node_features(sequence: str) -> torch.Tensor:
    """One categorical index per residue, over `possible_amino_acids`."""
    names = allowable_features["possible_amino_acids"]
    rows = [[safe_index(names, aa_short2long.get(a, "misc"))] for a in sequence]
    return torch.tensor(rows, dtype=torch.float32)


def atom_node_features(sequence: str, coords: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """All-atom node features and the residue each atom belongs to.

    Upstream (`get_moad_atom_feats`) looks the residue name up in
    `aa_short2long`, which only holds the 20 standard letters. A residue that
    reaches here as B, Z, X or J therefore raises, and the complex is skipped.
    That is upstream behaviour, and `build_complex_graph` isolates it per
    complex rather than hiding it.
    """
    table = _periodic_table()
    aas = allowable_features["possible_amino_acids"]
    nums = allowable_features["possible_atomic_num_list"]
    t2 = allowable_features["possible_atom_type_2"]
    t3 = allowable_features["possible_atom_type_3"]

    feats, residue_of = [], []
    for i, letter in enumerate(sequence):
        res_long = aa_short2long[letter]
        names = atom_order[letter]
        for j in range(MAX_RESIDUE_ATOMS):
            if np.any(np.isnan(coords[i, j])):
                continue
            if j >= len(names):
                row = [safe_index(aas, res_long), safe_index(nums, "misc"),
                       safe_index(t2, "misc"), safe_index(t3, "misc")]
            else:
                name = names[j]
                try:
                    atomic_num = table.GetAtomicNumber(name[:1])
                except Exception:
                    atomic_num = -1
                row = [safe_index(aas, res_long), safe_index(nums, atomic_num),
                       safe_index(t2, (name + "*")[:2]), safe_index(t3, name)]
            feats.append(row)
            residue_of.append(i)
    return torch.tensor(np.asarray(feats), dtype=torch.float32), np.asarray(residue_of)


# --------------------------------------------------------------- the graphs

def build_receptor_graph(
    receptor: Receptor,
    *,
    c_alpha_max_neighbors: int = 24,
    all_atoms: bool = False,
    atom_max_neighbors: int = 8,
    lm_embeddings: list | None = None,
) -> dict:
    """The receptor half of the complex graph, at the checkpoint's settings.

    `lm_embeddings` is one `(chain_len, 1280)` array per chain, in chain order,
    produced by tt-bio's own ESM-2 (D-dfd-4). Passing `None` gives the residue
    type alone, which is what the graph-structure tests compare.
    """
    n = receptor.n_residues
    if n > MAX_RECEPTOR_RESIDUES:
        raise ValueError(
            f"The receptor is too large for diffdock: {n} residues against a limit of "
            f"{MAX_RECEPTOR_RESIDUES}. No other model docks a ligand; fold the complex with "
            f"boltz2, openfold3 or protenix-v2 instead.")

    sequences, chain_ids = receptor.chain_sequences()
    pos = torch.tensor(receptor.coords[:, 1, :], dtype=torch.float)

    node_feat = residue_node_features(receptor.sequence)
    if lm_embeddings is not None:
        emb = torch.as_tensor(np.concatenate([np.asarray(e) for e in lm_embeddings], axis=0))
        if len(emb) != n:
            raise ValueError(
                f"ESM-2 embedding covers {len(emb)} residues, receptor has {n}")
        node_feat = torch.cat([node_feat, emb.to(node_feat.dtype)], dim=1)

    graph = {
        "sequence": sequences,
        "chain_ids": torch.from_numpy(chain_ids).long(),
        "x": node_feat,
        "pos": pos,
        "side_chain_vecs": side_chain_vectors(receptor.coords, receptor.sequence),
        "edge_index": knn_edge_index(pos, c_alpha_max_neighbors),
    }

    if all_atoms:
        atom_x, residue_of = atom_node_features(receptor.sequence, receptor.coords)
        flat = receptor.coords.reshape(-1, 3)
        atom_pos = torch.from_numpy(flat[~np.any(np.isnan(flat), axis=1)]).float()
        if len(atom_x) != len(atom_pos):
            raise ValueError("all-atom features and coordinates disagree in length")
        graph["atom"] = {
            "x": atom_x,
            "pos": atom_pos,
            "edge_index": knn_edge_index(atom_pos, atom_max_neighbors),
            "atom_rec_edge_index": torch.from_numpy(
                np.stack([np.arange(len(atom_x)), residue_of])).long(),
        }
    return graph
