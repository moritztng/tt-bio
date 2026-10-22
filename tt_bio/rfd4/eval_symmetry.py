"""Symmetry resolution before every structure comparison, as upstream does it.

Upstream's `StructureComparisonMetric` permutes the ground truth onto the prediction before it aligns
or scores anything (rfproteina `metrics/gt_structure_comparison.py:507-530`), at two levels:

* **subunits** (`utils/subunit_permutation.py`): identical molecule copies, such as the chains of a
  homo-oligomer or two copies of a ligand, are matched to the prediction by the assignment with the
  least error after a fit on the alignment atoms.
* **atoms** (`utils/atom_permutation.py`): chemically equivalent atoms within a residue or ligand,
  such as the two carboxylate oxygens of ASP, the ring edges of PHE, or LEU's two methyls, are swapped
  where that lowers the error after a fit on the alignment atoms.

Without this a refold that puts GLU's OE1 where the design has OE2 scores as an error of a few
Angstrom on atoms the model has no way to tell apart, and every all-atom RMSD disagrees with
upstream on exactly the refolds a real oracle produces.

The equivalent atoms come from graph automorphisms of each unit (a polymer residue, or a whole
non-polymer chain), matched on element, heavy-atom degree in the whole structure and residue rank,
with bonds compared on resonance-normalised orders, as atomworks' `find_automorphisms_with_networkx`
does: aromatic and RDKit-conjugated bonds compare equal, and so do the bonds from one atom to its
terminal same-element non-carbon neighbours (carboxylate, phosphate). On polymer residues this
gives ARG NH1/NH2, ASP OD1/OD2, GLU OE1/OE2, LEU CD1/CD2, VAL CG1/CG2, the PHE and TYR ring flips,
a C-terminal O/OXT and a nucleotide OP1/OP2, the same table upstream's function returns.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

MAX_AUTOMORPHS = 1000
MAX_SUBUNIT_PERMUTATIONS = 720


def _kabsch_fit(reference, mobile):
    """Rotation and translations that superimpose `mobile` onto `reference` (both (n, 3))."""
    from biotite.structure import superimpose

    _, transform = superimpose(reference, mobile)
    return transform


RESONANT = 1.5
_ORDER = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
_AROMATIC = {5, 6, 8, 9}  # biotite BondType AROMATIC_SINGLE, AROMATIC_DOUBLE, AROMATIC, AROMATIC_TRIPLE


def _heavy_degree(atoms):
    """atomworks `count_heavy_neighbors`: heavy partners over the whole structure."""
    if atoms.bonds is None:
        return np.zeros(atoms.array_length(), dtype=int)
    bonds = atoms.bonds.as_array()[:, :2]
    heavy = ~np.isin(atoms.element, ["H", "D", "T"])
    return np.bincount(bonds[heavy[bonds].all(axis=1)].ravel(), minlength=atoms.array_length())


def _units(atoms, is_polymer):
    """atomworks `build_automorphism_units`: each residue of an all-polymer chain, else the whole chain."""
    from biotite.structure import get_residue_starts

    n = atoms.array_length()
    residue_index = np.searchsorted(get_residue_starts(atoms), np.arange(n), side="right")
    _, chain_index = np.unique(atoms.chain_id, return_inverse=True)
    non_polymer = np.bincount(chain_index, weights=~is_polymer, minlength=chain_index.max() + 1)
    per_residue = (non_polymer == 0)[chain_index]
    unit_of_atom = np.where(per_residue, residue_index, n + 1 + chain_index)
    order = np.argsort(unit_of_atom, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(unit_of_atom[order])) + 1), residue_index


def _bond_orders(unit, degree):
    """atomworks `_resonance_normalized_bond_orders` on one unit, keyed by local index pairs."""
    orders, raw, neighbours = {}, {}, defaultdict(list)
    for i, j, kind in unit.bonds.as_array():
        bond = frozenset((int(i), int(j)))
        orders[bond] = RESONANT if int(kind) in _AROMATIC else _ORDER.get(int(kind), -1.0 - float(kind))
        raw[bond] = orders[bond]
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    try:
        from biotite.interface.rdkit import to_mol
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = to_mol(unit)
        Chem.SanitizeMol(mol)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i >= len(unit) or j >= len(unit):
                continue
            kind = bond.GetBondType()
            if kind == Chem.BondType.AROMATIC or (
                bond.GetIsConjugated() and kind in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
            ):
                orders[frozenset((i, j))] = RESONANT
    except Exception:
        return dict.fromkeys(orders, RESONANT)
    for centre, adjacent in neighbours.items():
        terminal = defaultdict(list)
        for k in adjacent:
            if degree[k] == 1:
                terminal[unit.element[k]].append(k)
        for element, group in terminal.items():
            bonds = [frozenset((centre, k)) for k in group]
            if len(group) < 2 or element == "C" or any(raw[b] == 3.0 for b in bonds):
                continue
            for b in bonds:
                orders[b] = RESONANT
    return orders


_AUTOMORPH_CACHE: dict = {}


def _automorphisms(atoms, indices, degree, rank):
    """Rows of global atom indices, identity first, mapping one unit onto itself."""
    import networkx as nx
    from networkx.algorithms import isomorphism as iso

    unit = atoms[indices]
    local_degree = degree[indices]
    bonds = tuple(sorted((min(i, j), max(i, j), k) for i, j, k in unit.bonds.as_array().tolist()))
    charge = tuple(unit.charge.tolist()) if "charge" in unit.get_annotation_categories() else ()
    key = (
        tuple(unit.res_name.tolist()), tuple(unit.atom_name.tolist()), tuple(unit.element.tolist()),
        bonds, tuple(local_degree.tolist()), tuple(rank.tolist()), charge,
    )
    if key not in _AUTOMORPH_CACHE:
        orders = _bond_orders(unit, local_degree)
        graph = nx.Graph()
        for k in range(len(indices)):
            graph.add_node(k, element=unit.element[k], n_heavy=int(local_degree[k]), rank=int(rank[k]))
        for i, j, _ in unit.bonds.as_array():
            graph.add_edge(int(i), int(j), bond_type=orders[frozenset((int(i), int(j)))])
        matcher = iso.GraphMatcher(
            graph,
            graph,
            node_match=iso.categorical_node_match(["element", "n_heavy", "rank"], ["", 0, 0]),
            edge_match=iso.categorical_edge_match("bond_type", None),
        )
        identity = list(range(len(indices)))
        rows = [identity]
        for n, mapping in enumerate(matcher.isomorphisms_iter()):
            if n >= MAX_AUTOMORPHS:
                break
            row = [mapping[k] for k in identity]
            if row != identity:
                rows.append(row)
        _AUTOMORPH_CACHE[key] = np.array(rows, dtype=int)
    return indices[_AUTOMORPH_CACHE[key]]


def automorphism_groups(atoms, is_polymer):
    """atomworks `find_automorphisms_with_networkx`: one (candidates, unit atoms) array per unit."""
    degree = _heavy_degree(atoms)
    units, residue_index = _units(atoms, is_polymer)
    groups = []
    for indices in units:
        residues = residue_index[indices]
        ranks = np.zeros(len(indices), dtype=int) if residues[0] == residues[-1] else np.unique(residues, return_inverse=True)[1]
        rank = np.where(is_polymer[indices], ranks, -1)
        groups.append(_automorphisms(atoms, indices, degree, rank))
    return groups


def _observed(atoms):
    observed = np.isfinite(atoms.coord).all(axis=-1)
    if "occupancy" in atoms.get_annotation_categories():
        observed &= np.asarray(atoms.occupancy) > 0.0
    return observed


def permute_atoms(pred, gt, gt_is_polymer, atom_classes=None, frame_mask=None):
    """rfproteina `atom_permutation.py:84-237`: gt coordinates with equivalent atoms swapped."""
    out = gt.coord.copy()
    if gt.bonds is None or pred.array_length() != gt.array_length():
        return out
    if atom_classes is not None and np.bincount(atom_classes[atom_classes >= 0] + 1).max(initial=0) < 2:
        return out
    gt_mask = _observed(gt)
    if not gt_mask.any():
        return out

    groups = [g for g in automorphism_groups(gt, gt_is_polymer) if len(g) > 1]
    if not groups:
        return out

    if frame_mask is None:
        frame_mask = gt_mask if atom_classes is None else gt_mask & (atom_classes >= 0)
    frame = frame_mask & gt_mask
    x_pred = pred.coord.astype(float)
    if frame.sum() >= 3:
        transform = _kabsch_fit(gt.coord[frame].astype(float), x_pred[frame])
        x_pred = transform.apply(x_pred)

    x_gt = gt.coord.astype(float)
    permutation = np.arange(gt.array_length())
    for group in groups:
        dest = group[0]
        moving = np.any(group != dest, axis=0)
        active = gt_mask[dest] & moving
        if atom_classes is not None:
            active &= atom_classes[dest] >= 0
        best, best_error = 0, np.inf
        for c, src in enumerate(group):
            if not np.all(gt_mask[src][active]):
                continue
            if atom_classes is not None and not np.all(atom_classes[src][active] == atom_classes[dest][active]):
                continue
            err = np.square(x_pred[dest] - x_gt[src]).sum(axis=-1)
            err = float(np.where(active, err, 0.0).sum() / max(int(active.sum()), 1))
            if not np.isfinite(err):
                err = np.inf
            if err < best_error:
                best, best_error = c, err
        chosen = group[best]
        permutation[dest[active]] = permutation[chosen[active]]
    return out[permutation]


def _molecules(atoms, is_polymer):
    """atomworks `molecule_iid`: polymer chains and non-polymer residues, joined where bonded."""
    import networkx as nx
    from biotite.structure import get_residue_starts

    n = atoms.array_length()
    residue_index = np.searchsorted(get_residue_starts(atoms), np.arange(n), side="right")
    _, chain_index = np.unique(atoms.chain_id, return_inverse=True)
    piece = np.where(is_polymer, chain_index, chain_index.max() + 1 + residue_index)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for value in np.unique(piece):
        members = np.flatnonzero(piece == value)
        graph.add_edges_from(zip(members[:-1].tolist(), members[1:].tolist()))
    if atoms.bonds is not None:
        graph.add_edges_from((int(i), int(j)) for i, j, _ in atoms.bonds.as_array())
    return [np.array(sorted(c)) for c in sorted(nx.connected_components(graph), key=min)], residue_index


def _starts(values):
    return np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))


def permute_subunits(pred, gt_coord, gt, gt_is_polymer, frame_mask, score_mask, align=True):
    """rfproteina `subunit_permutation.py:96-179`: identical molecule copies matched to the prediction."""
    n = gt.array_length()
    if pred.array_length() != n:
        return gt_coord
    frame = np.ones(n, dtype=bool) if frame_mask is None else np.asarray(frame_mask, dtype=bool)
    score = np.ones(n, dtype=bool) if score_mask is None else np.asarray(score_mask, dtype=bool)
    classes = (frame.astype(int) if align else 0) + 2 * score.astype(int)

    molecules, residue_index = _molecules(gt, gt_is_polymer)
    groups = defaultdict(list)
    for molecule in molecules:
        if not classes[molecule].any():
            continue
        signature = (
            _starts(gt.chain_id[molecule]).tobytes(),
            _starts(residue_index[molecule]).tobytes(),
            *(f[molecule].tobytes() for f in (gt.res_name, gt.atom_name, gt.element)),
        )
        groups[(signature, classes[molecule].tobytes())].append(molecule)

    count = 1
    for members in groups.values():
        for factor in range(2, len(members) + 1):
            count *= factor
            if count > MAX_SUBUNIT_PERMUTATIONS:
                raise NotImplementedError(
                    f"Subunit symmetry above {MAX_SUBUNIT_PERMUTATIONS} permutations, as upstream refuses too"
                )
    if count == 1:
        return gt_coord

    identity = np.arange(n)
    per_group = []
    for members in groups.values():
        options = []
        for order in itertools.permutations(range(len(members))):
            mapping = {}
            for dest_k, src_k in enumerate(order):
                mapping.update(zip(members[dest_k].tolist(), members[src_k].tolist()))
            options.append(mapping)
        per_group.append(options)
    candidates = []
    for combo in itertools.product(*per_group):
        cand = identity.copy()
        for mapping in combo:
            for dest, src in mapping.items():
                cand[dest] = src
        candidates.append(cand)

    valid = np.isfinite(gt_coord).all(axis=-1)
    if "occupancy" in gt.get_annotation_categories():
        valid &= np.asarray(gt.occupancy) > 0.0
    xp, xg = pred.coord.astype(float), gt_coord.astype(float)
    observed = valid & np.isfinite(xp).all(axis=-1)
    fit, scored = frame & observed, score & observed
    if not scored.any() or (align and not fit.any()):
        return gt_coord
    best, best_error = candidates[0], np.inf
    for cand in candidates:
        if not np.array_equal(valid[cand], valid):
            continue
        target = xg[cand]
        fitted = xp[scored]
        if align:
            fitted = _kabsch_fit(target[fit], xp[fit]).apply(xp[scored])
        err = float(np.square(fitted - target[scored]).sum() / scored.sum())
        if err < best_error:
            best, best_error = cand, err
    return gt_coord[best]


def resolve(gt, pred, gt_is_polymer, align_mask, subset_mask):
    """Both levels, in upstream's order, on the undivided path: the gt's coordinates resolved."""
    if pred.array_length() != gt.array_length():
        return gt.coord
    coord = permute_subunits(pred, gt.coord, gt, gt_is_polymer, align_mask, subset_mask, align=align_mask is not None)
    resolved = gt.copy()
    resolved.coord = coord
    atom_classes = None
    if subset_mask is not None and (~subset_mask).any():
        atom_classes = np.where(subset_mask, 0, -1)
    return permute_atoms(pred, resolved, gt_is_polymer, atom_classes=atom_classes, frame_mask=align_mask)
