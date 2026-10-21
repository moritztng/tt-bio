"""Upstream RFD4 `rfd4-evaluate` inline metrics, reimplemented on numpy + biotite.

Upstream computes these in `rfproteina/metrics/design_metrics.py` against atomworks 3.0.4, which
needs python 3.12 and is a release-gated bump on tt-bio's side (it breaks RF3's featurizer). The
metrics themselves do not need it: what atomworks contributes is three small rules, vendored here
with the upstream file and line they come from, so `tt-bio evaluate` runs on tt-bio's own stack.

Exactness is not asserted by this module. It is proven by running upstream's `rfd4-evaluate` and
this port on the same designs and diffing the two CSVs with `perf/rfd4/eval_compare.py`.

Column names are upstream's, character for character, including the `1p5A` threshold spelling and
the `bond_length.` prefix inside the `bond_length_metrics` namespace.
"""

import numpy as np

# Vendored from atomworks `io/utils/bonds.py:660-694` (filter_bonds_by_distance defaults): a bond is
# unreasonable when it is longer than the threshold for the elements at its two ends.
BOND_THRESHOLD_CHNO = 1.8
BOND_THRESHOLD_CHNOPS = 2.4
BOND_THRESHOLD_OTHER = 3.6

_CHNO = frozenset({"C", "H", "N", "O"})
_CHNOPS = frozenset({"C", "H", "N", "O", "P", "S"})

CLASH_THRESHOLD = 1.5

CLASH_CATEGORIES = (
    "intra_residue_polymer",
    "intra_residue_ligand",
    "inter_residue_polymer_polymer",
    "inter_residue_polymer_ligand",
    "inter_residue_ligand_ligand",
)


def threshold_key(threshold: float) -> str:
    """`1.5` -> `'1p5A'`. Upstream `design_metrics.py:176`: the key avoids `.` (its namespace
    separator) and `,` (the CSV delimiter)."""
    return f"{threshold:.1f}".replace(".", "p") + "A"


def _chain_identifier(atoms):
    categories = atoms.get_annotation_categories()
    return atoms.chain_iid if "chain_iid" in categories else atoms.chain_id


def _bond_threshold(elem_a, elem_b):
    """Per-bond threshold, vectorised over the two element arrays."""
    a = np.char.upper(elem_a.astype(str))
    b = np.char.upper(elem_b.astype(str))
    in_chno = np.isin(a, list(_CHNO)) & np.isin(b, list(_CHNO))
    in_chnops = np.isin(a, list(_CHNOPS)) & np.isin(b, list(_CHNOPS))
    out = np.full(a.shape, BOND_THRESHOLD_OTHER)
    out[in_chnops] = BOND_THRESHOLD_CHNOPS
    out[in_chno] = BOND_THRESHOLD_CHNO
    return out


def _bond_spatial_masks(atoms, a1, a2, is_polymer):
    """Upstream `design_metrics.py:14-47`. Mutually exclusive except that
    `between_chain.nonpolymer_to_polymer` is an XOR sub-category of between-chain."""
    chain = _chain_identifier(atoms)
    same_chain = chain[a1] == chain[a2]
    same_res = same_chain & (atoms.res_id[a1] == atoms.res_id[a2])
    cross_chain = ~same_chain
    poly_a1, poly_a2 = is_polymer[a1], is_polymer[a2]
    return {
        "within_residue": same_res,
        "between_residue.within_chain": ~same_res & same_chain,
        "between_chain.polymer_polymer": cross_chain & poly_a1 & poly_a2,
        "between_chain.nonpolymer_to_polymer": cross_chain & (poly_a1 ^ poly_a2),
    }


def compute_bond_length_stats(atoms, is_polymer, verbose: bool = True) -> dict:
    """`bond_length_metrics`, upstream `design_metrics.py:50-105`.

    A bond whose either endpoint has a non-finite coordinate is preserved, never flagged
    (atomworks `bonds.py:702`). With no bonds at all the metric contributes no columns, which is
    how upstream leaves the cell empty rather than writing a zero.
    """
    if atoms.bonds is None:
        return {}
    bonds = atoms.bonds.as_array()
    n_orig = len(bonds)
    if n_orig == 0:
        return {}

    a1, a2 = bonds[:, 0], bonds[:, 1]
    coords = atoms.coord
    dist = np.linalg.norm(coords[a1] - coords[a2], axis=1)
    finite = np.isfinite(coords[a1]).all(axis=1) & np.isfinite(coords[a2]).all(axis=1)
    is_unreasonable = finite & (dist > _bond_threshold(atoms.element[a1], atoms.element[a2]))

    out = {"bond_length.frac_unreasonable": float(is_unreasonable.sum() / n_orig)}
    if not verbose:
        return out

    for name, mask in _bond_spatial_masks(atoms, a1, a2, is_polymer).items():
        n_cat = int(mask.sum())
        out[f"bond_length.frac_unreasonable.{name}"] = (
            float(int((is_unreasonable & mask).sum()) / n_cat) if n_cat > 0 else float("nan")
        )
    return out


def _clash_category_masks(atoms, i_idx, j_idx, is_polymer):
    """Upstream `design_metrics.py:184-218`: same-vs-different residue x the two molecule types."""
    chain = _chain_identifier(atoms)
    poly_i, poly_j = is_polymer[i_idx], is_polymer[j_idx]
    both_polymer = poly_i & poly_j
    both_ligand = ~poly_i & ~poly_j
    polymer_ligand = poly_i ^ poly_j
    same_res = (chain[i_idx] == chain[j_idx]) & (atoms.res_id[i_idx] == atoms.res_id[j_idx])
    return {
        "intra_residue_polymer": same_res & both_polymer,
        "intra_residue_ligand": same_res & both_ligand,
        "inter_residue_polymer_polymer": ~same_res & both_polymer,
        "inter_residue_polymer_ligand": ~same_res & polymer_ligand,
        "inter_residue_ligand_ligand": ~same_res & both_ligand,
    }


def compute_clash_stats(atoms, is_polymer, is_crd=None, threshold: float = CLASH_THRESHOLD) -> dict:
    """`clash_metrics`, upstream `design_metrics.py:107-174`.

    A clash is a non-bonded pair closer than `threshold`; a pair whose two atoms are both
    coordinate-conditioned (fixed input, atomworks `C_CRD`) is excluded, because the model did not
    place either of them. The reported number is the fraction of *atoms* that have at least one
    clashing partner, not the fraction of pairs.

    Upstream reaches this through biotite's `CellList`, which raises on non-finite coordinates;
    that behaviour is kept here deliberately, since a grading arm must see a design with unresolved
    atoms fail rather than score it on a subset.
    """
    from biotite.structure import CellList

    n_atoms = atoms.array_length()
    coords = atoms.coord
    key = threshold_key(threshold)

    bonded_keys = np.empty(0, dtype=np.intp)
    if atoms.bonds is not None:
        bonds = atoms.bonds.as_array()
        if len(bonds) > 0:
            b1 = np.minimum(bonds[:, 0], bonds[:, 1]).astype(np.intp)
            b2 = np.maximum(bonds[:, 0], bonds[:, 1]).astype(np.intp)
            bonded_keys = b1 * n_atoms + b2

    neighbors = CellList(atoms, cell_size=threshold).get_atoms(coords, radius=threshold)
    n, max_k = neighbors.shape
    i_rep = np.repeat(np.arange(n), max_k)
    j_flat = neighbors.ravel()
    valid = (j_flat >= 0) & (i_rep < j_flat)
    i_idx, j_idx = i_rep[valid], j_flat[valid]

    if is_crd is None:
        is_crd = np.zeros(n_atoms, dtype=bool)
    keep = ~np.isin(i_idx * n_atoms + j_idx, bonded_keys) & ~(is_crd[i_idx] & is_crd[j_idx])
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    pair_dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=-1) if len(i_idx) else np.empty(0)

    def frac_atoms_clashing(pair_mask):
        min_dists = np.full(n_atoms, np.inf)
        if pair_mask.any():
            np.minimum.at(min_dists, i_idx[pair_mask], pair_dists[pair_mask])
            np.minimum.at(min_dists, j_idx[pair_mask], pair_dists[pair_mask])
        return float((min_dists < threshold).sum()) / n_atoms

    out = {f"frac_atoms_clashing_{key}": frac_atoms_clashing(np.ones(len(i_idx), dtype=bool))}
    masks = (
        _clash_category_masks(atoms, i_idx, j_idx, is_polymer)
        if len(i_idx)
        else {c: np.zeros(0, dtype=bool) for c in CLASH_CATEGORIES}
    )
    for category in CLASH_CATEGORIES:
        out[f"frac_atoms_clashing_{category}_{key}"] = frac_atoms_clashing(masks[category])
    return out
