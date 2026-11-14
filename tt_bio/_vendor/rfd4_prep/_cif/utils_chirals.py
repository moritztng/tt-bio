# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Detection and geometry of stereogenic centers and E/Z double bonds."""

import warnings
from itertools import permutations

import numpy as np
from tt_bio._vendor.atomworks.enums import ChiralType
from tt_bio._vendor.atomworks.ml.utils.annotator import ensure_annotations
from biotite.structure import AtomArray, BondType
from scipy.spatial.transform import Rotation

IDEAL_POLYHEDRA: dict[str, np.ndarray] = {
    "tetrahedral": np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32) / np.sqrt(3),
    "square_planar": np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32),
    "trigonal_bipyramidal": np.array(
        [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    ),
    "octahedral": np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32),
}


DOUBLE_BOND_TEMPLATES: dict[tuple[int, int, int], np.ndarray] = {
    (side, n_left, n_right): np.array(
        [[-0.5, 0, 0], [0.5, 0, 0]]
        + [[-1, np.sqrt(3) / 2, 0], [-1, -np.sqrt(3) / 2, 0]][:n_left]
        + [[1, side * np.sqrt(3) / 2, 0], [1, -side * np.sqrt(3) / 2, 0]][:n_right],
        dtype=np.float32,
    )
    for side in (1, -1)
    for n_left in (1, 2)
    for n_right in (1, 2)
}
"""Planar templates keyed by (side, left neighbor count, right neighbor count).

Rows are left endpoint, right endpoint, left neighbors, then right neighbors.
The double bond lies on x, centered at the origin; all bonds have unit length
and bond angles are 120 degrees. The first neighbors are on the same side for
``side=+1`` and opposite sides for ``side=-1``.
"""


def _enumerate_stereoisomer_templates(template: np.ndarray, tol: float = 1e-5) -> list[np.ndarray]:
    """One representative per proper-rotation orbit of all ``n!`` labelings."""
    reps: list[np.ndarray] = []

    # Only the residual is needed; the minimizing rotation need not be unique.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Optimal rotation is not uniquely or poorly defined", category=UserWarning
        )
        for perm in permutations(range(len(template))):
            candidate = template[list(perm)]
            if any(Rotation.align_vectors(candidate, existing)[1] < tol for existing in reps):
                continue
            reps.append(candidate)

    return reps


STEREOISOMER_TEMPLATES: dict[tuple[str, int], list[np.ndarray]] = {
    (geom, n): _enumerate_stereoisomer_templates(full[:n])
    for geom, full in IDEAL_POLYHEDRA.items()
    for n in range(3, len(full) + 1)
}
"""Pre-computed stereoisomer templates for all common geometries"""


def _template_rmsds(centered: np.ndarray, geometry: str) -> np.ndarray | None:
    """Per-template rmsd between unit-normalized ``centered`` and each template."""
    templates = STEREOISOMER_TEMPLATES.get((geometry, len(centered)))
    if templates is None or not np.isfinite(centered).all():
        return None
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Optimal rotation is not uniquely or poorly defined", category=UserWarning
        )
        rssds = np.array([Rotation.align_vectors(unit, t)[1] for t in templates])
    return rssds / np.sqrt(len(centered))


def classify_stereoisomer(centered: np.ndarray, geometry: str) -> int | None:
    """Index of the :py:data:`STEREOISOMER_TEMPLATES` entry best matching ``centered``.

    Returns:
      Index into ``STEREOISOMER_TEMPLATES[(geometry, len(centered))]``, or
      ``None`` if ``centered`` contains non-finite values or no templates exist
      for ``(geometry, len(centered))``.
    """
    rmsds = _template_rmsds(centered, geometry)
    return None if rmsds is None else int(np.argmin(rmsds))


def near_tied_templates(centered: np.ndarray, geometry: str, tol: float = 0.05) -> set[int] | None:
    """Indices of templates within ``tol`` rmsd of the best match.

    Returns:
      Set of template indices, or ``None`` if no templates exist for
      ``(geometry, len(centered))`` or ``centered`` is non-finite.
    """
    rmsds = _template_rmsds(centered, geometry)
    if rmsds is None:
        return None
    return set(np.where(rmsds <= rmsds.min() + tol)[0].tolist())


def identify_stereogenic_centers(atom_array: AtomArray) -> list[dict]:
    """Find stereogenic centers with a supported geometry and neighbor count.

    Excludes unspecified chiral tags and centers without a matching entry in
    :py:data:`STEREOISOMER_TEMPLATES`.

    Args:
        atom_array: Structure with bonds; a missing ``chiral_type`` annotation is added in place.

    Returns:
        list[dict]: One entry per eligible center; empty if none qualify. Each entry contains:
            center_idx (int): AtomArray index of the center atom.
            bonded_idxs (list[int]): Bonded neighbors' AtomArray indices, sorted by index.
            geometry (str): Geometry name used to select stereoisomer templates.
            chiral_type (ChiralType): Assigned stereogenic tag of the center.
    """
    ensure_annotations(atom_array, "chiral_type")

    centers: list[dict] = []
    for idx in np.where(atom_array.chiral_type != 0)[0]:
        ct = ChiralType(int(atom_array.chiral_type[idx]))
        if not ct.is_stereogenic():
            continue
        geometry = ct.to_geometry()
        bonded, _ = atom_array.bonds.get_bonds(int(idx))
        bonded_sorted = sorted(int(b) for b in bonded)
        if (geometry, len(bonded_sorted)) not in STEREOISOMER_TEMPLATES:
            continue
        centers.append(
            {
                "center_idx": int(idx),
                "bonded_idxs": bonded_sorted,
                "geometry": geometry,
                "chiral_type": ct,
            }
        )

    return centers


def identify_stereogenic_double_bonds(atom_array: AtomArray) -> list[dict]:
    """Find annotated stereogenic double bonds with present substituents.

    Eligibility comes from AtomWorks' ``stereogenic_bond_atoms`` annotation.
    If ``atomize`` is present, both endpoints must be atomized.
    All indices refer to the AtomArray and are sorted by index, not CIP priority.

    Args:
        atom_array: Structure with bonds; missing eligibility annotations are added in place.

    Returns:
        list[dict]: One entry per eligible bond; empty if none qualify. Each entry contains:
            bond_idxs (tuple[int, int]): Left and right endpoints.
            neighbor_idxs (tuple[list[int], list[int]]): Left and right
                substituents, excluding the opposite endpoint; one or two per side.
            reference_idxs (tuple[int, int, int, int]): Left endpoint, right
                endpoint, first left substituent, first right substituent,
                ordered for :func:`classify_double_bond_stereoisomer`.
    """
    if atom_array.bonds is None:
        return []
    bonds = atom_array.bonds.as_array()
    double_bonds = bonds[bonds[:, 2] == BondType.DOUBLE, :2]
    if not len(double_bonds):
        return []
    ensure_annotations(atom_array, "stereogenic_bond_atoms")
    eligible = atom_array.stereogenic_bond_atoms.copy()
    if "atomize" in atom_array.get_annotation_categories():
        eligible &= atom_array.atomize

    result: list[dict] = []
    for pair in double_bonds:
        if not eligible[pair].all():
            continue
        left, right = sorted(map(int, pair))
        left_neighbors = sorted(int(idx) for idx in atom_array.bonds.get_bonds(left)[0] if idx != right)
        right_neighbors = sorted(int(idx) for idx in atom_array.bonds.get_bonds(right)[0] if idx != left)
        if not (1 <= len(left_neighbors) <= 2 and 1 <= len(right_neighbors) <= 2):
            continue
        result.append(
            {
                "bond_idxs": (left, right),
                "neighbor_idxs": (left_neighbors, right_neighbors),
                "reference_idxs": (left, right, left_neighbors[0], right_neighbors[0]),
            }
        )
    return result


def classify_double_bond_stereoisomer(coords: np.ndarray) -> int | None:
    """Classify the relative sides of two substituents about a double bond.

    Args:
        coords: Shape (4, 3), ordered as (left endpoint, right endpoint,
            left substituent, right substituent).

    Returns:
        int | None: Relative side label:
            +1: Same-side substituents.
            -1: Opposite-side substituents.
            None: Non-finite, collapsed, or near-orthogonal geometry
    """
    if not np.isfinite(coords).all():
        return None
    axis = coords[1] - coords[0]
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return None
    axis = axis / length
    substituents = coords[2:] - coords[:2]
    projected = substituents - (substituents @ axis)[:, None] * axis
    lengths = np.linalg.norm(projected, axis=1)
    if np.any(lengths < 1e-6):
        return None
    cosine = np.dot(projected[0], projected[1]) / np.prod(lengths)
    if abs(cosine) < 0.5:
        return None
    return 1 if cosine > 0 else -1
