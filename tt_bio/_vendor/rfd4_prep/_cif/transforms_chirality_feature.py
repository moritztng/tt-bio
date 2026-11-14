# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Idealized local geometry features for stereogenic centers and E/Z double bonds."""

from collections.abc import Generator
from itertools import chain

import numpy as np
from tt_bio._vendor.atomworks.ml.conditions import C_CRD
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.ml.utils.geometry import get_random_rots
from biotite.structure import AtomArray
from scipy.spatial.transform import Rotation

from .conditions import C_CRDGRP, C_INTCRD
from .utils_chirals import (
    DOUBLE_BOND_TEMPLATES,
    STEREOISOMER_TEMPLATES,
    classify_double_bond_stereoisomer,
    classify_stereoisomer,
    identify_stereogenic_centers,
    identify_stereogenic_double_bonds,
)

MAX_OVERLAPPING_CENTERS: int = 6
"""Max disambiguable overlapping chiral centers per atom.

Sized to match the largest supported coordination number (octahedral, 6
neighbors); see :py:data:`rfproteina.utils.chirals.IDEAL_POLYHEDRA`.
"""


def _idealize_coords(centered: np.ndarray, geometry: str) -> np.ndarray:
    """Replace neighbor coords with the best-matching stereoisomeric ideal, rotated."""
    idx = classify_stereoisomer(centered, geometry)
    if idx is None:
        return centered.astype(np.float32)
    template = STEREOISOMER_TEMPLATES[(geometry, len(centered))][idx]
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    rot, _ = Rotation.align_vectors(unit, template)
    return rot.apply(template).astype(np.float32)


def _resolve_stereo_coords(
    indices: np.ndarray,
    crd: np.ndarray,
    intcrd: np.ndarray,
    groups: np.ndarray,
    raw_coords: np.ndarray,
) -> np.ndarray:
    """Choose one full coordinate array for a center or double-bond neighborhood.

    Prefer C_CRD, then C_INTCRD, then raw coordinates. Conditioning must cover
    every selected atom; C_INTCRD atoms must also share a coordinate group so
    their relative positions are meaningful.
    """
    if np.isfinite(crd[indices]).all():
        return crd
    if np.isfinite(intcrd[indices]).all() and np.all(groups[indices] == groups[indices[0]]):
        return intcrd
    return raw_coords


def _iter_idealized_chiral_features(
    atom_array: AtomArray,
    is_inference: bool,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Yield ``(center_idx, bonded_sorted, rotated_ideal_coords)`` per chiral center.

    At inference, no random rotation is applied so the feature is deterministic.
    """
    has_atomize = "atomize" in atom_array.get_annotation_categories()
    crd = C_CRD.annotation(atom_array, default="generate")
    intcrd = C_INTCRD.annotation(atom_array, default="generate")
    groups = C_CRDGRP.annotation(atom_array, default="generate")
    for center in identify_stereogenic_centers(atom_array):
        cidx, bonded = center["center_idx"], center["bonded_idxs"]
        if has_atomize and not atom_array.atomize[cidx]:
            continue
        center_idxs = np.concatenate([[cidx], bonded])
        source_coords = _resolve_stereo_coords(center_idxs, crd, intcrd, groups, atom_array.coord)
        centered = source_coords[bonded] - source_coords[cidx]
        idealized = _idealize_coords(centered, center["geometry"])
        rot = np.eye(3) if is_inference else get_random_rots(1, mode="numpy")[0]
        yield cidx, bonded, (idealized @ rot.T)


def _iter_idealized_double_bond_features(
    atom_array: AtomArray,
    is_inference: bool,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Encode both bond ends and their substituents in one shared idealized frame.

    Training adds a random rotation; inference is deterministic.

    Yields:
        tuple: One frame per double bond with classifiable geometry:
            left_endpoint_idx (int): AtomArray index of the left endpoint.
            atom_indices (np.ndarray): Left endpoint, right endpoint, left
                neighbors, then right neighbors, in coordinate-row order.
            idealized_coords (np.ndarray): Shape (n_atoms_in_frame, 3), centered
                on the bond midpoint and aligned to the input before rotation.
    """
    crd = C_CRD.annotation(atom_array, default="generate")
    intcrd = C_INTCRD.annotation(atom_array, default="generate")
    groups = C_CRDGRP.annotation(atom_array, default="generate")
    for bond in identify_stereogenic_double_bonds(atom_array):
        left, right = bond["bond_idxs"]
        left_neighbors, right_neighbors = bond["neighbor_idxs"]
        indices = np.array([left, right, *left_neighbors, *right_neighbors])
        source_coords = _resolve_stereo_coords(indices, crd, intcrd, groups, atom_array.coord)
        bond_coords = source_coords[indices]
        if not np.isfinite(bond_coords).all():
            continue
        side = classify_double_bond_stereoisomer(source_coords[list(bond["reference_idxs"])])
        if side is None:
            continue

        template = DOUBLE_BOND_TEMPLATES[(side, len(left_neighbors), len(right_neighbors))]
        midpoint = (source_coords[left] + source_coords[right]) / 2
        centered = bond_coords - midpoint
        rotation, _ = Rotation.align_vectors(centered, template)
        idealized = rotation.apply(template).astype(np.float32)
        if not is_inference:
            idealized = idealized @ get_random_rots(1, mode="numpy")[0].T
        yield left, indices, idealized


# ── Public API ──


def build_chirality_block_diagonal(
    atom_array: AtomArray,
    embed_dim: int = 18,
    is_inference: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-atom stereochemistry via non-overlapping 3D blocks with cycling fill.

    Uses idealized center polyhedra and shared double-bond frames. E/Z frames
    occupy the same channels, preserving the feature width for existing models.

    Args:
      atom_array: Input structure.
      embed_dim: Output feature dimension; must be a multiple of 3.
      is_inference: If True, output is deterministic (no random rotation, no shuffle).
      rng: Generator used for block-order and cycle shuffles during training.
        Defaults to a fresh :py:func:`numpy.random.default_rng`. Ignored when
        ``is_inference=True``.

    Returns:
      Float ``(n_atoms, embed_dim)``.
    """
    return _pack_chirality_frames(atom_array, embed_dim, is_inference, rng)["chirality"]


def _pack_chirality_frames(
    atom_array: AtomArray,
    embed_dim: int,
    is_inference: bool,
    rng: np.random.Generator | None,
    token_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pack vectors and token-indexed frames, retaining owners of cycling-fill copies.

    A frame stores two origin atoms followed by up to six neighbors; -1 marks
    unused entries. Slot owners encode anchor token * n_blocks + source block.
    """
    assert embed_dim % 3 == 0, f"embed_dim must be multiple of 3, got {embed_dim}"
    if rng is None:
        # Seed from the global (seed_everything) stream; an unseeded default_rng() ignores the seed.
        rng = np.random.default_rng(np.random.randint(0, 2**32))
    n = atom_array.array_length()
    output = np.zeros((n, embed_dim), dtype=np.float32)
    n_blocks = embed_dim // 3

    feats = {"chirality": output}
    if token_ids is not None:
        token_ids = token_ids.astype(np.int64)
        ensure_annotations(atom_array, "is_ligand")
        atomized = atom_array.atomize & (np.bincount(token_ids)[token_ids] == 1)
        frames = feats["chiral_frame_atoms"] = np.full((n, n_blocks, 8), -1, dtype=np.int64)
        owners = feats["chiral_slot_owner"] = np.full((n, n_blocks), -1, dtype=np.int64)
        vertices = feats["tetrahedral_vertices"] = np.full((n, 4), -1, dtype=np.int64)
        for center in identify_stereogenic_centers(atom_array):
            cidx, bonded = center["center_idx"], center["bonded_idxs"]
            if center["geometry"] != "tetrahedral" or len(bonded) not in (3, 4):
                continue
            support = np.array([cidx, *bonded])
            if (atom_array.is_ligand[support] & atomized[support]).all():
                vertices[cidx] = token_ids[[*bonded, cidx] if len(bonded) == 3 else bonded]
    atom_blocks: dict[int, list[int]] = {}

    for is_center, cidx, bonded, rotated in chain(
        ((True, *frame) for frame in _iter_idealized_chiral_features(atom_array, is_inference)),
        ((False, *frame) for frame in _iter_idealized_double_bond_features(atom_array, is_inference)),
    ):
        block_order = np.arange(n_blocks)
        if not is_inference:
            rng.shuffle(block_order)

        for block in block_order:
            offset = int(block) * 3
            if not np.any(output[bonded, offset : offset + 3]):
                origin = (cidx, cidx) if is_center else (int(bonded[0]), int(bonded[1]))
                if token_ids is not None and atomized[[*origin, *bonded]].all():
                    frames[cidx, block, : 2 + len(bonded)] = token_ids[[*origin, *bonded]]
                    owners[bonded, block] = token_ids[cidx] * n_blocks + block
                output[bonded, offset : offset + 3] = rotated
                for b in bonded:
                    atom_blocks.setdefault(b, []).append(int(block))
                break

    packed = output.reshape(n, n_blocks, 3)
    for atom_idx, blocks in atom_blocks.items():
        occupied = sorted(blocks)
        empty = [b for b in range(n_blocks) if b not in blocks]
        if not empty:
            continue
        cycle = list(occupied)
        if not is_inference:
            rng.shuffle(cycle)
        for i, block in enumerate(empty):
            source_block = cycle[i % len(cycle)]
            packed[atom_idx, block] = packed[atom_idx, source_block]
            if token_ids is not None:
                owners[atom_idx, block] = owners[atom_idx, source_block]

    return feats


def build_is_chiral_center_feature(atom_array: AtomArray) -> np.ndarray:
    """Per-atom boolean: True for chiral center atoms in atomized residues."""
    ensure_annotations(atom_array, "chiral_type")
    mask = atom_array.chiral_type != 0
    if "atomize" in atom_array.get_annotation_categories():
        mask = mask & atom_array.atomize
    return mask


# ── Transforms ──


class AddChiralityFeature(Transform):
    """Add per-atom idealized center and double-bond stereochemistry features."""

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        is_inference = data.get("is_inference", False)
        feats = data.get("feats", {})

        feats.update(
            _pack_chirality_frames(
                atom_array,
                MAX_OVERLAPPING_CENTERS * 3,
                is_inference,
                rng=None,
                token_ids=getattr(atom_array, "token_id", None),
            )
        )
        data["feats"] = feats
        return data
