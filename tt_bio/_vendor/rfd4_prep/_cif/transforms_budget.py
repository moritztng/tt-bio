# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
"""Post-crop token-budget enforcement.

- ``EnforceCropBudget``: drops spatially-peripheral non-reserved tokens so
  ``T + extra_tokens <= crop_size`` after a downstream expander runs.
- ``plan_expansion`` callable supplies ``(extra_tokens, must_keep_atom_mask)``.
- ``plan_unindex_expansion``: default plan for the ``UnindexFlaggedTokens`` flow.
"""

from collections.abc import Callable

import biotite.structure as struc
import numpy as np
from tt_bio._vendor.atomworks.ml.conditions import C_IDX
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.transforms.crop import compute_local_hash
from tt_bio._vendor.atomworks.ml.utils.token import (
    apply_token_wise,
    get_af3_token_center_coords,
    get_token_count,
    spread_token_wise,
)
from biotite.structure import AtomArray

# (extra_tokens, must_keep_atom_mask[n_atoms])
ExpansionPlan = tuple[int, np.ndarray]
PlanFn = Callable[[AtomArray], ExpansionPlan]


def plan_unindex_expansion(atom_array: AtomArray) -> ExpansionPlan:
    """Reserve space for :py:class:`UnindexFlaggedTokens`.

    - ``extra_tokens`` = residues with any ``C_IDX == False`` atom (each becomes one appended token).
    - ``must_keep`` = the unindexed atoms themselves.
    """
    is_unindexed = ~C_IDX.mask(atom_array, default="generate")
    extra_tokens = int(struc.apply_residue_wise(atom_array, is_unindexed, np.any).sum())
    return extra_tokens, is_unindexed


class EnforceCropBudget(Transform):
    """Drop spatially-peripheral non-reserved tokens so ``T + extra_tokens <= crop_size``.

    - ``plan_expansion(atom_array) -> (extra_tokens, must_keep_atom_mask)``.
    - Ranks non-reserved tokens by distance to the upstream crop center (``data["crop_info"]``)
      if present, else whole-array COM. Must-keep tokens survive unless the reservation itself
      exceeds ``crop_size // 2`` (each reserved token is duplicated by the expander, so it costs
      two slots), in which case the outermost reserved tokens are dropped to make it fit.
    - Trims ``data["crop_info"]["crop_atom_idxs"] / ["crop_token_idxs"]`` so downstream
      consumers see a consistent view.

    The token budget is resolved as ``data["crop_info"]["crop_size"]`` if present — the
    concrete size actually sampled by an upstream dynamic crop (e.g.
    :py:class:`~atomworks.ml.transforms.crop.CropSpatialLikeAF3` with ``crop_size_range``
    set) — else the fixed ``crop_size`` passed at construction.
    """

    def __init__(self, crop_size: int | None, plan_expansion: PlanFn):
        self.crop_size = crop_size
        self.plan_expansion = plan_expansion

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]
        extra_tokens, must_keep_atom_mask = self.plan_expansion(atom_array)

        crop_size = data.get("crop_info", {}).get("crop_size", self.crop_size)
        assert crop_size is not None, (
            "No crop_size available to enforce: pass crop_size explicitly to EnforceCropBudget, "
            "or ensure an upstream crop populates data['crop_info']['crop_size']."
        )

        T = get_token_count(atom_array)
        if T + extra_tokens <= crop_size:
            return data

        center = _resolve_ranking_center(data, atom_array)

        atom_array, atom_keep_mask, keep_token_mask = _drop_periphery_tokens_to_fit(
            atom_array,
            crop_size=crop_size,
            must_keep_atom_mask=must_keep_atom_mask,
            center=center,
        )
        data["atom_array"] = atom_array

        # Keep crop_info's index arrays aligned with the (now smaller) atom_array.
        crop_info = data.get("crop_info")
        if crop_info is not None:
            if "crop_atom_idxs" in crop_info:
                crop_info["crop_atom_idxs"] = crop_info["crop_atom_idxs"][atom_keep_mask]
            if "crop_token_idxs" in crop_info:
                crop_info["crop_token_idxs"] = crop_info["crop_token_idxs"][keep_token_mask]

        return data


def _resolve_ranking_center(data: dict, atom_array: AtomArray) -> np.ndarray:
    """Pick the center used to rank non-reserved tokens by distance.

    - Upstream crop's ``crop_center_atom_idx`` (resolved against the uncropped
      ``crop_info["atom_array"]``) if available.
    - Else: whole-array COM of valid token centers in the current ``atom_array``.
    """
    crop_info = data.get("crop_info")
    if crop_info is not None and "crop_center_atom_idx" in crop_info and "atom_array" in crop_info:
        idx = crop_info["crop_center_atom_idx"]
        if not np.isnan(idx):
            return crop_info["atom_array"].coord[int(idx)]

    coords = get_af3_token_center_coords(atom_array)
    valid = np.isfinite(coords).all(axis=1)
    return coords[valid].mean(axis=0)


def _drop_periphery_tokens_to_fit(
    atom_array: AtomArray,
    *,
    crop_size: int,
    must_keep_atom_mask: np.ndarray,
    center: np.ndarray,
) -> tuple[AtomArray, np.ndarray, np.ndarray]:
    """Drop tokens farthest from ``center`` so ``T + extra_tokens == crop_size`` after expansion.

    - Returns (new_atom_array, atom_keep_mask, token_keep_mask) for caller bookkeeping.
    - Preserves/extends ``at_crop_boundary`` if present.
    """
    reserved_token_mask = apply_token_wise(atom_array, must_keep_atom_mask, np.any)

    coords = get_af3_token_center_coords(atom_array)
    valid = np.isfinite(coords).all(axis=1)
    dists = np.linalg.norm(coords - center, axis=1)

    reserved_idxs = np.where(reserved_token_mask)[0]
    n_reserved_tokens = min(len(reserved_idxs), crop_size // 2)
    keep_token_mask = np.zeros(len(coords), dtype=bool)
    keep_token_mask[reserved_idxs[np.argsort(dists[reserved_idxs])[:n_reserved_tokens]]] = True

    # ... then spend what the reservation left over on the innermost non-reserved tokens
    target_idxs = np.where(~reserved_token_mask & valid)[0]
    n_keep = crop_size - 2 * n_reserved_tokens
    keep_token_mask[target_idxs[np.argsort(dists[target_idxs])[:n_keep]]] = True

    atom_keep_mask = spread_token_wise(atom_array, keep_token_mask)

    # Extend at_crop_boundary (if present) so atoms whose local neighborhood lost
    # members from this recrop are flagged. OR into existing — never clear.
    extend_boundary = "at_crop_boundary" in atom_array.get_annotation_categories()
    if extend_boundary:
        precrop_hash = compute_local_hash(atom_array)

    atom_array = atom_array[atom_keep_mask]

    if extend_boundary:
        postcrop_hash = compute_local_hash(atom_array)
        newly_at_boundary = precrop_hash[atom_keep_mask] != postcrop_hash
        atom_array.set_annotation("at_crop_boundary", atom_array.at_crop_boundary | newly_at_boundary)

    return atom_array, atom_keep_mask, keep_token_mask
