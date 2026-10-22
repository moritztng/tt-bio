# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
import logging
import numpy as np
import torch
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.ml.conditions import C_CRD
from tt_bio._vendor.atomworks.ml.conditions.base import ConditionBase
from tt_bio._vendor.atomworks.ml.utils.geometry import random_rigid_augmentation
from biotite.structure import superimpose
from biotite.structure.atoms import AtomArray
from biotite.structure.residues import get_residue_starts, spread_residue_wise

from .conditions import C_CRDGRP, C_IDX, C_IDXGRP, C_INTCRD
from .utils_common import true_with_prob
from .utils_standard_annotations import SA_CWI

global_logger = logging.getLogger(__name__)


def sample_island_tokens(
    array_length: int,
    island_len_min: int = 5,
    island_len_max: int = 30,
    n_islands_min: int = 1,
    n_islands_max: int = 30,
    max_length: int | None = None,
) -> np.ndarray:
    """
    Generate a boolean mask of length `array_length` with random contiguous islands (True segments)
    while optionally constraining the total number of True values.

    Args:
        array_length (int): Total length of the boolean array.
        island_len_min (int): Minimum island length (inclusive).
        island_len_max (int): Maximum island length (inclusive).
        n_islands (int): Number of islands to attempt to generate.
        max_length (int, optional): Maximum allowed total number of True values in the output.
                                    If None, no constraint is applied.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Boolean array of length `array_length` with island positions set to True.
    """
    n_islands = np.random.randint(n_islands_min, n_islands_max + 1)

    mask = np.zeros(array_length, dtype=bool)
    for _ in range(n_islands):
        current_total = mask.sum()
        if max_length is not None:
            if current_total >= max_length:
                break
            remaining = max_length - current_total
        else:
            remaining = None  # not used

        # Randomly select a candidate island length.
        candidate_length = np.random.randint(island_len_min, island_len_max + 1)
        candidate_length = min(candidate_length, array_length)  # Fit into array

        # Choose a random starting index ensuring the island fits.
        high_start = array_length - candidate_length
        start = np.random.randint(0, high_start + 1)

        # Evaluate the segment that would be activated.
        segment = mask[start : start + candidate_length]
        new_trues = np.sum(~segment)

        # If we have a maximum True budget and adding all new positions would exceed it, adjust the island.
        if max_length is not None and new_trues > remaining:
            # We try to trim the island so that it adds at most `remaining` new True values.
            count_new = 0
            adjusted_length = 0
            for i in range(candidate_length):
                if not mask[start + i]:
                    count_new += 1
                adjusted_length += 1
                # Once we've added as many new trues as allowed, break.
                if count_new >= remaining:
                    break
            # Only add the island if its adjusted length meets the minimum requirement.
            if adjusted_length < island_len_min:
                continue  # Skip this island and try the next one.
            mask[start : start + adjusted_length] = True
        else:
            # No max constraint or this candidate island fits within the remaining budget.
            mask[start : start + candidate_length] = True

    assert mask.sum() <= array_length, "Generated mask exceeds array length."
    return mask


#################################################################################
# Utility functions
#################################################################################


def sample_unindexed_groups(
    atom_array: AtomArray,
    p_merge_groups: float = 0.0,
    p_split_group: float = 0.0,
    p_increase_group_id_gap: float = 0.1,
) -> None:
    """Sample the unindexed groups in an AtomArray."""

    ensure_annotations(atom_array, "within_chain_source_res_idx")

    # NOTE: At this stage, we have not yet called UnindexFlaggedTokens, so the source residue for a tip-atom unindexing
    # case would contains both indexed and unindexed atoms. We therefore first subset to the unindexed atoms before
    # calling get_residue_starts
    unindexed_atom_array = atom_array[~C_IDX.mask(atom_array, default="raise")]
    unindexed_residue_starts = get_residue_starts(unindexed_atom_array)
    if unindexed_atom_array.array_length() == 0:
        return C_IDXGRP.default_annotation(atom_array)

    # Get ground-truth of which residues are adjacent and on the same chain
    unindexed_contiguity_breaks = (
        np.diff(unindexed_atom_array.within_chain_source_res_idx[unindexed_residue_starts]) != 1
    )

    # Get chain annotation
    chain_annot_name = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    chain_annot_residue_level = unindexed_atom_array.get_annotation(chain_annot_name)[unindexed_residue_starts]

    unindexed_chain_breaks = chain_annot_residue_level[1:] != chain_annot_residue_level[:-1]

    # Convert these to group ids, starting at 1 for the unindexed groups.
    unindexed_group_ids = np.cumsum(np.concatenate([[True], unindexed_contiguity_breaks | unindexed_chain_breaks]))

    # Merging groups allows for the use-case of two unindexed components with a fixed residue gap between them
    if true_with_prob(p_merge_groups) and unindexed_group_ids.max() > 1:
        # Sample the gap between the group ids to merge. This corresponds the number of unindexed
        # groups that belong between the final positions of the two (non-contiguous) merged groups
        # It is currently parametrized with a geometric distribution, as this "interleaving" becomes
        # less likely in practice the more groups there are in between
        max_group_id_gap = unindexed_group_ids.max() - unindexed_group_ids.min()
        geometric_parameter = 1 - p_increase_group_id_gap
        sampled_group_id_gap = min(np.random.geometric(geometric_parameter), max_group_id_gap)

        # Identify valid pairings for this gap size
        valid_smaller_group_ids = unindexed_group_ids[
            np.isin(unindexed_group_ids + sampled_group_id_gap, unindexed_group_ids)
        ]

        # Ensure that the two groups to be merged are on the same chain
        chain_by_group_id = {}
        for group_id in np.unique(unindexed_group_ids):
            chain_by_group_id[group_id] = chain_annot_residue_level[unindexed_group_ids == group_id][0]

        valid_smaller_group_ids = [
            gid
            for gid in valid_smaller_group_ids
            if chain_by_group_id[gid] == chain_by_group_id[gid + sampled_group_id_gap]
        ]

        if len(valid_smaller_group_ids) > 0:
            # Given the gap, uniformly sample the pair of groups to merge
            smaller_group_id = np.random.choice(valid_smaller_group_ids)
            larger_group_id = smaller_group_id + sampled_group_id_gap

            unindexed_group_ids[unindexed_group_ids == larger_group_id] = smaller_group_id
            if larger_group_id < unindexed_group_ids.max():
                unindexed_group_ids[unindexed_group_ids > larger_group_id] -= 1

    # Construct the corresponding atom-level annotation
    unindexed_group_ids_atom_level = spread_residue_wise(unindexed_atom_array, unindexed_group_ids)

    # Splitting a group avoids incorrectly teaching the model that different unindexed groups are never adjacent
    if true_with_prob(p_split_group):
        # To be split, an unindexed group must contain at least two residue starts
        all_unindexed_group_ids = np.unique(unindexed_group_ids)
        unindexed_residue_starts_mask = np.zeros(unindexed_atom_array.array_length(), dtype=bool)
        unindexed_residue_starts_mask[unindexed_residue_starts] = True
        candidate_group_ids_to_split = [
            gid
            for gid in all_unindexed_group_ids
            if ((unindexed_group_ids_atom_level == gid) & unindexed_residue_starts_mask).sum() > 1
        ]

        # We only split between residues
        if len(candidate_group_ids_to_split) > 0:
            group_to_split = np.random.choice(candidate_group_ids_to_split)
            group_to_split_mask = unindexed_group_ids_atom_level == group_to_split
            candidate_split_points = np.where(group_to_split_mask & unindexed_residue_starts_mask)[0]
            split_point = np.random.choice(
                candidate_split_points[1:]
            )  # Splitting at first residue start is meaningless
            group_to_split_indices = np.where(group_to_split_mask)[0]
            indices_to_reassign = group_to_split_indices[group_to_split_indices >= split_point]
            unindexed_group_ids_atom_level[indices_to_reassign] = unindexed_group_ids_atom_level.max() + 1

    # Get the default annotation for this AtomArray, which is correct for atoms with absolute indices
    all_index_group_ids = C_IDXGRP.default_annotation(atom_array)

    # Add the maximum indexed group id to each of the unindexed group ids (which start at 1)
    # Note that this may create gaps in the group ids, which is fine since the absolute values should never matter
    absolute_indexed_atom_mask = C_IDX.mask(atom_array, default="raise")
    if absolute_indexed_atom_mask.any():
        max_absolute_indexed_group_id = all_index_group_ids[absolute_indexed_atom_mask].max()
    else:
        max_absolute_indexed_group_id = -1
    all_index_group_ids[~absolute_indexed_atom_mask] = unindexed_group_ids_atom_level + max_absolute_indexed_group_id

    return all_index_group_ids


def set_specified_conditions_to_default(
    atom_array: AtomArray,
    conditions: list[type[ConditionBase]],
) -> AtomArray:
    """Set the specified Conditions' sole annotations to their default values.

    This is done by first removing all the Conditions and then iteratively re-adding the defaults, to avoid any affect
    on the computation of defaults due to the presence of non-default Conditions to be removed.

    Args:
        atom_array (AtomArray): The AtomArray to modify.
        conditions: The Conditions to reset.
    """
    for condition in conditions:
        if not issubclass(condition, ConditionBase):
            raise ValueError(f"Condition {condition} must be a subclass of ConditionBase.")

    conditions_to_reset = [condition for condition in conditions if condition.has_annotation(atom_array)]

    # Remove all conditioning, so that we can compute all of the default annotations independently
    categories = set(atom_array.get_annotation_categories())
    for condition in conditions_to_reset:
        for name in (condition.full_name, *condition.aliases):
            if name in categories:
                atom_array.del_annotation(name)

    # Now that we have an AtomArray with no Condition annotations, compute the default values
    # NOTE: We assume that order of applying defaults doesn't matter -- otherwise, the Condition default is not robust.
    for condition in conditions_to_reset:
        condition.set_annotation(atom_array, condition.default_annotation(atom_array))

    return atom_array


def convert_to_internal_coords(atom_array: AtomArray, mask: np.ndarray) -> AtomArray:
    """Assign the masked region to a new internal-coordinate group and clear its absolute targets.

    Atoms without finite live coordinates are excluded. Targets are materialized
    from live coordinates by ``MaterializeInternalCoords`` in the featurization pipeline.
    Regrouping must not invalidate existing saved internal targets.
    """
    mask = mask & np.isfinite(atom_array.coord).all(axis=1)
    if not mask.any():
        return atom_array

    # Absolute coord conditioning mask is False for the new internal coord region
    coordinate = C_CRD.annotation(atom_array, default="generate").copy()
    coordinate[mask] = C_CRD.default_annotation(atom_array)[mask]
    C_CRD.set_annotation(atom_array, coordinate)

    # Create a new coord group for the new internal coord region
    coord_groups = C_CRDGRP.annotation(atom_array, default="generate").copy()
    new_group_id = int(coord_groups.max()) + 1 if coord_groups.max() > 0 else 1
    coord_groups[mask] = new_group_id
    C_CRDGRP.set_annotation(atom_array, coord_groups)

    return atom_array


def separate_coordinate_groups(arrays: list[AtomArray]) -> list[AtomArray]:
    """Copy independent inputs into disjoint internal-coordinate group namespaces."""
    result = []
    next_id = 1
    for original in arrays:
        arr = original.copy()
        groups = C_CRDGRP.annotation(arr, default="generate").copy()
        old = groups.copy()
        internal = C_CRDGRP.is_internal(arr)
        if C_INTCRD.has_annotation(arr) and not np.isnan(C_INTCRD.annotation(arr)[~internal]).all():
            raise ValueError("Internal coordinate targets exist outside internal coordinate groups.")
        for gid in np.unique(old[internal]):
            groups[old == gid] = next_id
            next_id += 1
        C_CRDGRP.set_annotation(arr, groups)
        result.append(arr)
    return result


def set_absolute_coordinate_conditioning(atom_array: AtomArray, indices: np.ndarray, active: bool) -> None:
    """Edit input absolute coordinates, clearing affected saved internal targets for rematerialization."""
    groups = C_CRDGRP.annotation(atom_array, default="generate").copy()
    coords = C_CRD.annotation(atom_array, default="generate").copy()
    coords[indices] = atom_array.coord[indices] if active else C_CRD.default_annotation(atom_array)[indices]
    C_CRD.set_annotation(atom_array, coords)
    affected = indices if active else indices[~C_CRDGRP.is_internal(atom_array)[indices]]
    groups[affected] = C_CRDGRP.default_annotation(atom_array)[affected]
    C_CRDGRP.set_annotation(atom_array, groups, on_internal_coord_mismatch="clear")


def materialize_internal_coords(atom_array: AtomArray) -> AtomArray:
    """Fill entirely unset groups from live coordinates; preserve complete saved targets.

    Absolute targets must agree with reserved group labels. Partial, nonfinite,
    and orphan targets are errors; complete saved internal targets are never rebuilt.
    """
    groups = C_CRDGRP.annotation(atom_array, default="generate")
    if np.any(groups < C_CRDGRP.UNKNOWN_COORDS):
        raise ValueError("Coordinate groups must be -1 (unknown), 0 (absolute), or positive (internal).")
    if np.any(groups > C_CRDGRP.MAX_GROUP_ID):
        raise ValueError(f"Coordinate group IDs must not exceed {C_CRDGRP.MAX_GROUP_ID} (the int16 model limit).")
    absolute = C_CRD.annotation(atom_array, default="generate")
    absolute_present = np.isfinite(absolute).all(axis=1)
    if not np.all(absolute_present | np.isnan(absolute).all(axis=1)):
        raise ValueError("Absolute coordinate targets must be entirely finite or entirely unset (NaN) per atom.")
    if not np.array_equal(absolute_present, groups == C_CRDGRP.KNOWN_ABSOLUTE_COORDS):
        raise ValueError("Absolute coordinate targets must exist exactly on coordinate group 0.")
    payload = C_INTCRD.annotation(atom_array, default="generate").copy()
    internal = C_CRDGRP.is_internal(atom_array)
    if not np.isnan(payload[~internal]).all():
        raise ValueError("Internal coordinate targets exist outside internal coordinate groups.")
    for gid in np.unique(groups[internal]):
        members = groups == gid
        saved = payload[members]
        if np.isfinite(saved).all():
            continue
        if not np.isnan(saved).all():
            raise ValueError(f"Coordinate group {gid} has incomplete or nonfinite saved internal coordinate targets.")
        coords = atom_array.coord[members]
        if not np.isfinite(coords).all():
            raise ValueError(f"Coordinate group {gid} needs finite live coordinates to materialize its targets.")
        centered = coords - coords.mean(axis=0, keepdims=True)
        payload[members] = random_rigid_augmentation(torch.from_numpy(centered)[None], batch_size=1, s=0)[0].numpy()
    C_INTCRD.set_annotation(atom_array, payload)
    return atom_array


def insert_internal_coord_conditioned_groups(atom_array: AtomArray) -> AtomArray:
    """Update coord_with_insertion AtomArray annotation with internal coordinate conditioning.

    Aligns each internally-conditioned group to the model-generated coordinates before insertion.
    """
    intcrd_mask = C_CRDGRP.is_internal(atom_array)
    if not intcrd_mask.any():
        return atom_array

    intcrd_conditioning = C_INTCRD.annotation(atom_array, default="generate")

    if not SA_CWI.has_annotation(atom_array):
        SA_CWI.set_annotation(atom_array, SA_CWI.default_annotation(atom_array))

    # Find unique internal coordinate groups (exclude absolute coords and unknown coords)
    coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
    unique_groups = np.unique(coord_groups[intcrd_mask])

    # Align and insert each internal coordinate group separately
    for group_id in unique_groups:
        # Get mask for atoms in this coordinate group with internal coordinate conditioning
        group_intcrd_mask = intcrd_mask & (coord_groups == group_id)

        # Extract coordinates for alignment
        group_intcrd_conditioning = intcrd_conditioning[group_intcrd_mask]  # [n_atoms, 3]
        generated_coords = atom_array.coord[group_intcrd_mask]  # [n_atoms, 3]

        # Check for valid coordinates (no NaNs)
        if np.isnan(group_intcrd_conditioning).any() or np.isnan(generated_coords).any():
            raise ValueError(f"NaN values detected in input or generated coords for coord group {group_id}.")

        # Superimpose original internal coordinates onto model-generated coordinates
        # This aligns the internal coordinate conditioning to match the model output
        aligned_coords, _ = superimpose(
            fixed=generated_coords,
            mobile=group_intcrd_conditioning,
        )

        # Update the coords-with-insertion annotation with aligned coordinates
        cwi = SA_CWI.annotation(atom_array)
        cwi[group_intcrd_mask] = aligned_coords
        SA_CWI.set_annotation(atom_array, np.asarray(cwi, dtype=np.float32))

    return atom_array


def align_atom_array_to_input_frame(atom_array: AtomArray, crdgrp_id: int | None = None) -> AtomArray:
    """Apply a rigid transform so the chosen coord group's atoms sit at their input coords.

    Kabsch-aligns ``atom_array.coord[anchor]`` onto the coords-with-insertion at the anchor,
    and applies that transform to every atom in the array. This is purely cosmetic:
    it makes saved CIFs open consistently across designs (motif at its input position,
    scaffold rotated to match).

    Anchor selection:
      - ``crdgrp_id`` if explicitly provided
      - Else the ``KNOWN_ABSOLUTE_COORDS`` bucket (C_CRD atoms; all share one frame)
      - Else the largest single C_INTCRD group (deterministic fallback)
      - Else no-op if no anchor found or fewer than 3 atoms

    Args:
        atom_array: AtomArray with the coords-with-insertion annotation populated.
        crdgrp_id: Optional ``C_CRDGRP`` value to use as the anchor.

    Returns:
        The same AtomArray with ``coord`` updated in place (and returned for chaining).
    """
    if not SA_CWI.has_annotation(atom_array):
        return atom_array

    coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
    cwi = SA_CWI.annotation(atom_array)
    if cwi.ndim == 3:
        cwi = cwi[0]
    cwi_valid = ~np.isnan(cwi).any(axis=-1)

    if crdgrp_id is None:
        absolute_mask = (coord_groups == C_CRDGRP.KNOWN_ABSOLUTE_COORDS) & cwi_valid
        if absolute_mask.any():
            anchor_mask = absolute_mask
        else:
            is_internal = (coord_groups != C_CRDGRP.UNKNOWN_COORDS) & (coord_groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS)
            intcrd_groups = np.unique(coord_groups[cwi_valid & is_internal])
            if intcrd_groups.size == 0:
                return atom_array
            sizes = {int(g): int(((coord_groups == g) & cwi_valid).sum()) for g in intcrd_groups}
            anchor_group = max(sizes, key=sizes.get)
            anchor_mask = (coord_groups == anchor_group) & cwi_valid
    else:
        anchor_mask = (coord_groups == crdgrp_id) & cwi_valid
        if not anchor_mask.any():
            return atom_array

    if anchor_mask.sum() < 3:
        return atom_array

    mobile = atom_array.coord[anchor_mask]
    _, transform = superimpose(fixed=cwi[anchor_mask], mobile=mobile)
    atom_array.coord = transform.apply(atom_array.coord)
    return atom_array
