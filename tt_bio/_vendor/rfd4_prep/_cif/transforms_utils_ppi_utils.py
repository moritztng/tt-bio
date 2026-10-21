# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
import logging
import math
from typing import Literal

import numpy as np
from assertpy import assert_that
from tt_bio._vendor.atomworks.ml.transforms import mask_generator as mg
from tt_bio._vendor.atomworks.ml.transforms.atom_array import atom_id_to_atom_idx, atom_id_to_token_idx
from tt_bio._vendor.atomworks.ml.transforms.crop import (
    get_spatial_crop_center,
    get_token_count,
)
from tt_bio._vendor.atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)
from biotite.structure import AtomArray
from scipy.spatial import KDTree, cKDTree


ranked_logger = logging.getLogger(__name__)


def get_hotspot_atoms(
    atom_array: AtomArray,
    binder_pn_unit_iids: list[str],
    distance_cutoff: float = 4.5,
) -> np.ndarray:
    """Get hotspot atoms for a given distance cutoff.

    Args:
        atom_array (AtomArray): The atom array containing the protein structure.
        binder_pn_unit_iids (list[str]): The chain IDs of the binder (diffused chain).
        distance_cutoff (float): The interchain distance cutoff that defines hotspot atoms.

    Hotspots are atoms on non-binder chains that are within the distance cutoff of some residue on the binder.
    Residue distances are computed as the minimum pairwise distance between the two atoms.
    """

    # Filter NaN coordinates
    nan_coords_mask = np.any(np.isnan(atom_array.coord), axis=1)
    non_nan_atom_array = atom_array[~nan_coords_mask]

    # Get binder atoms
    binder_mask = np.isin(non_nan_atom_array.pn_unit_iid, binder_pn_unit_iids)
    binder_coords = non_nan_atom_array[binder_mask].coord

    # Get target atoms (non-binder)
    target_mask = ~binder_mask
    target_coords = non_nan_atom_array[target_mask].coord

    if len(binder_coords) == 0 or len(target_coords) == 0:
        full_hotspot_atom_mask = np.zeros(len(atom_array), dtype=bool)
        return full_hotspot_atom_mask

    # Build KDTree from binder atoms (one-time cost)
    binder_tree = cKDTree(binder_coords)

    # Query all target atoms at once - returns list of lists of indices
    # Each element is list of binder atom indices within distance
    query_results = binder_tree.query_ball_point(target_coords, distance_cutoff, p=2)

    # Convert to mask: True if any binder atom is within distance
    contacting_atoms_mask = np.array([len(indices) > 0 for indices in query_results])

    # Map back to full non_nan mask
    hotspot_atom_mask = np.zeros(len(non_nan_atom_array), dtype=bool)
    hotspot_atom_mask[target_mask] = contacting_atoms_mask

    # Map back to full atom_array mask (including NaN atoms)
    full_hotspot_atom_mask = np.zeros(len(atom_array), dtype=bool)
    full_hotspot_atom_mask[~nan_coords_mask] = hotspot_atom_mask

    return full_hotspot_atom_mask


def sample_hotspots(
    atom_array: AtomArray,
    binder_chain_ids: list[str] | str,
    hotspot_max_distance: float,
    max_hotspots_frac_to_provide: float = 0.1,
    hotspot_count_distribution: Literal["uniform", "geometric"] = "uniform",
    hotspot_geometric_p: float | None = None,
) -> np.ndarray:
    """
    Sample hotspots from the atom array.

    Args:
        atom_array: The atom array to sample hotspots from.
        binder_chain_ids: The chain IDs of all the binder chains. Can be list or single string.
        hotspot_max_distance: The maximum distance to the binder for an atom to be considered a hotspot.
        max_hotspots_frac_to_provide: Hard cap on the fraction of available hotspot atoms to
            provide. The sampled count is capped at this fraction of the total hotspot pool.
        hotspot_count_distribution: Distribution for sampling the number of hotspots. Must be
            ``"geometric"`` or ``"uniform"``.
        hotspot_geometric_p: Success probability for the geometric distribution. Required when
            ``hotspot_count_distribution`` is ``"geometric"``; must be ``None`` otherwise.

    Returns:
        A boolean mask of the same shape as the atom array, where True indicates a hotspot.
    """
    if hotspot_count_distribution not in ("geometric", "uniform"):
        raise ValueError(
            f"hotspot_count_distribution must be 'geometric' or 'uniform', got '{hotspot_count_distribution}'"
        )
    if hotspot_count_distribution == "geometric" and hotspot_geometric_p is None:
        raise ValueError("hotspot_geometric_p must not be None when hotspot_count_distribution is 'geometric'")
    if hotspot_count_distribution == "uniform" and hotspot_geometric_p is not None:
        raise ValueError("hotspot_geometric_p must be None when hotspot_count_distribution is 'uniform'")

    if isinstance(binder_chain_ids, str):
        binder_chain_ids = [binder_chain_ids]

    full_hotspot_atom_mask = get_hotspot_atoms(atom_array, binder_chain_ids, hotspot_max_distance)
    atom_array.set_annotation("is_hotspot_candidate", full_hotspot_atom_mask)

    n_available = int(full_hotspot_atom_mask.sum())
    if n_available == 0:
        return np.zeros(atom_array.array_length(), dtype=bool)

    n_max = max(1, math.floor(max_hotspots_frac_to_provide * n_available))
    if hotspot_count_distribution == "geometric":
        num_hotspots_to_sample = min(int(np.random.geometric(hotspot_geometric_p)), n_max)
    else:  # uniform
        num_hotspots_to_sample = int(np.random.randint(1, n_max + 1))

    fn_sample_seed = mg.SampleUniformly(is_eligible="is_hotspot_candidate")
    # Grow to just a single atom
    fn_grow_seed = mg.GrowToSegment(require_same_annotation=("atom_id",))
    fn_check_budget = mg.CheckAtomBudget(
        n_min_atoms=num_hotspots_to_sample,
        n_max_atoms=num_hotspots_to_sample,
    )
    total_mask, _ = mg.sample_mask_via_seed_grow_merge(
        atom_array,
        fn_sample_seed=fn_sample_seed,
        fn_grow_seed=fn_grow_seed,
        fn_check_budget=fn_check_budget,
        max_iterations=num_hotspots_to_sample,
    )
    return total_mask


def crop_spatial_keep_full_binder(
    atom_array: AtomArray,
    query_pn_unit_iids: list[str],
    binder_pn_unit_iid: str,
    crop_size: int,
    jitter_scale: float = 1e-3,
    crop_center_cutoff_distance: float = 15.0,
    crop_center_selection: Literal["random", "centroid"] = "random",
    force_crop: bool = False,
) -> dict:
    """
    Crop spatial tokens around a given `crop_center` by keeping the entire binder chain, then taking nearest
    neighbors (with jitter) until reaching the `crop_size`.

    Args:
        - atom_array (AtomArray): The atom array to crop.
        - query_pn_unit_iids (list[str]): List of query polymer/non-polymer unit instance IDs.
        - binder_pn_unit_iid (str): The polymer/non-polymer unit instance ID corresponding to the binder.
        - crop_size (int): The maximum number of tokens to crop.
        - jitter_scale (float, optional): Scale of jitter to apply when calculating distances.
            Defaults to 1e-3.
        - crop_center_cutoff_distance (float, optional): Maximum distance from query units to
            consider for crop center. Defaults to 15.0 Angstroms.
        - crop_center_selection ({"random", "centroid"}, optional): How to pick the crop centre from the
            interface-candidate atoms. ``"random"`` (AF3-style) draws one uniformly at random, so the crop
            wanders across the interface shell between calls. ``"centroid"`` picks the candidate atom nearest
            the mean of all interface-candidate atoms, giving a deterministic crop centred on the
            binder-target interface. Defaults to ``"random"``.
        - force_crop (bool, optional): Whether to force crop even if the atom array is already small enough.
            Defaults to False.

    Returns:
        dict: A dictionary containing crop information, including:
            - requires_crop (bool): Whether cropping was necessary.
            - crop_center_atom_id (int or np.nan): ID of the atom chosen as crop center.
            - crop_center_atom_idx (int or np.nan): Index of the atom chosen as crop center.
            - crop_center_token_idx (int or np.nan): Index of the token containing the crop center.
            - crop_token_idxs (np.ndarray): Indices of tokens included in the crop.
            - crop_atom_idxs (np.ndarray): Indices of atoms included in the crop.

    Note:
        This function implements the spatial cropping procedure as described in AlphaFold 3 and AlphaFold 2 Multimer.

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
    """
    if binder_pn_unit_iid not in query_pn_unit_iids:
        raise ValueError(
            f"Binder polymer/non-polymer unit instance ID '{binder_pn_unit_iid}' "
            f"not found in query polymer/non-polymer unit instance IDs: {query_pn_unit_iids}"
        )
    n_tokens = get_token_count(atom_array)
    requires_crop = n_tokens > crop_size

    # ... get binder information
    binder_token_mask = atom_array[get_token_starts(atom_array)].pn_unit_iid == binder_pn_unit_iid
    binder_atom_mask = atom_array.pn_unit_iid == binder_pn_unit_iid
    n_binder_tokens = get_token_count(atom_array[binder_atom_mask])

    if force_crop or requires_crop:
        # Get possible crop centers
        can_be_crop_center = get_spatial_crop_center(atom_array, query_pn_unit_iids, crop_center_cutoff_distance)

        # ... pick the crop center atom
        candidate_atoms = atom_array[can_be_crop_center]
        if crop_center_selection == "centroid":
            # Deterministic: the interface-candidate atom nearest the interface centroid.
            interface_centroid = candidate_atoms.coord.mean(axis=0)
            nearest_to_centroid = np.argmin(np.linalg.norm(candidate_atoms.coord - interface_centroid, axis=1))
            crop_center_atom_id = candidate_atoms.atom_id[nearest_to_centroid]
        else:
            # AF3-style: uniform random draw over interface-candidate atoms.
            crop_center_atom_id = np.random.choice(candidate_atoms.atom_id)
        crop_center_atom_idx = atom_id_to_atom_idx(atom_array, crop_center_atom_id)

        # ... sample crop, excluding the binder polymer/non-polymer unit
        # Use first-atom-per-token coords
        token_coords = atom_array.coord[get_token_starts(atom_array)]
        crop_center_token_idx = atom_id_to_token_idx(atom_array, crop_center_atom_id)
        is_token_in_crop = get_spatial_crop_excluding_mask(
            token_coords,
            crop_center_token_idx,
            crop_size=crop_size - n_binder_tokens,  # reserve space for the binder tokens
            mask_to_exclude=binder_token_mask,
            jitter_scale=jitter_scale,
        )
        # ... spread token-level crop mask to atom-level
        is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop)

        # ... add in binder tokens and atoms
        is_token_in_crop = is_token_in_crop | binder_token_mask
        is_atom_in_crop = is_atom_in_crop | binder_atom_mask
    else:
        # ... no need to crop since the atom array is already small enough
        crop_center_atom_id = np.nan
        crop_center_atom_idx = np.nan
        crop_center_token_idx = np.nan
        is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
        is_token_in_crop = np.ones(n_tokens, dtype=bool)

    return {
        "requires_crop": requires_crop,  # whether cropping was necessary
        "crop_center_atom_id": crop_center_atom_id,  # atom_id of crop center
        "crop_center_atom_idx": crop_center_atom_idx,  # atom_idx of crop center
        "crop_center_token_idx": crop_center_token_idx,  # token_idx of crop center
        "crop_token_idxs": np.where(is_token_in_crop)[0],  # token_idxs in crop
        "crop_atom_idxs": np.where(is_atom_in_crop)[0],  # atom_idxs in crop
    }


def get_spatial_crop_excluding_mask(
    coord: np.ndarray,
    crop_center_idx: int,
    crop_size: int,
    mask_to_exclude: np.ndarray,
    jitter_scale: float = 1e-3,
) -> np.ndarray:
    """
    Crop spatial tokens around a given `crop_center`, keeping nearest neighbors (with jitter) and excluding atoms in a
    specified mask, until reaching the `crop_size`.

    Args:
        coord (np.ndarray): A 2D numpy array of shape (N, 3) representing the 3D token-level coordinates.
            Coordinates are expected to be in Angstroms.
        crop_center_idx (int): The index of the token to be used as the center of the crop.
        crop_size (int): The number of nearest neighbors to include in the crop.
        mask_to_exclude (siwnp.ndarray): A mask indicating atoms to be excluded from the crop.
        jitter_scale (float): The scale of the jitter to add to the coordinates.

    Returns:
        crop_mask (np.ndarray): A boolean mask of shape (N,) where True indicates that the token is within the crop.

    """
    assert_that(coord.ndim).is_equal_to(2)
    assert_that(coord.shape[1]).is_equal_to(3)
    assert_that(crop_center_idx).is_less_than(coord.shape[0])
    assert_that(crop_size).is_greater_than(0)
    assert_that(jitter_scale).is_greater_than_or_equal_to(0)

    # Add small jitter to coordinates to break ties
    if jitter_scale > 0:
        coord = coord + np.random.normal(scale=jitter_scale, size=coord.shape)

    # ... get query center
    query_center = coord[crop_center_idx]

    # ... extract a mask for valid coordinates (i.e. no `nan`'s, which indicate unknown token centers)
    #     including including unoccupied tokens in the crop
    is_valid = np.isfinite(coord).all(axis=1)

    # ... exclude the specified pn_unit
    is_valid = is_valid & ~mask_to_exclude

    # ... build a KDTree for efficient querying, excluding invalid coordinates
    tree = KDTree(coord[is_valid])

    # ... query the `crop_size` nearest neighbors of the crop center
    _, nearest_neighbor_idxs = tree.query(query_center, k=crop_size, p=2)
    # ... filter out missing neighbours (index equal to `tree.n`)
    nearest_neighbor_idxs = nearest_neighbor_idxs[nearest_neighbor_idxs < tree.n]

    # ... crop mask is True for the `crop_size` nearest neighbors of the crop center
    crop_mask = np.zeros(coord.shape[0], dtype=bool)
    is_valid_and_in_crop_idxs = np.where(is_valid)[0][nearest_neighbor_idxs]
    crop_mask[is_valid_and_in_crop_idxs] = True

    return crop_mask
