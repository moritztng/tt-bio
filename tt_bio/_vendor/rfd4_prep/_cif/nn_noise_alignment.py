# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Align conditioning to the noisy state with proper rotations and optional translation."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .utils_align import proper_rotation


@torch.no_grad()
def align_chirality(
    chirality_vectors: Float[Tensor, "b n d"], batch: dict, translate: bool = False, *, rotate: bool = True
) -> Float[Tensor, "b n d"]:
    """Align packed stereochemical vectors to the current noisy state.

    Args:
        chirality_vectors: Packed, unit-scale 3D vectors per token; d = 3 * S.
        batch: Noisy anchors in nm (``x_t.bb_ca``), real-token indicators
            (``padding_mask``), frame indices (``chiral_frame_atoms``), and
            vector ownership (``chiral_slot_owner``).
        translate: Add the noisy center or bond-endpoint midpoint in nm.
        rotate: Fit orientations to noisy neighbor directions, choosing an
            arbitrary optimum when the fit is non-unique.

    Returns:
        Aligned vectors with unsupported slots preserved and padding zeroed.
        Both options disabled returns the input unchanged.

    Shape labels: B=batch, N=tokens, S=vector slots, F=active frames across
    the batch, K=fitting atoms per frame.
    """
    if not (rotate or translate):
        return chirality_vectors
    batch_size, n_tokens, feature_dim = chirality_vectors.shape
    n_blocks = feature_dim // 3
    noisy_anchor_coords = batch["x_t"]["bb_ca"].float()  # [B, N, 3]
    valid_tokens = batch["padding_mask"]  # [B, N]
    frames = batch["chiral_frame_atoms"]  # [B, N, S, 2 + K]
    bi, ci, blocks = torch.where((frames[..., 0] >= 0) & valid_tokens[..., None])  # Each [F]
    if bi.numel() == 0:
        return chirality_vectors * valid_tokens[..., None]  # [B, N, 3*S]
    origins = frames[bi, ci, blocks, :2]  # [F, 2]
    neighbors = frames[bi, ci, blocks, 2:]  # [F, K]
    present_neighbors = neighbors >= 0  # [F, K]
    neighbors = neighbors.clamp_min(0)  # [F, K]; missing entries use a masked placeholder
    packed = chirality_vectors.reshape(batch_size, n_tokens, n_blocks, 3)  # [B, N, S, 3]
    # Index shapes [F, 1], [F, K], [F, 1] broadcast to [F, K]; xyz stays intact.
    supplied_vectors = packed[bi[:, None], neighbors, blocks[:, None]].float()  # [F, K, 3]
    noisy_origin = noisy_anchor_coords[bi[:, None], origins].mean(-2, keepdim=True)  # [F, 2, 3] -> [F, 1, 3]
    noisy_vectors = noisy_anchor_coords[bi[:, None], neighbors] - noisy_origin  # [F, K, 3]
    if not (
        (valid_tokens[bi[:, None], neighbors] | ~present_neighbors).all() & valid_tokens[bi[:, None], origins].all()
    ):
        raise ValueError("Chiral frame references padding tokens")
    if not (
        (torch.isfinite(supplied_vectors).all(-1) | ~present_neighbors).all()
        & (torch.isfinite(noisy_vectors).all(-1) | ~present_neighbors).all()
    ):
        raise ValueError("Nonfinite chiral conditioning or noisy coordinates")
    with torch.autocast(device_type=chirality_vectors.device.type, enabled=False):
        frame_index = torch.zeros(
            batch_size,
            n_tokens * n_blocks,
            device=chirality_vectors.device,
            dtype=torch.long,
        )  # [B, N*S]
        # Map owner IDs (anchor_token * S + source_block) to compact frame IDs 1..F; 0 means no transform.
        frame_index[bi, ci * n_blocks + blocks] = torch.arange(1, len(bi) + 1, device=chirality_vectors.device)
        owners = batch["chiral_slot_owner"]  # [B, N, S]; copied slots retain the source frame's owner
        slot_indices = frame_index[
            torch.arange(batch_size, device=chirality_vectors.device)[:, None, None],  # [B, 1, 1]
            owners.clamp_min(0),  # [B, N, S]; unsupported owners (-1) are restored below
        ]  # [B, N, S], values 0..F
        aligned = packed  # [B, N, S, 3]
        if rotate:
            supplied_vectors = torch.where(present_neighbors[..., None], supplied_vectors, 0.0)  # [F, K, 3]
            noisy_vectors = torch.where(present_neighbors[..., None], noisy_vectors, 0.0)  # [F, K, 3]
            norms = torch.linalg.vector_norm(noisy_vectors, dim=-1, keepdim=True)  # [F, K, 1]
            noisy_vectors = noisy_vectors / norms.clamp_min(1e-8)  # [F, K, 3]
            covariance = supplied_vectors.transpose(-1, -2) @ noisy_vectors  # [F, 3, K] @ [F, K, 3] -> [F, 3, 3]
            rotation = proper_rotation(covariance)  # [F, 3, 3]
            rotations = torch.cat([torch.eye(3, device=chirality_vectors.device)[None], rotation])  # [F+1, 3, 3]
            # Gather [B, N, S, 3, 3] rotations; multiply each slot's row vector by its owner's rotation.
            aligned = torch.einsum("bnsx,bnsxy->bnsy", packed.float(), rotations[slot_indices])  # [B, N, S, 3]
        if translate:
            translations = torch.cat([noisy_origin.new_zeros(1, 3), noisy_origin[:, 0]])  # [F+1, 3]
            aligned = aligned + translations[slot_indices]  # [B, N, S, 3]
        aligned = torch.where((owners >= 0)[..., None], aligned, packed)  # [B, N, S, 3]
    return aligned.reshape_as(chirality_vectors) * valid_tokens[..., None]  # [B, N, 3*S]


@torch.no_grad()
def align_internal_coords(
    supplied_coords: Float[Tensor, "b n 37 3"],
    coord_group_ids: Int[Tensor, "b n 37"],
    conditioned_atoms: Bool[Tensor, "b n 37"],
    noisy_anchor_coords: Float[Tensor, "b n 3"],
    valid_tokens: Bool[Tensor, "b n"],
    translate: bool = False,
    *,
    rotate: bool = True,
) -> Float[Tensor, "b n 37 3"]:
    """Align each internal-coordinate group using its slot-1 CA/atomized anchors.

    Args:
        supplied_coords: Conditioned atom37 coordinates in nm.
        coord_group_ids: Group membership for each atom37 slot.
        conditioned_atoms: True for slots containing internal-coordinate conditioning.
        noisy_anchor_coords: Current noisy CA/atomized coordinates per token, in nm.
        valid_tokens: True for real tokens, false for padding.
        translate: Place each group's anchor centroid at its noisy anchor centroid.
        rotate: Fit each group's orientation to its noisy anchors, choosing an
            arbitrary optimum when non-unique. Without translation, center at the origin.

    Returns:
        Group-aligned atom37 coordinates in nm. Groups without conditioned anchors
        retain their supplied coordinates, since no alignment can be determined.
        Unconditioned slots and padding are zero. Both options disabled returns
        the input unchanged.
    """
    if not (rotate or translate):
        return supplied_coords
    result = torch.zeros_like(supplied_coords)
    for b in range(supplied_coords.shape[0]):
        valid_conditioned_atoms = conditioned_atoms[b] & valid_tokens[b, :, None]
        for group in torch.unique(coord_group_ids[b][valid_conditioned_atoms]):
            group_atoms = valid_conditioned_atoms & (coord_group_ids[b] == group)
            group_anchor_tokens = group_atoms[:, 1]
            if not group_anchor_tokens.any():
                result[b][group_atoms] = supplied_coords[b][group_atoms]
                continue
            supplied_anchors = supplied_coords[b, group_anchor_tokens, 1].float()
            noisy_anchors = noisy_anchor_coords[b, group_anchor_tokens].float()
            if not torch.isfinite(supplied_anchors).all() or not torch.isfinite(noisy_anchors).all():
                raise ValueError("Nonfinite internal conditioning or noisy anchor coordinates")
            supplied_centroid = supplied_anchors.mean(0)
            noisy_centroid = noisy_anchors.mean(0)
            with torch.autocast(device_type=supplied_coords.device.type, enabled=False):
                aligned = supplied_coords[b][group_atoms].float() - supplied_centroid
                if rotate:
                    rotation = proper_rotation(
                        (supplied_anchors - supplied_centroid).T @ (noisy_anchors - noisy_centroid)
                    )
                    aligned = aligned @ rotation
                if translate:
                    aligned = aligned + noisy_centroid
                result[b][group_atoms] = aligned
    return result
