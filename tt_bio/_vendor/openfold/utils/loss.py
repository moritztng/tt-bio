# Copyright 2021 AlQuraishi Laboratory / DeepMind. Apache-2.0.
"""Inference-only confidence metrics extracted from OpenFold's utils/loss.py
(pLDDT, predicted aligned error, pTM). Training losses are intentionally omitted —
this vendoring is inference-only (see docs/openfold-port.md)."""
from typing import Dict, Optional, Tuple

import torch


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)
    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )
    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)
    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8
    probs = torch.nn.functional.softmax(logits, dim=-1)
    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]
