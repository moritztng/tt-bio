# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Derive autoencoder and flow-model masks from token-level primitives.

``padding_mask`` says that a token exists rather than being batch padding.
``is_in_context_conditioning`` says that a token is context rather than generated output.
``supports_local_latent`` classifies generated, non-atomized residue types represented
by the autoencoder's latent alphabet; it is not itself an attention, output, or loss mask.
``is_resolved_mask`` says that every atom in a token is resolved.

In the formulas below, ``P`` is ``padding_mask``, ``C`` is ``is_in_context_conditioning``,
``L`` is ``supports_local_latent``, and ``R`` is ``is_resolved_mask``. The autoencoder
attends to ``P & ~C`` and emits local latents for ``P & ~C & L``; the flow trunk attends
to ``P``, while its BBCA/local-latent emit masks are ``P & ~C`` / ``P & ~C & L``.
Autoencoder and flow supervision are defined separately so their policies can diverge.
"""

from jaxtyping import Bool
from torch import Tensor


def compute_ae_masks(batch: dict[str, Tensor]) -> tuple[Bool[Tensor, "b n"], Bool[Tensor, "b n"]]:
    """Return the autoencoder attention mask and local-latent emit mask."""
    attention_mask = batch["padding_mask"] & ~batch["is_in_context_conditioning"]
    local_latent_emit_mask = attention_mask & batch["supports_local_latent"]
    return attention_mask, local_latent_emit_mask


def compute_flow_emit_masks(batch: dict[str, Tensor]) -> dict[str, Bool[Tensor, "b n"]]:
    """Return the tokens emitted for each flow modality."""
    bb_ca_emit_mask = batch["padding_mask"] & ~batch["is_in_context_conditioning"]
    return {
        "bb_ca": bb_ca_emit_mask,
        "local_latents": bb_ca_emit_mask & batch["supports_local_latent"],
    }


def compute_ae_supervision_mask(batch: dict[str, Tensor]) -> Bool[Tensor, "b n"]:
    """Return the resolved local-latent tokens supervised by the autoencoder."""
    _, local_latent_emit_mask = compute_ae_masks(batch)
    return local_latent_emit_mask & batch["is_resolved_mask"]


def compute_flow_supervision_masks(batch: dict[str, Tensor]) -> dict[str, Bool[Tensor, "b n"]]:
    """Return the resolved tokens supervised for each flow modality."""
    emit_masks = compute_flow_emit_masks(batch)
    return {mode: emit_mask & batch["is_resolved_mask"] for mode, emit_mask in emit_masks.items()}
