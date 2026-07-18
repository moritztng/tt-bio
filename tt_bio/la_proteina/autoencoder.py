# La-Proteina autoencoder (encoder + decoder) -- ttnn port (pass 4).
#
# SPDX-License-Identifier: Apache-2.0
#
# Port of the La-Proteina structure autoencoder. Reference:
# proteinfoundation/partial_autoencoder/{encoder,decoder}.py
# `EncoderTransformer` / `DecoderTransformer` (Apache-2.0, vendored under
# _vendor/la-proteina-ref). The AE trunk is structurally identical to the
# denoiser trunk (cond stack + nlayers x MultiheadAttnAndTransition, optional
# PairReprUpdate), so it reuses `TTTransformerTrunk` from `denoiser.py`. The
# new surface is the heads:
#   - encoder: `latent_decoder_mean_n_log_scale` (LN + Linear 768 -> 2*latent_z_dim)
#     -> chunk -> mean, log_scale -> z = mean + eps * exp(log_scale) (stochastic;
#     eps is an explicit shared draw) -> ln_z (Identity when normalize_latent=false)
#     -> * mask.
#   - decoder: `logit_linear` (LN + Linear 768 -> 20, sequence logits) and
#     `struct_linear` (LN + Linear 768 -> 37*3, reshaped to [B,N,37,3] with the
#     abs_coors / relative-to-CA post-process).
#
# As with the denoiser, inputs are injected at the post-builder interface
# (seqs, pair_rep, c_pre, mask) so the full trunk + heads run without porting
# the FeatureFactory / PairRepBuilder dataset feature pipeline (the same gate
# that blocks real-weight parity and the full end-to-end forward). The
# abs_coors post-process is parameter-free deterministic host math, applied
# identically on both sides in the parity harness.
#
# Config (configs/nn_ae/nn_130m.yaml): encoder/decoder nlayers=12, token_dim=768,
# nheads=12, dim_cond=128, pair_repr_dim=256, update_pair_repr=False,
# use_qkln=True, latent_z_dim=8, decoder.abs_coors=False.

from __future__ import annotations

import torch
import ttnn

from .denoiser import (
    TTTransformerTrunk, _LNLinearHead, _tt, _CORE_GRID,
)


class TTLatentHead:
    """ttnn port of the encoder's `latent_decoder_mean_n_log_scale` head.

    LN + Linear(768 -> 2*latent_z_dim, bias=False) -> chunk -> mean, log_scale.
    z = mean + eps * exp(log_scale)  (eps = explicit shared stochastic draw).
    Then ln_z (Identity when normalize_latent=False) -> * mask.
    Returns (mean, log_scale, z) as device tensors.
    """

    def __init__(self, device, ck, state_dict, dim=768, latent_dim=8,
                 normalize_latent=False, dtype=ttnn.bfloat16):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.latent_dim = latent_dim
        self.normalize_latent = normalize_latent
        self.head = _LNLinearHead(device, ck, state_dict, dim=dim,
                                   out_dim=2 * latent_dim, dtype=dtype)

    def __call__(self, seqs, mask, eps):
        flat = self.head(seqs, mask)                       # [B,N,2*latent_dim]
        b, n, _ = flat.shape
        di = self.latent_dim
        mean = ttnn.slice(flat, (0, 0, 0), (b, n, di), (1, 1, 1))
        log_scale = ttnn.slice(flat, (0, 0, di), (b, n, 2 * di), (1, 1, 1))
        ttnn.deallocate(flat)
        # z = mean + eps * exp(log_scale)
        ex = ttnn.exp(log_scale)
        z = ttnn.add(mean, ttnn.multiply(eps, ex))
        ttnn.deallocate(ex)
        z = ttnn.multiply(z, mask)
        if self.normalize_latent:
            z = ttnn.layer_norm(z, epsilon=1e-5, compute_kernel_config=self.ck)
            z = ttnn.multiply(z, mask)
        return mean, log_scale, z


def TTLogitHead(device, ck, state_dict, dim=768, dtype=ttnn.bfloat16):
    """Decoder `logit_linear`: LN + Linear(768 -> 20, bias=False) * mask."""
    return _LNLinearHead(device, ck, state_dict, dim=dim, out_dim=20, dtype=dtype)


def TTStructHead(device, ck, state_dict, dim=768, dtype=ttnn.bfloat16):
    """Decoder `struct_linear`: LN + Linear(768 -> 37*3=111, bias=False) * mask.
    Returns the flat [B,N,111] (TILE-padded to 128); the parity harness slices
    the real 111 lanes and reshapes to [B,N,37,3] on host, then applies the
    abs_coors post-process (parameter-free deterministic host math)."""
    return _LNLinearHead(device, ck, state_dict, dim=dim, out_dim=111, dtype=dtype)


class TTEncoderTransformer:
    """ttnn port of `EncoderTransformer` (post-builder interface)."""

    def __init__(self, device, ck, state_dict, token_dim=768, pair_dim=256,
                 nheads=12, dim_cond=128, nlayers=12, latent_dim=8,
                 use_qkln=True, normalize_latent=False, dtype=ttnn.bfloat16):
        self.trunk = TTTransformerTrunk(
            device, ck, state_dict, token_dim=token_dim, pair_dim=pair_dim,
            nheads=nheads, dim_cond=dim_cond, nlayers=nlayers,
            use_qkln=use_qkln, update_pair_repr=False, dtype=dtype,
        )
        self.latent_head = TTLatentHead(
            device, ck, state_dict["latent_decoder_mean_n_log_scale"],
            dim=token_dim, latent_dim=latent_dim,
            normalize_latent=normalize_latent, dtype=dtype,
        )

    def __call__(self, seqs, pair_rep, c_pre, mask, pair_mask_bias, eps):
        seqs, _, _ = self.trunk(seqs, pair_rep, c_pre, mask, pair_mask_bias)
        mean, log_scale, z = self.latent_head(seqs, mask, eps)
        return mean, log_scale, z


class TTDecoderTransformer:
    """ttnn port of `DecoderTransformer` (post-builder interface).

    Returns (logits [B,N,20], coors_flat [B,N,111]) as device tensors. The
    abs_coors post-process (set CA slot / add CA) is parameter-free host math
    applied in the parity harness.
    """

    def __init__(self, device, ck, state_dict, token_dim=768, pair_dim=256,
                 nheads=12, dim_cond=128, nlayers=12, use_qkln=True,
                 dtype=ttnn.bfloat16):
        self.trunk = TTTransformerTrunk(
            device, ck, state_dict, token_dim=token_dim, pair_dim=pair_dim,
            nheads=nheads, dim_cond=dim_cond, nlayers=nlayers,
            use_qkln=use_qkln, update_pair_repr=False, dtype=dtype,
        )
        self.logit_head = TTLogitHead(
            device, ck, state_dict["logit_linear"], dim=token_dim, dtype=dtype,
        )
        self.struct_head = TTStructHead(
            device, ck, state_dict["struct_linear"], dim=token_dim, dtype=dtype,
        )

    def __call__(self, seqs, pair_rep, c_pre, mask, pair_mask_bias):
        seqs, _, _ = self.trunk(seqs, pair_rep, c_pre, mask, pair_mask_bias)
        logits = self.logit_head(seqs, mask)
        coors_flat = self.struct_head(seqs, mask)
        return logits, coors_flat
