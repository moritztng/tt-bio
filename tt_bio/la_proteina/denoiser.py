# La-Proteina denoiser — ttnn port (pass 2).

# SPDX-License-Identifier: Apache-2.0
#
# Port of the La-Proteina flow-matching denoiser transformer block to ttnn.
# Reference: proteinfoundation/nn/modules/{pair_bias_attn,adaptive_ln_scale,
# attn_n_transition}.py (Apache-2.0, vendored under _vendor/la-proteina-ref).
#
# Pass 2 scope: the core sequence-side attention block
# `MultiHeadBiasedAttentionADALN_MM` = AdaptiveLayerNorm + PairBiasAttention
# (QK-LN + pair bias + gated output) + AdaptiveOutputScale. This is the first
# port target per the pass-1 scoping; it reuses tt-bio's existing pair-biased
# attention trunk idiom (tt_bio/tenstorrent.py AttentionPairBias / AdaLN) and
# adds the two La-Proteina-specific pieces: QK-LN (LayerNorm on the full q/k
# vectors before head split) and the cond-conditioned AdaptiveOutputScale.
#
# The block is intentionally self-contained (imports only ttnn/torch) so it can
# be parity-checked in isolation against the vendored PyTorch reference without
# pulling the full tt_bio import chain. Pass 3 will fold it into
# tt_bio/tenstorrent.py alongside the boltz2 DiffusionTransformer and wire the
# full LocalLatentsTransformer trunk (PairReprUpdate + tri-mult + TransitionADALN
# + the Euler sampler loop).

from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = a.norm() * b.norm()
    return float(num / (den + 1e-12))


class TTPairBiasAttentionAdaLN:
    """ttnn port of `MultiHeadBiasedAttentionADALN_MM`.

    Shapes (160M denoiser, configs/nn/local_latents_score_nn_160M.yaml):
      x (node)   [B, N, token_dim=768]
      pair_rep   [B, N, N, pair_dim=256]
      cond       [B, N, dim_cond=256]
      mask       [B, N]            (bool; here assumed all-True for the parity path)
    head_dim = token_dim // nheads = 64 (tile-aligned, so the manual head reshape
    is metadata-only on ROW_MAJOR and SDPA's program config is the standard one).
    """

    def __init__(
        self,
        device,
        compute_kernel_config,
        state_dict: dict[str, torch.Tensor],
        token_dim: int = 768,
        pair_dim: int = 256,
        nheads: int = 12,
        dim_cond: int = 256,
        use_qkln: bool = True,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = compute_kernel_config
        self.token_dim = token_dim
        self.pair_dim = pair_dim
        self.nheads = nheads
        self.dim_cond = dim_cond
        self.head_dim = token_dim // nheads
        self.use_qkln = use_qkln
        self.dtype = dtype
        inner_dim = token_dim  # head_dim * nheads

        def tt(key, transform=lambda x: x, d=dtype):
            return ttnn.from_torch(
                transform(state_dict[key]), layout=ttnn.TILE_LAYOUT,
                device=device, dtype=d,
            )

        # AdaptiveLayerNorm: norm(x) no-affine; norm_cond(cond) WITH affine;
        # gamma = sigmoid(Linear(cond->dim)); beta = Linear(cond->dim, no bias).
        self.adaln_norm_cond_w = tt("adaln.norm_cond.weight")
        self.adaln_norm_cond_b = tt("adaln.norm_cond.bias")
        self.adaln_gamma_w = tt("adaln.to_gamma.0.weight", transform=lambda x: x.t())
        self.adaln_gamma_b = tt("adaln.to_gamma.0.bias")
        self.adaln_beta_w = tt("adaln.to_beta.weight", transform=lambda x: x.t())

        # PairBiasAttention.
        self.node_norm_w = tt("mha.node_norm.weight")
        self.node_norm_b = tt("mha.node_norm.bias")
        self.qkv_w = tt("mha.to_qkv.weight", transform=lambda x: x.t())
        self.qkv_b = tt("mha.to_qkv.bias")
        if use_qkln:
            self.q_ln_w = tt("mha.q_layer_norm.weight")
            self.q_ln_b = tt("mha.q_layer_norm.bias")
            self.k_ln_w = tt("mha.k_layer_norm.weight")
            self.k_ln_b = tt("mha.k_layer_norm.bias")
        self.g_w = tt("mha.to_g.weight", transform=lambda x: x.t())
        self.pair_norm_w = tt("mha.pair_norm.weight")
        self.pair_norm_b = tt("mha.pair_norm.bias")
        self.bias_w = tt("mha.to_bias.weight", transform=lambda x: x.t())  # [pair_dim, nheads]
        self.out_w = tt("mha.to_out_node.weight", transform=lambda x: x.t())
        self.out_b = tt("mha.to_out_node.bias")

        # AdaptiveOutputScale: gamma = sigmoid(Linear(cond->dim, zero-init w, bias=-2)).
        self.scale_gamma_w = tt("scale_output.to_adaln_zero_gamma.0.weight", transform=lambda x: x.t())
        self.scale_gamma_b = tt("scale_output.to_adaln_zero_gamma.0.bias")

    # ---- helpers ----
    def _lin(self, x, w, bias=None, dtype=None):
        return ttnn.linear(
            x, w, bias=bias, compute_kernel_config=self.ck,
            dtype=dtype if dtype is not None else self.dtype,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

    def _ln(self, x, w=None, b=None, eps=1e-5):
        return ttnn.layer_norm(
            x, weight=w, bias=b, epsilon=eps, compute_kernel_config=self.ck,
        )

    def _split_heads(self, t):
        """[B, N, inner] -> [B, H, N, d_head] via row-major reshape + permute."""
        b, n, inner = t.shape
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.reshape(t, (b, n, self.nheads, self.head_dim))
        t = ttnn.permute(t, (0, 2, 1, 3))  # [B, H, N, d_head]
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=self.dtype)
        return t

    # ---- sub-blocks ----
    def adaln(self, x, cond, mask):
        normed = self._ln(x)  # no-affine LN over token_dim
        nc = self._ln(cond, self.adaln_norm_cond_w, self.adaln_norm_cond_b)
        gamma = self._lin(nc, self.adaln_gamma_w, self.adaln_gamma_b)
        gamma = ttnn.sigmoid(gamma)
        beta = self._lin(nc, self.adaln_beta_w)
        out = ttnn.multiply(normed, gamma)
        out = ttnn.add(out, beta)
        out = ttnn.multiply(out, mask)
        return out

    def pair_bias_attn(self, node, pair_rep, pair_mask_bias):
        h = self.nheads
        node_n = self._ln(node, self.node_norm_w, self.node_norm_b)
        qkv = self._lin(node_n, self.qkv_w, self.qkv_b)  # [B, N, 3*inner]
        # split q,k,v along last dim
        q = ttnn.slice(qkv, (0, 0, 0), (qkv.shape[0], qkv.shape[1], self.token_dim), (1, 1, 1))
        k = ttnn.slice(qkv, (0, 0, self.token_dim), (qkv.shape[0], qkv.shape[1], 2 * self.token_dim), (1, 1, 1))
        v = ttnn.slice(qkv, (0, 0, 2 * self.token_dim), (qkv.shape[0], qkv.shape[1], 3 * self.token_dim), (1, 1, 1))
        ttnn.deallocate(qkv)
        if self.use_qkln:
            q = self._ln(q, self.q_ln_w, self.q_ln_b)
            k = self._ln(k, self.k_ln_w, self.k_ln_b)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        g = self._lin(node_n, self.g_w)  # [B, N, inner]
        # pair bias: LN(pair) -> Linear -> [B, N, N, H] -> permute to [B, H, N, N]
        z = self._ln(pair_rep, self.pair_norm_w, self.pair_norm_b)
        z = self._lin(z, self.bias_w)  # [B, N, N, H]
        z = ttnn.permute(z, (0, 3, 1, 2))  # [B, H, N, N]
        z = ttnn.add(z, pair_mask_bias)  # fold the (additive) pair mask into the bias
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=z, is_causal=False,
            scale=self.head_dim ** -0.5,
        )  # [B, H, N, d_head]
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        # merge heads: [B, H, N, d_head] -> [B, N, inner]
        o = ttnn.permute(o, (0, 2, 1, 3))  # [B, N, H, d_head]
        o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.reshape(o, (o.shape[0], o.shape[1], self.token_dim))
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, dtype=self.dtype)
        g = ttnn.sigmoid(g)
        o = ttnn.multiply(o, g)
        out = self._lin(o, self.out_w, self.out_b)
        return out

    def scale_output(self, x, cond, mask):
        gamma = self._lin(cond, self.scale_gamma_w, self.scale_gamma_b)
        gamma = ttnn.sigmoid(gamma)
        out = ttnn.multiply(x, gamma)
        out = ttnn.multiply(out, mask)
        return out

    def __call__(self, x, pair_rep, cond, mask, pair_mask_bias):
        # mirror MultiHeadBiasedAttentionADALN_MM.forward
        x = self.adaln(x, cond, mask)
        x = self.pair_bias_attn(x, pair_rep, pair_mask_bias)
        x = self.scale_output(x, cond, mask)
        x = ttnn.multiply(x, mask)
        return x
