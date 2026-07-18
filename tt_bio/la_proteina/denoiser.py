# La-Proteina denoiser — ttnn port (pass 2 + pass 3).

# SPDX-License-Identifier: Apache-2.0
#
# Port of the La-Proteina flow-matching denoiser transformer to ttnn.
# Reference: proteinfoundation/nn/modules/{pair_bias_attn,adaptive_ln_scale,
# attn_n_transition,seq_transition_af3,swiglu,pair_update}.py and
# openfold/model/{pair_transition,triangular_multiplicative_update}.py
# (Apache-2.0, vendored under _vendor/la-proteina-ref).
#
# Pass 2: the core sequence-side attention block
# `MultiHeadBiasedAttentionADALN_MM` = AdaptiveLayerNorm + PairBiasAttention
# (QK-LN + pair bias + gated output) + AdaptiveOutputScale.
#
# Pass 3: the rest of the denoiser trunk (component-by-component, same
# random-weight PCC bar, golden = unmodified vendored reference):
#   - TTTransition            : plain SwiGLU transition (the conditioning path
#                               `transition_c_1/2` and the inner block of
#                               TransitionADALN).  Reference: seq_transition_af3.Transition.
#   - TTTransitionADALN       : AdaLN + Transition + AdaptiveOutputScale.
#                               Reference: seq_transition_af3.TransitionADALN.
#   - TTLocalLatentsHead / TTCaHead : the two output heads (LN + Linear).
#                               Reference: local_latents_transformer.LocalLatentsTransformer.
#   - TTMultiheadAttnAndTransition : one trunk layer (attn + transition,
#                               sequential, both residual) = the stitch of the
#                               pass-2 attention block + TransitionADALN.
#                               Reference: attn_n_transition.MultiheadAttnAndTransition.
#   - TTPairReprUpdate        : pair-representation update (outer-product-style
#                               pair bias injection + optional openfold tri-mult
#                               + PairTransition).  NOT exercised by the 160M
#                               config (update_pair_repr=False, use_tri_mult=False);
#                               ported + component-paritied as a stretch.
#
# The blocks are self-contained (import only ttnn/torch) so each can be
# parity-checked in isolation against the vendored PyTorch reference without
# pulling the full tt_bio import chain.

from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_CORE_GRID = ttnn.CoreGrid(y=8, x=8)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = a.norm() * b.norm()
    return float(num / (den + 1e-12))


def _tt(t: torch.Tensor, device, dtype, transform=lambda x: x) -> "ttnn.Tensor":
    return ttnn.from_torch(
        transform(t), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype,
    )


class _AdaLN:
    """ttnn port of `AdaptiveLayerNorm`.

    norm(x) is no-affine; norm_cond(cond) carries affine params;
    gamma = sigmoid(Linear(cond->dim)); beta = Linear(cond->dim, no bias);
    out = norm(x) * gamma + beta, then * mask.
    """

    def __init__(self, device, ck, state_dict: dict, dtype):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.norm_cond_w = _tt(state_dict["norm_cond.weight"], device, dtype)
        self.norm_cond_b = _tt(state_dict["norm_cond.bias"], device, dtype)
        self.gamma_w = _tt(state_dict["to_gamma.0.weight"], device, dtype, lambda x: x.t())
        self.gamma_b = _tt(state_dict["to_gamma.0.bias"], device, dtype)
        self.beta_w = _tt(state_dict["to_beta.weight"], device, dtype, lambda x: x.t())

    def __call__(self, x, cond, mask):
        normed = ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=self.ck)
        nc = ttnn.layer_norm(
            cond, weight=self.norm_cond_w, bias=self.norm_cond_b,
            epsilon=1e-5, compute_kernel_config=self.ck,
        )
        gamma = ttnn.linear(
            nc, self.gamma_w, bias=self.gamma_b,
            compute_kernel_config=self.ck, dtype=self.dtype, core_grid=_CORE_GRID,
        )
        gamma = ttnn.sigmoid(gamma)
        beta = ttnn.linear(
            nc, self.beta_w, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )
        out = ttnn.multiply(normed, gamma)
        out = ttnn.add(out, beta)
        out = ttnn.multiply(out, mask)
        return out


class _AdaptiveOutputScale:
    """ttnn port of `AdaptiveOutputScale`.

    gamma = sigmoid(Linear(cond->dim, zero-init weight, bias=-2));
    out = x * gamma * mask.
    """

    def __init__(self, device, ck, state_dict: dict, dtype):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.gamma_w = _tt(
            state_dict["to_adaln_zero_gamma.0.weight"], device, dtype, lambda x: x.t()
        )
        self.gamma_b = _tt(state_dict["to_adaln_zero_gamma.0.bias"], device, dtype)

    def __call__(self, x, cond, mask):
        gamma = ttnn.linear(
            cond, self.gamma_w, bias=self.gamma_b,
            compute_kernel_config=self.ck, dtype=self.dtype, core_grid=_CORE_GRID,
        )
        gamma = ttnn.sigmoid(gamma)
        out = ttnn.multiply(x, gamma)
        out = ttnn.multiply(out, mask)
        return out



class TTTransition:
    """ttnn port of `seq_transition_af3.Transition` (SwiGLU, optional input LN).

    swish_linear = Linear(dim, dim_inner*2, bias=False) then SwiGLU
    (x, gates = chunk(2, -1); silu(gates) * x);
    linear_out = Linear(dim_inner, dim, bias=False);
    out = linear_out(swish_linear(x)) * mask  (with optional leading LN).

    State-dict keys: `swish_linear.0.weight` [dim_inner*2, dim],
    `linear_out.weight` [dim, dim_inner], and (if layer_norm) `ln.weight/bias`.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        dim: int,
        expansion_factor: int = 4,
        layer_norm: bool = False,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)
        self.layer_norm = layer_norm
        self.swish_w = _tt(state_dict["swish_linear.0.weight"], device, dtype, lambda x: x.t())
        self.out_w = _tt(state_dict["linear_out.weight"], device, dtype, lambda x: x.t())
        if layer_norm:
            self.ln_w = _tt(state_dict["ln.weight"], device, dtype)
            self.ln_b = _tt(state_dict["ln.bias"], device, dtype)

    def __call__(self, x, mask):
        if self.layer_norm:
            x = ttnn.layer_norm(
                x, weight=self.ln_w, bias=self.ln_b,
                epsilon=1e-5, compute_kernel_config=self.ck,
            )
        h = ttnn.linear(
            x, self.swish_w, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )  # [B, N, 2*dim_inner]
        di = self.dim_inner
        x_part = ttnn.slice(h, (0, 0, 0), (h.shape[0], h.shape[1], di), (1, 1, 1))
        gates = ttnn.slice(h, (0, 0, di), (h.shape[0], h.shape[1], 2 * di), (1, 1, 1))
        ttnn.deallocate(h)
        gates = ttnn.silu(gates)
        m = ttnn.multiply(gates, x_part)
        ttnn.deallocate(gates); ttnn.deallocate(x_part)
        out = ttnn.linear(
            m, self.out_w, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )
        ttnn.deallocate(m)
        out = ttnn.multiply(out, mask)
        return out


class TTTransitionADALN:
    """ttnn port of `seq_transition_af3.TransitionADALN`.

    = AdaptiveLayerNorm + Transition(layer_norm=False, expansion_factor=4)
    + AdaptiveOutputScale, then * mask.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        dim: int,
        dim_cond: int,
        expansion_factor: int = 4,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.adaln = _AdaLN(device, ck, state_dict["adaln"], dtype)
        self.transition = TTTransition(
            device, ck, state_dict["transition"], dim=dim,
            expansion_factor=expansion_factor, layer_norm=False, dtype=dtype,
        )
        self.scale_output = _AdaptiveOutputScale(
            device, ck, state_dict["scale_output"], dtype
        )

    def __call__(self, x, cond, mask):
        x = self.adaln(x, cond, mask)
        x = self.transition(x, mask)
        x = self.scale_output(x, cond, mask)
        x = ttnn.multiply(x, mask)
        return x



class TTPairBiasAttentionAdaLN:
    """ttnn port of `MultiHeadBiasedAttentionADALN_MM`.

    = AdaptiveLayerNorm + PairBiasAttention (QK-LN + pair bias + gated output)
    + AdaptiveOutputScale.

    Shapes (160M denoiser, configs/nn/local_latents_score_nn_160M.yaml):
      x (node)   [B, N, token_dim=768]
      pair_rep   [B, N, N, pair_dim=256]
      cond       [B, N, dim_cond=256]
      mask       [B, N, 1]            (float; 1.0 valid, 0.0 masked)
      pair_mask_bias [B, 1, N, N]     (additive; 0 valid, -1e4 masked)
    head_dim = token_dim // nheads = 64 (tile-aligned, so the manual head
    reshape is metadata-only on ROW_MAJOR and SDPA's program config is standard).
    """

    def __init__(
        self,
        device,
        compute_kernel_config,
        state_dict: dict,
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

        self.adaln = _AdaLN(device, self.ck, state_dict["adaln"], dtype)
        self.scale_output = _AdaptiveOutputScale(
            device, self.ck, state_dict["scale_output"], dtype
        )

        mha = state_dict["mha"]
        self.node_norm_w = _tt(mha["node_norm.weight"], device, dtype)
        self.node_norm_b = _tt(mha["node_norm.bias"], device, dtype)
        self.qkv_w = _tt(mha["to_qkv.weight"], device, dtype, lambda x: x.t())
        self.qkv_b = _tt(mha["to_qkv.bias"], device, dtype)
        if use_qkln:
            self.q_ln_w = _tt(mha["q_layer_norm.weight"], device, dtype)
            self.q_ln_b = _tt(mha["q_layer_norm.bias"], device, dtype)
            self.k_ln_w = _tt(mha["k_layer_norm.weight"], device, dtype)
            self.k_ln_b = _tt(mha["k_layer_norm.bias"], device, dtype)
        self.g_w = _tt(mha["to_g.weight"], device, dtype, lambda x: x.t())
        self.pair_norm_w = _tt(mha["pair_norm.weight"], device, dtype)
        self.pair_norm_b = _tt(mha["pair_norm.bias"], device, dtype)
        self.bias_w = _tt(mha["to_bias.weight"], device, dtype, lambda x: x.t())
        self.out_w = _tt(mha["to_out_node.weight"], device, dtype, lambda x: x.t())
        self.out_b = _tt(mha["to_out_node.bias"], device, dtype)

    # ---- helpers ----
    def _lin(self, x, w, bias=None, dtype=None):
        return ttnn.linear(
            x, w, bias=bias, compute_kernel_config=self.ck,
            dtype=dtype if dtype is not None else self.dtype,
            core_grid=_CORE_GRID,
        )

    def _ln(self, x, w=None, b=None, eps=1e-5):
        return ttnn.layer_norm(
            x, weight=w, bias=b, epsilon=eps, compute_kernel_config=self.ck,
        )

    def _split_heads(self, t):
        """[B, N, inner] -> [B, H, N, d_head] via row-major reshape + permute."""
        b, n, _inner = t.shape
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.reshape(t, (b, n, self.nheads, self.head_dim))
        t = ttnn.permute(t, (0, 2, 1, 3))  # [B, H, N, d_head]
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=self.dtype)
        return t

    def pair_bias_attn(self, node, pair_rep, pair_mask_bias):
        node_n = self._ln(node, self.node_norm_w, self.node_norm_b)
        qkv = self._lin(node_n, self.qkv_w, self.qkv_b)  # [B, N, 3*inner]
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
        z = self._ln(pair_rep, self.pair_norm_w, self.pair_norm_b)
        z = self._lin(z, self.bias_w)  # [B, N, N, H]
        z = ttnn.permute(z, (0, 3, 1, 2))  # [B, H, N, N]
        z = ttnn.add(z, pair_mask_bias)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=z, is_causal=False,
            scale=self.head_dim ** -0.5,
        )  # [B, H, N, d_head]
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = ttnn.permute(o, (0, 2, 1, 3))  # [B, N, H, d_head]
        o = ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.reshape(o, (o.shape[0], o.shape[1], self.token_dim))
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, dtype=self.dtype)
        g = ttnn.sigmoid(g)
        o = ttnn.multiply(o, g)
        out = self._lin(o, self.out_w, self.out_b)
        return out

    def __call__(self, x, pair_rep, cond, mask, pair_mask_bias):
        x = self.adaln(x, cond, mask)
        x = self.pair_bias_attn(x, pair_rep, pair_mask_bias)
        x = self.scale_output(x, cond, mask)
        x = ttnn.multiply(x, mask)
        return x



class _LNLinearHead:
    """ttnn port of a `nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out, bias=False))`
    output head.  Used for both `local_latents_linear` (out=latent_dim=8) and
    `ca_linear` (out=3).  out = Linear(LN(x)) * mask.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        dim: int,
        out_dim: int,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        # nn.Sequential stores children as "0" (LayerNorm) and "1" (Linear).
        self.ln_w = _tt(state_dict["0.weight"], device, dtype)
        self.ln_b = _tt(state_dict["0.bias"], device, dtype)
        self.lin_w = _tt(state_dict["1.weight"], device, dtype, lambda x: x.t())

    def __call__(self, x, mask):
        h = ttnn.layer_norm(
            x, weight=self.ln_w, bias=self.ln_b,
            epsilon=1e-5, compute_kernel_config=self.ck,
        )
        out = ttnn.linear(
            h, self.lin_w, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )
        ttnn.deallocate(h)
        out = ttnn.multiply(out, mask)
        return out


def TTLocalLatentsHead(device, ck, state_dict, dim, latent_dim=8, dtype=ttnn.bfloat16):
    return _LNLinearHead(device, ck, state_dict, dim=dim, out_dim=latent_dim, dtype=dtype)


def TTCaHead(device, ck, state_dict, dim, dtype=ttnn.bfloat16):
    return _LNLinearHead(device, ck, state_dict, dim=dim, out_dim=3, dtype=dtype)


class TTMultiheadAttnAndTransition:
    """ttnn port of `attn_n_transition.MultiheadAttnAndTransition` (one trunk layer).

    160M config: parallel=False, residual_mha=True, residual_transition=True,
    use_attn_pair_bias=True, use_qkln=True.  forward:
        x = x * mask
        x_attn = mhba(x, pair_rep, cond, mask) + x   ; x_attn *= mask
        x_tr  = transition(x_attn, cond, mask) + x_attn ; x_tr *= mask
        return x_tr * mask
    `mhba` is the pass-2 attention block; `transition` is TransitionADALN.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        token_dim: int = 768,
        pair_dim: int = 256,
        nheads: int = 12,
        dim_cond: int = 256,
        use_qkln: bool = True,
        expansion_factor: int = 4,
        residual_mha: bool = True,
        residual_transition: bool = True,
        parallel: bool = False,
        dtype=ttnn.bfloat16,
    ):
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition
        self.parallel = parallel
        self.mhba = TTPairBiasAttentionAdaLN(
            device, ck, state_dict["mhba"],
            token_dim=token_dim, pair_dim=pair_dim, nheads=nheads,
            dim_cond=dim_cond, use_qkln=use_qkln, dtype=dtype,
        )
        self.transition = TTTransitionADALN(
            device, ck, state_dict["transition"],
            dim=token_dim, dim_cond=dim_cond,
            expansion_factor=expansion_factor, dtype=dtype,
        )

    def _apply_mha(self, x, pair_rep, cond, mask, pair_mask_bias):
        x_attn = self.mhba(x, pair_rep, cond, mask, pair_mask_bias)
        if self.residual_mha:
            x_attn = ttnn.add(x_attn, x)
        x_attn = ttnn.multiply(x_attn, mask)
        return x_attn

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = ttnn.add(x_tr, x)
        x_tr = ttnn.multiply(x_tr, mask)
        return x_tr

    def __call__(self, x, pair_rep, cond, mask, pair_mask_bias):
        x = ttnn.multiply(x, mask)
        if self.parallel:
            out = ttnn.add(
                self._apply_mha(x, pair_rep, cond, mask, pair_mask_bias),
                self._apply_transition(x, cond, mask),
            )
        else:
            x = self._apply_mha(x, pair_rep, cond, mask, pair_mask_bias)
            out = self._apply_transition(x, cond, mask)
        out = ttnn.multiply(out, mask)
        return out



class TTPairTransition:
    """ttnn port of openfold `PairTransition` (Algorithm 15).

    out = Linear_2(relu(Linear_1(LN(z)))) * mask, with z [B, N, N, c_z].
    State-dict keys: `layer_norm.weight/bias`, `linear_1.weight/bias`,
    `linear_2.weight/bias`.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        c_z: int,
        n: int = 2,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.c_z = c_z
        self.n = n
        self.ln_w = _tt(state_dict["layer_norm.weight"], device, dtype)
        self.ln_b = _tt(state_dict["layer_norm.bias"], device, dtype)
        self.l1_w = _tt(state_dict["linear_1.weight"], device, dtype, lambda x: x.t())
        self.l1_b = _tt(state_dict["linear_1.bias"], device, dtype)
        self.l2_w = _tt(state_dict["linear_2.weight"], device, dtype, lambda x: x.t())
        self.l2_b = _tt(state_dict["linear_2.bias"], device, dtype)

    def __call__(self, z, pair_mask):
        # pair_mask: [B, N, N] -> [B, N, N, 1]
        h = ttnn.layer_norm(
            z, weight=self.ln_w, bias=self.ln_b,
            epsilon=1e-5, compute_kernel_config=self.ck,
        )
        h = ttnn.linear(
            h, self.l1_w, bias=self.l1_b, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )
        h = ttnn.relu(h)
        h = ttnn.linear(
            h, self.l2_w, bias=self.l2_b, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )
        h = ttnn.multiply(h, pair_mask)
        return h




# ---------------------------------------------------------------------------
# pass 4: tri-mult pair update + full multi-layer trunk
# ---------------------------------------------------------------------------


class TTTriangleMultiplicativeUpdate:
    """ttnn port of openfold `TriangleMultiplicativeUpdate` (Algorithms 11/12).

    Self-contained direct port (mirrors the openfold math exactly; does NOT
    reuse `tt_bio.tenstorrent.TriangleMultiplication` because that port uses
    the boltz2/protenix state-dict key layout (`norm_in/g_in/p_in/...`), not
    openfold's (`layer_norm_in/linear_a_p/...`), and is coupled to the
    tenstorrent.py `Module`/`Weights` framework. Reuse was not free; a clean
    parity-focused port here keeps the la_proteina denoiser self-contained.
    The perf-grade fused/chunked path is a follow-on (see memory
    `trimul-largeN-permute-bottleneck`); at parity N=64 it is L1-resident.

    forward (c_z = pair_dim, c_hidden = min(pair_dim, tri_mult_c)):
        z = LN_in(z)
        a = linear_a_p(z) * sigmoid(linear_a_g(z))   ; a *= mask
        b = linear_b_p(z) * sigmoid(linear_b_g(z))   ; b *= mask
        # contraction (batch over c_hidden), outgoing vs incoming differ only
        # in which spatial axes of a/b become (N_i, N_k) / (N_k, N_j):
        #   outgoing: a->(C,N_i,N_k)  b->(C,N_k,N_j)
        #   incoming: a->(C,N_i,N_k)  b->(C,N_k,N_j)   (permutes swapped)
        p = matmul(a_perm, b_perm)                   # [B,C,N_i,N_j]
        x = permute(p, -> [B,N_i,N_j,C])
        x = LN_out(x) ; x = linear_z(x)              # [B,N,N,c_z]
        g = sigmoid(linear_g(z))                     # z = LN_in(z)
        out = x * g                                  # [B,N,N,c_z]
    No trailing mask (the outer PairReprUpdate re-masks after adding).
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        c_z: int,
        c_hidden: int,
        outgoing: bool = True,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.outgoing = outgoing
        self.ln_in_w = _tt(state_dict["layer_norm_in.weight"], device, dtype)
        self.ln_in_b = _tt(state_dict["layer_norm_in.bias"], device, dtype)
        self.ln_out_w = _tt(state_dict["layer_norm_out.weight"], device, dtype)
        self.ln_out_b = _tt(state_dict["layer_norm_out.bias"], device, dtype)
        self.a_p_w = _tt(state_dict["linear_a_p.weight"], device, dtype, lambda x: x.t())
        self.a_p_b = _tt(state_dict["linear_a_p.bias"], device, dtype)
        self.a_g_w = _tt(state_dict["linear_a_g.weight"], device, dtype, lambda x: x.t())
        self.a_g_b = _tt(state_dict["linear_a_g.bias"], device, dtype)
        self.b_p_w = _tt(state_dict["linear_b_p.weight"], device, dtype, lambda x: x.t())
        self.b_p_b = _tt(state_dict["linear_b_p.bias"], device, dtype)
        self.b_g_w = _tt(state_dict["linear_b_g.weight"], device, dtype, lambda x: x.t())
        self.b_g_b = _tt(state_dict["linear_b_g.bias"], device, dtype)
        self.g_w = _tt(state_dict["linear_g.weight"], device, dtype, lambda x: x.t())
        self.g_b = _tt(state_dict["linear_g.bias"], device, dtype)
        self.z_w = _tt(state_dict["linear_z.weight"], device, dtype, lambda x: x.t())
        self.z_b = _tt(state_dict["linear_z.bias"], device, dtype)

    def _lin(self, x, w, bias=None):
        return ttnn.linear(
            x, w, bias=bias, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )

    def _ln(self, x, w, b):
        return ttnn.layer_norm(
            x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=self.ck,
        )

    def __call__(self, z, pair_mask_4):
        # z: [B,N,N,c_z]; pair_mask_4: [B,N,N,1]
        zn = self._ln(z, self.ln_in_w, self.ln_in_b)
        a = self._lin(zn, self.a_p_w, self.a_p_b)
        a = ttnn.multiply(a, ttnn.sigmoid(self._lin(zn, self.a_g_w, self.a_g_b)))
        a = ttnn.multiply(a, pair_mask_4)
        b = self._lin(zn, self.b_p_w, self.b_p_b)
        b = ttnn.multiply(b, ttnn.sigmoid(self._lin(zn, self.b_g_w, self.b_g_b)))
        b = ttnn.multiply(b, pair_mask_4)
        # to [B, c_hidden, N, N] (channel-move permute)
        if self.outgoing:
            a_p = ttnn.permute(a, (0, 3, 1, 2))   # [B,C,N_i,N_k]
            b_p = ttnn.permute(b, (0, 3, 2, 1))   # [B,C,N_k,N_j]
        else:
            a_p = ttnn.permute(a, (0, 3, 2, 1))   # [B,C,N_i,N_k]
            b_p = ttnn.permute(b, (0, 3, 1, 2))   # [B,C,N_k,N_j]
        ttnn.deallocate(a); ttnn.deallocate(b)
        # batched matmul over the channel dim. squeeze the leading B=1 batch
        # so ttnn.matmul sees a 3D [C,N,N]@[C,N,N] contraction (M=K=N=64,
        # tile-aligned; batch=C is the leading dim).
        squeeze_batch = a_p.shape[0] == 1
        if squeeze_batch:
            a_p = ttnn.squeeze(a_p, 0)
            b_p = ttnn.squeeze(b_p, 0)
        p = ttnn.matmul(a_p, b_p, compute_kernel_config=self.ck)  # [C,N,N]
        ttnn.deallocate(a_p); ttnn.deallocate(b_p)
        if squeeze_batch:
            p = ttnn.unsqueeze(p, 0)
        # [B,C,N,N] -> [B,N,N,C]
        p = ttnn.permute(p, (0, 2, 3, 1))
        x = self._ln(p, self.ln_out_w, self.ln_out_b)
        ttnn.deallocate(p)
        x = self._lin(x, self.z_w, self.z_b)          # [B,N,N,c_z]
        g = ttnn.sigmoid(self._lin(zn, self.g_w, self.g_b))
        ttnn.deallocate(zn)
        out = ttnn.multiply(x, g)
        return out



class TTPairReprUpdate:
    """ttnn port of `pair_update.PairReprUpdate`.

    forward (use_tri_mult=False):
        pair_mask = mask[:,None,:] * mask[:,:,None]
        x = x * mask[...,None]
        x1, x2 = linear_x(LN_in(x)).chunk(2, -1)
        pair_rep = (pair_rep + x1[:,None,:] + x2[:,:,None,:]) * pair_mask[...,None]
        pair_rep = (pair_rep + PairTransition(pair_rep, pair_mask)) * pair_mask[...,None]
    use_tri_mult=True inserts openfold TriangleMultiplicationOutgoing/Incoming
    between the injection and the PairTransition (each tri-mult add followed by
    a re-mask), via the self-contained `TTTriangleMultiplicativeUpdate` port
    (same openfold state-dict layout: layer_norm_in / linear_a_p / linear_a_g /
    linear_b_p / linear_b_g / linear_g / linear_z / layer_norm_out).
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        token_dim: int = 768,
        pair_dim: int = 256,
        expansion_factor_transition: int = 2,
        use_tri_mult: bool = False,
        tri_mult_c: int = 196,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.token_dim = token_dim
        self.pair_dim = pair_dim
        self.use_tri_mult = use_tri_mult
        self.ln_in_w = _tt(state_dict["layer_norm_in.weight"], device, dtype)
        self.ln_in_b = _tt(state_dict["layer_norm_in.bias"], device, dtype)
        self.linear_x_w = _tt(state_dict["linear_x.weight"], device, dtype, lambda x: x.t())
        self.transition_out = TTPairTransition(
            device, ck, state_dict["transition_out"],
            c_z=pair_dim, n=expansion_factor_transition, dtype=dtype,
        )
        if use_tri_mult:
            c_hidden = min(pair_dim, tri_mult_c)
            self.tri_mult_out = TTTriangleMultiplicativeUpdate(
                device, ck, state_dict["tri_mult_out"],
                c_z=pair_dim, c_hidden=c_hidden, outgoing=True, dtype=dtype,
            )
            self.tri_mult_in = TTTriangleMultiplicativeUpdate(
                device, ck, state_dict["tri_mult_in"],
                c_z=pair_dim, c_hidden=c_hidden, outgoing=False, dtype=dtype,
            )

    def __call__(self, x, pair_rep, mask):
        b, n, _ = x.shape
        m1 = mask
        m2 = ttnn.permute(mask, (0, 2, 1))
        pair_mask = ttnn.multiply(m1, m2)            # [B,N,N]
        pair_mask_4 = ttnn.reshape(pair_mask, (b, n, n, 1))

        x = ttnn.multiply(x, mask)
        xn = ttnn.layer_norm(
            x, weight=self.ln_in_w, bias=self.ln_in_b,
            epsilon=1e-5, compute_kernel_config=self.ck,
        )
        proj = ttnn.linear(
            xn, self.linear_x_w, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )
        x1 = ttnn.slice(proj, (0, 0, 0), (b, n, self.pair_dim), (1, 1, 1))
        x2 = ttnn.slice(proj, (0, 0, self.pair_dim), (b, n, 2 * self.pair_dim), (1, 1, 1))
        ttnn.deallocate(proj)
        x1 = ttnn.reshape(x1, (b, 1, n, self.pair_dim))
        x2 = ttnn.reshape(x2, (b, n, 1, self.pair_dim))
        pair_rep = ttnn.add(pair_rep, x1)
        pair_rep = ttnn.add(pair_rep, x2)
        ttnn.deallocate(x1); ttnn.deallocate(x2)
        pair_rep = ttnn.multiply(pair_rep, pair_mask_4)
        if self.use_tri_mult:
            tmo = self.tri_mult_out(pair_rep, pair_mask_4)
            pair_rep = ttnn.add(pair_rep, tmo)
            ttnn.deallocate(tmo)
            pair_rep = ttnn.multiply(pair_rep, pair_mask_4)
            tmi = self.tri_mult_in(pair_rep, pair_mask_4)
            pair_rep = ttnn.add(pair_rep, tmi)
            ttnn.deallocate(tmi)
            pair_rep = ttnn.multiply(pair_rep, pair_mask_4)
        pt = self.transition_out(pair_rep, pair_mask_4)
        pair_rep = ttnn.add(pair_rep, pt)
        ttnn.deallocate(pt)
        pair_rep = ttnn.multiply(pair_rep, pair_mask_4)
        return pair_rep

class TTTransformerTrunk:
    """Shared denoiser/AE trunk orchestrator: cond stack + nlayers x
    MultiheadAttnAndTransition (+ optional PairReprUpdate every n layers).

    Mirrors `local_latents_transformer.LocalLatentsTransformer.forward` (and the
    AE `EncoderTransformer` / `DecoderTransformer` trunks, which are
    structurally identical). Inputs are injected at the post-builder interface
    (seqs, pair_rep, c_pre, mask) so the full trunk runs without porting the
    FeatureFactory / PairReprBuilder dataset feature pipeline.

    forward:
        c = transition_c_2(transition_c_1(c_pre, mask), mask)
        seqs = seqs_in * mask
        for i in range(nlayers):
            seqs = transformer_layers[i](seqs, pair_rep, c, mask, pair_mask_bias)
            if update_pair_repr and i < nlayers-1 and pair_update_layers[i]:
                pair_rep = pair_update_layers[i](seqs, pair_rep, mask)
        return seqs, pair_rep, c
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        token_dim: int = 768,
        pair_dim: int = 256,
        nheads: int = 12,
        dim_cond: int = 256,
        nlayers: int = 14,
        use_qkln: bool = True,
        update_pair_repr: bool = False,
        update_pair_repr_every_n: int = 3,
        use_tri_mult: bool = False,
        tri_mult_c: int = 196,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.nlayers = nlayers
        self.update_pair_repr = update_pair_repr
        self.transition_c_1 = TTTransition(
            device, ck, state_dict["transition_c_1"], dim=dim_cond,
            expansion_factor=2, layer_norm=False, dtype=dtype,
        )
        self.transition_c_2 = TTTransition(
            device, ck, state_dict["transition_c_2"], dim=dim_cond,
            expansion_factor=2, layer_norm=False, dtype=dtype,
        )
        self.transformer_layers = [
            TTMultiheadAttnAndTransition(
                device, ck, state_dict["transformer_layers"][i],
                token_dim=token_dim, pair_dim=pair_dim, nheads=nheads,
                dim_cond=dim_cond, use_qkln=use_qkln, expansion_factor=4,
                residual_mha=True, residual_transition=True, parallel=False,
                dtype=dtype,
            )
            for i in range(nlayers)
        ]
        self.pair_update_layers = []
        if update_pair_repr:
            for i in range(nlayers - 1):
                if i % update_pair_repr_every_n == 0:
                    self.pair_update_layers.append(
                        TTPairReprUpdate(
                            device, ck, state_dict["pair_update_layers"][i],
                            token_dim=token_dim, pair_dim=pair_dim,
                            expansion_factor_transition=2,
                            use_tri_mult=use_tri_mult,
                            tri_mult_c=tri_mult_c, dtype=dtype,
                        )
                    )
                else:
                    self.pair_update_layers.append(None)

    def __call__(self, seqs, pair_rep, c_pre, mask, pair_mask_bias):
        c = self.transition_c_1(c_pre, mask)
        c = self.transition_c_2(c, mask)
        seqs = ttnn.multiply(seqs, mask)
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask, pair_mask_bias)
            if self.update_pair_repr and i < self.nlayers - 1:
                upd = self.pair_update_layers[i]
                if upd is not None:
                    pair_rep = upd(seqs, pair_rep, mask)
        return seqs, pair_rep, c


class TTLocalLatentsTransformer:
    """ttnn port of `local_latents_transformer.LocalLatentsTransformer` (denoiser).

    = `TTTransformerTrunk` (cond stack + 14x MultiheadAttnAndTransition, optional
    PairReprUpdate) + the two output heads (local_latents_linear -> latent_dim,
    ca_linear -> 3). Inputs injected at the post-builder interface (seqs,
    pair_rep, c_pre, mask) so the full trunk + cond + heads run without porting
    the 1990-line feature_factory. See `TTTransformerTrunk` for the forward.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        token_dim: int = 768,
        pair_dim: int = 256,
        nheads: int = 12,
        dim_cond: int = 256,
        latent_dim: int = 8,
        nlayers: int = 14,
        use_qkln: bool = True,
        update_pair_repr: bool = False,
        update_pair_repr_every_n: int = 3,
        use_tri_mult: bool = False,
        tri_mult_c: int = 196,
        dtype=ttnn.bfloat16,
    ):
        self.trunk = TTTransformerTrunk(
            device, ck, state_dict, token_dim=token_dim, pair_dim=pair_dim,
            nheads=nheads, dim_cond=dim_cond, nlayers=nlayers,
            use_qkln=use_qkln, update_pair_repr=update_pair_repr,
            update_pair_repr_every_n=update_pair_repr_every_n,
            use_tri_mult=use_tri_mult, tri_mult_c=tri_mult_c, dtype=dtype,
        )
        self.local_latents_head = TTLocalLatentsHead(
            device, ck, state_dict["local_latents_linear"],
            dim=token_dim, latent_dim=latent_dim, dtype=dtype,
        )
        self.ca_head = TTCaHead(
            device, ck, state_dict["ca_linear"], dim=token_dim, dtype=dtype,
        )

    def __call__(self, seqs, pair_rep, c_pre, mask, pair_mask_bias):
        seqs, _, _ = self.trunk(seqs, pair_rep, c_pre, mask, pair_mask_bias)
        local_latents = self.local_latents_head(seqs, mask)
        ca = self.ca_head(seqs, mask)
        return local_latents, ca
