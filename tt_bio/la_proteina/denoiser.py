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


class TTPairReprUpdate:
    """ttnn port of `pair_update.PairReprUpdate` (non-tri-mult path).

    forward (use_tri_mult=False):
        pair_mask = mask[:,None,:] * mask[:,:,None]
        x = x * mask
        x1, x2 = linear_x(LN(x)).chunk(2, -1)
        pair_rep = (pair_rep + x1[:,None,:] + x2[:,:,None]) * pair_mask
        pair_rep = (pair_rep + PairTransition(pair_rep, pair_mask)) * pair_mask
    The `use_tri_mult=True` path inserts openfold TriangleMultiplicationOutgoing/
    Incoming between the injection and the PairTransition; that reuses the
    existing `tt_bio.tenstorrent.TriangleMultiplication` port (same openfold
    state-dict layout: norm_in/g_in/p_in/norm_out/g_out/z_out). Tri-mult device
    parity is already covered by tt-bio's boltz2/protenix ports, so the fused
    tri-mult path is deferred to a follow-on pass (wired when a `_tri` checkpoint
    is available for real-weight parity).
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
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.token_dim = token_dim
        self.pair_dim = pair_dim
        self.use_tri_mult = use_tri_mult
        if use_tri_mult:
            raise NotImplementedError(
                "TTPairReprUpdate tri-mult path deferred -- reuse "
                "tt_bio.tenstorrent.TriangleMultiplication (openfold idiom)."
            )
        self.ln_in_w = _tt(state_dict["layer_norm_in.weight"], device, dtype)
        self.ln_in_b = _tt(state_dict["layer_norm_in.bias"], device, dtype)
        self.linear_x_w = _tt(state_dict["linear_x.weight"], device, dtype, lambda x: x.t())
        self.transition_out = TTPairTransition(
            device, ck, state_dict["transition_out"],
            c_z=pair_dim, n=expansion_factor_transition, dtype=dtype,
        )

    def __call__(self, x, pair_rep, mask):
        b, n, _ = x.shape
        # pair_mask [B,N,N] -> [B,N,N,1]
        # build pair_mask from mask [B,N,1]
        # mask is [B,N,1]; pair_mask = mask_b * mask_a^T
        m1 = mask  # [B,N,1]
        m2 = ttnn.permute(mask, (0, 2, 1))  # [B,1,N]
        pair_mask = ttnn.multiply(m1, m2)  # [B,N,N]
        pair_mask_4 = ttnn.reshape(pair_mask, (b, n, n, 1))  # [B,N,N,1]

        x = ttnn.multiply(x, mask)
        xn = ttnn.layer_norm(
            x, weight=self.ln_in_w, bias=self.ln_in_b,
            epsilon=1e-5, compute_kernel_config=self.ck,
        )
        proj = ttnn.linear(
            xn, self.linear_x_w, compute_kernel_config=self.ck,
            dtype=self.dtype, core_grid=_CORE_GRID,
        )  # [B,N,2*pair_dim]
        x1 = ttnn.slice(proj, (0, 0, 0), (b, n, self.pair_dim), (1, 1, 1))
        x2 = ttnn.slice(proj, (0, 0, self.pair_dim), (b, n, 2 * self.pair_dim), (1, 1, 1))
        ttnn.deallocate(proj)
        # x1[:,None,:] -> [B,1,N,pair_dim]; x2[:,:,None,:] -> [B,N,1,pair_dim]
        x1 = ttnn.reshape(x1, (b, 1, n, self.pair_dim))
        x2 = ttnn.reshape(x2, (b, n, 1, self.pair_dim))
        pair_rep = ttnn.add(pair_rep, x1)
        pair_rep = ttnn.add(pair_rep, x2)
        ttnn.deallocate(x1); ttnn.deallocate(x2)
        pair_rep = ttnn.multiply(pair_rep, pair_mask_4)
        pt = self.transition_out(pair_rep, pair_mask_4)
        pair_rep = ttnn.add(pair_rep, pt)
        ttnn.deallocate(pt)
        pair_rep = ttnn.multiply(pair_rep, pair_mask_4)
        return pair_rep
