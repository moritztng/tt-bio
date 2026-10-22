"""RFD4-Proteina flow-transformer trunk blocks on ttnn.

Upstream pin `state/rfd4/UPSTREAM.json` (2db07bb4), `rfproteina/nn/modules/`:
`MultiheadAttnAndTransition` is the trunk block -- pair-biased multi-head attention wrapped in an
adaptive norm and an adaptive output scale, then a SwiGLU transition wrapped the same way. The
released flow model stacks 24 of them at token_dim 1024, 16 heads, pair 384, cond 256, with
`norm_type: rmsnorm_torch_fused` (so every norm here is an RMSNorm, eps 1e-5) and `use_qkln: true`.

Masks. The caller passes `mask` as a [b, n, 1] 0/1 tensor and `pair_bias_mask` as a [b, 1, n, n]
additive mask (0 where both tokens are real, -1e4 elsewhere, upstream's NEG_INF). Padding the token
axis to a multiple of 32 (`tt_bio.token_axis`) is the caller's job: the pad tokens are masked
tokens, which is exactly what upstream does to a short sequence in a batch.

Every residual stream is kept masked: `Trunk` masks s and z once on entry and every update is
masked before it is added, so (x + u) * mask, upstream's form, is x + u * mask here. With a 0/1
mask the two are the same numbers, and the second never touches the fp32 residual twice. A block
or pair update called on its own expects masked s and z, as the trunk hands it.
"""

from __future__ import annotations

import ttnn

import torch

from .. import ops
from ..tenstorrent import (CORE_GRID_MAIN, Module, TriangleMultiplication, WeightScope,
                           _sdpa_program_config_for_lengths, bfp8_fidelity, download, fused_sdpa, lever,
                           lever_scope, model_levers, under_lever_scope, upload)

EPS = 1e-5
NEG_INF = -1e4


def _rms(x, w, ckc):
    """RMSNorm, with or without the elementwise weight (`elementwise_affine=False` upstream)."""
    if w is None:
        return ttnn.rms_norm(x, epsilon=EPS, compute_kernel_config=ckc)
    return ttnn.rms_norm(x, weight=w, epsilon=EPS, compute_kernel_config=ckc)


def _as(x, dtype):
    """`x` in `dtype`: the same handle when it already is, else a typecast copy the caller frees."""
    return x if x.dtype == dtype else ttnn.typecast(x, dtype)


def _residual(x, update, res_dtype, mask=None):
    """x + update * mask in the residual stream's dtype, freeing the update. `mask` is in the
    update's dtype, so the masking runs at the sublayer's width; None when the sublayer masked its
    own output already."""
    if mask is not None:
        masked = ttnn.multiply(update, mask)
        ttnn.deallocate(update)
        update = masked
    u = _as(update, res_dtype)
    if u is not update:
        ttnn.deallocate(update)
    out = ttnn.add(x, u)
    ttnn.deallocate(u)
    return out


def _masked_copy(x, mask, dtype):
    """x * mask in `dtype`, a new tensor; `x` survives."""
    xd = _as(x, dtype)
    out = ttnn.multiply(xd, mask)
    if xd is not x:
        ttnn.deallocate(xd)
    return out


class AdaptiveNorm(Module):
    """`AdaptiveLayerNorm`: norm(x) * sigmoid(W_g norm_c(c) + b_g) + W_b norm_c(c), then mask.

    `self.norm` has no elementwise weight; `norm_cond` does, and it is folded into both
    projections: W (w * rms0(c)) = (W diag(w)) rms0(c), where rms0 is the RMSNorm without a weight.
    So every adaptive norm of a pass reads the same rms0(cond), which the trunk computes once
    (`cn`) instead of 28 times (two per block).
    """

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        w = self.weights["norm_cond.weight"].to(torch.float32)
        self.gamma_w = self.torch_to_tt("to_gamma.0.weight", transform=lambda m: (m.float() * w).t(),
                                        dtype=self.dtype)
        self.gamma_b = self.torch_to_tt("to_gamma.0.bias", transform=lambda x: x, dtype=self.dtype)
        self.beta_w = self.torch_to_tt("to_beta.weight", transform=lambda m: (m.float() * w).t(), dtype=self.dtype)

    def __call__(self, x, cond, mask, cn=None):
        """`cn` is rms0(cond) when the caller already has it."""
        ckc = self.compute_kernel_config
        xn = _rms(x, None, ckc)
        own = cn is None
        if own:
            cn = _rms(cond, None, ckc)
        gamma = self._lin(cn, self.gamma_w, bias=self.gamma_b, dtype=self.dtype, activation="sigmoid")
        beta = self._lin(cn, self.beta_w, dtype=self.dtype)
        if own:
            ttnn.deallocate(cn)
        out = ttnn.add(ttnn.multiply(xn, gamma), beta)
        ttnn.deallocate(xn)
        ttnn.deallocate(gamma)
        ttnn.deallocate(beta)
        return ttnn.multiply(out, mask)


class AdaptiveScale(Module):
    """`AdaptiveOutputScale`: x * sigmoid(W c + b) * mask. The bias trains to -2, so this gates
    a freshly initialised block almost shut; at inference it is an ordinary per-channel gate. The
    sigmoid runs in the matmul's epilogue."""

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.w = self.torch_to_tt("to_adaln_zero_gamma.0.weight", dtype=self.dtype)
        self.b = self.torch_to_tt("to_adaln_zero_gamma.0.bias", transform=lambda x: x, dtype=self.dtype)

    def __call__(self, x, cond, mask):
        gamma = self._lin(cond, self.w, bias=self.b, dtype=self.dtype, activation="sigmoid")
        out = ttnn.multiply(ttnn.multiply(x, gamma), mask)
        ttnn.deallocate(gamma)
        return out


class Transition(Module):
    """`Transition` with `layer_norm=False`: linear_out(silu(gates) * x), where x and gates
    are the two halves of one [dim -> 2 * dim_inner] projection. Upstream masks the output; here
    the adaptive scale after it does, which multiplies by the same 0/1 mask.

    The projection is loaded as its two halves. Splitting a matmul over its OUTPUT columns gives
    each output element the same dot product in the same order, so this is the same arithmetic,
    and it saves slicing a [b, n, 2 * dim_inner] tensor in DRAM on every call. The silu runs in the
    gate matmul's epilogue. The output is not masked (see `AdaptiveScale`).
    """

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        inner = self.weights["swish_linear.0.weight"].shape[0] // 2
        self.fc_x = self.torch_to_tt("swish_linear.0.weight", transform=lambda w: w[:inner].t(), dtype=self.dtype)
        self.fc_g = self.torch_to_tt("swish_linear.0.weight", transform=lambda w: w[inner:].t(), dtype=self.dtype)
        self.out_w = self.torch_to_tt("linear_out.weight", dtype=self.dtype)

    def __call__(self, x):
        a = self._lin(x, self.fc_x, dtype=self.dtype)
        g = self._lin(x, self.fc_g, dtype=self.dtype, activation="silu")
        h = ttnn.multiply(g, a)
        ttnn.deallocate(a)
        ttnn.deallocate(g)
        out = self._lin(h, self.out_w, dtype=self.dtype)
        ttnn.deallocate(h)
        return out


class TransitionADALN(Module):
    """`TransitionADALN`: adaptive norm, transition, adaptive scale. The output is masked once, by
    the scale; upstream's second mask on it multiplies by the same 0/1 values."""

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.adaln = AdaptiveNorm(self.scope("adaln"), ckc, dtype=self.dtype)
        self.transition = Transition(self.scope("transition"), ckc, dtype=self.dtype)
        self.scale_output = AdaptiveScale(self.scope("scale_output"), ckc, dtype=self.dtype)

    def __call__(self, x, cond, mask, cn=None):
        h = self.adaln(x, cond, mask, cn)
        t = self.transition(h)
        ttnn.deallocate(h)
        out = self.scale_output(t, cond, mask)
        ttnn.deallocate(t)
        return out


def pair_biases(zb, w, mask_scaled, ckc, dtype):
    """[b, H, n, n] attention biases from the bf16 pair `zb` for every head whose folded projection
    is a column of `w` ([pair, H], see `PairBiasAttention.bias_w_host`), plus `mask_scaled`, the
    additive pair mask already times sqrt(head_dim) (or None).

    One call serves every block that reads the same z: the trunk stacks the two blocks between
    pair updates into one [pair, 2 * heads] projection, so the RMSNorm over the whole pair tensor
    and the permute run once per z instead of once per block."""
    zn = _rms(zb, None, ckc)
    bias = ttnn.linear(zn, w, compute_kernel_config=ckc, dtype=dtype, core_grid=CORE_GRID_MAIN)
    ttnn.deallocate(zn)
    out = ttnn.permute(bias, (0, 3, 1, 2))
    ttnn.deallocate(bias)
    if mask_scaled is not None:
        masked = ttnn.add(out, mask_scaled)
        ttnn.deallocate(out)
        out = masked
    return out


class PairBiasAttention(Module):
    """`PairBiasAttention`: RMSNorm, packed QKV with bias, optional QK norm over the packed head
    axis, pair-bias from RMSNorm(z), sigmoid gate, output projection.

    The QK norms run on the packed [b, n, heads * dim_head] tensor, before the head split, which is
    what upstream does -- normalising per head instead would be a different model. The normed q, k
    and v are then concatenated and split into heads by one tile-aware op, and the heads merged
    back by another, instead of a reshape and a permute each.

    Pair bias. ttnn's fused SDPA computes softmax((q.k + attn_mask) * scale), not
    q.k * scale + attn_mask: the mask goes through the scale as well. A bias handed over unscaled
    is silently shrunk by 1/sqrt(head_dim), which here costs rel_rms 0.69 against float64 and still
    looks like a plausible attention. RFD3's atom attention pre-scales for the same reason
    (`rfd3/model.py::_sdpa_bias`). Here the pre-scale and the pair norm's weight are both folded
    into the bias projection (`bias_w_host`), so the bias is one matmul on rms0(z); the mask's
    -1e4 fill is scaled with it by the caller and stays far below every real score.
    """

    def __init__(self, state_dict, ckc, n_heads: int, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.n_heads = n_heads
        inner = self.weights["to_qkv.weight"].shape[0] // 3
        self.inner = inner
        self.head_dim = inner // n_heads
        self.node_norm_w = self.torch_to_tt("node_norm.weight", transform=lambda x: x, dtype=self.dtype)
        self.q_w = self.torch_to_tt("to_qkv.weight", transform=lambda w: w[:inner].t(), dtype=self.dtype)
        self.k_w = self.torch_to_tt("to_qkv.weight", transform=lambda w: w[inner:2 * inner].t(), dtype=self.dtype)
        self.v_w = self.torch_to_tt("to_qkv.weight", transform=lambda w: w[2 * inner:].t(), dtype=self.dtype)
        self.q_b = self.torch_to_tt("to_qkv.bias", transform=lambda b: b[:inner], dtype=self.dtype)
        self.k_b = self.torch_to_tt("to_qkv.bias", transform=lambda b: b[inner:2 * inner], dtype=self.dtype)
        self.v_b = self.torch_to_tt("to_qkv.bias", transform=lambda b: b[2 * inner:], dtype=self.dtype)
        self.g_w = self.torch_to_tt("to_g.weight", dtype=self.dtype)
        self.g_b = self.torch_to_tt("to_g.bias", transform=lambda x: x, dtype=self.dtype)
        self.out_w = self.torch_to_tt("to_out_node.weight", dtype=self.dtype)
        self.out_b = self.torch_to_tt("to_out_node.bias", transform=lambda x: x, dtype=self.dtype)
        self.qk_norm = "q_layer_norm.weight" in self.weights
        if self.qk_norm:
            self.q_norm_w = self.torch_to_tt("q_layer_norm.weight", transform=lambda x: x, dtype=self.dtype)
            self.k_norm_w = self.torch_to_tt("k_layer_norm.weight", transform=lambda x: x, dtype=self.dtype)
        #: [pair, heads] fp32: sqrt(head_dim) * diag(pair_norm.weight) to_bias.weight^T.
        self.bias_w_host = (self.weights["to_bias.weight"].float().t()
                            * self.weights["pair_norm.weight"].float()[:, None] * self.head_dim ** 0.5)
        self._bias_w = None

    def pair_bias(self, z, pair_bias_mask):
        """This block's [b, heads, n, n] bias, pre-scaled, from the pair `z` and the unscaled
        additive mask. The trunk does not call this; it batches blocks through `pair_biases`."""
        if self._bias_w is None:
            self._bias_w = ttnn.from_torch(self.bias_w_host, dtype=self.dtype, layout=ttnn.TILE_LAYOUT,
                                           device=self.device)
        ms = None if pair_bias_mask is None else ttnn.multiply(pair_bias_mask, self.head_dim ** 0.5)
        out = pair_biases(z, self._bias_w, ms, self.compute_kernel_config, self.dtype)
        if ms is not None:
            ttnn.deallocate(ms)
        return out

    def __call__(self, x, z, pair_bias_mask, bias=None):
        """`bias` is this block's pre-scaled pair bias when the caller has it (it is then neither
        recomputed nor freed); otherwise it is built from `z` and `pair_bias_mask`."""
        ckc = self.compute_kernel_config
        a = _rms(x, self.node_norm_w, ckc)
        q = self._lin(a, self.q_w, bias=self.q_b, dtype=self.dtype)
        k = self._lin(a, self.k_w, bias=self.k_b, dtype=self.dtype)
        v = self._lin(a, self.v_w, bias=self.v_b, dtype=self.dtype)
        g = self._lin(a, self.g_w, bias=self.g_b, dtype=self.dtype, activation="sigmoid")
        ttnn.deallocate(a)
        if self.qk_norm:
            qn, kn = _rms(q, self.q_norm_w, ckc), _rms(k, self.k_norm_w, ckc)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = qn, kn
        qkv = ttnn.concat([q, k, v], dim=-1)
        for t in (q, k, v):
            ttnn.deallocate(t)
        q, k, v = self._split_heads(qkv, self.n_heads)
        own = bias is None
        if own:
            bias = self.pair_bias(z, pair_bias_mask)
        # The kernel config and the exact exponential are worth 3.19e-2 -> 2.05e-2 rel_rms against
        # float64 on the parity case: the default call runs the scores at a lower fidelity and an
        # approximate exp, and this attention's output is a near-cancelling average over the whole
        # token axis, which magnifies both.
        n = int(q.shape[2])
        o = fused_sdpa(q, k, v, attn_mask=bias, scale=self.head_dim ** -0.5,
                       compute_kernel_config=ckc,
                       program_config=_sdpa_program_config_for_lengths(n, n, site="rfd4_trunk",
                                                                      d=self.head_dim))
        for t in (q, k, v) + ((bias,) if own else ()):
            ttnn.deallocate(t)
        om = self._merge_heads(o)
        ttnn.deallocate(o)
        gated = ttnn.multiply(om, g)
        ttnn.deallocate(om)
        ttnn.deallocate(g)
        out = self._lin(gated, self.out_w, bias=self.out_b, dtype=self.dtype)
        ttnn.deallocate(gated)
        return out


class AttnADALN(Module):
    """`MultiHeadBiasedAttentionADALN_MM`: adaptive norm, pair-bias attention, adaptive scale. The
    scale masks the output, so upstream's second mask is not repeated."""

    def __init__(self, state_dict, ckc, n_heads: int, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.adaln = AdaptiveNorm(self.scope("adaln"), ckc, dtype=self.dtype)
        self.mha = PairBiasAttention(self.scope("mha"), ckc, n_heads, dtype=self.dtype)
        self.scale_output = AdaptiveScale(self.scope("scale_output"), ckc, dtype=self.dtype)

    def __call__(self, x, z, cond, mask, pair_bias_mask, cn=None, bias=None):
        """`cn` and `bias` are the trunk's per-pass shares (see `TrunkBlock`)."""
        h = self.adaln(x, cond, mask, cn)
        a = self.mha(h, z, pair_bias_mask, bias)
        ttnn.deallocate(h)
        out = self.scale_output(a, cond, mask)
        ttnn.deallocate(a)
        return out


class TrunkBlock(Module):
    """`MultiheadAttnAndTransition` with the release settings: sequential (not parallel) attention
    and transition, both residual, both masked."""

    def __init__(self, state_dict, ckc, n_heads: int = 16, dtype=None, residual_dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.residual_dtype = residual_dtype or self.dtype
        self.mhba = AttnADALN(self.scope("mhba"), ckc, n_heads, dtype=self.dtype)
        self.transition = TransitionADALN(self.scope("transition"), ckc, dtype=self.dtype)

    def __call__(self, x, z, cond, mask, pair_bias_mask, res_mask=None, cn=None, bias=None):
        """`x` is the masked residual stream in `residual_dtype`; the sublayers read a `dtype` copy
        of it, as autocast hands a matmul bf16 inputs. Both sublayers mask their own output, so the
        residual adds need no mask (`res_mask` is accepted for the old call shape and unused).
        `cn` (rms0(cond)) and `bias` (this block's pre-scaled pair bias) are the trunk's per-pass
        shares; without them the block builds its own. Does not free `x`."""
        given = x
        for sub in (lambda h: self.mhba(h, z, cond, mask, pair_bias_mask, cn, bias),
                    lambda h: self.transition(h, cond, mask, cn)):
            h = _as(x, self.dtype)
            u = sub(h)
            if h is not x:
                ttnn.deallocate(h)
            x_new = _residual(x, u, self.residual_dtype)
            if x is not given:
                ttnn.deallocate(x)
            x = x_new
        return x



def _trimul_weights(scope) -> WeightScope:
    """An openfold triangle multiplication's weights under the keys tt-bio's
    `TriangleMultiplication` reads (Boltz's layout: packed [a; b] in-projections, norm_in/out,
    p_out/g_out). The release checkpoint has no biases (`tri_mult_bias: false`); its LayerNorms
    get zero biases, which is what a bias-free LayerNorm computes."""
    w = scope.data
    cat = lambda a, b: torch.cat([w[a], w[b]])  # noqa: E731
    out = {
        "norm_in.weight": w["layer_norm_in.weight"],
        "norm_in.bias": w.get("layer_norm_in.bias", torch.zeros_like(w["layer_norm_in.weight"])),
        "norm_out.weight": w["layer_norm_out.weight"],
        "norm_out.bias": w.get("layer_norm_out.bias", torch.zeros_like(w["layer_norm_out.weight"])),
        "p_in.weight": cat("linear_a_p.weight", "linear_b_p.weight"),
        "g_in.weight": cat("linear_a_g.weight", "linear_b_g.weight"),
        "p_out.weight": w["linear_z.weight"],
        "g_out.weight": w["linear_g.weight"],
    }
    if "linear_a_p.bias" in w:
        out["p_in.bias"] = cat("linear_a_p.bias", "linear_b_p.bias")
        out["g_in.bias"] = cat("linear_a_g.bias", "linear_b_g.bias")
    for src, dst in (("linear_z.bias", "p_out.bias"), ("linear_g.bias", "g_out.bias")):
        if src in w:
            out[dst] = w[src]
    return WeightScope(out)


def _per_design(tri, z, pair_mask):
    """A triangle multiplication on each design of a [b, n, n, c] pair stack in turn.

    The op prices its fused path (T1) on the whole leading axis, so at n = 128 a batch of four is
    refused it and runs the chunked channel loop instead: 42 ops for 10, slower, and a different
    rounding, so a design came out depending on its batch. One design at a time keeps every design
    on the path it takes alone, bit for bit."""
    b = int(z.shape[0])
    if b == 1:
        return tri(z, pair_mask)
    n, c = int(z.shape[1]), int(z.shape[-1])
    outs = []
    for k in range(b):
        zk = ttnn.slice(z, [k, 0, 0, 0], [k + 1, n, n, c])
        mk = None if pair_mask is None else ttnn.slice(pair_mask, [k, 0, 0], [k + 1, n, n])
        out = tri(zk, mk)
        ttnn.deallocate(zk)
        if mk is not None:
            ttnn.deallocate(mk)
        if out.memory_config().buffer_type == ttnn.BufferType.L1:
            # An L1 result left there would sit under the next design's circular buffers: the
            # trimul would clash, retry narrower and stay on the slower path for every design.
            out_dram = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(out)
            out = out_dram
        outs.append(out)
    out = ttnn.concat(outs, dim=0)
    for t in outs:
        ttnn.deallocate(t)
    return out

class PairUpdate(Module):
    """`PairReprUpdate` with the release settings: a Hadamard single-to-pair write, both triangle
    multiplications (tt-bio's `TriangleMultiplication`, the one Boltz-2, OpenFold3 and AF2 run) and
    the ReLU pair transition, each a masked residual.

    `pair_mask` is the [b, n, n] outer product of the token mask, which the triangle
    multiplication takes; `pair_mask_c` is the same mask as [b, n, n, 1] for the residual masks.
    Build both once per stack call.
    """

    def __init__(self, state_dict, ckc, dtype=None, pair_write="hadamard", residual_dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.residual_dtype = residual_dtype or self.dtype
        self.pair_write = pair_write
        ident = lambda x: x  # noqa: E731
        self.ln_w = self.torch_to_tt("layer_norm_in.weight", transform=ident, dtype=self.dtype)
        self.ln_b = self.torch_to_tt("layer_norm_in.bias", transform=ident, dtype=self.dtype)
        p = self.weights["linear_x.weight"].shape[0] // 2
        self.xi_w = self.torch_to_tt("linear_x.weight", transform=lambda w: w[:p].t(), dtype=self.dtype)
        self.xj_w = self.torch_to_tt("linear_x.weight", transform=lambda w: w[p:].t(), dtype=self.dtype)
        self.tri_out = TriangleMultiplication(False, _trimul_weights(self.scope("tri_mult_out")), ckc)
        self.tri_in = TriangleMultiplication(True, _trimul_weights(self.scope("tri_mult_in")), ckc)
        t = "transition_out"
        # transition_b8: both weights and the hidden activation in bfp8. The linears write their
        # output, so the hidden's bytes are most of their time (fc1 at 128 tokens 0.407 -> 0.240 ms,
        # perf/rfd4/bfp8/sites.py); fc2 then reads two bfp8 operands at HiFi2, the same bits.
        b8 = lever("transition_b8")
        self.tr_hidden_dtype = ttnn.bfloat8_b if b8 else self.dtype
        self.tr_ln_w = self.torch_to_tt(f"{t}.layer_norm.weight", transform=ident, dtype=self.dtype)
        self.tr_ln_b = self.torch_to_tt(f"{t}.layer_norm.bias", transform=ident, dtype=self.dtype)
        self.tr_w1 = self.torch_to_tt(f"{t}.linear_1.weight", dtype=self.tr_hidden_dtype)
        self.tr_b1 = self.torch_to_tt(f"{t}.linear_1.bias", transform=ident, dtype=self.dtype)
        self.tr_w2 = self.torch_to_tt(f"{t}.linear_2.weight", dtype=self.tr_hidden_dtype)
        self.tr_b2 = self.torch_to_tt(f"{t}.linear_2.bias", transform=ident, dtype=self.dtype)

    def _write(self, x, mask):
        """pi[b, j] * pj[b, i] (or pi[b, j] + pj[b, i]) for every pair (i, j): [b, n, n, pair].

        The Hadamard write comes back masked: its factors are masked per token, which is the pair
        mask's mi * mj, so the [n, n, pair] product never needs a mask of its own. Returns
        `(update, masked)`."""
        ckc = self.compute_kernel_config
        h = ttnn.layer_norm(ttnn.multiply(x, mask), weight=self.ln_w, bias=self.ln_b, epsilon=EPS,
                            compute_kernel_config=ckc)
        pi = self._lin(h, self.xi_w, dtype=self.dtype)
        pj = self._lin(h, self.xj_w, dtype=self.dtype)
        ttnn.deallocate(h)
        hadamard = self.pair_write == "hadamard"
        if hadamard:
            pi, pj = ttnn.multiply(pi, mask), ttnn.multiply(pj, mask)
        b, n, c = (int(d) for d in pi.shape)
        pi4 = ttnn.reshape(pi, (b, 1, n, c))
        pj4 = ttnn.reshape(pj, (b, n, 1, c))
        op = ttnn.multiply if hadamard else ttnn.add
        return op(pj4, pi4), hadamard

    def _transition(self, z, pair_mask_c):
        ckc = self.compute_kernel_config
        zn = ttnn.layer_norm(z, weight=self.tr_ln_w, bias=self.tr_ln_b, epsilon=EPS, compute_kernel_config=ckc)
        h = self._lin(zn, self.tr_w1, bias=self.tr_b1, dtype=self.tr_hidden_dtype, activation="relu")
        ttnn.deallocate(zn)
        out = ops.linear(h, self.tr_w2, bias=self.tr_b2, compute_kernel_config=bfp8_fidelity(ckc, h, self.tr_w2),
                         dtype=self.dtype, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(h)
        return ttnn.multiply(out, pair_mask_c)

    def __call__(self, x, z, mask, pair_mask, pair_mask_c, res_mask_c=None):
        """`z` is the masked pair stream in `residual_dtype` (not freed); the triangle
        multiplications and the transition read a `dtype` copy of it, the write reads only x.
        The triangle updates are masked with `pair_mask_c` before they are added; the write and the
        transition come back masked. `res_mask_c` is accepted for the old call shape and unused."""
        xs = _as(x, self.dtype)
        write, write_masked = self._write(xs, mask)
        if xs is not x:
            ttnn.deallocate(xs)
        z = _residual(z, write, self.residual_dtype, None if write_masked else pair_mask_c)
        for sub, own_mask in ((lambda h: _per_design(self.tri_out, h, pair_mask), False),
                              (lambda h: _per_design(self.tri_in, h, pair_mask), False),
                              (lambda h: self._transition(h, pair_mask_c), True)):
            h = _as(z, self.dtype)
            u = sub(h)
            if h is not z:
                ttnn.deallocate(h)
            z_new = _residual(z, u, self.residual_dtype, None if own_mask else pair_mask_c)
            ttnn.deallocate(z)
            z = z_new
        return z


def trunk_masks(mask, device, dtype=ttnn.bfloat16, residual_dtype=None):
    """The device masks one trunk call reads, from a host [b, n] 0/1 token mask: the token mask
    [b, n, 1], the attention's additive [b, 1, n, n] pair mask, the pair mask as [b, n, n]
    (triangle multiplication) and [b, n, n, 1], then the token and [b, n, n, 1] pair masks again in
    `residual_dtype` for the residual streams (the same handles when that is `dtype`). Built once
    per stack call.

    The triangle multiplication's mask is None when every token is real (no padding, every design
    of a batch the same length): it only multiplies operand `a` by the pair mask, and multiplying by
    1 changes no bit, so the op is skipped rather than run. 0.76 ms a call at 512 tokens, 0.027 at
    128 (perf/rfd4/tri2/bench.py `mask:none`, bit-identical)."""
    m = mask.to(torch.float32)
    pair = m[:, :, None] * m[:, None, :]
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tok, pair_c = up(m[..., None]), up(pair[..., None])
    res = residual_dtype or dtype
    if res == dtype:
        res_tok, res_pair_c = tok, pair_c
    else:
        res_tok, res_pair_c = ttnn.typecast(tok, res), ttnn.typecast(pair_c, res)
    return (tok, up(torch.where(pair.bool(), 0.0, NEG_INF)[:, None]), None if bool(m.all()) else up(pair),
            pair_c, res_tok, res_pair_c)


#: RFD4's levers per mode (`tenstorrent.model_levers`); TT_BIO_LEVERS overrides both when a harness grades a
#: set. `trimul_t1` runs the triangle multiplication's input layer_norm, gated projections and channel moves as
#: one kernel from 160 tokens (it declines below, where the trimul runs from L1): 15.0 -> 11.3 ms a call at 512
#: tokens, same float64 error (state/rfd4/bets/tri2.md).
NORMAL_LEVERS = frozenset({"trimul_t1"})
#: `--fast` adds the precision levers. `pair_mm` + `pair_mm_tr` put the pair transition on minimal_matmul
#: (bf16, same float64 error, not the same bits); `transition_b8` makes its weights and hidden bfp8;
#: `trimul_gin` + `trimul_gin_w8` give the triangle multiplication's in-projection bfp8 resident weights,
#: 15.4 -> 13.9 ms a call at 512 tokens and inert (bit-identical) at 128 (state/rfd4/bets/bfp8.md).
FAST_LEVERS = NORMAL_LEVERS | {"pair_mm", "pair_mm_tr", "transition_b8", "trimul_gin", "trimul_gin_w8"}


class Trunk(Module):
    """`LocalLatentsTransformer._run_trunk` in inference: `nlayers` trunk blocks with a pair update
    after every `every_n`-th one but the last, run once per recycle.

    Each cycle starts from the initial states plus a projection of the previous cycle's,
    s_init + W_s rms(s) and z_init + W_z rms(z). On the first cycle the previous states are zero, so
    the projection is exactly zero (rms(0) = 0) and the cycle starts from the initial states as
    they are; nothing the model computes is skipped.

    `residual_dtype=ttnn.float32` keeps s and z in fp32 between sublayers, as the release
    inference config's `precision: bf16-mixed` does on a GPU (autocast runs the matmuls in bf16
    and leaves the residual adds in fp32). Over 6 recycles x 12 pair updates an all-bf16 pair stream
    drifts far from float64 even when every op is at its bf16 floor; see the parity test.
    """

    def __init__(self, state_dict, ckc, nlayers: int = 24, n_heads: int = 16, every_n: int = 2,
                 dtype=None, residual_dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.residual_dtype = residual_dtype or self.dtype
        self.nlayers = nlayers
        self._levers = model_levers(normal=NORMAL_LEVERS, fast=FAST_LEVERS)
        with lever_scope(self._levers):   # a lever that picks a stored dtype is read at build
            self.blocks = [TrunkBlock(self.scope(f"transformer_layers.{i}"), ckc, n_heads, dtype=self.dtype,
                                      residual_dtype=self.residual_dtype)
                           for i in range(nlayers)]
            self.pair_updates = {i: PairUpdate(self.scope(f"pair_update_layers.{i}"), ckc, dtype=self.dtype,
                                               residual_dtype=self.residual_dtype)
                                 for i in range(nlayers - 1) if i % every_n == 0}
        ident = lambda x: x  # noqa: E731
        self.s_norm = self.torch_to_tt("s_cycle_norm.weight", transform=ident, dtype=self.dtype)
        self.s_proj = self.torch_to_tt("s_cycle_proj.weight", dtype=self.dtype)
        self.z_norm = self.torch_to_tt("z_cycle_norm.weight", transform=ident, dtype=self.dtype)
        self.z_proj = self.torch_to_tt("z_cycle_proj.weight", dtype=self.dtype)
        # The blocks between two pair updates read the same z, so their pair biases are one
        # stacked projection: [[0], [1, 2], [3, 4], ..., [23]] at every_n 2.
        self.bias_groups, group = [], []
        for i in range(nlayers):
            group.append(i)
            if i in self.pair_updates or i == nlayers - 1:
                self.bias_groups.append(group)
                group = []
        mhas = [blk.mhba.mha for blk in self.blocks]
        self.n_heads, self.head_dim = n_heads, mhas[0].head_dim
        self.bias_ws = [ttnn.from_torch(torch.cat([mhas[i].bias_w_host for i in g], dim=1), dtype=self.dtype,
                                        layout=ttnn.TILE_LAYOUT, device=self.device)
                        for g in self.bias_groups]
        self._group_at = {g[0]: k for k, g in enumerate(self.bias_groups)}

    def _group_biases(self, k, zb, mask_scaled):
        """The pre-scaled [b, heads, n, n] pair bias of every block in `bias_groups[k]`."""
        both = pair_biases(zb, self.bias_ws[k], mask_scaled, self.compute_kernel_config, self.dtype)
        g = self.bias_groups[k]
        if len(g) == 1:
            return [both]
        b, _, n, m = (int(d) for d in both.shape)
        h = self.n_heads
        out = [ttnn.slice(both, (0, j * h, 0, 0), (b, (j + 1) * h, n, m)) for j in range(len(g))]
        ttnn.deallocate(both)
        return out

    def _recycle(self, init, prev, norm_w, proj_w):
        h = _rms(prev, norm_w, self.compute_kernel_config)
        hb = _as(h, self.dtype)
        if hb is not h:
            ttnn.deallocate(h)
        p = self._lin(hb, proj_w, dtype=self.dtype)
        ttnn.deallocate(hb)
        pr = _as(p, self.residual_dtype)
        if pr is not p:
            ttnn.deallocate(p)
        out = ttnn.add(init, pr)
        ttnn.deallocate(pr)
        return out

    def run_pass(self, s, z, cond, masks):
        """One trunk pass. Consumes `s` and `z` (frees them) and returns the new pair."""
        mask, bias_mask, pair_mask, pair_mask_c, res_mask, res_pair_mask_c = masks
        ckc = self.compute_kernel_config
        cn = _rms(cond, None, ckc)  # every adaptive norm of the pass reads rms0(cond)
        mask_scaled = None if bias_mask is None else ttnn.multiply(bias_mask, self.head_dim ** 0.5)
        zb = _as(z, self.dtype)  # the attention's pair bias reads z; cast once per pair update
        biases = []
        for i, block in enumerate(self.blocks):
            if i in self._group_at:
                biases = self._group_biases(self._group_at[i], zb, mask_scaled)
            bias = biases.pop(0)
            s_new = block(s, zb, cond, mask, bias_mask, res_mask, cn=cn, bias=bias)
            ttnn.deallocate(bias)
            ttnn.deallocate(s)
            s = s_new
            pu = self.pair_updates.get(i)
            if pu is not None:
                z_new = pu(s, z, mask, pair_mask, pair_mask_c, res_pair_mask_c)
                if zb is not z:
                    ttnn.deallocate(zb)
                ttnn.deallocate(z)
                z = z_new
                zb = _as(z, self.dtype)
        if zb is not z:
            ttnn.deallocate(zb)
        ttnn.deallocate(cn)
        if mask_scaled is not None:
            ttnn.deallocate(mask_scaled)
        return s, z

    @under_lever_scope
    def __call__(self, s_init, z_init, conds, masks):
        """`conds[k]` is cycle k's conditioning ([b, n, dim_cond]); upstream recomputes it each cycle
        because it carries the remaining-pass count. `masks` is `trunk_masks(...)` built with this
        trunk's residual dtype. Returns the final (s, z) in the residual dtype; the inputs survive."""
        res_mask, res_pair_mask_c = masks[4], masks[5]
        # Masked once here, in the residual dtype; every update after this keeps them masked.
        s_init = _masked_copy(s_init, res_mask, self.residual_dtype)
        z_init = _masked_copy(z_init, res_pair_mask_c, self.residual_dtype)
        s = z = None
        for c in conds:
            if s is None:
                s_in, z_in = ttnn.clone(s_init), ttnn.clone(z_init)
            else:
                s_in = self._recycle(s_init, s, self.s_norm, self.s_proj)
                z_in = self._recycle(z_init, z, self.z_norm, self.z_proj)
                ttnn.deallocate(s)
                ttnn.deallocate(z)
            s, z = self.run_pass(s_in, z_in, c, masks)
        ttnn.deallocate(s_init)
        ttnn.deallocate(z_init)
        return s, z

    def masks_for(self, mask, device):
        """`trunk_masks` for a padded host [b, n] mask, kept on the device while the mask stays the
        same: a design reads one mask on all of its sampler steps."""
        cached = getattr(self, "_masks", None)
        if cached is not None and cached[0] is device and torch.equal(cached[1], mask):
            return cached[2]
        if cached is not None and cached[0] is device:
            for t in {id(t): t for t in cached[2] if t is not None}.values():
                if t.is_allocated():
                    ttnn.deallocate(t)
        masks = trunk_masks(mask, device, dtype=self.dtype, residual_dtype=self.residual_dtype)
        self._masks = (device, mask.clone(), masks)
        return masks

    def run_bucketed(self, s_init, z_init, conds, mask, device, pair=True):
        """Host entry: s_init [b, N, token], z_init [b, N, N, pair], conds [b, N, cond] per cycle and the
        [b, N] 0/1 token mask as torch tensors. Pads the token axis to the fleet bucket
        (`token_axis.bucket_multiple("rfd4")`, 32), masks the padding, runs and slices back to N.
        `pair=False` returns `(s, None)` and skips reading the [b, N, N, pair] fp32 pair back, which
        the flow network, reading only s, does not need.
        Padded tokens are masked tokens, which upstream does to a short sequence in a batch
        (tests/test_rfd4_trunk_parity.py::test_padding_does_not_touch_the_real_tokens). rfd4 gets
        its `token_axis.TOKEN_AXIS` row when it ships as a CLI model; that table admits shipped
        models only."""
        from .. import token_axis
        n = int(mask.shape[1])
        npad = token_axis.pad_amount(n, token_axis.bucket_multiple("rfd4")) if token_axis.bucket_enabled() else 0

        def pad(t, dims):
            for d in dims:
                t = torch.nn.functional.pad(t, [0, 0] * (t.dim() - 1 - d) + [0, npad])
            return t

        def up(t, dtype):
            return upload(t.to(torch.float32).contiguous(), dtype=dtype, device=device)

        m = pad(mask.to(torch.float32), [1])
        masks = self.masks_for(m, device)
        mc = m[..., None]
        s = up(pad(s_init.float(), [1]) * mc, self.residual_dtype)
        z = up(pad(z_init.float(), [1, 2]) * (mc[:, :, None] * mc[:, None, :]), self.residual_dtype)
        cs = [up(pad(c.float(), [1]), self.dtype) for c in conds]
        s_out, z_out = self(s, z, cs, masks)
        out = download(s_out)[:, :n], download(z_out)[:, :n, :n] if pair else None
        for t in (s, z, s_out, z_out, *cs):
            if t.is_allocated():
                ttnn.deallocate(t)
        return out


def compute_kernel_config():
    """The config the rest of tt-bio's trunks run on: HiFi4, fp32 dest, packer L1 accumulate."""
    from ..rfd3.model import _default_compute_kernel_config
    return _default_compute_kernel_config()
