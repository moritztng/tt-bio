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
"""

from __future__ import annotations

import ttnn

from ..tenstorrent import CORE_GRID_MAIN, Module, _sdpa_program_config_for_lengths, fused_sdpa

EPS = 1e-5
NEG_INF = -1e4


def _rms(x, w, ckc):
    """RMSNorm, with or without the elementwise weight (`elementwise_affine=False` upstream)."""
    if w is None:
        return ttnn.rms_norm(x, epsilon=EPS, compute_kernel_config=ckc)
    return ttnn.rms_norm(x, weight=w, epsilon=EPS, compute_kernel_config=ckc)


class AdaptiveNorm(Module):
    """`AdaptiveLayerNorm`: norm(x) * sigmoid(W_g norm_c(c) + b_g) + W_b norm_c(c), then mask.

    `self.norm` has no elementwise weight; `norm_cond` does.
    """

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.norm_cond_w = self.torch_to_tt("norm_cond.weight", transform=lambda x: x, dtype=self.dtype)
        self.gamma_w = self.torch_to_tt("to_gamma.0.weight", dtype=self.dtype)
        self.gamma_b = self.torch_to_tt("to_gamma.0.bias", transform=lambda x: x, dtype=self.dtype)
        self.beta_w = self.torch_to_tt("to_beta.weight", dtype=self.dtype)

    def __call__(self, x, cond, mask):
        ckc = self.compute_kernel_config
        xn = _rms(x, None, ckc)
        cn = _rms(cond, self.norm_cond_w, ckc)
        gamma = ttnn.sigmoid(self._lin(cn, self.gamma_w, bias=self.gamma_b, dtype=self.dtype))
        beta = self._lin(cn, self.beta_w, dtype=self.dtype)
        ttnn.deallocate(cn)
        out = ttnn.add(ttnn.multiply(xn, gamma), beta)
        ttnn.deallocate(xn)
        ttnn.deallocate(gamma)
        ttnn.deallocate(beta)
        return ttnn.multiply(out, mask)


class AdaptiveScale(Module):
    """`AdaptiveOutputScale`: x * sigmoid(W c + b) * mask. The bias trains to -2, so this gates
    a freshly initialised block almost shut; at inference it is an ordinary per-channel gate."""

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.w = self.torch_to_tt("to_adaln_zero_gamma.0.weight", dtype=self.dtype)
        self.b = self.torch_to_tt("to_adaln_zero_gamma.0.bias", transform=lambda x: x, dtype=self.dtype)

    def __call__(self, x, cond, mask):
        gamma = ttnn.sigmoid(self._lin(cond, self.w, bias=self.b, dtype=self.dtype))
        out = ttnn.multiply(ttnn.multiply(x, gamma), mask)
        ttnn.deallocate(gamma)
        return out


class Transition(Module):
    """`Transition` with `layer_norm=False`: linear_out(silu(gates) * x) * mask, where x and gates
    are the two halves of one [dim -> 2 * dim_inner] projection.

    The projection is loaded as its two halves. Splitting a matmul over its OUTPUT columns gives
    each output element the same dot product in the same order, so this is the same arithmetic,
    and it saves slicing a [b, n, 2 * dim_inner] tensor in DRAM on every call.
    """

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        inner = self.weights["swish_linear.0.weight"].shape[0] // 2
        self.fc_x = self.torch_to_tt("swish_linear.0.weight", transform=lambda w: w[:inner].t(), dtype=self.dtype)
        self.fc_g = self.torch_to_tt("swish_linear.0.weight", transform=lambda w: w[inner:].t(), dtype=self.dtype)
        self.out_w = self.torch_to_tt("linear_out.weight", dtype=self.dtype)

    def __call__(self, x, mask):
        a = self._lin(x, self.fc_x, dtype=self.dtype)
        g = self._lin(x, self.fc_g, dtype=self.dtype)
        h = ttnn.multiply(ttnn.silu(g), a)
        ttnn.deallocate(a)
        ttnn.deallocate(g)
        out = self._lin(h, self.out_w, dtype=self.dtype)
        ttnn.deallocate(h)
        return ttnn.multiply(out, mask)


class TransitionADALN(Module):
    """`TransitionADALN`: adaptive norm, transition, adaptive scale."""

    def __init__(self, state_dict, ckc, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.adaln = AdaptiveNorm(self.scope("adaln"), ckc, dtype=self.dtype)
        self.transition = Transition(self.scope("transition"), ckc, dtype=self.dtype)
        self.scale_output = AdaptiveScale(self.scope("scale_output"), ckc, dtype=self.dtype)

    def __call__(self, x, cond, mask):
        h = self.adaln(x, cond, mask)
        t = self.transition(h, mask)
        ttnn.deallocate(h)
        out = self.scale_output(t, cond, mask)
        ttnn.deallocate(t)
        return ttnn.multiply(out, mask)


class PairBiasAttention(Module):
    """`PairBiasAttention`: RMSNorm, packed QKV with bias, optional QK norm over the packed head
    axis, pair-bias from RMSNorm(z), sigmoid gate, output projection.

    The QK norms run on the packed [b, n, heads * dim_head] tensor, before the head split, which is
    what upstream does -- normalising per head instead would be a different model.
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
        self.pair_norm_w = self.torch_to_tt("pair_norm.weight", transform=lambda x: x, dtype=self.dtype)
        self.bias_w = self.torch_to_tt("to_bias.weight", dtype=self.dtype)

    def _heads(self, t):
        """[b, n, inner] -> [b, heads, n, head_dim]."""
        b, n = int(t.shape[0]), int(t.shape[1])
        t = ttnn.reshape(t, (b, n, self.n_heads, self.head_dim))
        out = ttnn.permute(t, (0, 2, 1, 3))
        ttnn.deallocate(t)
        return out

    def pair_bias(self, z, pair_bias_mask):
        """[b, heads, n, n] additive attention bias from the pair tensor, PRE-SCALED by
        sqrt(head_dim).

        ttnn's fused SDPA computes softmax((q.k + attn_mask) * scale), not q.k * scale + attn_mask:
        the mask goes through the scale as well. A bias handed over unscaled is silently shrunk by
        1/sqrt(head_dim), which here costs rel_rms 0.69 against float64 and still looks like a
        plausible attention. RFD3's atom attention pre-scales for the same reason
        (`rfd3/model.py::_sdpa_bias`). The mask's -1e4 fill is scaled with it and stays far below
        every real score."""
        ckc = self.compute_kernel_config
        zn = _rms(z, self.pair_norm_w, ckc)
        bias = ttnn.linear(zn, self.bias_w, compute_kernel_config=ckc, dtype=self.dtype,
                           core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(zn)
        out = ttnn.permute(bias, (0, 3, 1, 2))
        ttnn.deallocate(bias)
        if pair_bias_mask is not None:
            masked = ttnn.add(out, pair_bias_mask)
            ttnn.deallocate(out)
            out = masked
        scaled = ttnn.multiply(out, self.head_dim ** 0.5)
        ttnn.deallocate(out)
        return scaled

    def __call__(self, x, z, pair_bias_mask):
        ckc = self.compute_kernel_config
        a = _rms(x, self.node_norm_w, ckc)
        q = self._lin(a, self.q_w, bias=self.q_b, dtype=self.dtype)
        k = self._lin(a, self.k_w, bias=self.k_b, dtype=self.dtype)
        v = self._lin(a, self.v_w, bias=self.v_b, dtype=self.dtype)
        g = self._lin(a, self.g_w, bias=self.g_b, dtype=self.dtype)
        ttnn.deallocate(a)
        if self.qk_norm:
            qn, kn = _rms(q, self.q_norm_w, ckc), _rms(k, self.k_norm_w, ckc)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = qn, kn
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
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
        for t in (q, k, v, bias):
            ttnn.deallocate(t)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (int(o.shape[0]), int(o.shape[1]), self.inner))
        gated = ttnn.multiply(o, ttnn.sigmoid(g))
        ttnn.deallocate(o)
        ttnn.deallocate(g)
        out = self._lin(gated, self.out_w, bias=self.out_b, dtype=self.dtype)
        ttnn.deallocate(gated)
        return out


class AttnADALN(Module):
    """`MultiHeadBiasedAttentionADALN_MM`: adaptive norm, pair-bias attention, adaptive scale."""

    def __init__(self, state_dict, ckc, n_heads: int, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.adaln = AdaptiveNorm(self.scope("adaln"), ckc, dtype=self.dtype)
        self.mha = PairBiasAttention(self.scope("mha"), ckc, n_heads, dtype=self.dtype)
        self.scale_output = AdaptiveScale(self.scope("scale_output"), ckc, dtype=self.dtype)

    def __call__(self, x, z, cond, mask, pair_bias_mask):
        h = self.adaln(x, cond, mask)
        a = self.mha(h, z, pair_bias_mask)
        ttnn.deallocate(h)
        out = self.scale_output(a, cond, mask)
        ttnn.deallocate(a)
        return ttnn.multiply(out, mask)


class TrunkBlock(Module):
    """`MultiheadAttnAndTransition` with the release settings: sequential (not parallel) attention
    and transition, both residual, both masked."""

    def __init__(self, state_dict, ckc, n_heads: int = 16, dtype=None):
        super().__init__(state_dict, ckc)
        self.dtype = dtype or ttnn.bfloat16
        self.mhba = AttnADALN(self.scope("mhba"), ckc, n_heads, dtype=self.dtype)
        self.transition = TransitionADALN(self.scope("transition"), ckc, dtype=self.dtype)

    def __call__(self, x, z, cond, mask, pair_bias_mask):
        x = ttnn.multiply(x, mask)
        a = self.mhba(x, z, cond, mask, pair_bias_mask)
        x = ttnn.multiply(ttnn.add(a, x), mask)
        ttnn.deallocate(a)
        t = self.transition(x, cond, mask)
        x = ttnn.multiply(ttnn.add(t, x), mask)
        ttnn.deallocate(t)
        return ttnn.multiply(x, mask)


def compute_kernel_config():
    """The config the rest of tt-bio's trunks run on: HiFi4, fp32 dest, packer L1 accumulate."""
    from ..rfd3.model import _default_compute_kernel_config
    return _default_compute_kernel_config()
