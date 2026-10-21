"""Faithful host reference for the RFD4-Proteina flow-transformer trunk block.

Transcribed from the pinned upstream (`state/rfd4/UPSTREAM.json`, 2db07bb4):
`rfproteina/nn/modules/{adaptive_ln_scale,swiglu,seq_transition_af3,pair_bias_attn,
attn_n_transition}.py` with the released flow config -- `norm_type: rmsnorm_torch_fused`
(so `make_norm` is `torch.nn.RMSNorm`, eps 1e-5), `use_qkln: true`, `parallel_mha_transition:
false`, both residuals on, expansion_factor 4.

It is plain torch with no rfproteina import, so it runs in any dtype: the parity test grades the
card against this in float64, which the charter's hard stop asks for (never against another
approximation). Passing `rnd=bf16` rounds every op boundary the way the card does, which gives the
same chain's arithmetic floor to compare a device number against -- a device result at that floor
is as good as this precision gets, whatever the absolute number looks like. The weight keys are upstream's, so the ttnn port's weight remap is graded too.
"""

from __future__ import annotations

import math

import torch

NEG_INF = -1e4
EPS = 1e-5


def bf16(t):
    """The value bf16 holds, back in float64: the rounding the card does at every op boundary."""
    return t.to(torch.bfloat16).double()


def ident(t):
    return t


def rms_norm(x, weight, eps=EPS):
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out if weight is None else out * weight


def adaptive_norm(w, p, x, cond, mask, rnd=ident):
    """`AdaptiveLayerNorm.forward` for a [b, n, dim_cond] conditioning tensor."""
    normed = rnd(rms_norm(x, None))
    c = rnd(rms_norm(cond, w[f"{p}.norm_cond.weight"]))
    gamma = rnd(torch.sigmoid(rnd(c @ w[f"{p}.to_gamma.0.weight"].T + w[f"{p}.to_gamma.0.bias"])))
    beta = rnd(c @ w[f"{p}.to_beta.weight"].T)
    return rnd(rnd(normed * gamma) + beta) * mask[..., None]


def adaptive_scale(w, p, x, cond, mask, rnd=ident):
    """`AdaptiveOutputScale.forward`."""
    k = f"{p}.to_adaln_zero_gamma.0"
    gamma = rnd(torch.sigmoid(rnd(cond @ w[f"{k}.weight"].T + w[f"{k}.bias"])))
    return rnd(x * gamma) * mask[..., None]


def transition(w, p, x, mask, rnd=ident):
    """`Transition.forward` with `layer_norm=False`: SwiGLU, then the output projection."""
    h = rnd(x @ w[f"{p}.swish_linear.0.weight"].T)
    a, gates = h.chunk(2, dim=-1)
    out = rnd(rnd(rnd(torch.nn.functional.silu(gates)) * a) @ w[f"{p}.linear_out.weight"].T)
    return out * mask[..., None]


def transition_adaln(w, p, x, cond, mask, rnd=ident):
    """`TransitionADALN.forward`."""
    h = adaptive_norm(w, f"{p}.adaln", x, cond, mask, rnd)
    h = transition(w, f"{p}.transition", h, mask, rnd)
    h = adaptive_scale(w, f"{p}.scale_output", h, cond, mask, rnd)
    return h * mask[..., None]


def pair_bias_attention(w, p, x, z, pair_mask, n_heads, rnd=ident):
    """`PairBiasAttention.forward` (the CPU einsum branch, which is the exact one)."""
    inner = w[f"{p}.to_qkv.weight"].shape[0] // 3
    head_dim = inner // n_heads
    a = rnd(rms_norm(x, w[f"{p}.node_norm.weight"]))
    qkv = rnd(a @ w[f"{p}.to_qkv.weight"].T + w[f"{p}.to_qkv.bias"])
    q, k, v = qkv.chunk(3, dim=-1)
    if f"{p}.q_layer_norm.weight" in w:
        q = rnd(rms_norm(q, w[f"{p}.q_layer_norm.weight"]))
        k = rnd(rms_norm(k, w[f"{p}.k_layer_norm.weight"]))
    g = rnd(a @ w[f"{p}.to_g.weight"].T + w[f"{p}.to_g.bias"])
    b, n = x.shape[0], x.shape[1]

    def heads(t):
        return t.reshape(b, n, n_heads, head_dim).permute(0, 2, 1, 3)

    q, k, v, g = heads(q), heads(k), heads(v), heads(g)
    zn = rnd(rms_norm(z, w[f"{p}.pair_norm.weight"]))
    bias = rnd(zn @ w[f"{p}.to_bias.weight"].T).permute(0, 3, 1, 2)
    bias = bias.masked_fill(~pair_mask[:, None], NEG_INF)
    bias = bias.masked_fill(~pair_mask[:, None].any(dim=-1, keepdim=True), 0.0)
    sim = rnd(torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(head_dim) + bias)
    o = torch.einsum("bhij,bhjd->bhid", rnd(torch.softmax(sim, dim=-1)), v)
    o = rnd(rnd(torch.sigmoid(g)) * o).permute(0, 2, 1, 3).reshape(b, n, inner)
    return rnd(o @ w[f"{p}.to_out_node.weight"].T + w[f"{p}.to_out_node.bias"])


def attn_adaln(w, p, x, z, cond, mask, pair_mask, n_heads, rnd=ident):
    """`MultiHeadBiasedAttentionADALN_MM.forward`."""
    h = adaptive_norm(w, f"{p}.adaln", x, cond, mask, rnd)
    h = pair_bias_attention(w, f"{p}.mha", h, z, pair_mask, n_heads, rnd)
    h = adaptive_scale(w, f"{p}.scale_output", h, cond, mask, rnd)
    return h * mask[..., None]


def trunk_block(w, x, z, cond, mask, n_heads=16, p="", rnd=ident):
    """`MultiheadAttnAndTransition.forward`, sequential with both residuals."""
    pair_mask = mask[:, :, None].bool() & mask[:, None, :].bool()
    pre = f"{p}." if p else ""
    x = x * mask[..., None]
    x = rnd(attn_adaln(w, f"{pre}mhba", x, z, cond, mask, pair_mask, n_heads, rnd) + x) * mask[..., None]
    x = rnd(transition_adaln(w, f"{pre}transition", x, cond, mask, rnd) + x) * mask[..., None]
    return x * mask[..., None]


def block_weights(dim=1024, dim_pair=384, dim_cond=256, n_heads=16, expansion=4, qkln=True,
                  seed=0, dtype=torch.float64):
    """Upstream-keyed random weights for one trunk block, scaled like trained ones.

    Linear weights take torch's own `nn.Linear` init (uniform, 1/sqrt(fan_in)); the adaptive
    output scale keeps its trained-like bias near upstream's -2 init so the gate is not saturated.
    """
    g = torch.Generator().manual_seed(seed)

    def lin(out_f, in_f):
        bound = 1.0 / math.sqrt(in_f)
        return (torch.rand(out_f, in_f, generator=g, dtype=dtype) * 2 - 1) * bound

    def norm(d):
        return 1.0 + 0.1 * torch.randn(d, generator=g, dtype=dtype)

    inner = n_heads * (dim // n_heads)
    w = {}
    for p in ("mhba.adaln", "transition.adaln"):
        w[f"{p}.norm_cond.weight"] = norm(dim_cond)
        w[f"{p}.to_gamma.0.weight"] = lin(dim, dim_cond)
        w[f"{p}.to_gamma.0.bias"] = lin(dim, dim_cond)[:, 0].contiguous()
        w[f"{p}.to_beta.weight"] = lin(dim, dim_cond)
    for p in ("mhba.scale_output", "transition.scale_output"):
        w[f"{p}.to_adaln_zero_gamma.0.weight"] = lin(dim, dim_cond)
        w[f"{p}.to_adaln_zero_gamma.0.bias"] = torch.full((dim,), -2.0, dtype=dtype) + 0.1 * torch.randn(
            dim, generator=g, dtype=dtype)
    p = "mhba.mha"
    w[f"{p}.node_norm.weight"] = norm(dim)
    w[f"{p}.to_qkv.weight"] = lin(3 * inner, dim)
    w[f"{p}.to_qkv.bias"] = lin(3 * inner, dim)[:, 0].contiguous()
    w[f"{p}.to_g.weight"] = lin(inner, dim)
    w[f"{p}.to_g.bias"] = lin(inner, dim)[:, 0].contiguous()
    w[f"{p}.to_out_node.weight"] = lin(dim, inner)
    w[f"{p}.to_out_node.bias"] = lin(dim, inner)[:, 0].contiguous()
    w[f"{p}.pair_norm.weight"] = norm(dim_pair)
    w[f"{p}.to_bias.weight"] = lin(n_heads, dim_pair)
    if qkln:
        w[f"{p}.q_layer_norm.weight"] = norm(inner)
        w[f"{p}.k_layer_norm.weight"] = norm(inner)
    p = "transition.transition"
    w[f"{p}.swish_linear.0.weight"] = lin(2 * dim * expansion, dim)
    w[f"{p}.linear_out.weight"] = lin(dim, dim * expansion)
    return w


def block_inputs(b=1, n=128, dim=1024, dim_pair=384, dim_cond=256, lengths=None, seed=1,
                 dtype=torch.float64):
    """x, z, cond and a 0/1 token mask; `lengths` makes the batch ragged (padded tokens masked)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, dim, generator=g, dtype=dtype)
    z = torch.randn(b, n, n, dim_pair, generator=g, dtype=dtype) * 0.5
    cond = torch.randn(b, n, dim_cond, generator=g, dtype=dtype)
    mask = torch.ones(b, n, dtype=dtype)
    for i, ln in enumerate(lengths or []):
        mask[i, ln:] = 0.0
    return x, z, cond, mask
