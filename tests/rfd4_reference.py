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


def f32(t):
    return t.float().double()


def bf16_mixed(t):
    """bf16 at every op boundary but fp32 on the residual streams: what `precision: bf16-mixed`
    (the release inference config) keeps, since autocast leaves the residual adds in fp32."""
    return bf16(t)


bf16_mixed.residual = f32


def _res(rnd):
    """The rounding a residual add sees: `rnd.residual` when the mode keeps a wider stream."""
    return getattr(rnd, "residual", rnd)


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
    res = _res(rnd)
    x = res(attn_adaln(w, f"{pre}mhba", x, z, cond, mask, pair_mask, n_heads, rnd) + x) * mask[..., None]
    x = res(transition_adaln(w, f"{pre}transition", x, cond, mask, rnd) + x) * mask[..., None]
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


# --- pair update (PairReprUpdate, release settings: hadamard write, triangle multiplication on) ---

def layer_norm(x, weight, bias, eps=EPS):
    """openfold `LayerNorm` / `torch.nn.LayerNorm` (the pair update uses real LayerNorms, not
    `make_norm`, so `norm_type` does not reach them)."""
    mu = x.mean(-1, keepdim=True)
    var = (x - mu).pow(2).mean(-1, keepdim=True)
    out = (x - mu) * torch.rsqrt(var + eps) * weight
    return out if bias is None else out + bias


def _lin(x, w, p):
    b = w.get(f"{p}.bias")
    out = x @ w[f"{p}.weight"].T
    return out if b is None else out + b


def tri_mult(w, p, z, pair_mask, outgoing, rnd=ident):
    """`TriangleMultiplicationOutgoing` / `Incoming` on the torch backend (the exact one).

    outgoing: x[i, j] = sum_k a[i, k] b[j, k];  incoming: x[i, j] = sum_k a[k, i] b[k, j].
    """
    m = pair_mask[..., None]
    zn = rnd(layer_norm(z, w[f"{p}.layer_norm_in.weight"], w.get(f"{p}.layer_norm_in.bias")))
    a = rnd(rnd(_lin(zn, w, f"{p}.linear_a_p")) * rnd(torch.sigmoid(_lin(zn, w, f"{p}.linear_a_g"))) * m)
    b = rnd(rnd(_lin(zn, w, f"{p}.linear_b_p")) * rnd(torch.sigmoid(_lin(zn, w, f"{p}.linear_b_g"))) * m)
    eq = "bikc,bjkc->bijc" if outgoing else "bkic,bkjc->bijc"
    x = rnd(torch.einsum(eq, a, b))
    x = rnd(layer_norm(x, w[f"{p}.layer_norm_out.weight"], w.get(f"{p}.layer_norm_out.bias")))
    x = rnd(_lin(x, w, f"{p}.linear_z"))
    g = rnd(torch.sigmoid(_lin(zn, w, f"{p}.linear_g")))
    return rnd(x * g)


def pair_transition(w, p, z, pair_mask, rnd=ident):
    """openfold `PairTransition`: LayerNorm, linear + relu, linear, mask."""
    zn = rnd(layer_norm(z, w[f"{p}.layer_norm.weight"], w[f"{p}.layer_norm.bias"]))
    h = rnd(torch.relu(_lin(zn, w, f"{p}.linear_1")))
    return rnd(_lin(h, w, f"{p}.linear_2")) * pair_mask[..., None]


def pair_update(w, p, x, z, mask, rnd=ident, pair_write="hadamard"):
    """`PairReprUpdate.forward` with `use_tri_mult=True`."""
    pm = mask[:, None, :] * mask[:, :, None]
    res = _res(rnd)
    x = x * mask[..., None]
    h = rnd(layer_norm(x, w[f"{p}.layer_norm_in.weight"], w[f"{p}.layer_norm_in.bias"]))
    pi, pj = rnd(h @ w[f"{p}.linear_x.weight"].T).chunk(2, dim=-1)
    if pair_write == "hadamard":
        z = res(z + rnd(pi[:, None, :, :] * pj[:, :, None, :]))
    else:
        z = res(z + rnd(pi[:, None, :, :] + pj[:, :, None, :]))
    z = z * pm[..., None]
    z = res(z + tri_mult(w, f"{p}.tri_mult_out", z, pm, True, rnd)) * pm[..., None]
    z = res(z + tri_mult(w, f"{p}.tri_mult_in", z, pm, False, rnd)) * pm[..., None]
    z = res(z + pair_transition(w, f"{p}.transition_out", z, pm, rnd)) * pm[..., None]
    return z


def pair_update_weights(dim=1024, dim_pair=384, tri_c=256, expansion=2, tri_bias=False, seed=0,
                        dtype=torch.float64, prefix=""):
    """Upstream-keyed random weights for one `PairReprUpdate` (release: tri_mult_bias false)."""
    g = torch.Generator().manual_seed(seed)
    pre = f"{prefix}." if prefix else ""

    def lin(out_f, in_f):
        bound = 1.0 / math.sqrt(in_f)
        return (torch.rand(out_f, in_f, generator=g, dtype=dtype) * 2 - 1) * bound

    def norm(d):
        return 1.0 + 0.1 * torch.randn(d, generator=g, dtype=dtype)

    def small(d):
        return 0.1 * torch.randn(d, generator=g, dtype=dtype)

    w = {f"{pre}layer_norm_in.weight": norm(dim), f"{pre}layer_norm_in.bias": small(dim),
         f"{pre}linear_x.weight": lin(2 * dim_pair, dim)}
    for t in ("tri_mult_out", "tri_mult_in"):
        p = f"{pre}{t}"
        for name in ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"):
            w[f"{p}.{name}.weight"] = lin(tri_c, dim_pair)
            if tri_bias:
                w[f"{p}.{name}.bias"] = small(tri_c)
        w[f"{p}.linear_g.weight"] = lin(dim_pair, dim_pair)
        w[f"{p}.linear_z.weight"] = lin(dim_pair, tri_c)
        w[f"{p}.layer_norm_in.weight"] = norm(dim_pair)
        w[f"{p}.layer_norm_out.weight"] = norm(tri_c)
        if tri_bias:
            for name, d in (("linear_g", dim_pair), ("linear_z", dim_pair), ("layer_norm_in", dim_pair),
                            ("layer_norm_out", tri_c)):
                w[f"{p}.{name}.bias"] = small(d)
    p = f"{pre}transition_out"
    w[f"{p}.layer_norm.weight"] = norm(dim_pair)
    w[f"{p}.layer_norm.bias"] = small(dim_pair)
    w[f"{p}.linear_1.weight"] = lin(expansion * dim_pair, dim_pair)
    w[f"{p}.linear_1.bias"] = small(expansion * dim_pair)
    w[f"{p}.linear_2.weight"] = lin(dim_pair, expansion * dim_pair)
    w[f"{p}.linear_2.bias"] = small(dim_pair)
    return w


# --- the stack and its recycles (LocalLatentsTransformer._run_trunk) ---

def trunk(w, s_init, z_init, conds, mask, nlayers, every_n=2, n_heads=16, rnd=ident):
    """`_run_trunk` with recycling on, in inference: `len(conds)` cycles, cycle k reading `conds[k]`
    (the caller's per-cycle conditioning, which carries `recycle_remaining`). Each cycle starts
    from the initial states plus a projection of the previous cycle's (RMSNorm + bias-free linear;
    zero on the first cycle because the previous state is zero)."""
    s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)
    for c in conds:
        z_in = _res(rnd)(z_init + rnd(rnd(rms_norm(z, w["z_cycle_norm.weight"])) @ w["z_cycle_proj.weight"].T))
        s_in = _res(rnd)(s_init + rnd(rnd(rms_norm(s, w["s_cycle_norm.weight"])) @ w["s_cycle_proj.weight"].T))
        s, z = s_in, z_in
        for i in range(nlayers):
            s = trunk_block(w, s, z, c, mask, n_heads, p=f"transformer_layers.{i}", rnd=rnd)
            if i < nlayers - 1 and i % every_n == 0:
                z = pair_update(w, f"pair_update_layers.{i}", s, z, mask, rnd)
    return s, z


def trunk_weights(nlayers, dim=1024, dim_pair=384, dim_cond=256, n_heads=16, tri_c=256, every_n=2,
                  seed=0, dtype=torch.float64):
    """Upstream-keyed random weights for a `nlayers`-deep trunk with its recycle projections."""
    w = {}
    for i in range(nlayers):
        for k, v in block_weights(dim, dim_pair, dim_cond, n_heads, seed=seed + 2 * i, dtype=dtype).items():
            w[f"transformer_layers.{i}.{k}"] = v
        if i < nlayers - 1 and i % every_n == 0:
            w.update(pair_update_weights(dim, dim_pair, tri_c, seed=seed + 2 * i + 1, dtype=dtype,
                                         prefix=f"pair_update_layers.{i}"))
    g = torch.Generator().manual_seed(seed + 10_000)
    for name, d in (("s", dim), ("z", dim_pair)):
        w[f"{name}_cycle_norm.weight"] = 1.0 + 0.1 * torch.randn(d, generator=g, dtype=dtype)
        # trained, not zero: upstream initialises these to zero and training moves them
        w[f"{name}_cycle_proj.weight"] = (torch.rand(d, d, generator=g, dtype=dtype) * 2 - 1) / math.sqrt(d)
    return w
