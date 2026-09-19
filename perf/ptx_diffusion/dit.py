"""One Protenix-v2 token-DiT block, written twice: float64 reference and tt_bio tape.

The diffusion module's compute is its 24-block token DiT (`protenix.py:1262
_token_dit_device`), so that block is what a precision or throughput question about the
diffusion backward is actually asking about. The composition here is that function's,
op for op:

    b    = AdaLN(a, s)                          adaln_a
    attn = AttentionPairBias(b, bias)           apb, bias precomputed, not recomputed
    sg   = sigmoid(linear(s, Wa_last, ba_last))
    ao   = a + attn * sg
    an2  = AdaLN(ao, s)                         ctb_adaln
    bb   = silu(linear(an2, Wa1)) * linear(an2, Wa2)
    cs   = sigmoid(linear(s, Ws, bs))
    a'   = ao + cs * linear(bb, Wb)

and AdaLN is `tenstorrent.py:9489 AdaLN.__call__`: a parameter-free layer norm on `a`,
scaled by sigmoid(s_scale) and shifted by s_bias, where both terms are linears of a
gamma-only layer norm of `s`.

What is NOT copied, and why it does not affect either measurement. The shipped attention is
`AttentionPairBias` with `fp32_raw_matmul_attention=True`, which is a hand-routed kernel
with no backward (the same obstacle `ptx-fastpath` owns on the pairformer). The attention
here is the same arithmetic composed from taped ops -- per-head q/k/v projections, scaled
q@k^T, the precomputed pair bias added as a mask, softmax, p@v, output projection -- so the
op mix, the shapes and the reduction lengths are the block's. The gradient is scored against
a float64 reference of THIS composition, which is what makes the precision reading valid
whatever the shipped kernel does; the throughput reading is quoted as the composed block's,
not as the shipped kernel's.
"""
from __future__ import annotations

import numpy as np
import torch

# Protenix-v2, `protenix.py:874`. 24 blocks, 16 heads, head_dim 48 -> 768 channels; the
# single conditioning `s` is 384 wide.
BLOCKS, N_HEADS, HEAD_DIM = 24, 16, 48
C = N_HEADS * HEAD_DIM          # 768
S = 384
EPS = 1e-5                      # every tt-bio layer norm on the inference path


def params(rng, nt, *, with_bias=True):
    """One block's weights plus its activations, at production widths.

    Scaled 1/sqrt(fan_in) so the block's activations stay O(1) over 24 of them: a synthetic
    block whose activations blow up would make a precision reading about the scale, not
    about the dtype.
    """
    def w(i, o):
        return rng.standard_normal((i, o)) / np.sqrt(i)

    p = {
        "a": rng.standard_normal((nt, C)) / np.sqrt(C),
        "s": rng.standard_normal((nt, S)) / np.sqrt(S),
        # AdaLN #1 (adaln_a) and #2 (ctb_adaln)
        "g1": 1.0 + 0.1 * rng.standard_normal(S), "ws1": w(S, C), "bs1": rng.standard_normal(C) * 0.1,
        "wb1": w(S, C),
        "g2": 1.0 + 0.1 * rng.standard_normal(S), "ws2": w(S, C), "bs2": rng.standard_normal(C) * 0.1,
        "wb2": w(S, C),
        # attention
        "wq": w(C, C), "wk": w(C, C), "wv": w(C, C), "wo": w(C, C),
        "wgate": w(C, C),
        # the two sigmoid gates off s, and the SwiGLU transition
        "walast": w(S, C), "balast": rng.standard_normal(C) * 0.1,
        "wa1": w(C, 2 * C), "wa2": w(C, 2 * C), "wbt": w(2 * C, C),
        "wsc": w(S, C), "bsc": rng.standard_normal(C) * 0.1,
    }
    if not with_bias:
        p.pop("balast"); p.pop("bsc")
    # The per-block pair bias is precomputed once per fold and is a CONSTANT to the block,
    # exactly as `cond["dit_block_biases"]` is. It carries no gradient.
    p["_bias"] = rng.standard_normal((N_HEADS, nt, nt)) * 0.5
    return p


# ---------------------------------------------------------------------------------------
# float64 reference
# ---------------------------------------------------------------------------------------
def _ln64(x, gamma=None):
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdim=True)
    y = xc * torch.rsqrt(var + EPS)
    return y * gamma if gamma is not None else y


def _adaln64(a, s, g, ws, bs, wb):
    sn = _ln64(s, g)
    return _ln64(a) * torch.sigmoid(sn @ ws + bs) + sn @ wb


def block_ref(t, nt):
    """The block in torch, at whatever dtype `t`'s tensors carry (float64 for the reference)."""
    a, s, bias = t["a"], t["s"], t["_bias"]
    b = _adaln64(a, s, t["g1"], t["ws1"], t["bs1"], t["wb1"])
    q = (b @ t["wq"]).reshape(nt, N_HEADS, HEAD_DIM).permute(1, 0, 2)
    k = (b @ t["wk"]).reshape(nt, N_HEADS, HEAD_DIM).permute(1, 0, 2)
    v = (b @ t["wv"]).reshape(nt, N_HEADS, HEAD_DIM).permute(1, 0, 2)
    sc = (q @ k.transpose(-1, -2)) * (HEAD_DIM ** -0.5) + bias
    o = (torch.softmax(sc, -1) @ v).permute(1, 0, 2).reshape(nt, C)
    attn = (o @ t["wo"]) * torch.sigmoid(b @ t["wgate"])
    sg = torch.sigmoid(s @ t["walast"] + t["balast"])
    ao = a + attn * sg
    an2 = _adaln64(ao, s, t["g2"], t["ws2"], t["bs2"], t["wb2"])
    h = an2 @ t["wa1"]
    bb = (h * torch.sigmoid(h)) * (an2 @ t["wa2"])
    cs = torch.sigmoid(s @ t["wsc"] + t["bsc"])
    return ao + cs * (bb @ t["wbt"])


# ---------------------------------------------------------------------------------------
# the same block on tt_bio.autograd
# ---------------------------------------------------------------------------------------
def _adaln_tape(ag, a, s, g, ws, bs, wb, *, cfg, bwcfg):
    sn = ag.layer_norm(s, g, eps=EPS, config=cfg, backward_config=bwcfg)
    scale = ag.linear(sn, ws, bs, config=cfg, backward_config=bwcfg)
    shift = ag.linear(sn, wb, config=cfg, backward_config=bwcfg)
    an = ag.layer_norm(a, eps=EPS, config=cfg, backward_config=bwcfg)
    return ag.add(ag.mul(an, ag.sigmoid(scale)), shift)


def block_tape(ag, t, nt, *, cfg, bwcfg, attn_cfg=None):
    """`cfg`/`bwcfg` are the per-site levers the state doc's MIXED line names.

    `attn_cfg` overrides the two attention matmuls and the softmax alone, which is the site
    the shipped forward already singles out (`fp32_raw_matmul_attention=True`).
    """
    acfg = attn_cfg or cfg
    a, s, bias = t["a"], t["s"], t["_bias"]
    b = _adaln_tape(ag, a, s, t["g1"], t["ws1"], t["bs1"], t["wb1"], cfg=cfg, bwcfg=bwcfg)

    def heads(w):
        x = ag.linear(b, w, config=cfg, backward_config=bwcfg)
        return ag.permute(ag.reshape(x, [nt, N_HEADS, HEAD_DIM]), [1, 0, 2])

    q, k, v = heads(t["wq"]), heads(t["wk"]), heads(t["wv"])
    sc = ag.add(ag.scale(ag.matmul(q, k, transpose_b=True, config=acfg), HEAD_DIM ** -0.5), bias)
    o = ag.matmul(ag.softmax(sc, -1, config=acfg), v, config=acfg)
    o = ag.reshape(ag.permute(o, [1, 0, 2]), [nt, C])
    attn = ag.mul(ag.linear(o, t["wo"], config=cfg, backward_config=bwcfg),
                  ag.sigmoid(ag.linear(b, t["wgate"], config=cfg, backward_config=bwcfg)))
    sg = ag.sigmoid(ag.linear(s, t["walast"], t["balast"], config=cfg, backward_config=bwcfg))
    ao = ag.add(a, ag.mul(attn, sg))
    an2 = _adaln_tape(ag, ao, s, t["g2"], t["ws2"], t["bs2"], t["wb2"],
                      cfg=cfg, bwcfg=bwcfg)
    bb = ag.mul(ag.silu(ag.linear(an2, t["wa1"], config=cfg, backward_config=bwcfg)),
                ag.linear(an2, t["wa2"], config=cfg, backward_config=bwcfg))
    cs = ag.sigmoid(ag.linear(s, t["wsc"], t["bsc"], config=cfg, backward_config=bwcfg))
    return ag.add(ao, ag.mul(cs, ag.linear(bb, t["wbt"], config=cfg, backward_config=bwcfg)))


# Every name in `params` that carries a gradient. `_bias` and the activations are separate:
# `a` and `s` are the block's inputs, and in a 24-block stack `a` is the one that chains.
WEIGHTS = ("g1", "ws1", "bs1", "wb1", "g2", "ws2", "bs2", "wb2",
           "wq", "wk", "wv", "wo", "wgate", "walast", "balast",
           "wa1", "wa2", "wbt", "wsc", "bsc")
INPUTS = ("a", "s")
