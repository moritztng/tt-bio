#!/usr/bin/env python3
"""The AttentionPairBias backward in float64, with its four additive terms separable.

One implementation, imported by the device arm, the reference arm and the scorer, so the three
cannot drift. It is upstream 0.4.3's own arithmetic
(`openfold3/core/model/layers/attention_pair_bias.py` + `primitives/attention.py`), rewritten
with FOUR SEPARATE LEAVES for the single-track activation instead of one, which is the whole
point: `a` reaches the attention through `linear_q`, `linear_k`, `linear_v` and `linear_g`, so
the cotangent it receives is a sum of four terms and asking autograd for each leaf's gradient
splits that sum EXACTLY rather than to first order.

The rewrite is validated in `split.py` before any arm is read -- forward against upstream's own
module on the same operands, total cotangent against torch's float64 autograd through that
module, and against float64 central finite differences with `h` swept (`of3t-apbleaf`'s FDSWEEP
showed the campaign's 1e-9 clause misses at h = 1e-6 and the residual falls as 1/h, so the band
is what gets quoted).

`z` enters only through `bias_raw`, so that projection is the stored form of the pair-track
channel: 16 channels per pair instead of 128, and no information about `g` is lost.
"""
from __future__ import annotations

import math

import torch

LN_EPS = 1e-5
INF = 1e9


def proj_bias(z, w_ln, b_ln, w_lz):
    """z [1, N, N, C_z] -> the raw additive pair bias [1, H, N, N], in float64.

    `layer_norm_z` is upstream's `LayerNorm` (plain `F.layer_norm`, eps 1e-5) and `linear_z`
    carries no bias in this checkpoint, both read off the state dict rather than assumed.
    `permute_final_dims(z, [2, 0, 1])` is the [*, N, N, H] -> [*, H, N, N] move.
    """
    z = z.to(torch.float64)
    z = torch.nn.functional.layer_norm(
        z, (z.shape[-1],), w_ln.to(torch.float64), b_ln.to(torch.float64), LN_EPS)
    z = z @ w_lz.to(torch.float64).t()
    return z.permute(0, 3, 1, 2).contiguous()


def mask_bias(single_mask, inf=INF):
    """upstream `_prep_bias`: `(inf * (mask - 1))[..., None, None, :]`.

    Our harness hands the device the identical tensor -- `dev_grad.py` builds
    `attn = (1.0 - sm) * -1e9` -- so this is held fixed and is not an error channel.
    """
    m = single_mask.to(torch.float64).reshape(1, -1)
    return (inf * (m - 1))[..., None, None, :]


def weights_from_sd(sd, prefix, dtype=torch.float64):
    """The eight tensors F needs, by their upstream names under `<prefix>attn_pair_bias.`."""
    g = lambda k: sd[prefix + k].detach().to(dtype).clone()
    return {"Wq": g("mha.linear_q.weight"), "bq": g("mha.linear_q.bias"),
            "Wk": g("mha.linear_k.weight"), "Wv": g("mha.linear_v.weight"),
            "Wg": g("mha.linear_g.weight"), "Wo": g("mha.linear_o.weight"),
            "w_ln_z": g("layer_norm_z.weight"), "b_ln_z": g("layer_norm_z.bias"),
            "w_lz": g("linear_z.weight")}


def apb_terms(a, bias_raw, mbias, W, do, heads=16):
    """F(a, bias_raw, do, W) and its four additive terms, every tensor float64.

    Returns the forward output, the four terms of the cotangent at `a`, their sum, and
    `dbias` -- the cotangent this backward EXPORTS to the pair track, which is the `dz`
    reading at this site and the one quantity `R44` structurally cannot see.
    """
    a = a.to(torch.float64)
    leaves = [a.detach().clone().requires_grad_(True) for _ in range(4)]
    lq, lk, lv, lg = leaves
    b = bias_raw.to(torch.float64).detach().clone().requires_grad_(True)
    d = W["Wq"].shape[0] // heads

    def shp(t):
        return t.view(t.shape[:-1] + (heads, -1)).transpose(-2, -3)

    q = shp(lq @ W["Wq"].t() + W["bq"]) / math.sqrt(d)
    k = shp(lk @ W["Wk"].t())
    v = shp(lv @ W["Wv"].t())
    sc = torch.einsum("...qc,...kc->...qk", q, k)
    sc = sc + mbias
    sc = sc + b
    p = torch.softmax(sc, dim=-1)
    o = torch.einsum("...qk,...kc->...qc", p, v).transpose(-2, -3)
    gate = torch.sigmoid(lg @ W["Wg"].t())
    o = o * gate.view(gate.shape[:-1] + (heads, -1))
    out = o.reshape(o.shape[:-2] + (-1,)) @ W["Wo"].t()

    gr = torch.autograd.grad(out, leaves + [b], grad_outputs=do.to(torch.float64))
    total = gr[0] + gr[1] + gr[2] + gr[3]
    return {"out": out.detach(), "T_Q": gr[0], "T_K": gr[1], "T_V": gr[2], "T_GATE": gr[3],
            "dbias": gr[4], "g": total}


def apb_forward_scalar(a, bias_raw, mbias, W, do, heads=16):
    """<F_forward(a), do>, the scalar the finite differences are taken of. No autograd."""
    with torch.no_grad():
        a = a.to(torch.float64)
        d = W["Wq"].shape[0] // heads

        def shp(t):
            return t.view(t.shape[:-1] + (heads, -1)).transpose(-2, -3)

        q = shp(a @ W["Wq"].t() + W["bq"]) / math.sqrt(d)
        k = shp(a @ W["Wk"].t())
        v = shp(a @ W["Wv"].t())
        sc = torch.einsum("...qc,...kc->...qk", q, k) + mbias + bias_raw.to(torch.float64)
        p = torch.softmax(sc, dim=-1)
        o = torch.einsum("...qk,...kc->...qc", p, v).transpose(-2, -3)
        gate = torch.sigmoid(a @ W["Wg"].t())
        o = o * gate.view(gate.shape[:-1] + (heads, -1))
        out = o.reshape(o.shape[:-2] + (-1,)) @ W["Wo"].t()
        return float((out * do.to(torch.float64)).sum())
