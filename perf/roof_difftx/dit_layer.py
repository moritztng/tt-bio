#!/usr/bin/env python3
"""The Boltz-2 token diffusion-transformer layer, on its own shapes, with its own code.

`roof-shape-honest-roofs` priced the diffusion token layer at 20 % of the fold's 11.134 s
arithmetic floor from four bare `ttnn.linear` arms. This file measures the SHIPPED layer --
`tt_bio.tenstorrent.DiffusionTransformerLayer` itself, random weights, real shapes -- against
that same session's dense cube, and then takes the layer apart one class of work at a time so
the gap has a mechanism instead of a percentage.

The one construction that is not the shipped code is `mm_only`: the layer's eleven matmuls at
their real shapes and residency with every layer_norm, sigmoid, SwiGLU and residual removed,
which is `roof_pair_transition`'s `mm3only` applied to this unit. It is the shape-honest
arithmetic roof for the layer and nothing else in this file is allowed to beat it.

Method, unchanged from `perf/roof_shape/shape_roofs.py`: one process, one device, one session;
`reps` enqueues per synchronize so host dispatch is amortised; the whole arm set runs once per
block in a fixed order so a JIT warm-up or a clock ramp cannot bias one arm against another;
minimum over blocks. The dense cube runs in this same session, so every `pct_of_cube` is
internally consistent. Two arms are entered twice under different names to carry an A/A floor.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

import tt_bio.tenstorrent as T                                                # noqa: E402

S = 512           # tokens at 512 aa, already a multiple of 32
DIM = 768         # 2 * token_s
HEADS = 16        # AttentionPairBias|1x512x768,1x16x512x512 in the roof-budget table
HD = DIM // HEADS # 48, padded to 64 inside the fused qkv
INNER = 2 * DIM   # ConditionedTransitionBlock expansion_factor = 2
NLAYERS = 24      # token_transformer_depth
STEPS = 200       # sampling steps

# FLOP counts, logical (unpadded), per call. 2*M*K*N per matmul.
def _mm(m, k, n):
    return 2 * m * k * n

F_ADALN = 2 * _mm(S, DIM, DIM)                       # s_scale + s_bias
F_ATTN = (_mm(S, DIM, 3 * DIM)                       # fused qkv
          + _mm(S, DIM, DIM)                         # gate
          + _mm(S, DIM, DIM)                         # out
          + 2 * HEADS * _mm(S, S, HD))               # QK^T and AV
F_OUTPROJ = _mm(S, DIM, DIM)
F_TRANS = (F_ADALN + 3 * _mm(S, DIM, INNER) + _mm(S, INNER, DIM) + F_OUTPROJ)
F_LAYER = F_ADALN + F_ATTN + F_OUTPROJ + F_TRANS
# the conditioning half: 6 linears per layer that are a pure function of `s`
F_SONLY = 2 * F_ADALN + 2 * F_OUTPROJ


def weights(scale=0.02):
    """The exact key set `DiffusionTransformerLayer` reads, at the exact shapes."""
    def w(*shape):
        return torch.randn(*shape, dtype=torch.float32) * scale

    d = {}
    for pfx in ("adaln", "transition.adaln"):
        d[f"{pfx}.s_norm.weight"] = torch.ones(DIM)
        d[f"{pfx}.s_scale.weight"] = w(DIM, DIM)
        d[f"{pfx}.s_scale.bias"] = w(DIM)
        d[f"{pfx}.s_bias.weight"] = w(DIM, DIM)
    for k in ("proj_q", "proj_k", "proj_v", "proj_g", "proj_o"):
        d[f"pair_bias_attn.{k}.weight"] = w(DIM, DIM)
    d["pair_bias_attn.proj_q.bias"] = w(DIM)
    d["output_projection_linear.weight"] = w(DIM, DIM)
    d["output_projection_linear.bias"] = w(DIM)
    d["transition.swish_gate.0.weight"] = w(2 * INNER, DIM)
    d["transition.a_to_b.weight"] = w(INNER, DIM)
    d["transition.b_to_a.weight"] = w(DIM, INNER)
    d["transition.output_projection.0.weight"] = w(DIM, DIM)
    d["transition.output_projection.0.bias"] = w(DIM)
    return d


def build(dev, blocks_of):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    layer = T.DiffusionTransformerLayer(DIM, HEADS, False, weights(), kc)
    cg = T.CORE_GRID_MAIN

    def t(shape, sc=1.0):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16) * sc,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    a = t((1, S, DIM))
    s = t((1, S, DIM))
    z = t((1, HEADS, S, S), 0.1)                 # the rollout-invariant token bias, per layer
    sB = t((1, blocks_of * S, DIM))              # `s` for a block of future sampling steps
    cube4 = t((4096, 4096)); cube4b = t((4096, 4096))
    big = t((8192, 8192)); bigb = t((8192, 8192))
    keep = [a, s, z, sB, cube4, cube4b, big, bigb]

    ad1, ad2 = layer.adaln, layer.transition.adaln
    tr = layer.transition

    def lin(x, w, bias=None, act=None, grid=cg):
        return ttnn.linear(x, w, bias=bias, compute_kernel_config=kc, core_grid=grid,
                           activation=act)

    # ---- the shipped layer, and its three named children ------------------------------
    def ship():
        return layer(a, s, z)

    def adaln_only():
        return ad1(a, s)

    def sterms_only():
        sc, sb = ad1.s_terms(s)
        ttnn.deallocate(sb)
        return sc

    def attn_only():
        b = ad1(a, s)
        o = layer.attn_pair_bias(b, z)
        ttnn.deallocate(b)
        return o

    def trans_only():
        return tr(a, s)

    # ---- the layer with the whole conditioning half already in hand -------------------
    # Every tensor below is a pure function of `s`. Precomputing them outside the timed
    # region is NOT a shippable arrangement on its own (s carries the noise level and
    # changes every step); it is the UPPER BOUND on any arrangement that computes them
    # somewhere other than inside the layer.
    def s_pack():
        p = (ad1.s_terms(s), lin(s, layer.output_projection_weight,
                                 layer.output_projection_bias, "sigmoid"),
             ad2.s_terms(s), lin(s, tr.output_projection_weight,
                                 tr.output_projection_bias))
        ttnn.synchronize_device(dev)
        return p

    SP = s_pack()

    def _adaln_with(module, x, terms):
        x = ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=kc)
        x = ttnn.multiply_(x, terms[0], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        x = ttnn.add_(x, terms[1])
        return x

    def _trans_with(x, terms, s_o2):
        x = _adaln_with(ad2, x, terms)
        sw = lin(x, tr.swish_weight)
        ga = lin(x, tr.gates_weight)
        sw = ttnn.multiply_(ga, sw, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ab = lin(x, tr.a_to_b_weight)
        ttnn.deallocate(x)
        b = ttnn.multiply_(sw, ab)
        ttnn.deallocate(ab)
        ba = lin(b, tr.b_to_a_weight)
        ttnn.deallocate(b)
        out = ttnn.multiply(s_o2, ba, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(ba)
        return out

    def hoisted():
        t1, s_o, t2, s_o2 = SP
        b = _adaln_with(ad1, a, t1)
        b = layer.attn_pair_bias(b, z)
        b = ttnn.multiply(s_o, b)
        x = ttnn.add(a, b)
        ttnn.deallocate(b)
        at = _trans_with(x, t2, s_o2)
        out = ttnn.add(x, at)
        ttnn.deallocate(x); ttnn.deallocate(at)
        return out

    # ---- the conditioning half alone, at one step and at a block of steps -------------
    # sterms_b1 is what the fold runs: for ONE layer and ONE step, two layer_norms of `s`
    # and six [512,768]x[768,768] linears. sterms_bN is the same arithmetic with the
    # sampling steps blocked N deep, which is legal because the sigma schedule is fixed
    # and `s` therefore does not depend on the coordinates of the step before it.
    def _sterms(x, n):
        o = []
        for ad in (ad1, ad2):
            xn = ttnn.layer_norm(x, weight=ad.s_norm_weight, epsilon=1e-5,
                                 compute_kernel_config=kc)
            o.append(ttnn.linear(xn, ad.s_scale_weight, bias=ad.s_scale_bias,
                                 compute_kernel_config=kc))
            o.append(ttnn.linear(xn, ad.s_bias_weight, compute_kernel_config=kc))
            ttnn.deallocate(xn)
        o.append(lin(x, layer.output_projection_weight, layer.output_projection_bias,
                     "sigmoid"))
        o.append(lin(x, tr.output_projection_weight, tr.output_projection_bias))
        return o

    def sterms_b1():
        o = _sterms(s, 1)
        for x in o[1:]:
            ttnn.deallocate(x)
        return o[0]

    def sterms_bN():
        o = _sterms(sB, blocks_of)
        for x in o[1:]:
            ttnn.deallocate(x)
        return o[0]

    def sterms_bN_slice():
        o = _sterms(sB, blocks_of)
        for x in o:
            for i in range(blocks_of):
                ttnn.deallocate(x[:, i * S:(i + 1) * S, :])
            ttnn.deallocate(x)
        return ttnn.clone(s)

    # ---- inside AttentionPairBias: where its 40 % of the layer actually goes ----------
    # Five arms that partition the token-level `__call__`: the fused qkv linear, the head
    # split, the fused SDPA, the four-op layout epilogue that puts the heads back, and the
    # gate + output projection. Only two of the five are matmuls.
    apb = layer.attn_pair_bias
    PHD = apb.padded_head_dim
    qh = t((1, HEADS, S, PHD)); kh = t((1, HEADS, S, PHD)); vh = t((1, HEADS, S, PHD))
    oh = t((1, HEADS, S, PHD)); o2 = t((1, S, DIM))
    keep += [qh, kh, vh, oh, o2]
    spc = T._sdpa_program_config_for_lengths(S, S, HEADS, site="token_dit", d=PHD)

    def att_qkv():
        return ttnn.linear(a, apb.qkv_weight, bias=apb.qkv_bias,
                           compute_kernel_config=kc, core_grid=cg)

    def att_split():
        q_, k_, v_ = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(att_qkv(), 1), num_heads=HEADS, num_kv_heads=HEADS,
            transpose_k_heads=False)
        ttnn.deallocate(k_); ttnn.deallocate(v_)
        return q_

    def att_sdpa():
        return ttnn.transformer.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=z, is_causal=False, scale=HD ** -0.5,
            program_config=spc)

    def att_epi():
        o = oh[:, :, :, :HD]
        o = ttnn.permute(o, (0, 1, 3, 2))
        o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
        return ttnn.permute(o, (0, 2, 1))

    def att_gate_out():
        g = ttnn.linear(a, apb.g_weight, compute_kernel_config=kc, core_grid=cg)
        o = ttnn.multiply(o2, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        x = ttnn.linear(o, apb.o_weight, compute_kernel_config=kc, core_grid=cg)
        ttnn.deallocate(o)
        return x

    # ---- the two levers the floor allows, and the layer carrying both -----------------
    # L1 (layout). The shipped head re-assembly is four ops -- drop the 16 pad lanes, two
    # permutes and a reshape -- on a tensor that carries no arithmetic. `nlp_concat_heads`
    # is one op, but it KEEPS the pad lanes, so the gate and the output projection have to
    # run at width 1024 instead of 768. That is exact, not approximate: SDPA's v pad lanes
    # are zero because the fused qkv weight was zero-padded there, so o's pad lanes are zero,
    # anything multiplies them to zero, and zeroing the matching rows of proj_o drops them.
    #
    # L2 (placement). AdaLN applies sigmoid(s_scale) and adds s_bias as two eltwise ops on
    # [1,512,768]. `ttnn.mac` does both in one, and the sigmoid moves into the s_scale
    # linear's own activation epilogue, where it is free.
    gw_pad = t((DIM, HEADS * PHD)); ow_pad = t((HEADS * PHD, DIM))
    keep += [gw_pad, ow_pad]

    def _apb_fused(x):
        qkv = ttnn.linear(x, apb.qkv_weight, bias=apb.qkv_bias,
                          compute_kernel_config=kc, core_grid=cg)
        q_, k_, v_ = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1), num_heads=HEADS, num_kv_heads=HEADS,
            transpose_k_heads=False)
        ttnn.deallocate(qkv)
        o = ttnn.transformer.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=z, is_causal=False, scale=HD ** -0.5, program_config=spc)
        ttnn.deallocate(q_); ttnn.deallocate(k_); ttnn.deallocate(v_)
        o = ttnn.reshape(ttnn.experimental.nlp_concat_heads(o), (1, S, HEADS * PHD))
        g = ttnn.linear(x, gw_pad, compute_kernel_config=kc, core_grid=cg)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(g)
        out = ttnn.linear(o, ow_pad, compute_kernel_config=kc, core_grid=cg)
        ttnn.deallocate(o)
        return out

    def _adaln_mac(ad, x):
        xn = ttnn.layer_norm(s, weight=ad.s_norm_weight, epsilon=1e-5,
                             compute_kernel_config=kc)
        sc = ttnn.linear(xn, ad.s_scale_weight, bias=ad.s_scale_bias,
                         compute_kernel_config=kc, activation="sigmoid")
        sb = ttnn.linear(xn, ad.s_bias_weight, compute_kernel_config=kc)
        ttnn.deallocate(xn)
        xl = ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=kc)
        out = ttnn.mac(xl, sc, sb)
        ttnn.deallocate(xl); ttnn.deallocate(sc); ttnn.deallocate(sb)
        return out

    def _layer(l1, l2):
        b = _adaln_mac(ad1, a) if l2 else ad1(a, s)
        b = _apb_fused(b) if l1 else layer.attn_pair_bias(b, z)
        so = lin(s, layer.output_projection_weight, layer.output_projection_bias, "sigmoid")
        b = ttnn.multiply(so, b)
        ttnn.deallocate(so)
        x = ttnn.add(a, b)
        ttnn.deallocate(b)
        if l2:
            x2 = _adaln_mac(ad2, x)
            sw = lin(x2, tr.swish_weight)
            ga = lin(x2, tr.gates_weight)
            sw = ttnn.multiply_(ga, sw, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ab = lin(x2, tr.a_to_b_weight)
            ttnn.deallocate(x2)
            bb = ttnn.multiply_(sw, ab)
            ttnn.deallocate(ab)
            ba = lin(bb, tr.b_to_a_weight)
            ttnn.deallocate(bb)
            s2 = lin(s, tr.output_projection_weight, tr.output_projection_bias)
            at = ttnn.multiply_(s2, ba, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(ba)
        else:
            at = tr(x, s)
        out = ttnn.add(x, at)
        ttnn.deallocate(x); ttnn.deallocate(at)
        return out

    def att_out_k768():
        return ttnn.linear(o2, apb.o_weight, compute_kernel_config=kc, core_grid=cg)

    # ---- the per-op floor on a 384-tile tensor ---------------------------------------
    # Everything the layer does that is not a matmul happens on [1,512,768] = 384 tiles,
    # 0.75 MB. If the cost of one such op is flat in the number of rows, the layer is
    # paying a per-launch floor and not its bytes. Same op at 1x and 4x the rows decides it.
    a4 = t((1, 4 * S, DIM)); b4 = t((1, 4 * S, DIM)); b1 = t((1, S, DIM))
    a2x = t((1, 2 * S, DIM))
    o_pad = t((1, S, HEADS * PHD)); w_o_pad = t((HEADS * PHD, DIM))
    keep += [a4, b4, b1, a2x, o_pad, w_o_pad]

    def att_epi_fused():
        """One op in place of slice+permute+reshape+permute.

        nlp_concat_heads keeps the 64-wide padded head lanes, so the output is
        [1,512,1024] and the 16 pad lanes per head carry SDPA's zeros. Zeroing the
        matching rows of proj_o turns that into an exact identity at the cost of a
        K=1024 output projection instead of K=768, which `att_out_k1024` prices."""
        o = ttnn.experimental.nlp_concat_heads(oh)
        return ttnn.reshape(o, (1, S, HEADS * PHD))

    def att_out_k1024():
        return ttnn.linear(o_pad, w_o_pad, compute_kernel_config=kc, core_grid=cg)


    def op_add_1x():
        return ttnn.add(a, b1)

    def op_add_4x():
        return ttnn.add(a4, b4)

    def op_ln_1x():
        return ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=kc)

    def op_ln_2x():
        return ttnn.layer_norm(a2x, epsilon=1e-5, compute_kernel_config=kc)

    def op_ln_4x():
        return ttnn.layer_norm(a4, epsilon=1e-5, compute_kernel_config=kc)

    def op_slice_1x():
        return ttnn.clone(a4[:, S:2 * S, :])

    # ---- the conditioning half, concatenated across all 24 layers ---------------------
    # Every one of the 24 layers takes the SAME `s` and runs the same six [512,768]x[768,768]
    # projections through its own weights, so the whole step's conditioning half is one
    # [512,768] input against a concatenated [768, 24*6*768] weight. The four AdaLN
    # projections read LN_gamma_i(s); LN_gamma_i(s) = gamma_i * s_hat with s_hat the
    # parameter-free normalisation shared by every layer, so folding gamma_i into the i-th
    # weight block leaves one layer_norm per step instead of 48. Same FLOPs, same dot
    # products, 2 launches instead of 144.
    NCAT_AD = NLAYERS * 4                    # [s_scale|s_bias] x 2 AdaLNs x 24 layers
    NCAT_OP = NLAYERS * 2                    # [layer out-proj|transition out-proj] x 24
    w_ad = t((DIM, NCAT_AD * DIM))           # gamma_i folded into block i
    w_op = t((DIM, NCAT_OP * DIM))
    w_ad_b = t((NCAT_AD, DIM, DIM))          # the same weights as a batched operand
    w_op_b = t((NCAT_OP, DIM, DIM))
    keep += [w_ad, w_op, w_ad_b, w_op_b]

    def _ncat(slices):
        sh = ttnn.layer_norm(s, epsilon=1e-5, compute_kernel_config=kc)
        o1 = ttnn.linear(sh, w_ad, compute_kernel_config=kc)
        ttnn.deallocate(sh)
        o2 = ttnn.linear(s, w_op, compute_kernel_config=kc, activation="sigmoid")
        if slices:
            for o, n in ((o1, NCAT_AD), (o2, NCAT_OP)):
                for i in range(n):
                    ttnn.deallocate(o[:, :, i * DIM:(i + 1) * DIM])
        ttnn.deallocate(o2)
        return o1

    def _bmm(slices):
        """The same concatenation carried on the BATCH axis instead of on N.

        Slicing a [96,512,768] result on dim 0 hands the consumer a contiguous tile range;
        slicing a [512, 96*768] result on the last dim is a strided gather over every tile
        row. Same arithmetic, same weights, different address of the seam."""
        sh = ttnn.layer_norm(s, epsilon=1e-5, compute_kernel_config=kc)
        o1 = ttnn.matmul(sh, w_ad_b, compute_kernel_config=kc)
        ttnn.deallocate(sh)
        o2 = ttnn.matmul(s, w_op_b, compute_kernel_config=kc, activation="sigmoid")
        if slices:
            for o, n in ((o1, NCAT_AD), (o2, NCAT_OP)):
                for i in range(n):
                    ttnn.deallocate(o[i:i + 1])
        ttnn.deallocate(o2)
        return o1

    def sterms_step():
        """The shipped arrangement for one whole sampling step: 24 layers x six linears."""
        out = None
        for _ in range(NLAYERS):
            o = _sterms(s, 1)
            for x in o[1:]:
                ttnn.deallocate(x)
            if out is not None:
                ttnn.deallocate(out)
            out = o[0]
        return out

    # ---- the shape-honest arithmetic roof for this unit -------------------------------
    def mm_only():
        outs = [lin(s, ad1.s_scale_weight), lin(s, ad1.s_bias_weight),
                lin(s, ad2.s_scale_weight), lin(s, ad2.s_bias_weight),
                lin(s, layer.output_projection_weight),
                lin(s, tr.output_projection_weight),
                lin(a, layer.attn_pair_bias.qkv_weight),
                lin(a, layer.attn_pair_bias.g_weight),
                lin(a, layer.attn_pair_bias.o_weight),
                lin(a, tr.swish_weight), lin(a, tr.gates_weight),
                lin(a, tr.a_to_b_weight)]
        ba = lin(outs[-1], tr.b_to_a_weight)
        for x in outs[1:]:
            ttnn.deallocate(x)
        ttnn.deallocate(ba)
        return outs[0]

    F_MM = (F_SONLY + _mm(S, DIM, 3 * DIM) + 2 * _mm(S, DIM, DIM)
            + 3 * _mm(S, DIM, INNER) + _mm(S, INNER, DIM))

    A = {
        "bw_add8192": (lambda: ttnn.add(big, bigb, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                       3 * 8192 * 8192 * 2, 3),
        "cube4096": (lambda: ttnn.matmul(cube4, cube4b, compute_kernel_config=kc,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG),
                     _mm(4096, 4096, 4096), 3),
        "layer_ship": (ship, F_LAYER, 20),
        "layer_hoisted": (hoisted, F_LAYER - F_SONLY, 20),
        "layer_L1": (lambda: _layer(True, False), F_LAYER, 20),
        "layer_L2": (lambda: _layer(False, True), F_LAYER, 20),
        "layer_L1L2": (lambda: _layer(True, True), F_LAYER, 20),
        "mm_only": (mm_only, F_MM, 20),
        "adaln": (adaln_only, F_ADALN, 40),
        "sterms": (sterms_only, F_ADALN, 40),
        "attn": (attn_only, F_ADALN + F_ATTN, 20),
        "trans": (trans_only, F_TRANS, 20),
        "att_qkv": (att_qkv, _mm(S, DIM, 3 * DIM), 40),
        "att_split": (att_split, _mm(S, DIM, 3 * DIM), 40),
        "att_sdpa": (att_sdpa, 2 * HEADS * _mm(S, S, HD), 40),
        "att_epi": (att_epi, 0, 40),
        "att_gate_out": (att_gate_out, 2 * _mm(S, DIM, DIM), 40),
        "op_add_1x": (op_add_1x, 0, 60),
        "op_add_4x": (op_add_4x, 0, 60),
        "op_ln_1x": (op_ln_1x, 0, 60),
        "op_ln_2x": (op_ln_2x, 0, 60),
        "op_ln_4x": (op_ln_4x, 0, 60),
        "att_epi_fused": (att_epi_fused, 0, 40),
        "att_out_k1024": (att_out_k1024, _mm(S, HEADS * PHD, DIM), 40),
        "att_out_k768": (att_out_k768, _mm(S, DIM, DIM), 40),
        "op_slice_1x": (op_slice_1x, 0, 60),
        "sterms_step": (sterms_step, NLAYERS * F_SONLY, 2),
        "sterms_ncat": (lambda: _ncat(True), NLAYERS * F_SONLY, 2),
        "sterms_ncat_noslice": (lambda: _ncat(False), NLAYERS * F_SONLY, 2),
        "sterms_b1": (sterms_b1, F_SONLY, 20),
        "sterms_b%d" % blocks_of: (sterms_bN, blocks_of * F_SONLY, 20 // blocks_of or 1),
        "sterms_b%d_slice" % blocks_of: (sterms_bN_slice, blocks_of * F_SONLY,
                                         20 // blocks_of or 1),
    }
    A["layer_L1L2_AA"] = A["layer_L1L2"]
    A["layer_ship_AA"] = A["layer_ship"]
    A["cube4096_AA"] = A["cube4096"]
    return A, keep, SP, layer


def time_arms(arms, order, blocks, dev, warm=2):
    err, live = {}, []
    for n in order:
        fn = arms[n][0]
        try:
            for _ in range(warm):
                ttnn.deallocate(fn())
            live.append(n)
        except Exception as e:                                                # noqa: BLE001
            err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200])
            print("SKIP %-20s %s" % (n, err[n]), flush=True)
    ttnn.synchronize_device(dev)
    best = {n: None for n in live}
    for _ in range(blocks):
        for n in list(live):
            if n not in best:
                continue
            fn, _flop, reps = arms[n]
            outs = []
            try:
                t0 = time.perf_counter()
                for _ in range(reps):
                    outs.append(fn())
                    if len(outs) > 4:
                        ttnn.deallocate(outs.pop(0))
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / reps
            except Exception as e:                                            # noqa: BLE001
                err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200])
                print("DROP %-20s %s" % (n, err[n]), flush=True)
                best.pop(n, None)
                for o in outs:
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                continue
            for o in outs:
                ttnn.deallocate(o)
            best[n] = dt if best[n] is None else min(best[n], dt)
    return {n: v for n, v in best.items() if v is not None}, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "dit_layer.json")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--step-block", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    dev = T.get_device()
    try:
        arms, keep, _sp, _layer = build(dev, a.step_block)
        order = [n for n in arms if not a.only or n in a.only.split(",")]
        best, err = time_arms(arms, order, a.blocks, dev)
        order = [n for n in order if n in best]
        rows = [{"arm": n, "ms": best[n] * 1e3, "TFLOPs": arms[n][1] / best[n] / 1e12,
                 "reps": arms[n][2], "GFLOP": arms[n][1] / 1e9} for n in order]
        cube = next((r["TFLOPs"] for r in rows if r["arm"] == "cube4096"), None)
        for r in rows:
            r["pct_of_cube"] = 100 * r["TFLOPs"] / cube if cube else None
        cc = dev.compute_with_storage_grid_size()
        out = {"host": platform.node(), "arch": str(dev.arch()),
               "grid": [cc.x, cc.y], "core_grid_main": [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y],
               "blocks": a.blocks, "step_block": a.step_block,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "cube4096_TFLOPs": cube, "F_LAYER_GFLOP": F_LAYER / 1e9,
               "F_SONLY_GFLOP": F_SONLY / 1e9, "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            print("%-*s  %9.4f ms  %8.2f TFLOP/s  %6.1f %% of cube"
                  % (w, r["arm"], r["ms"], r["TFLOPs"], r["pct_of_cube"] or 0), flush=True)
    finally:
        T.cleanup() if hasattr(T, "cleanup") else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
