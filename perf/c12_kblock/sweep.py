#!/usr/bin/env python3
"""Does raising the inner K block buy a RATE on the census's top thin-K `linear` shapes?

The `linear` class is 42.8 % of the device term at 512 aa (108,608 calls, 4.6604 s) and sits at
21.62 TFLOP/s and 159.7 GB/s -- neither roof binding, 4,021 Mcycles above roof. That is the
signature of poor reuse, and the inner K block (`in0_block_w`) is the parameter that decides how
much arithmetic one loaded in0 block feeds. This sweep asks, per census key, whether any legal K
block beats the shipped call's rate by more than the sweep's own noise.

What the shipped call actually is, per key, is the first thing this measures. The five keys that
carry the class are plain `ttnn.linear(core_grid=CORE_GRID_MAIN)` with NO program config, so their
K block is whatever ttnn v0.68.0 derives in `create_matmul_program_config`
(`matmul_program_config.cpp:361` on the narrow 1D path, `:539` on the 2D path). `--probe` prints
that derivation next to the measured `prod` arm; the `cg_equiv` arm is the control for it.

Instrument controls, because four prior campaigns here were derailed by an uncalibrated one:
  * FLOPs and minimum bytes are counted twice, once from the logical dims and once from the tile
    grid, and the sweep refuses to report if the two disagree.
  * the compute and DRAM roofs are MEASURED in this same process on this card, and any arm whose
    rate exceeds its roof is flagged UNFIT rather than reported as a win.
  * every arm is checked against a float64 reference of its own bf16 operands, so a config that
    computes the wrong thing cannot look like a fast one.
  * arms are interleaved rep by rep (compile/warm-up bias otherwise lands entirely on arm 1), and
    the AICLK is forced and sampled at 1 kHz DURING every timed interval.

Run (card 1 on qb2):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c12-kblock-unlock \
  PYTHONPATH=$PWD python3 perf/c12_kblock/sweep.py --out perf/c12_kblock/sweep_512_qb2c1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clk  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
T = 32

# (key, a_shape, w_shape, act_mc, out_mc, activation, calls@512aa, census_s, census_Mc, site)
# Shapes, call counts and seconds are the c12-linear-fusion-census table (sweep2 budget, 1350 MHz);
# the 17,920-call `1x16x512x512 K=128` key is fc1 + fc2, so it is carried here as two 8,960-call
# rows that differ only in fc1's fused silu.
SHAPES = [
    ("pair_tr_fc1", (1, 16, 512, 128), (128, 512), "L1", "L1", "silu", 8960, 0.5836, 787.8,
     "Transition.swiglu fc1 tenstorrent.py:8211"),
    ("pair_tr_fc2", (1, 16, 512, 128), (128, 512), "L1", "L1", None, 8960, 0.5836, 787.8,
     "Transition.swiglu fc2 tenstorrent.py:8222"),
    ("pair_tr_fc3", (1, 16, 512, 512), (512, 128), "L1", "DRAM", None, 8960, 0.7920, 1069.3,
     "Transition.swiglu fc3 tenstorrent.py:8233"),
    ("dit_s_768x768", (1, 512, 768), (768, 768), "DRAM", "DRAM", None, 38600, 0.5658, 763.9,
     "six DiT sites, AdaLN :9395/:9403 + AttentionPairBias :8159/:8169 + DiT :9610/:9519"),
    ("ctb_768x1536", (1, 512, 768), (768, 1536), "DRAM", "DRAM", None, 15200, 0.3418, 461.5,
     "ConditionedTransitionBlock :9492/:9498/:9506"),
    ("trimul_p_out", (1, 512, 512, 128), (128, 128), "DRAM", "DRAM", None, 560, 0.5276, 712.3,
     "trimul p_out :6789 via _pair_proj_linear -- already carries our tuned config; the L1 "
     "destination production prefers needs its own rung ladder, so this row is the DRAM fallback"),
    # THE FOLD'S ACTUAL PAIR-TRANSITION SHAPES, from the firing witness in fold_ab_kblock.py
    # (perf/c12_kblock/fold_screen_512_qb2c3.json). The census describes this key as
    # `1x16x512x512 K=128`, which gives mt_total = 256 -- and NO call in the fold has that key. The
    # pair tensor is row-blocked into 10 chunks of 47 plus one of 42 (10*47 + 42 = 512), so the
    # shapes issued are b=47 (mt_total 752, 5600 fc1+fc2 and 2800 fc3 calls per fold) and b=42
    # (mt_total 672, 560 and 280). mt_total is exactly what per_core_M and therefore the drain block
    # derive from, so a config swept at b=16 is tuned for a shape the fold never asks for.
    ("pair_tr_fc2_b47", (1, 47, 512, 128), (128, 512), "L1", "L1", None, 0, 0.0, 0.0,
     "Transition.swiglu fc2 tenstorrent.py:8334, the b=47 row block -- 2800 calls/fold"),
    ("pair_tr_fc3_b47", (1, 47, 512, 512), (512, 128), "L1", "DRAM", None, 0, 0.0, 0.0,
     "Transition.swiglu fc3 tenstorrent.py:8345, the b=47 row block -- 2800 calls/fold"),
    ("pair_tr_fc2_b42", (1, 42, 512, 128), (128, 512), "L1", "L1", None, 0, 0.0, 0.0,
     "Transition.swiglu fc2, the b=42 tail row block -- 280 calls/fold"),
    ("pair_tr_fc3_b42", (1, 42, 512, 512), (512, 128), "L1", "DRAM", None, 0, 0.0, 0.0,
     "Transition.swiglu fc3, the b=42 tail row block -- 280 calls/fold"),
    ("pair_tr_fc12_n1024", (1, 16, 512, 128), (128, 1024), "L1", "L1", None, 0, 0.0, 0.0,
     "NOT a production call. fc1+fc2 of Transition.swiglu widened to one N=1024 matmul, the second "
     "measured N at fixed (b, M, K) that c12-linear-fusion-census could not get. --fusion pairs it "
     "against 2x the N=512 key in one interleaved arm list."),
]
MC = {"L1": L1, "DRAM": DRAM}

# Which axes of `a_shape` are the TOKEN axis, per key. This has to be explicit rather than "every
# 512": `pair_tr_fc3` is (1, 16, 512, 512) where axis 2 is the token axis and axis 3 is the c_z*4
# contraction, and rescaling the latter would change the op instead of its size. The channel dims
# (K, N) do not move with sequence length, so a size sweep moves M and mt_total only -- which is
# exactly the axis `out_block_h` and `per_core_M` are derived from, i.e. the half of this lever that
# a one-size tuning would get wrong.
TOK_AXES = {"pair_tr_fc1": (2,), "pair_tr_fc2": (2,), "pair_tr_fc3": (2,),
            "pair_tr_fc12_n1024": (2,), "dit_s_768x768": (1,), "ctb_768x1536": (1,),
            "trimul_p_out": (1, 2),
            # The row-blocked keys: axis 2 is the token axis, axis 1 is the ROW BLOCK and does not
            # move with sequence length -- the number of blocks does, not their height.
            "pair_tr_fc2_b47": (2,), "pair_tr_fc3_b47": (2,),
            "pair_tr_fc2_b42": (2,), "pair_tr_fc3_b42": (2,)}


def at_tokens(key, a_shape, tok):
    """`a_shape` with its token axes moved to `tok`. Unknown key -> unchanged, and main() refuses."""
    s = list(a_shape)
    for i in TOK_AXES[key]:
        s[i] = tok
    return tuple(s)


def med(xs):
    return sorted(xs)[len(xs) // 2]


def divisors(n, cap=None):
    return [d for d in range(1, n + 1) if n % d == 0 and (cap is None or d <= cap)]


def counts(a_shape, w_shape):
    """FLOPs and minimum bytes, counted from the logical dims AND from the tile grid.

    A dense matmul has an exact FLOP count and an exact minimum byte count. Two independent counts
    that agree is the cheapest calibration this sweep can carry; if they ever disagree the shape is
    not tile-aligned and every rate below it would be wrong.
    """
    b = math.prod(a_shape[:-2])
    m, k = a_shape[-2], a_shape[-1]
    n = w_shape[-1]
    flops = 2 * b * m * n * k
    tile_flops = 2 * (b * (m // T) * (k // T) * (n // T)) * T ** 3
    rd = 2 * (b * m * k + k * n)
    wr = 2 * b * m * n
    tile_bytes = 2 * T * T * (b * (m // T) * (k // T) + (k // T) * (n // T) + b * (m // T) * (n // T))
    fit = (m % T == 0 and k % T == 0 and n % T == 0
           and flops == tile_flops and rd + wr == tile_bytes)
    return {"flops": flops, "read_B": rd, "write_B": wr, "bytes": rd + wr, "fit": fit,
            "b": b, "m": m, "k": k, "n": n, "mt": m // T, "kt": k // T, "nt": n // T,
            "mt_total": b * m // T}


def roofs(dev, ckc, node):
    """Compute and DRAM roofs, measured here. A roof that is asserted rather than measured is how
    a campaign talks itself into a win that is not there."""
    out = {}
    n = 4096
    a = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(1, 1, n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    s = clk.Sampler(node)
    s.start()
    ms = timed_one(dev, lambda: ttnn.deallocate(ttnn.matmul(
        a, b, memory_config=DRAM, compute_kernel_config=ckc)), pipe=4, reps=5)
    out["clk_compute"] = s.stop()
    out["compute_TFLOPs"] = round(2 * n ** 3 / 1e9 / ms, 2)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    nb = 32 * 1024 * 1024
    rows = nb // (2 * 1024)
    t = ttnn.from_torch(torch.zeros(1, 1, rows, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    s = clk.Sampler(node)
    s.start()
    ms = timed_one(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)), pipe=4, reps=5)
    out["clk_dram"] = s.stop()
    out["dram_rw_GBs"] = round(2 * nb / 1e9 / (ms / 1e3), 1)
    ttnn.deallocate(t)
    # An L1 roof as well, because the DRAM roof does not price L1 traffic AT ALL and most of this
    # class runs L1-resident. Scoring an L1->L1 key against the DRAM roof flags a correct arm as
    # UNFIT and then drops it from the candidate set, which is how an uncalibrated instrument
    # deletes the win it was built to find -- it happened on the first 768 aa run of this sweep.
    nb1 = 4 * 1024 * 1024
    rows1 = nb1 // (2 * 1024)
    t1 = ttnn.from_torch(torch.zeros(1, 1, rows1, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    s = clk.Sampler(node)
    s.start()
    ms = timed_one(dev, lambda: ttnn.deallocate(ttnn.clone(t1, memory_config=L1)), pipe=4, reps=5)
    out["clk_l1"] = s.stop()
    out["l1_rw_GBs"] = round(2 * nb1 / 1e9 / (ms / 1e3), 1)
    ttnn.deallocate(t1)
    return out


def byte_roof(R, amc, omc):
    """The bandwidth figure that applies to THIS key's operand placement, and whether it may REFUSE.

    `c12-linear-fusion-census` closed by pointing out that 1.9591 s of the class was priced against
    the measured DRAM roof while the fold runs those keys L1-resident, so that roof did not apply.
    The split below fixes that. But an L1 figure from `ttnn.clone` is a LOWER BOUND on L1 bandwidth,
    not a roof: a measured `ttnn.slice` moved bytes at about 1056 GB/s against a 581.7 GB/s clone
    probe. Using it to refuse an arm emptied the candidate set on the b=47 fc2 sweep and deleted
    every result in the run, which is the third time an uncalibrated gate has eaten this campaign's
    own measurement. So DRAM keys may be refused against the DRAM roof and L1 keys may NOT be
    refused at all -- the probe only annotates them.
    """
    if not R:
        return None, False
    if amc == "DRAM" and omc == "DRAM":
        return R["dram_rw_GBs"], True
    return R.get("l1_rw_GBs"), False


def timed_one(dev, fn, pipe=4, reps=5):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(o)


def timed_interleaved(dev, arms, pipe=4, reps=5, warm=2):
    """Median ms/call per arm, arms interleaved rep by rep.

    Measuring arm A to completion and then arm B puts all of the compile, cache-warm and
    governor-ramp bias on A. The fleet has made that mistake before, so the rep loop is outside the
    arm loop here and the warm-up is done for every arm before any arm is timed.
    """
    for _, fn in arms:
        for _ in range(warm):
            fn()
    ttnn.synchronize_device(dev)
    acc = {name: [] for name, _ in arms}
    for _ in range(reps):
        for name, fn in arms:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(pipe):
                fn()
            ttnn.synchronize_device(dev)
            acc[name].append((time.perf_counter() - t0) * 1e3 / pipe)
    return {k: {"ms": med(v), "ms_all": [round(x, 4) for x in v]} for k, v in acc.items()}


def derived_incumbent(c, nc, gx, gy, all_dram):
    """What ttnn v0.68.0 derives for `ttnn.linear(core_grid=...)` at this shape.

    `create_matmul_program_config`, matmul_program_config.cpp: narrow (height/width ratio > 8, or
    all-DRAM with a dim inside one tile) takes the 1D systolic factory, whose K block is
    `K / 32 % 2 == 0 ? 2 : 1` (:361); everything else takes the 2D factory, whose K block starts at
    4 and walks down to the first divisor of Kt (:539-541). Neither is a function of the shape's
    arithmetic intensity, and neither is ever the whole contraction above Kt = 4.
    """
    height, width = c["mt_total"] * T, c["nt"] * T
    ratio = max(height, width) / min(height, width)
    narrow = ratio > 8 or (all_dram and min(height, width) <= T)
    if narrow:
        bw = 2 if c["kt"] % 2 == 0 else 1
        return {"family": "1d", "in0_block_w": bw, "per_core_M": -(-c["mt_total"] // nc),
                "per_core_N": c["nt"], "narrow_ratio": round(ratio, 1)}
    bw = 4
    while c["kt"] % bw:
        bw -= 1
    return {"family": "2d", "in0_block_w": bw, "per_core_M": -(-c["mt_total"] // gy),
            "per_core_N": -(-c["nt"] // gx), "narrow_ratio": round(ratio, 1)}


def subblocks(obh, obw):
    sh = max((h for h in range(min(4, obh), 0, -1) if obh % h == 0), default=1)
    sw = max((w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0), default=1)
    return sh, sw


def make_cfg(family, gx, gy, bw, pcm, pcn, obh, obw, act):
    sh, sw = subblocks(obh, obw)
    if family == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
            out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=pcm, per_core_N=pcn,
            fuse_batch=True, fused_activation=act, mcast_in0=False)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=pcm, per_core_N=pcn,
        transpose_mcast=False, fused_activation=act)


def fusion_probe(dev, ckc, grid, gx, gy, nc, node, a, R):
    """One N=1024 matmul against TWO N=512 matmuls, at fixed (b=16, M=tokens, K=128), L1 operands.

    This is the one measurement `c12-linear-fusion-census` said it could not make: `Transition`
    fc1+fc2 is 16,896 calls and 1.1005 s, the biggest sibling set in the fold, and (16, M, 128) has
    exactly ONE measured N in the census, so no rate and no byte argument existed -- only a ceiling
    borrowed from the best K=128 key anywhere (24.37 TFLOP/s -> 0.7446 s). Two measured N points at
    fixed (b, M, K) replace that ceiling with a rate.

    The arithmetic is IDENTICAL both ways and so are the weight and output bytes: 2*b*M*1024*128
    FLOPs, 128*1024*2 B of weights, b*M*1024*2 B written. The ONLY thing widening deletes is the
    second read of the shared activation, 2*b*M*K bytes per call. That is the whole mechanism, it is
    a byte deletion and not a call-count argument, and because the activation is L1-resident the
    deleted bytes are L1 bytes -- which is exactly why this needed measuring instead of pricing.

    Scope: the matmul only. fc1 carries `activation="silu"` and fc2 does not, so one wider matmul
    cannot apply the epilogue to half its output; the epilogue is `c12-unfused-silu-bh`'s row and
    `b2z2-unfused-silu-recover` already cleared its accuracy. Both arms here are silu-free, so this
    is a like-for-like matmul comparison and not a comparison against today's fused fc1.
    """
    tok = a.tokens
    b, k, n = 16, 128, 512
    a_shape = (1, b, tok, k)
    c512 = counts(a_shape, (k, n))
    c1024 = counts(a_shape, (k, 2 * n))
    if not (c512["fit"] and c1024["fit"]):
        print("UNFIT: the two independent FLOP/byte counts disagree, refusing to report", flush=True)
        return
    assert c1024["flops"] == 2 * c512["flops"]
    saved_B = 2 * b * tok * k
    assert 2 * c512["bytes"] - c1024["bytes"] == saved_B
    print(f"\n=== FUSION PROBE  a{list(a_shape)} @ w[{k},{n}] x2  vs  @ w[{k},{2*n}]  "
          f"tokens={tok}\n    identical: {c1024['flops']/1e9:.3f} GFLOP, "
          f"{c1024['write_B']/1e6:.3f} MB written, {2*k*n*2/1e6:.3f} MB of weights\n"
          f"    deleted by widening: {saved_B/1e6:.3f} MB/call of L1 activation re-read "
          f"({100*saved_B/(2*c512['bytes']):.1f} % of the pair's bytes)", flush=True)

    torch.manual_seed(0)
    at = torch.randn(*a_shape) * 0.1
    w1t = torch.randn(k, n) * 0.1
    w2t = torch.randn(k, n) * 0.1
    ta = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=L1)
    tw1 = ttnn.from_torch(w1t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                          memory_config=DRAM)
    tw2 = ttnn.from_torch(w2t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                          memory_config=DRAM)
    # The wide weight is the CONCATENATION of the two narrow ones, so the fused arm must reproduce
    # both narrow outputs exactly. That makes the fusion itself a known-answer control, not just
    # its rate.
    twc = ttnn.from_torch(torch.cat([ttnn.to_torch(tw1), ttnn.to_torch(tw2)], dim=-1),
                          layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                          memory_config=DRAM)
    ab = ttnn.to_torch(ta).double()
    ref_wide = ab @ ttnn.to_torch(twc).double()

    inc512 = derived_incumbent(c512, nc, gx, gy, False)
    inc1024 = derived_incumbent(c1024, nc, gx, gy, False)
    print(f"    ttnn derives  N=512: {inc512}\n    ttnn derives N=1024: {inc1024}", flush=True)

    def pair_prod():
        ttnn.deallocate(ttnn.linear(ta, tw1, compute_kernel_config=ckc, core_grid=grid,
                                    memory_config=L1))
        ttnn.deallocate(ttnn.linear(ta, tw2, compute_kernel_config=ckc, core_grid=grid,
                                    memory_config=L1))

    arms = [("pair_prod_x2", pair_prod), ("pair_prod_x2_aa", pair_prod),
            ("wide_prod", lambda: ttnn.deallocate(ttnn.linear(
                ta, twc, compute_kernel_config=ckc, core_grid=grid, memory_config=L1)))]
    cfgs = {}
    for tag, cc, tw, inc in (("pair", c512, None, inc512), ("wide", c1024, twc, inc1024)):
        fam = inc["family"]
        pcm = -(-cc["mt_total"] // nc) if fam == "1d" else -(-cc["mt_total"] // gy)
        pcn = cc["nt"] if fam == "1d" else -(-cc["nt"] // gx)
        for bw in sorted({1, 2, 4, inc["in0_block_w"], cc["kt"]} & set(divisors(cc["kt"]))):
            for obh in sorted(set(divisors(pcm)), reverse=True)[:2]:
                for obw in sorted({pcn, min(pcn, 16), min(pcn, 8)} & set(divisors(pcn)),
                                  reverse=True):
                    name = f"{tag}_{fam}_bw{bw}_obh{obh}_obw{obw}"
                    if name in cfgs:
                        continue
                    try:
                        cfg = make_cfg(fam, gx, gy, bw, pcm, pcn, obh, obw, None)
                    except Exception as e:
                        print(f"    {name:28s} CFG-REFUSED {type(e).__name__}", flush=True)
                        continue
                    cfgs[name] = cfg
                    if tag == "wide":
                        arms.append((name, lambda cfg=cfg: ttnn.deallocate(ttnn.linear(
                            ta, twc, compute_kernel_config=ckc, memory_config=L1,
                            program_config=cfg))))
                    else:
                        def two(cfg=cfg):
                            ttnn.deallocate(ttnn.linear(ta, tw1, compute_kernel_config=ckc,
                                                        memory_config=L1, program_config=cfg))
                            ttnn.deallocate(ttnn.linear(ta, tw2, compute_kernel_config=ckc,
                                                        memory_config=L1, program_config=cfg))
                        arms.append((name, two))

    live = []
    for name, fn in arms:
        try:
            fn()
            ttnn.synchronize_device(dev)
        except Exception as e:
            print(f"    {name:28s} SKIP {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        live.append((name, fn))
    s = clk.Sampler(node)
    s.start()
    res = timed_interleaved(dev, live, pipe=a.pipe, reps=a.reps)
    clock = s.stop()
    print(f"    clock during the probe: {clock}", flush=True)

    # Accuracy, once per distinct config: the wide arms against float64 over the concatenated
    # weight, the pair arms against the two halves of the same reference.
    gflop = c1024["flops"] / 1e9
    out = {}
    for name, _ in live:
        ms = res[name]["ms"]
        wide = name.startswith("wide")
        if name.startswith("pair_prod"):
            y1 = ttnn.linear(ta, tw1, compute_kernel_config=ckc, core_grid=grid, memory_config=L1)
            y2 = ttnn.linear(ta, tw2, compute_kernel_config=ckc, core_grid=grid, memory_config=L1)
            yt = torch.cat([ttnn.to_torch(y1), ttnn.to_torch(y2)], dim=-1).double()
            ttnn.deallocate(y1)
            ttnn.deallocate(y2)
        elif name == "wide_prod":
            y = ttnn.linear(ta, twc, compute_kernel_config=ckc, core_grid=grid, memory_config=L1)
            yt = ttnn.to_torch(y).double()
            ttnn.deallocate(y)
        elif wide:
            y = ttnn.linear(ta, twc, compute_kernel_config=ckc, memory_config=L1,
                            program_config=cfgs[name])
            yt = ttnn.to_torch(y).double()
            ttnn.deallocate(y)
        else:
            y1 = ttnn.linear(ta, tw1, compute_kernel_config=ckc, memory_config=L1,
                             program_config=cfgs[name])
            y2 = ttnn.linear(ta, tw2, compute_kernel_config=ckc, memory_config=L1,
                             program_config=cfgs[name])
            yt = torch.cat([ttnn.to_torch(y1), ttnn.to_torch(y2)], dim=-1).double()
            ttnn.deallocate(y1)
            ttnn.deallocate(y2)
        err = (yt - ref_wide).abs().max().item()
        gb = (c1024["bytes"] if wide else 2 * c512["bytes"]) / 1e9
        tf, gbs = gflop / ms, gb / (ms / 1e3)
        unfit = tf > 1.05 * R["compute_TFLOPs"] or gbs > 1.05 * R["l1_rw_GBs"]
        out[name] = {"ms": round(ms, 5), "TFLOPs": round(tf, 2), "GBs": round(gbs, 1),
                     "max_abs_vs_f64": err, "unfit": unfit, "wide": wide,
                     "ms_all": res[name]["ms_all"]}
        print(f"    {name:28s} {ms:8.4f} ms  {tf:7.2f} TF/s  {gbs:7.1f} GB/s  maxabs {err:.4f}"
              f"{' UNFIT' if unfit else ''}", flush=True)

    ok = {k: v for k, v in out.items() if not v["unfit"] and not k.endswith("_aa")}
    bp = min((k for k in ok if not ok[k]["wide"]), key=lambda k: ok[k]["ms"], default=None)
    bw_ = min((k for k in ok if ok[k]["wide"]), key=lambda k: ok[k]["ms"], default=None)
    aa = None
    if "pair_prod_x2" in out and "pair_prod_x2_aa" in out:
        aa = round(out["pair_prod_x2"]["ms"] / out["pair_prod_x2_aa"]["ms"], 4)
        print(f"    A/A floor (pair_prod_x2 vs itself): {aa:.4f}x", flush=True)
    verdict = None
    if bp and bw_:
        r = out[bp]["ms"] / out[bw_]["ms"]
        rp = out["pair_prod_x2"]["ms"] / out[bw_]["ms"]
        print(f"\n    best pair  {bp}  {out[bp]['ms']:.4f} ms\n"
              f"    best wide  {bw_}  {out[bw_]['ms']:.4f} ms\n"
              f"    WIDENING vs best-tuned pair: {r:.4f}x | vs SHIPPED pair: {rp:.4f}x"
              f"{'' if aa is None else f'  [A/A floor {aa:.4f}x]'}", flush=True)
        verdict = {"best_pair": bp, "best_wide": bw_, "x_vs_best_pair": round(r, 4),
                   "x_vs_shipped_pair": round(rp, 4), "aa_ratio": aa}
        if aa is not None and abs(r - 1) <= abs(aa - 1):
            print("    ^ NOT A RESULT: inside this session's own A/A floor", flush=True)

    # --- the decisive second pass: the wide arm owes a SPLIT, and the pair arm does not -------
    # swiglu needs silu(x@W1) * (x@W2), i.e. TWO tensors. The pair arms hand back two. The wide arm
    # hands back one [.., 1024] tensor, so the wide route has to slice it in half before the silu
    # and the multiply, and those two slices read and write the whole 16.777 MB output again. The
    # silu and the elementwise multiply are common to both routes and cancel; the slice does not.
    # Comparing a wide matmul against two narrow ones WITHOUT the slice is comparing routes that
    # produce different things, so it is not a lever, and the byte arithmetic says the slice is the
    # same order as the saving. Hence this is measured, not argued.
    split2 = None
    if bp and bw_:
        def pair_route(cfg=cfgs.get(bp)):
            if cfg is None:
                pair_prod()
            else:
                ttnn.deallocate(ttnn.linear(ta, tw1, compute_kernel_config=ckc, memory_config=L1,
                                            program_config=cfg))
                ttnn.deallocate(ttnn.linear(ta, tw2, compute_kernel_config=ckc, memory_config=L1,
                                            program_config=cfg))

        def wide_route(cfg=cfgs.get(bw_)):
            y = (ttnn.linear(ta, twc, compute_kernel_config=ckc, core_grid=grid, memory_config=L1)
                 if cfg is None else
                 ttnn.linear(ta, twc, compute_kernel_config=ckc, memory_config=L1,
                             program_config=cfg))
            h1 = ttnn.slice(y, [0, 0, 0, 0], [1, b, tok, n], memory_config=L1)
            h2 = ttnn.slice(y, [0, 0, 0, n], [1, b, tok, 2 * n], memory_config=L1)
            ttnn.deallocate(y)
            ttnn.deallocate(h1)
            ttnn.deallocate(h2)

        arms2 = [("pair_route", pair_route), ("pair_route_aa", pair_route),
                 ("wide_route_split", wide_route)]
        live2 = []
        for name, fn in arms2:
            try:
                fn()
                ttnn.synchronize_device(dev)
            except Exception as e:
                print(f"    {name:28s} SKIP {type(e).__name__}: {str(e)[:90]}", flush=True)
                continue
            live2.append((name, fn))
        if len(live2) >= 2:
            s2 = clk.Sampler(node)
            s2.start()
            r2 = timed_interleaved(dev, live2, pipe=a.pipe, reps=a.reps)
            clk2 = s2.stop()
            print(f"\n    --- with the split the wide route owes ---  clock {clk2}", flush=True)
            for name, _ in live2:
                print(f"    {name:28s} {r2[name]['ms']:8.4f} ms  {r2[name]['ms_all']}", flush=True)
            split2 = {"clock": clk2, "best_pair_cfg": bp, "best_wide_cfg": bw_,
                      "ms": {k: r2[k]["ms"] for k in r2}, "ms_all": {k: r2[k]["ms_all"] for k in r2}}
            if "pair_route" in r2 and "wide_route_split" in r2:
                aa2 = (round(r2["pair_route"]["ms"] / r2["pair_route_aa"]["ms"], 4)
                       if "pair_route_aa" in r2 else None)
                rr = r2["pair_route"]["ms"] / r2["wide_route_split"]["ms"]
                split2["x_vs_pair_route"] = round(rr, 4)
                split2["aa_ratio"] = aa2
                print(f"    WIDE+SPLIT vs tuned pair: {rr:.4f}x"
                      f"{'' if aa2 is None else f'  [A/A floor {aa2:.4f}x]'}", flush=True)
                if aa2 is not None and abs(rr - 1) <= abs(aa2 - 1):
                    print("    ^ NOT A RESULT: inside this session's own A/A floor", flush=True)
                if rr < 1:
                    print("    ^ the slice costs MORE than widening saves: the fusion is a LOSS "
                          "once it produces what swiglu actually consumes", flush=True)

    payload = {"mode": "fusion", "tokens": tok, "clock_target_MHz": a.mhz, "grid": [gx, gy],
               "cores": nc, "roofs": R, "clock": clock, "deleted_B_per_call": saved_B,
               "counts_512": c512, "counts_1024": c1024, "derived_512": inc512,
               "derived_1024": inc1024, "arms": out, "verdict": verdict,
               "with_split": split2,
               "pipe": a.pipe, "reps": a.reps}
    if a.out:
        json.dump(payload, open(a.out, "w"), indent=1)
        print("\nwrote", a.out, flush=True)
    for t in (ta, tw1, tw2, twc):
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default="trimul_p_out,pair_tr_fc12_n1024,pair_tr_fc1,pair_tr_fc2,pair_tr_fc3",
                    help="comma-separated keys to skip. Both defaults are non-production calls: "
                         "trimul_p_out's shipped path already names a config via _pair_proj_linear "
                         "so its ratio here is double counting, and pair_tr_fc12_n1024 belongs to "
                         "--fusion where it is measured paired against 2x the N=512 key.")
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pipe", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="print the derivation and stop")
    ap.add_argument("--tokens", type=int, default=512,
                    help="token axis; 512 aa -> 512, 298 aa -> 320 (bucketed to 32), 768 aa -> 768. "
                         "Only 512 has census call counts, so other sizes report ratios only.")
    ap.add_argument("--fusion", action="store_true",
                    help="run ONLY the paired fc1+fc2 fusion-ceiling probe: one N=1024 matmul "
                         "against two N=512 matmuls, same arm list, interleaved, with an A/A arm")
    a = ap.parse_args()

    import tt_bio.tenstorrent as TB
    print(f"tt_bio from {TB.__file__}", flush=True)
    dev = TB.get_device()
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(a.mhz, nodes) if a.mhz else []
    node = nodes[0]
    ag = dev.compute_with_storage_grid_size()
    gx, gy = int(ag.x), int(ag.y)
    nc = gx * gy
    grid = ttnn.CoreGrid(y=gy, x=gx)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print(f"grid {gx}x{gy} = {nc} cores | nodes {nodes} | clock forced {a.mhz} on {held} "
          f"| aiclk now {clk.aiclk(node)} MHz", flush=True)

    if a.fusion:
        R = roofs(dev, ckc, node)
        print("ROOFS " + json.dumps(R), flush=True)
        fusion_probe(dev, ckc, grid, gx, gy, nc, node, a, R)
        clk.release()
        return

    R = roofs(dev, ckc, node) if not a.probe else {}
    if R:
        print("ROOFS " + json.dumps(R), flush=True)
        print(f"   byte roof applied: {R['dram_rw_GBs']} GB/s to all-DRAM keys, "
              f"{R['l1_rw_GBs']} GB/s to keys with an L1 operand", flush=True)

    torch.manual_seed(0)
    recs = []
    for (key, a_shape, w_shape, amc, omc, act, calls, cs, cmc, site) in SHAPES:
        if a.only and a.only not in key:
            continue
        if key in {x for x in a.skip.split(",") if x}:
            continue
        if key not in TOK_AXES:
            print(f"    {key}: no token axis declared, refusing to size-sweep it", flush=True)
            continue
        a_shape = at_tokens(key, a_shape, a.tokens)
        if a.tokens != 512:
            # The census priced 512 aa only. Extrapolating its call counts to another size would be
            # inventing the number, so at 298/768 this sweep reports RATIOS and rates, not seconds.
            calls, cs, cmc = 0, 0.0, 0.0
        c = counts(a_shape, w_shape)
        all_dram = amc == "DRAM" and omc == "DRAM"
        inc = derived_incumbent(c, nc, gx, gy, all_dram)
        print(f"\n=== {key}  a{list(a_shape)} @ w{list(w_shape)}  "
              f"mt_total={c['mt_total']} kt={c['kt']} nt={c['nt']}  act={act}  {amc}->{omc}\n"
              f"    census {calls} calls, {cs:.4f} s, {cmc:.1f} Mc | {site}\n"
              f"    ttnn v0.68.0 would derive: {inc}", flush=True)
        if not c["fit"]:
            print("    UNFIT: the two FLOP/byte counts disagree, skipping", flush=True)
            continue
        if a.probe:
            recs.append({"key": key, "counts": c, "derived": inc})
            continue

        at = torch.randn(*a_shape) * 0.1
        wt = torch.randn(*w_shape) * 0.1
        ta = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=MC[amc])
        tw = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=DRAM)
        # The reference is float64 over the operands the device actually holds, so an arm is scored
        # against arithmetic and not against another approximation.
        ab = ttnn.to_torch(ta).double()
        wb = ttnn.to_torch(tw).double()
        ref = ab @ wb
        if act == "silu":
            ref = ref * torch.sigmoid(ref)
        gflop = c["flops"] / 1e9
        gb = c["bytes"] / 1e9

        arms, cfgs = [], {}
        def prod_call():
            ttnn.deallocate(ttnn.linear(ta, tw, compute_kernel_config=ckc, core_grid=grid,
                                        memory_config=MC[omc], activation=act))
        # The A/A arm is the shipped call a SECOND time, interleaved with every other arm in this
        # same list. benchlock checks the box once, before the command, so it is blind to a
        # co-tenant that starts mid-run or is already device-bound; a 10:38Z sweep on this host was
        # destroyed exactly that way. prod/prod_aa is therefore this key's own noise floor, and a
        # ratio smaller than it is not a result.
        arms.append(("prod", prod_call))
        arms.append(("prod_aa", prod_call))
        # BOTH factories, not just the one ttnn picks. `create_matmul_program_config` routes on a
        # height/width ratio of 8 (`matmul_program_config.cpp:28`) and nothing else, so the pair
        # keys at N=512 (ratio 16.0) get the 1D systolic factory while the SAME operands at N=1024
        # (ratio 8.0) get the 2D one -- and the fusion probe measured the N=1024 shape at 0.0406 ms
        # on ttnn's own derived config against 0.0550 ms for the best 1D config on half the work.
        # That gap is the factory, not the K block, and the routing threshold is a constant rather
        # than a measurement. A sweep that only explores the derived family cannot see it.
        fams = ["1d", "2d"]
        geo = {"1d": (-(-c["mt_total"] // nc), c["nt"]),
               "2d": (-(-c["mt_total"] // gy), -(-c["nt"] // gx))}
        fused_act = None
        if act == "silu":
            try:
                fused_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
            except Exception as e:
                print(f"    fused silu not constructible ({type(e).__name__}); this key's "
                      f"explicit arms would not be the same op, so they are skipped. fc2 carries "
                      f"the identical shape without the silu.", flush=True)
                fused_act = "UNAVAILABLE"
        # Bounded arm set: 1, 2, whatever ttnn derives, 4, 8, half of K and the whole of K --
        # every rung that could change the reuse story, without paying compile for all of them.
        bw_set = sorted({1, 2, 4, 8, inc["in0_block_w"], c["kt"] // 2, c["kt"]}
                        & set(divisors(c["kt"])))
        if fused_act == "UNAVAILABLE":
            bw_set = []
        for fam in fams:
            pcm, pcn = geo[fam]
            for bw in bw_set:
                for obh in sorted(set(divisors(pcm)), reverse=True)[:2]:
                    for obw in sorted({pcn, min(pcn, 16), min(pcn, 8)} & set(divisors(pcn)),
                                      reverse=True):
                        name = f"{fam}_bw{bw}_obh{obh}_obw{obw}"
                        if name in cfgs:
                            continue
                        try:
                            cfg = make_cfg(fam, gx, gy, bw, pcm, pcn, obh, obw, fused_act)
                        except Exception as e:
                            print(f"    {name:26s} CFG-REFUSED {type(e).__name__}", flush=True)
                            continue
                        cfgs[name] = cfg
                        arms.append((name, (lambda cfg=cfg: ttnn.deallocate(ttnn.linear(
                            ta, tw, compute_kernel_config=ckc, memory_config=MC[omc],
                            program_config=cfg)))))

        # Drop any arm the device refuses outright, before timing, so a refusal is not a fast arm.
        live, out = [], {}
        for name, fn in arms:
            try:
                fn()
                ttnn.synchronize_device(dev)
            except Exception as e:
                print(f"    {name:26s} SKIP {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            live.append((name, fn))
        s = clk.Sampler(node)
        s.start()
        res = timed_interleaved(dev, live, pipe=a.pipe, reps=a.reps)
        clock = s.stop()
        print(f"    clock during this key: {clock}", flush=True)

        prod_t = None
        for name, _ in live:
            ms = res[name]["ms"]
            if name in ("prod", "prod_aa"):
                y = ttnn.linear(ta, tw, compute_kernel_config=ckc, core_grid=grid,
                                memory_config=MC[omc], activation=act)
            else:
                y = ttnn.linear(ta, tw, compute_kernel_config=ckc, memory_config=MC[omc],
                                program_config=cfgs[name])
            yt = ttnn.to_torch(y).double()
            ttnn.deallocate(y)
            err = (yt - ref).abs().max().item()
            rms = (yt - ref).pow(2).mean().sqrt().item()
            tf = gflop / ms
            gbs = gb / (ms / 1e3)
            br, br_binds = byte_roof(R, amc, omc)
            over_probe = bool(br) and gbs > 1.05 * br
            # The compute roof is a real roof and always binds; the byte figure only binds when it
            # is the DRAM one. An arm over the L1 probe is recorded, not discarded.
            unfit = bool(R) and (tf > 1.05 * R["compute_TFLOPs"]
                                 or (br_binds and over_probe))
            rec = {"ms": round(ms, 5), "TFLOPs": round(tf, 2), "GBs": round(gbs, 1),
                   "max_abs_vs_f64": err, "rmsd_vs_f64": rms, "unfit": unfit,
                   "over_byte_probe": over_probe, "byte_probe_GBs": br,
                   "byte_probe_binds": br_binds, "ms_all": res[name]["ms_all"]}
            if name == "prod":
                prod_t = ms
                rec["derived"] = inc
            elif prod_t is None:
                rec["x_vs_prod"] = None      # the shipped call itself was refused at this shape
            else:
                rec["x_vs_prod"] = round(prod_t / ms, 4)
                if cs:
                    rec["fold_s_saved"] = round(cs * (1 - ms / prod_t), 4)
                    rec["fold_Mc_saved"] = round(cmc * (1 - ms / prod_t), 1)
            out[name] = rec
            flag = " UNFIT" if unfit else (" >L1probe" if over_probe else "")
            extra = "" if rec.get("x_vs_prod") is None else f"  {rec['x_vs_prod']:.4f}x"
            if "fold_s_saved" in rec:
                extra += f"  {rec['fold_s_saved']:+.4f} s fold"
            print(f"    {name:26s} {ms:8.4f} ms  {tf:7.2f} TF/s  {gbs:7.1f} GB/s  "
                  f"maxabs {err:.4f}{extra}{flag}", flush=True)

        aa = out.get("prod_aa", {}).get("x_vs_prod")
        cand = {k: v for k, v in out.items()
                if k not in ("prod", "prod_aa") and not v["unfit"]
                and v.get("x_vs_prod") is not None}
        best = min(cand, key=lambda k: cand[k]["ms"]) if cand else None
        rec = {"key": key, "a_shape": list(a_shape), "w_shape": list(w_shape), "act": act,
               "amc": amc, "omc": omc, "site": site, "calls": calls, "census_s": cs,
               "census_Mc": cmc, "counts": c, "derived": inc, "clock": clock,
               "tokens": a.tokens, "aa_ratio": aa, "arms": out}
        if best:
            rec["best"] = {"arm": best, **cand[best]}
            b = cand[best]
            tail = (f"  {b['fold_s_saved']:+.4f} s / {b['fold_Mc_saved']:+.1f} Mc of fold"
                    if "fold_s_saved" in b else "  (no census counts at this size)")
            over = "" if aa is None else f"  [A/A floor {aa:.4f}x]"
            print(f"    BEST {best}  {b['x_vs_prod']:.4f}x vs prod{tail}{over}", flush=True)
            if aa is not None and abs(b["x_vs_prod"] - 1) <= abs(aa - 1):
                print("    ^ NOT A RESULT: this key's best arm is inside its own A/A floor",
                      flush=True)
        recs.append(rec)
        ttnn.deallocate(ta)
        ttnn.deallocate(tw)

    payload = {"clock_target_MHz": a.mhz, "grid": [gx, gy], "cores": nc, "roofs": R,
               "nodes": nodes, "pipe": a.pipe, "reps": a.reps, "shapes": recs,
               "ttnn": getattr(ttnn, "__version__", "0.68.0")}
    if a.out:
        json.dump(payload, open(a.out, "w"), indent=1)
        print("\nwrote", a.out, flush=True)
    tot = sum(r["best"].get("fold_s_saved", 0.0) for r in recs if r.get("best")
              and r["best"].get("fold_s_saved", 0.0) > 0)
    print(f"TOTAL pre-registered fold seconds from the K block alone: {tot:+.4f} s", flush=True)
    clk.release()


if __name__ == "__main__":
    main()
