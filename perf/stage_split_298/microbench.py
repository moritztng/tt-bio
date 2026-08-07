#!/usr/bin/env python3
"""Step-2 microbenchmarks for the 298-aa trunk campaign (protenix-v2 shapes, card 3).

Discipline (per skill ttnn-perf-profiling): warm >=3 calls per leg before timing,
ttnn.synchronize_device before the clock starts AND before it stops, A/B legs
INTERLEAVED (all-A-then-all-B reads thermal drift as a code difference), median of
>=5 timed iterations. Every candidate is validated against the fp32 host reference
(PCC + max abs diff + bit test) in the same run.

  M1  pair projection shape: ttnn.linear on 4D [1,16,320,256] vs reshaped tall GEMM
      [1,1,5120,256] (metadata-only reshape claim: check it is free and bit-exact).
  M2  trimul contraction program-config sweep on [1,32,320,320]@[1,32,320,320].
  M4  pair-transition H chunk size 16 vs 32/48/64 (the swiglu op sequence itself).
  M5  math fidelity HiFi4+fp32_dest vs HiFi2(+/-fp32_dest) vs LoFi on the M1/M2 GEMMs.
      (Run regardless; the fidelity lever is only LIVE if Step 1b says FPU-bound.)
  M6  qkv+g packed into one [c_z,1024] minimal_matmul + chunk vs two separate matmuls.
  M7  tri-attention SDPA program-config sweep at the measured offender shape
      q [320,8,320,32] + bias [1,8,320,320] (27% of block device time at 4.5 TFLOP/s).

    TT_VISIBLE_DEVICES=3 python3 perf/stage_split_298/microbench.py --only m1
"""

import argparse
import time

import torch

import ttnn
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

N = 320          # padded tokens at 298 aa
C_Z = 256        # protenix-v2 pair width
H_CHUNK = 16     # TRANSITION_H_CHUNK_SIZE (non-fast)
C_HID = 1024     # transition hidden
TRI_C = 32       # TRIANGLE_MULT_CHUNK_SIZE -> contraction batch


def synced_med_ms(fn, dev, warm=3, iters=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def interleave(legs, dev, warm=3, iters=7):
    """legs: list of (name, fn). Round-robin so drift hits every leg equally."""
    for name, fn in legs:
        for _ in range(warm):
            fn()
    ttnn.synchronize_device(dev)
    times = {name: [] for name, _ in legs}
    for i in range(iters):
        for name, fn in legs:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            times[name].append((time.perf_counter() - t0) * 1e3)
    return {k: sorted(v)[len(v) // 2] for k, v in times.items()}


def check(name, got_tt, ref_t):
    got = ttnn.to_torch(got_tt).float()
    ref = ref_t.float()
    if got.shape != ref.shape:
        print(f"    {name}: SHAPE MISMATCH {tuple(got.shape)} vs {tuple(ref.shape)}")
        return
    a, b = got.flatten(), ref.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    mad = (a - b).abs().max().item()
    bit = torch.equal(got, ref)
    print(f"    {name}: PCC={pcc:.6f} max_abs_diff={mad:.5f} bit_exact={bit}")


def ckc_of(dev, fidelity, fp32_acc):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True)


def m1(dev, ckc):
    print("M1 pair projection shape (fc1-style [256->1024] silu, L1 out)")
    x_t = torch.randn(1, H_CHUNK, N, C_Z, dtype=torch.bfloat16)
    w_t = torch.randn(C_Z, C_HID, dtype=torch.bfloat16)
    x4 = ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(w_t, layout=ttnn.TILE_LAYOUT, device=dev)

    def leg_4d():
        return ttnn.linear(x4, w, activation="silu", compute_kernel_config=ckc,
                           memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                           core_grid=CORE_GRID_MAIN)

    def leg_tall():
        xr = ttnn.reshape(x4, (1, 1, H_CHUNK * N, C_Z))
        y = ttnn.linear(xr, w, activation="silu", compute_kernel_config=ckc,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                        core_grid=CORE_GRID_MAIN)
        # NOTE: no deallocate(y) -- reshape returns a VIEW of y's buffer, and
        # freeing y would invalidate the returned tensor.
        return ttnn.reshape(y, (1, H_CHUNK, N, C_HID))

    def leg_reshape_only():
        xr = ttnn.reshape(x4, (1, 1, H_CHUNK * N, C_Z))
        return xr

    med = interleave([("4d", leg_4d), ("tall", leg_tall), ("reshape_only", leg_reshape_only)], dev)
    print(f"    median ms: {med}  speedup 4d/tall = {med['4d'] / med['tall']:.2f}x")
    a, b = leg_4d(), leg_tall()
    ttnn.synchronize_device(dev)
    check("tall vs 4d", b, ttnn.to_torch(a))

    # M1b: the profile's slowest matmuls are the 3D [320,320,256] pair projections
    # ([256->256] at 744 us, [256->8] at 466 us). Test as-is vs one flattened tall GEMM.
    print("M1b 3D pair projection [320,320,256]@[256,k] vs flattened [102400,256] tall GEMM")
    x3_t = torch.randn(N, N, C_Z, dtype=torch.bfloat16)
    for k_dim in (256, 8):
        wk_t = torch.randn(C_Z, k_dim, dtype=torch.bfloat16)
        x3 = ttnn.from_torch(x3_t, layout=ttnn.TILE_LAYOUT, device=dev)
        wk = ttnn.from_torch(wk_t, layout=ttnn.TILE_LAYOUT, device=dev)

        def leg_3d(x3=x3, wk=wk):
            return ttnn.linear(x3, wk, compute_kernel_config=ckc,
                               dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)

        def leg_flat(x3=x3, wk=wk, k_dim=k_dim):
            xf = ttnn.reshape(x3, (1, 1, N * N, C_Z))
            y = ttnn.linear(xf, wk, compute_kernel_config=ckc,
                            dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
            return ttnn.reshape(y, (N, N, k_dim))

        med = interleave([(f"k{k_dim}_3d", leg_3d), (f"k{k_dim}_flat", leg_flat)], dev)
        print(f"    k={k_dim}: {med}  speedup 3d/flat = {med[f'k{k_dim}_3d'] / med[f'k{k_dim}_flat']:.2f}x")
        a, b = leg_3d(), leg_flat()
        ttnn.synchronize_device(dev)
        check(f"k={k_dim} flat vs 3d", b, ttnn.to_torch(a))


def m2(dev, ckc):
    print("M2 trimul contraction config sweep [1,32,320,320]@[1,32,320,320]")
    a = ttnn.from_torch(torch.randn(1, TRI_C, N, N, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(1, TRI_C, N, N, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    gx, gy = 13, 10
    pm, pn = -(-10 // gy), -(-10 // gx)
    flops = 2 * TRI_C * N * N * N
    legs = []
    # fuse_batch is invalid here: it requires batch(in1)==1, and b is [1,32,320,320].
    # Kt = 10 tiles: in0_block_w must divide it (4 is invalid).
    for ibw in (1, 2, 5, 10):
        # out block is per_core_M x per_core_N = 1x1 tile at N=320, so
        # out_subblock > 1 is invalid (out_block % subblock == 0 required).
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
            out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn,
            transpose_mcast=False, fused_activation=None, fuse_batch=False)

        def leg(pc=pc):
            return ttnn.matmul(a, b, compute_kernel_config=ckc,
                               memory_config=ttnn.L1_MEMORY_CONFIG, program_config=pc,
                               dtype=ttnn.bfloat16)
        legs.append((f"ibw{ibw}", leg))
    med = interleave(legs, dev, warm=2, iters=5)
    cur = med.get("ibw1")
    for k, v in sorted(med.items(), key=lambda kv: kv[1]):
        print(f"    {k}: {v:.3f} ms  {flops / v / 1e9:.1f} TFLOP/s  ({cur / v:.2f}x vs current)")


def m4(dev, ckc):
    print("M4 pair-transition H chunk (full swiglu sequence, per-chunk ms)")
    w1 = ttnn.from_torch(torch.randn(C_Z, C_HID, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    w2 = ttnn.from_torch(torch.randn(C_Z, C_HID, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    w3 = ttnn.from_torch(torch.randn(C_HID, C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    lnw = ttnn.from_torch(torch.randn(C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    lnb = ttnn.from_torch(torch.randn(C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)

    def swiglu(x):
        xn = ttnn.layer_norm(x, weight=lnw, bias=lnb, epsilon=1e-5,
                             compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1 = ttnn.linear(xn, w1, activation="silu", compute_kernel_config=ckc,
                         memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        x2 = ttnn.linear(xn, w2, compute_kernel_config=ckc,
                         memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(xn)
        x = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        xd = ttnn.linear(x, w3, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                         core_grid=CORE_GRID_MAIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        return xd

    legs = []
    for h in (16, 32, 48, 64):
        xh = ttnn.from_torch(torch.randn(1, h, N, C_Z, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev)
        n_chunks = -(-N // h)

        def leg(xh=xh, n_chunks=n_chunks):
            outs = []
            for c in ttnn.chunk(xh, n_chunks, dim=1):
                outs.append(swiglu(c))
            y = ttnn.concat(outs, dim=1)
            for o in outs:
                ttnn.deallocate(o)
            return y
        legs.append((f"h{h}(x{n_chunks})", leg))
    med = interleave(legs, dev, warm=2, iters=5)
    cur = med.get("h16(x20)")
    for k, v in sorted(med.items(), key=lambda kv: kv[1]):
        print(f"    {k}: {v:.3f} ms/full-transition  ({cur / v:.2f}x vs current)")


def m5(dev):
    print("M5 math fidelity on the two hot GEMM shapes")
    x = ttnn.from_torch(torch.randn(1, 1, H_CHUNK * N, C_Z, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(torch.randn(C_Z, C_HID, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    a = ttnn.from_torch(torch.randn(1, TRI_C, N, N, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(1, TRI_C, N, N, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    for shape, fn_make in (("proj [5120,256]x[256,1024]", lambda c: ttnn.matmul(x, w, compute_kernel_config=c, dtype=ttnn.bfloat16)),
                           ("trimul [32,320,320]x[32,320,320]", lambda c: ttnn.matmul(a, b, compute_kernel_config=c, dtype=ttnn.bfloat16))):
        legs = []
        for name, fid, acc in (("HiFi4+fp32", ttnn.MathFidelity.HiFi4, True),
                               ("HiFi2+fp32", ttnn.MathFidelity.HiFi2, True),
                               ("HiFi2", ttnn.MathFidelity.HiFi2, False),
                               ("LoFi", ttnn.MathFidelity.LoFi, False)):
            c = ckc_of(dev, fid, acc)
            legs.append((name, lambda fn_make=fn_make, c=c: fn_make(c)))
        med = interleave(legs, dev, warm=2, iters=5)
        base = med.get("HiFi4+fp32")
        print(f"    {shape}:")
        for k, v in sorted(med.items(), key=lambda kv: kv[1]):
            print(f"      {k}: {v:.3f} ms ({base / v:.2f}x vs HiFi4+fp32)")


def m6(dev, ckc):
    print("M6 qkv+g packed [256->1024] vs separate [256->768]+[256->256] (S=320)")
    S = N
    x = ttnn.from_torch(torch.randn(S, C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    wqkv = ttnn.from_torch(torch.randn(C_Z, 3 * C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    wg = ttnn.from_torch(torch.randn(C_Z, C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    wpacked = ttnn.from_torch(
        torch.cat([ttnn.to_torch(wqkv), ttnn.to_torch(wg)], dim=-1).contiguous(),
        layout=ttnn.TILE_LAYOUT, device=dev)

    def leg_sep():
        qkv = ttnn.experimental.minimal_matmul(x, wqkv, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        g = ttnn.experimental.minimal_matmul(x, wg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        return qkv, g

    def leg_packed():
        y = ttnn.experimental.minimal_matmul(x, wpacked, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        qkv, g = ttnn.chunk(y, 2, dim=-1)
        ttnn.deallocate(y)
        return qkv, g

    med = interleave([("separate", leg_sep), ("packed+chunk", leg_packed)], dev)
    print(f"    median ms: {med}  speedup sep/packed = {med['separate'] / med['packed+chunk']:.2f}x")
    q1, g1 = leg_sep()
    q2, g2 = leg_packed()
    ttnn.synchronize_device(dev)
    check("packed qkv vs separate", q2, ttnn.to_torch(q1))
    check("packed g vs separate", g2, ttnn.to_torch(g1))


def m7(dev, ckc, fine=False):
    print("M7 tri-attention SDPA program-config sweep (q [N,8,N,32], bias [1,8,N,N])")
    n = 128 if fine else N
    q = ttnn.from_torch(torch.randn(n, 8, n, 32, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    k = ttnn.from_torch(torch.randn(n, 8, n, 32, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(torch.randn(n, 8, n, 32, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev)
    bias = ttnn.from_torch(torch.randn(1, 8, n, n, dtype=torch.bfloat16),
                           layout=ttnn.TILE_LAYOUT, device=dev)
    flops = 2 * 2 * n * 8 * n * n * 32

    legs = []
    combos = ((32, 32), (64, 64), (64, 128), (128, 64), (128, 128)) if fine else \
        ((32, 32), (64, 64), (128, 128), (256, 256), (64, 256), (128, 256), (256, 128))
    for qc, kc in combos:
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(13, 10),
                                    exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)

        def leg(pc=pc):
            return ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=32 ** -0.5, program_config=pc)
        legs.append((f"q{qc}k{kc}", leg))
    med = interleave(legs, dev, warm=2, iters=5)
    cur = med.get("q256k256") or med.get("q128k128")  # current: _capped_sdpa_chunk_size(N)
    for kk, vv in sorted(med.items(), key=lambda kv: kv[1]):
        print(f"    {kk}: {vv:.3f} ms  {flops / vv / 1e9:.1f} TFLOP/s  ({cur / vv:.2f}x vs current)")
    # numerics: chunking changes the online-softmax reduction order -- measure it
    ref_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(13, 10),
                                    exp_approx_mode=False, q_chunk_size=256 if not fine else 128,
                                    k_chunk_size=256 if not fine else 128)
    best_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(13, 10),
                                     exp_approx_mode=False, q_chunk_size=64, k_chunk_size=64)
    a = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False,
                                                      scale=32 ** -0.5, program_config=ref_pc)
    b = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False,
                                                      scale=32 ** -0.5, program_config=best_pc)
    ttnn.synchronize_device(dev)
    check("q64k64 vs current-config", b, ttnn.to_torch(a))


def m7c(dev, ckc):
    """Cross-size guard for the M7 winner: boltz2/esmfold2 tri-att shapes (L=512/1024,
    same head_dim=32). If q64k64 does not win there too, the landing must be size-gated."""
    print("M7c SDPA chunk winner at OTHER models' shapes")
    for n in (512, 1024):
        q = ttnn.from_torch(torch.randn(n, 8, n, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        k = ttnn.from_torch(torch.randn(n, 8, n, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        v = ttnn.from_torch(torch.randn(n, 8, n, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        bias = ttnn.from_torch(torch.randn(1, 8, n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        cur_chunk = min(256, ((n + 31) // 32) * 32)
        legs = []
        for qc, kc in ((64, 64), (128, 128), (cur_chunk, cur_chunk)):
            pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(13, 10),
                                        exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)

            def leg(pc=pc):
                return ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, is_causal=False, scale=32 ** -0.5, program_config=pc)
            legs.append((f"q{qc}k{kc}", leg))
        med = interleave(legs, dev, warm=2, iters=5)
        cur = med.get(f"q{cur_chunk}k{cur_chunk}")
        print(f"  N={n} (current chunk {cur_chunk}):")
        for kk, vv in sorted(med.items(), key=lambda kv: kv[1]):
            print(f"    {kk}: {vv:.3f} ms ({cur / vv:.2f}x vs current)")
        del q, k, v, bias


def m2b(dev, ckc):
    """Cross-size guard for the M2 winner: boltz2 trimul contraction at L=512 (Kt=16)."""
    print("M2b trimul contraction at [1,32,512,512] (Kt=16): ibw 1 vs 8")
    a = ttnn.from_torch(torch.randn(1, 32, 512, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(1, 32, 512, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    gx, gy = 13, 10
    tiles = 16
    pm, pn = -(-tiles // gy), -(-tiles // gx)
    legs = []
    for ibw in (1, 8):
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw,
            out_subblock_h=1, out_subblock_w=1, out_block_h=pm, out_block_w=pn,
            per_core_M=pm, per_core_N=pn, transpose_mcast=False,
            fused_activation=None, fuse_batch=False)

        def leg(pc=pc):
            return ttnn.matmul(a, b, compute_kernel_config=ckc,
                               memory_config=ttnn.L1_MEMORY_CONFIG, program_config=pc,
                               dtype=ttnn.bfloat16)
        legs.append((f"ibw{ibw}", leg))
    med = interleave(legs, dev, warm=2, iters=5)
    print(f"    {med}  speedup ibw1/ibw8 = {med['ibw1'] / med['ibw8']:.2f}x")


def m4b(dev, ckc):
    """Cross-shape guard for the M4 winner: boltz2 transition at W=1024, c=128."""
    print("M4b transition at boltz2 shape [1,h,1024,128], h=16 vs 64")
    w1 = ttnn.from_torch(torch.randn(128, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    w2 = ttnn.from_torch(torch.randn(128, 512, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    w3 = ttnn.from_torch(torch.randn(512, 128, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    lnw = ttnn.from_torch(torch.randn(128, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    lnb = ttnn.from_torch(torch.randn(128, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)

    def swiglu(x):
        xn = ttnn.layer_norm(x, weight=lnw, bias=lnb, epsilon=1e-5,
                             compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
        x1 = ttnn.linear(xn, w1, activation="silu", compute_kernel_config=ckc,
                         memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        x2 = ttnn.linear(xn, w2, compute_kernel_config=ckc,
                         memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(xn)
        x = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        xd = ttnn.linear(x, w3, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                         core_grid=CORE_GRID_MAIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        return xd

    legs = []
    for h in (16, 64):
        xh = ttnn.from_torch(torch.randn(1, h, 1024, 128, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev)
        n_chunks = -(-1024 // h)

        def leg(xh=xh, n_chunks=n_chunks):
            outs = []
            for c in ttnn.chunk(xh, n_chunks, dim=1):
                outs.append(swiglu(c))
            y = ttnn.concat(outs, dim=1)
            for o in outs:
                ttnn.deallocate(o)
            return y
        legs.append((f"h{h}(x{n_chunks})", leg))
    med = interleave(legs, dev, warm=2, iters=5)
    print(f"    {med}  speedup h16/h64 = {med['h16(x64)'] / med['h64(x16)']:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["m1", "m2", "m2b", "m4", "m4b", "m5", "m6", "m7", "m7b", "m7c"], default=None)
    args = ap.parse_args()
    dev = get_device()
    ckc = ckc_of(dev, ttnn.MathFidelity.HiFi4, True)
    todo = [args.only] if args.only else ["m1", "m2", "m4", "m5", "m6", "m7"]
    for name in todo:
        {"m1": lambda: m1(dev, ckc), "m2": lambda: m2(dev, ckc), "m2b": lambda: m2b(dev, ckc),
         "m4": lambda: m4(dev, ckc), "m4b": lambda: m4b(dev, ckc), "m5": lambda: m5(dev),
         "m6": lambda: m6(dev, ckc), "m7": lambda: m7(dev, ckc), "m7b": lambda: m7(dev, ckc, fine=True),
         "m7c": lambda: m7c(dev, ckc)}[name]()


if __name__ == "__main__":
    main()
