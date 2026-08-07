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
        yr = ttnn.reshape(y, (1, H_CHUNK, N, C_HID))
        ttnn.deallocate(y)
        return yr

    def leg_reshape_only():
        xr = ttnn.reshape(x4, (1, 1, H_CHUNK * N, C_Z))
        return xr

    med = interleave([("4d", leg_4d), ("tall", leg_tall), ("reshape_only", leg_reshape_only)], dev)
    print(f"    median ms: {med}  speedup 4d/tall = {med['4d'] / med['tall']:.2f}x")
    a, b = leg_4d(), leg_tall()
    ttnn.synchronize_device(dev)
    check("tall vs 4d", b, ttnn.to_torch(a))


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
    for ibw in (1, 2, 4, 10):
        for osh, osw in ((1, 1), (2, 2)):
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=ibw, out_subblock_h=osh, out_subblock_w=osw,
                out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn,
                transpose_mcast=False, fused_activation=None, fuse_batch=False)

            def leg(pc=pc):
                return ttnn.matmul(a, b, compute_kernel_config=ckc,
                                   memory_config=ttnn.L1_MEMORY_CONFIG, program_config=pc,
                                   dtype=ttnn.bfloat16)
            legs.append((f"ibw{ibw}_sb{osh}x{osw}", leg))
    med = interleave(legs, dev, warm=2, iters=5)
    cur = med.get("ibw1_sb1x1")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["m1", "m2", "m4", "m5", "m6"], default=None)
    args = ap.parse_args()
    dev = get_device()
    ckc = ckc_of(dev, ttnn.MathFidelity.HiFi4, True)
    todo = [args.only] if args.only else ["m1", "m2", "m4", "m5", "m6"]
    for name in todo:
        {"m1": lambda: m1(dev, ckc), "m2": lambda: m2(dev, ckc),
         "m4": lambda: m4(dev, ckc), "m5": lambda: m5(dev),
         "m6": lambda: m6(dev, ckc)}[name]()


if __name__ == "__main__":
    main()
