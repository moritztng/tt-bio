#!/usr/bin/env python3
"""W6 micro-experiments on the three biggest gaps in the non-trimul half at N=320.

E1  permute(1,0,2) of the pair tensor runs at 69.8 GB/s while ttnn.clone of the same
    52.4 MB tensor runs at 377 GB/s. Is a dedicated transpose kernel reachable?
E2  ttnn.linear(core_grid=11x10) vs ttnn.experimental.minimal_matmul at the identical
    [102400,256]@[256,256] shape: 0.685 vs 0.398 ms in the block. Confirm + bit-exactness.
E3  tri-attention SDPA runs at 11.5 TF/s / 72.6 GB/s, at neither roof. Sweep q/k chunk.
"""

import argparse
import json
import time

import torch

import ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

DEV = None


def tbytes(t):
    eb = {ttnn.bfloat16: 2, ttnn.float32: 4}.get(t.dtype, 2)
    sh = [int(d) for d in t.shape]
    v = 1
    for d in sh[:-2]:
        v *= d
    v *= ((sh[-2] + 31) // 32) * 32 * ((sh[-1] + 31) // 32) * 32
    return v * eb


def timeit(fn, warm=3, iters=9):
    for _ in range(warm):
        r = fn()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    return sorted(ts)[len(ts) // 2]


def e1(N, C, out):
    print(f"\n=== E1 layout roof: pair tensor {N}x{N}x{C} bf16 ===")
    torch.manual_seed(0)
    ref = torch.randn(N, N, C)
    x3 = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    x4 = ttnn.reshape(x3, (1, N, N, C))
    b = tbytes(x3)
    cases = {
        "clone (copy roof)": lambda: ttnn.clone(x3),
        "clone -> L1": lambda: ttnn.clone(x3, memory_config=ttnn.L1_MEMORY_CONFIG),
        "permute3 (1,0,2)": lambda: ttnn.permute(x3, (1, 0, 2)),
        "permute4 (0,2,1,3)": lambda: ttnn.permute(x4, (0, 2, 1, 3)),
        "transpose3 (0,1)": lambda: ttnn.transpose(x3, 0, 1),
        "transpose4 hc (1,2)": lambda: ttnn.transpose(x4, 1, 2),
        "permute3 -> L1": lambda: ttnn.permute(x3, (1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG),
        "transpose4 hc -> L1": lambda: ttnn.transpose(x4, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG),
    }
    res = {}
    gold = ref.permute(1, 0, 2)
    for name, fn in cases.items():
        try:
            ms = timeit(fn)
        except Exception as ex:
            print(f"  {name:24s} FAILED {type(ex).__name__}: {str(ex)[:120]}")
            continue
        # correctness (transposing cases only)
        tag = ""
        if "clone" not in name:
            r = fn()
            got = ttnn.to_torch(r).reshape(N, N, C).float()
            tag = "  EXACT" if torch.equal(got, gold.bfloat16().float()) else f"  PCC-diff max={float((got-gold).abs().max()):.4f}"
            ttnn.deallocate(r)
        res[name] = {"ms": ms, "GBps": 2 * b / ms * 1e-6}
        print(f"  {name:24s} {ms:7.3f} ms  {2*b/ms*1e-6:6.1f} GB/s{tag}")
    out["e1"] = res


def e2(N, C, out):
    print(f"\n=== E2 linear(core_grid) vs minimal_matmul: [{N*N},{C}] @ [{C},{C}] ===")
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(N, N, C) * 0.1, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(C, C) * 0.05, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    w8 = ttnn.from_torch(torch.randn(C, 8) * 0.05, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    cases = {
        "linear core_grid 11x10": lambda: ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN),
        "linear no core_grid": lambda: ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16),
        "minimal_matmul": lambda: ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16),
        "bias linear core_grid": lambda: ttnn.linear(x, w8, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN),
        "bias minimal_matmul": lambda: ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w8, compute_kernel_config=ckc, dtype=ttnn.bfloat16),
    }
    res = {}
    golds = {}
    for name, fn in cases.items():
        try:
            ms = timeit(fn)
        except Exception as ex:
            print(f"  {name:24s} FAILED {type(ex).__name__}: {str(ex)[:120]}")
            continue
        r = fn()
        key = "bias" if "bias" in name else "sq"
        g = ttnn.to_torch(r)
        eq = ""
        if key in golds:
            eq = "  BIT-EXACT vs first" if torch.equal(g, golds[key]) else f"  maxdiff={float((g.float()-golds[key].float()).abs().max()):.5f}"
        else:
            golds[key] = g
        b = tbytes(x) + tbytes(r)
        ttnn.deallocate(r)
        res[name] = {"ms": ms, "GBps": b / ms * 1e-6}
        print(f"  {name:24s} {ms:7.3f} ms  {b/ms*1e-6:6.1f} GB/s{eq}")
    out["e2"] = res


def e3(N, out):
    print(f"\n=== E3 tri-att SDPA chunk sweep: q[{N},8,{N},32] ===")
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    mk = lambda: ttnn.from_torch(torch.randn(N, 8, N, 32) * 0.1, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    q, k, v = mk(), mk(), mk()
    bias = ttnn.from_torch(torch.randn(1, 8, N, N) * 0.1, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    flops = 2 * (2 * N * 8 * N * N * 32)
    byt = 3 * tbytes(q) + tbytes(bias) + tbytes(q)
    res = {}
    base = None
    for qc, kc in ((32, 32), (64, 64), (64, 128), (128, 128), (128, 320), (320, 320), (64, 320), (256, 256)):
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN,
                                    exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)
        fn = lambda pc=pc: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=32 ** -0.5, program_config=pc)
        try:
            ms = timeit(fn, warm=2, iters=5)
        except Exception as ex:
            print(f"  q={qc:<4d} k={kc:<4d} FAILED {type(ex).__name__}: {str(ex)[:100]}")
            continue
        r = fn()
        g = ttnn.to_torch(r)
        tag = ""
        if base is None:
            base = g
        else:
            tag = "  BIT-EXACT vs q32k32" if torch.equal(g, base) else f"  maxdiff={float((g.float()-base.float()).abs().max()):.5f}"
        ttnn.deallocate(r)
        res[f"q{qc}_k{kc}"] = {"ms": ms, "tflops": flops / ms * 1e-9, "GBps": byt / ms * 1e-6}
        print(f"  q={qc:<4d} k={kc:<4d} {ms:7.3f} ms  {flops/ms*1e-9:6.2f} TF/s  {byt/ms*1e-6:6.1f} GB/s{tag}")
    out["e3"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    global DEV
    DEV = get_device()
    out = {}
    only = args.only.split(",") if args.only else ["e1", "e2", "e3"]
    if "e1" in only:
        e1(args.n, args.c, out)
    if "e2" in only:
        e2(args.n, args.c, out)
    if "e3" in only:
        e3(args.n, out)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
