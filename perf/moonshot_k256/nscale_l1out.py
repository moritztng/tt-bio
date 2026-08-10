#!/usr/bin/env python3
"""Is the production (L1-out) trunk matmul A-read bound or arithmetic bound at K=256?

bw_bound.py swept math fidelity only on the `-> DRAM` arms, which are write-bound, then
applied "not FPU-bound" to the `-> L1` arms production actually runs. That inference does not
follow. Two arms settle it, on the same shape family (M=102400, K=256, bf16 A in DRAM, bf16
result in L1, production compute kernel config unless stated):

  1. N sweep at fixed M,K. If the op is bound by reading A (52.4 MB, identical for every N),
     the milliseconds are FLAT in N and TFLOP/s scales linearly. If it is arithmetic bound at
     N=256, the milliseconds grow with N past 256 and TFLOP/s plateaus.
  2. Fidelity at L1 out. If LoFi/HiFi2 beat HiFi4+fp32_dest_acc on the SAME L1-out arm, the
     52.81 figure is a fidelity ceiling, not a bandwidth one.

Also names the arithmetic roof at the production config, which the record only has for
fp32_dest_acc_en=False.
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
MM = ttnn.experimental.minimal_matmul


def timed(dev, fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(out), (max(out) - min(out)) / st.median(out)


dev = get_device()
g = dev.compute_with_storage_grid_size()
print(f"grid {g.x}x{g.y} = {g.x*g.y} cores", flush=True)
res = {"grid": [g.x, g.y], "nsweep": [], "fidelity": [], "roof": []}


def ckc(fid, fp32acc, l1acc=True):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, fp32_dest_acc_en=fp32acc, packer_l1_acc=l1acc)


PROD = ckc(ttnn.MathFidelity.HiFi4, True)
FIDS = [("HiFi4", ttnn.MathFidelity.HiFi4), ("HiFi3", ttnn.MathFidelity.HiFi3),
        ("HiFi2", ttnn.MathFidelity.HiFi2), ("LoFi", ttnn.MathFidelity.LoFi)]

torch.manual_seed(0)

# ---------------------------------------------------------------- 1. N sweep, L1 out, prod ckc
print("\n== N sweep, M=102400 K=256, bf16 A(DRAM) -> bf16 L1, production ckc ==", flush=True)
print("  N     ms     TFLOP/s   A-read GB/s   A/A", flush=True)
M, K = 102400, 256
a = ttnn.from_torch(torch.randn(M, K) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16, memory_config=DRAM)
abytes = M * K * 2
for N in (128, 256, 384, 512):
    try:
        b = ttnn.from_torch(torch.randn(K, N) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        ms, aa = timed(dev, lambda: ttnn.deallocate(
            MM(a, b, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=PROD)))
    except Exception as e:
        print(f"  {N:5d}  ERR {str(e)[:70]}", flush=True)
        res["nsweep"].append({"n": N, "error": str(e)[:200]})
        continue
    tf = 2 * M * K * N / (ms / 1e3) / 1e12
    rgb = abytes / (ms / 1e3) / 1e9
    print(f"  {N:5d}  {ms:7.3f}  {tf:7.2f}   {rgb:8.1f}     {aa:.2%}", flush=True)
    res["nsweep"].append({"n": N, "ms": round(ms, 4), "tflops": round(tf, 2),
                          "a_read_GBs": round(rgb, 1), "aa_spread": round(aa, 4)})
    ttnn.deallocate(b)

# --------------------------------------------------- 2. fidelity ON the L1-out arm, N=128/256
print("\n== fidelity on the L1-out arm (the sweep bw_bound.py never ran) ==", flush=True)
for N in (128, 256):
    b = ttnn.from_torch(torch.randn(K, N) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    for name, fid in FIDS:
        for acc in (True, False):
            try:
                c = ckc(fid, acc)
                ms, aa = timed(dev, lambda: ttnn.deallocate(
                    MM(a, b, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=c)))
            except Exception as e:
                print(f"  N={N} {name} fp32acc={acc}  ERR {str(e)[:60]}", flush=True)
                continue
            tf = 2 * M * K * N / (ms / 1e3) / 1e12
            print(f"  N={N:4d} {name:6s} fp32acc={str(acc):5s}  {ms:7.3f} ms  {tf:7.2f} TFLOP/s"
                  f"   read {abytes/(ms/1e3)/1e9:7.1f} GB/s  (A/A {aa:.2%})", flush=True)
            res["fidelity"].append({"n": N, "fidelity": name, "fp32_dest_acc": acc,
                                    "ms": round(ms, 4), "tflops": round(tf, 2),
                                    "a_read_GBs": round(abytes / (ms / 1e3) / 1e9, 1),
                                    "aa_spread": round(aa, 4)})
    ttnn.deallocate(b)
ttnn.deallocate(a)

# ------------------------------------------------- 3. arithmetic roof AT THE PRODUCTION CONFIG
print("\n== square roof, 2048^3, by fidelity, both fp32_dest_acc settings ==", flush=True)
S = 2048
sa = ttnn.from_torch(torch.randn(S, S) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                     dtype=ttnn.bfloat16, memory_config=DRAM)
sb = ttnn.from_torch(torch.randn(S, S) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                     dtype=ttnn.bfloat16, memory_config=DRAM)
for name, fid in FIDS:
    for acc in (True, False):
        try:
            c = ckc(fid, acc)
            ms, aa = timed(dev, lambda: ttnn.deallocate(
                MM(sa, sb, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=c)))
        except Exception as e:
            print(f"  {name} fp32acc={acc}  ERR {str(e)[:60]}", flush=True)
            continue
        tf = 2 * S ** 3 / (ms / 1e3) / 1e12
        print(f"  {name:6s} fp32acc={str(acc):5s}  {ms:7.3f} ms  {tf:7.2f} TFLOP/s  (A/A {aa:.2%})",
              flush=True)
        res["roof"].append({"s": S, "fidelity": name, "fp32_dest_acc": acc,
                            "ms": round(ms, 4), "tflops": round(tf, 2), "aa_spread": round(aa, 4)})

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1], flush=True)
