#!/usr/bin/env python3
"""Two follow-ups to the 135.5 TFLOP/s roof measurement:

(a) has the square-matmul roof saturated by N=6144, or does it keep climbing?
(b) the qkv op reaches 102.8 TFLOP/s L1-resident, 76% of that roof. Is the remaining 24% the
    thinness of K (8 tiles), i.e. one pack of a 96-tile output block per 768 tile-MACs? Sweep K at
    fixed M and N, everything L1-resident, and see where the rate lands as K grows.
"""
import json, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=5, pipe=6, reps=5):
    for _ in range(warm):
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


dev = get_device()
dg = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
res = {}

print("=== (a) square-matmul roof saturation (op ckc, result to DRAM) ===", flush=True)
sat = {}
for n in (6144, 8192):
    a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gf = 2 * n ** 3 / 1e9
    try:
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=DRAM)),
                   warm=3, pipe=3, reps=3)
        sat[n] = {"ms": round(ms, 4), "tflops": round(gf / (ms / 1e3) / 1e3, 2)}
        print(f"  N={n:<6} {ms:9.4f} ms {gf/(ms/1e3)/1e3:8.2f} TFLOP/s", flush=True)
    except Exception as e:
        print(f"  N={n} ERR {str(e)[:70]}", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
res["roof_saturation"] = sat

print("\n=== (b) K sweep at the qkv M and N, everything L1-resident ===", flush=True)
M, NN = 16384, 768
ks = {}
for K in (256, 512, 1024, 2048):
    kt = K // 32
    try:
        x = ttnn.from_torch(torch.randn(1, M, K), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        w = ttnn.from_torch(torch.randn(K, NN), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(dg.x, dg.y), in0_block_w=kt,
            out_subblock_h=1, out_subblock_w=4, out_block_h=4, out_block_w=24,
            per_core_M=4, per_core_N=24, fuse_batch=True, fused_activation=None, mcast_in0=False)
        gf = 2 * M * K * NN / 1e9
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                                            memory_config=L1, program_config=cfg)),
                   warm=5, pipe=8, reps=5)
        ks[K] = {"k_tiles": kt, "GFLOP": round(gf, 3), "ms": round(ms, 4),
                 "tflops": round(gf / (ms / 1e3) / 1e3, 2)}
        print(f"  K={K:<5} ({kt:2d} tiles) {gf:7.2f} GF {ms:9.4f} ms {gf/(ms/1e3)/1e3:8.2f} TFLOP/s", flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
    except Exception as e:
        ks[K] = {"error": str(e)[:110]}
        print(f"  K={K} ERR {str(e)[:80]}", flush=True)
res["k_sweep_L1_resident"] = ks

json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/roof_ksweep.json", "w"), indent=2)
print("wrote", sys.argv[1] if len(sys.argv) > 1 else "/tmp/roof_ksweep.json", flush=True)
