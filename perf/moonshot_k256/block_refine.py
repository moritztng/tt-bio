#!/usr/bin/env python3
"""Refine the MinimalMatmulConfig optimum found by block_sweep.py.

The coarse sweep found narrower N_block winning at both trimul shapes and both winners sat on
the edge of the grid it searched (N_block=2 at N=256, N_block=1 at N=128), so the optimum was
not bracketed. K_block is pinned at 8 (the whole K=256 in one accumulation pass): every
K_block < 8 arm was both slower AND not bit-exact, so the K-depth knob is rejected on two axes
and is not swept again here.

Adds triatt.qkv N=768 on the DRAM-out arm, which bw_bound.py measured as write-bound, to test
whether blocking moves a shape whose bound is the output write rather than the compute.
"""
import json, os, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
MM = ttnn.experimental.minimal_matmul
M, K, M_PAR = 102400, 256, 8192


def timed(dev, fn, warm=3, pipe=4, reps=7):
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
GX, GY = g.x, g.y
PROD = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
print(f"grid {GX}x{GY}  loadavg {os.getloadavg()[0]:.1f}", flush=True)
torch.manual_seed(0)
res = {"grid": [GX, GY], "load_start": round(os.getloadavg()[0], 2), "shapes": []}

for N, mem, memname in ((256, L1, "L1"), (128, L1, "L1"), (768, DRAM, "DRAM")):
    nt = N // 32
    cfgs = []
    for nb in (1, 2, 4, 8):
        if nb > nt:
            continue
        for mb in (2, 4, 8, 16, 32):
            for sh, sw in ((1, min(4, nb)), (2, min(2, nb)), (4, 1), (1, 1)):
                if sh > mb or sw > nb or mb % sh or nb % sw or sh * sw > 4:
                    continue
                c = (mb, 8, nb, sh, sw)
                if c not in cfgs:
                    cfgs.append(c)
    print(f"\n=== N={N} -> {memname}, {len(cfgs)} configs, 2 passes ===", flush=True)
    a = ttnn.from_torch(torch.randn(M, K) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(K, N) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ap = ttnn.from_torch(torch.randn(M_PAR, K) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=DRAM)
    ref = ttnn.to_torch(MM(ap, b, memory_config=mem, dtype=ttnn.bfloat16,
                           compute_kernel_config=PROD)).float()
    rows = {}
    for p in (1, 2):
        ms, aa = timed(dev, lambda: ttnn.deallocate(
            MM(a, b, memory_config=mem, dtype=ttnn.bfloat16, compute_kernel_config=PROD)))
        rows.setdefault("default", []).append(ms)
        for c in cfgs:
            mb, kb, nb, sh, sw = c
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
            try:
                ms, aa = timed(dev, lambda: ttnn.deallocate(
                    MM(a, b, memory_config=mem, dtype=ttnn.bfloat16,
                       compute_kernel_config=PROD, config=cfg)))
            except Exception:
                continue
            rows.setdefault(c, []).append(ms)
    out = []
    for key, mms in rows.items():
        if len(mms) < 2:
            continue
        med = st.median(mms)
        rec = {"config": "default" if key == "default" else list(key),
               "ms": round(med, 4), "tflops": round(2 * M * K * N / (med / 1e3) / 1e12, 2),
               "drift": round((max(mms) - min(mms)) / med, 4)}
        if key != "default":
            mb, kb, nb, sh, sw = key
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
            got = ttnn.to_torch(MM(ap, b, memory_config=mem, dtype=ttnn.bfloat16,
                                   compute_kernel_config=PROD, config=cfg)).float()
            rec["bit_exact"] = bool(torch.equal(got, ref))
        out.append(rec)
    out.sort(key=lambda r: -r["tflops"])
    base = next(r["tflops"] for r in out if r["config"] == "default")
    print(f"  default {base} TFLOP/s", flush=True)
    for r in out[:10]:
        print(f"    {str(r['config']):24s} {r['tflops']:7.2f}  {r['tflops']/base:5.3f}x  "
              f"exact={r.get('bit_exact')}  drift {r['drift']:.2%}", flush=True)
    res["shapes"].append({"M": M, "K": K, "N": N, "out": memname, "default_tflops": base,
                          "results": out})
    for t in (a, b, ap):
        ttnn.deallocate(t)

res["load_end"] = round(os.getloadavg()[0], 2)
json.dump(res, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1], res["load_start"], "->", res["load_end"], flush=True)
