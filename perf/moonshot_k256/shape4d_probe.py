#!/usr/bin/env python3
"""Does the block_refine winner survive on the shape production actually issues?

block_sweep.py / block_refine.py measured a FLATTENED 2D stand-in, [102400, 256] @ [256, N].
Production's trimul input projection (tenstorrent.py:1656) passes x_norm_in 4D,
[1, seq, seq, c_z] @ [c_z, 4*chunk]. ttnn folds leading dims into batch, so the real call
presents batch=seq with ceil(seq/32) M-tiles each, not one ceil(seq^2/32)=3200 M-tile grid.
M_block_size is in M-tiles, so M_block_size=8 divides 3200 but NOT the 10 M-tiles of seq=320.
That is the documented flattened-M stand-in trap (memory
perfwar-flattened-m-standin-and-remote-lease-card-mismatch: a config test on a flattened stand-in
declined all 4832 calls in the first real in-fold run).

Also fixes a second mismatch: at seq 512 the in_proj result goes to DRAM, not L1
(_triangle_mul_memory_config, TRIANGLE_MULT_L1_MAX_SEQ=352), so the N=128 gain measured on an
L1-out arm is not the 512 aa production configuration either.

Arms, per production shape: default config, the sweep winner, and M_block_size values that do
divide the real per-batch M-tile count. Reports rate, whether the config was legal, and whether
it is bit-exact against the default.
"""
import json, os, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
MM = ttnn.experimental.minimal_matmul
K = 256


def timed(dev, fn, warm=2, pipe=2, reps=5):
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
print("grid %dx%d  loadavg %.1f" % (GX, GY, os.getloadavg()[0]), flush=True)
torch.manual_seed(0)
res = {"grid": [GX, GY], "load_start": round(os.getloadavg()[0], 2), "cases": []}

# (label, seq, N, memory_config, memname).  N = 4 * _trimul_chunk_size at that seq on this grid.
CASES = [
    ("298aa in_proj  4D [1,320,320,256]@[256,256] -> L1", 320, 256, L1, "L1"),
    ("298aa in_proj  2D    [102400,256]@[256,256] -> L1", None, 256, L1, "L1"),
    ("512aa in_proj  4D [1,512,512,256]@[256,128] -> DRAM", 512, 128, DRAM, "DRAM"),
    ("512aa in_proj  2D    [262144,256]@[256,128] -> DRAM", None, 128, DRAM, "DRAM"),
    ("512aa chunk64  4D [1,512,512,256]@[256,256] -> DRAM", 512, 256, DRAM, "DRAM"),
]

for label, seq, N, mem, memname in CASES:
    if seq is None:
        Mtot = 102400 if (N == 256 and mem is L1) else 262144
        a_shape, mtiles = (Mtot, K), Mtot // 32
    else:
        a_shape, mtiles = (1, seq, seq, K), (seq + 31) // 32
        Mtot = seq * seq
    print("\n=== %s   per-batch M-tiles=%d ===" % (label, mtiles), flush=True)
    a = ttnn.from_torch(torch.randn(*a_shape) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(K, N) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    ref = ttnn.to_torch(MM(a, b, memory_config=mem, dtype=ttnn.bfloat16,
                           compute_kernel_config=PROD)).float()
    base_ms, base_aa = timed(dev, lambda: ttnn.deallocate(
        MM(a, b, memory_config=mem, dtype=ttnn.bfloat16, compute_kernel_config=PROD)))
    base_tf = 2 * Mtot * K * N / (base_ms / 1e3) / 1e12
    print("  default            %7.2f TFLOP/s  A/A %.2f%%" % (base_tf, base_aa * 100), flush=True)
    rows = [{"config": "default", "ms": round(base_ms, 4), "tflops": round(base_tf, 2),
             "aa": round(base_aa, 4)}]
    # sweep winner (8) plus every M_block that divides the real per-batch M-tile count
    mbs = sorted({8, 4, 2, 1} | set(d for d in (1, 2, 4, 5, 8, 10, 16) if mtiles % d == 0))
    for mb in mbs:
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb, K_block_size=8, N_block_size=1, subblock_h=1, subblock_w=1,
            compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY))
        try:
            ms, aa = timed(dev, lambda: ttnn.deallocate(
                MM(a, b, memory_config=mem, dtype=ttnn.bfloat16,
                   compute_kernel_config=PROD, config=cfg)))
            got = ttnn.to_torch(MM(a, b, memory_config=mem, dtype=ttnn.bfloat16,
                                   compute_kernel_config=PROD, config=cfg)).float()
            tf = 2 * Mtot * K * N / (ms / 1e3) / 1e12
            exact = bool(torch.equal(got, ref))
            mx = float((got - ref).abs().max())
            print("  M_blk=%-3d N_blk=1  %7.2f TFLOP/s  %5.3fx  exact=%s  max_abs=%.3e  A/A %.2f%%"
                  % (mb, tf, tf / base_tf, exact, mx, aa * 100), flush=True)
            rows.append({"config": [mb, 8, 1, 1, 1], "ms": round(ms, 4),
                         "tflops": round(tf, 2), "gain": round(tf / base_tf, 4),
                         "bit_exact": exact, "max_abs": mx, "aa": round(aa, 4)})
        except Exception as e:
            msg = str(e).split("\n")[0][:160]
            print("  M_blk=%-3d N_blk=1  THREW: %s" % (mb, msg), flush=True)
            rows.append({"config": [mb, 8, 1, 1, 1], "threw": msg})
    res["cases"].append({"label": label, "seq": seq, "N": N, "out": memname,
                         "m_tiles_per_batch": mtiles, "M_total": Mtot, "rows": rows})
    for t in (a, b):
        ttnn.deallocate(t)

res["load_end"] = round(os.getloadavg()[0], 2)
json.dump(res, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1], res["load_start"], "->", res["load_end"], flush=True)
