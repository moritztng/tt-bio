#!/usr/bin/env python3
"""Where does the REMAINING gap live after fixing in0_block_w?
(a) move the operands/result out of DRAM one at a time
(b) re-measure the compute roof at the op's OWN ckc (HiFi4+fp32_dest_acc+packer_l1_acc),
    since roofline_bh.py measured 100.6 TF/s with fp32_dest_acc_en=False."""
import json, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

N, C_Z, H, D = 128, 256, 8, 32
GF = 2 * (N * N) * C_Z * (3 * H * D) / 1e9
BYTES = ((N * N * C_Z) + (C_Z * 3 * H * D) + (N * N * 3 * H * D)) * 2


def med(xs): return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=8, pipe=12, reps=5):
    for _ in range(warm): fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe): fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(o)


dev = get_device()
dg = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
res = {}

BEST = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(dg.x, dg.y), in0_block_w=8,
    out_subblock_h=1, out_subblock_w=4, out_block_h=4, out_block_w=24,
    per_core_M=4, per_core_N=24, fuse_batch=True, fused_activation=None, mcast_in0=False)

for xmem, omem, tag in ((ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "xDRAM_oDRAM"),
                        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, "xDRAM_oL1"),
                        (ttnn.L1_MEMORY_CONFIG,   ttnn.L1_MEMORY_CONFIG, "xL1_oL1"),
                        (ttnn.L1_MEMORY_CONFIG,   ttnn.DRAM_MEMORY_CONFIG, "xL1_oDRAM")):
    try:
        x = ttnn.from_torch(torch.randn(1, N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=xmem)
        w = ttnn.from_torch(torch.randn(C_Z, 3 * H * D), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for cfgname, cfg in (("autocfg", None), ("bw8", BEST)):
            kw = {"program_config": cfg} if cfg else {"core_grid": CORE_GRID_MAIN}
            f = lambda: ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                                    memory_config=omem, **kw))
            ms = timed(dev, f)
            tf, gbs = GF / (ms / 1e3) / 1e3, BYTES / (ms / 1e3) / 1e9
            res[f"{tag}_{cfgname}"] = {"ms": round(ms, 4), "tflops": round(tf, 2), "GBs": round(gbs, 1)}
            print(f"{tag}_{cfgname:8s} {ms:8.4f} ms {tf:7.2f} TF/s  {gbs:6.1f} GB/s", flush=True)
        ttnn.deallocate(x); ttnn.deallocate(w)
    except Exception as e:
        res[tag] = {"error": str(e)[:120]}
        print(f"{tag:20s} ERR {str(e)[:110]}", flush=True)

# --- roof correction: square matmul at the op's OWN ckc vs the roofline script's ckc
print("\n# compute roof at this op's ckc vs roofline_bh.py's ckc", flush=True)
for n in (4096,):
    a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gf = 2 * n ** 3 / 1e9
    for lbl, kc in (("HiFi4_fp32acc_pl1acc(OP)", ckc),
                    ("HiFi4_plain(ROOFLINE_SCRIPT)", ttnn.init_device_compute_kernel_config(
                        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=False, packer_l1_acc=False))):
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc)), warm=5, pipe=6, reps=5)
        res[f"roof_{n}_{lbl}"] = {"ms": round(ms, 4), "tflops": round(gf / (ms / 1e3) / 1e3, 2)}
        print(f"  N={n} {lbl:32s} {ms:8.4f} ms  {gf/(ms/1e3)/1e3:7.2f} TFLOP/s", flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)

json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qkv_probe3.json", "w"), indent=2)
