#!/usr/bin/env python3
"""The qkv op reached 102.8 TFLOP/s L1-resident, above the 100.6 TFLOP/s figure this thread has been
calling the compute roof. That figure came from roofline_bh.py, which measures square matmuls with
fp32_dest_acc_en=False, packer_l1_acc=False and the result written to DRAM -- so it is not a compute
roof at all, it is a DRAM-bound matmul. Re-measure the roof properly.

Also closes the acceptance check the placement ladder owed: at fixed in0_block_w the four memory
placements must produce bit-identical output, because memory placement cannot change arithmetic.
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
res = {}
torch.manual_seed(0)

print("=== compute roof: square bf16 matmul, HiFi4, by result placement ===", flush=True)
ckc_op = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
ckc_plain = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False)
roof = {}
for n in (2048, 4096, 6144):
    a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gf = 2 * n ** 3 / 1e9
    for lbl, kc, omem in (("op_ckc_oDRAM", ckc_op, DRAM), ("op_ckc_oL1", ckc_op, L1),
                          ("plain_ckc_oDRAM", ckc_plain, DRAM), ("plain_ckc_oL1", ckc_plain, L1)):
        try:
            ms = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=omem)))
        except Exception as e:
            print(f"  N={n} {lbl:16s} ERR {str(e)[:70]}", flush=True)
            continue
        tf = gf / (ms / 1e3) / 1e3
        roof[f"{n}_{lbl}"] = {"ms": round(ms, 4), "tflops": round(tf, 2)}
        print(f"  N={n:<5} {lbl:16s} {ms:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
peak = max((v["tflops"] for v in roof.values()), default=0.0)
res["compute_roof"] = {"runs": roof, "peak_TFLOPs": peak}
print(f"MEASURED_HiFi4_COMPUTE_ROOF {peak:.2f} TFLOP/s", flush=True)

print("\n=== placement ladder must be bit-identical at fixed in0_block_w ===", flush=True)
N, C_Z, H, D = 128, 256, 8, 32
at, wt = torch.randn(1, N, N, C_Z), torch.randn(C_Z, 3 * H * D)
BEST = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(dg.x, dg.y), in0_block_w=8,
    out_subblock_h=1, out_subblock_w=4, out_block_h=4, out_block_w=24,
    per_core_M=4, per_core_N=24, fuse_batch=True, fused_activation=None, mcast_in0=False)
outs, w = {}, ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
for tag, xmem, omem in (("xDRAM_oDRAM", DRAM, DRAM), ("xDRAM_oL1", DRAM, L1),
                        ("xL1_oL1", L1, L1), ("xL1_oDRAM", L1, DRAM)):
    x = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=xmem)
    o = ttnn.linear(x, w, compute_kernel_config=ckc_op, memory_config=omem, program_config=BEST)
    outs[tag] = ttnn.to_torch(o)
    ttnn.deallocate(o)
    ttnn.deallocate(x)
ref_tag = "xDRAM_oDRAM"
pl = {t: bool(torch.equal(outs[t], outs[ref_tag])) for t in outs}
res["placement_bit_identical"] = pl
print(" ", json.dumps(pl), flush=True)
res["placement_all_bit_identical"] = all(pl.values())

json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/compute_roof.json", "w"), indent=2)
print("wrote", sys.argv[1] if len(sys.argv) > 1 else "/tmp/compute_roof.json", flush=True)
