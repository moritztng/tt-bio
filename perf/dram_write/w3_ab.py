"""A/B the matmul output drain against the stock writer, on the rung the root-cause used.

Rung K256_obh4 is the instrumented case from tt-bio-dram-write-serialization: M=16384 K=256 N=768,
per_core_M=4 per_core_N=24, 128 cores, 25.166 MB bf16 output. Baseline on card 3 was 167.8 us with
the writeback at 163.9 GB/s = 59% of the 277.6 GB/s device write roof.

Both arms (DRAM out, L1 out) are timed because the L1 arm is the control: a change that only helps
the DRAM arm is acting on the write, one that moves both is acting on something else.

  python3 w3_ab.py <tag>   -> /tmp/w3_<tag>.pt + one JSON line
"""
import json, sys, time, statistics as st
import torch, ttnn

TAG = sys.argv[1]
DEV = ttnn.open_device(device_id=0)
g = DEV.compute_with_storage_grid_size()
GRID = ttnn.CoreCoord(g.x, g.y)
M, N = 16384, 768
PER_CORE_M, PER_CORE_N = 4, 24

CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def cfg(obh):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID, in0_block_w=8,
        out_subblock_h=1, out_subblock_w=4, out_block_h=obh, out_block_w=PER_CORE_N,
        per_core_M=PER_CORE_M, per_core_N=PER_CORE_N,
        fuse_batch=True, fused_activation=None, mcast_in0=False)


torch.manual_seed(0)
out, res = {}, {}
for name, K, obh in [("K256_obh4", 256, 4), ("K4096_obh1", 4096, 1)]:
    ta = torch.randn(1, M, K)
    tw = torch.randn(K, N)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(tw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pc = cfg(obh)
    r = {}
    for arm, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
        y = ttnn.linear(a, w, program_config=pc, memory_config=mem, compute_kernel_config=CKC,
                        dtype=ttnn.bfloat16)
        ttnn.synchronize_device(DEV)
        out[name + "_" + arm] = ttnn.to_torch(y)[0].clone()
        ttnn.deallocate(y)
        for _ in range(8):
            ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                        compute_kernel_config=CKC, dtype=ttnn.bfloat16))
        inner = 6 if K > 512 else 12
        ts = []
        for _ in range(5):
            ttnn.synchronize_device(DEV)
            t0 = time.perf_counter()
            for _ in range(inner):
                ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                            compute_kernel_config=CKC, dtype=ttnn.bfloat16))
            ttnn.synchronize_device(DEV)
            ts.append((time.perf_counter() - t0) / inner * 1e6)
        r["t_%s_us" % arm] = round(st.median(ts), 2)
        r["t_%s_spread_us" % arm] = round(max(ts) - min(ts), 2)
    res[name] = r
    ttnn.deallocate(a)
    ttnn.deallocate(w)

torch.save(out, "/tmp/w3_%s.pt" % TAG)
print("W3 " + json.dumps({"tag": TAG, "grid": [g.x, g.y], "rungs": res}))
ttnn.close_device(DEV)
