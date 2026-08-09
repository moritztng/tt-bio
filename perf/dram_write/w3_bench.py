"""Time the production matmul rung and dump its result, so two kernel variants can be compared
bit-exactly and not just by PCC.

  python3 w3_bench.py <tag>   ->  writes /tmp/w3_<tag>.pt and prints W3 {...}
"""
import json, sys, time, statistics as st
import torch, ttnn

TAG = sys.argv[1]
DEV = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(13, 10)
M, N = 16384, 768
PER_CORE_M, PER_CORE_N = 4, 24

def ckc():
    return ttnn.WormholeComputeKernelConfig(
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
    ta = torch.randn(1, M, K); tw = torch.randn(K, N)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(tw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pc, k = cfg(obh), ckc()
    r = {}
    for arm, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
        y = ttnn.linear(a, w, program_config=pc, memory_config=mem, compute_kernel_config=k,
                        dtype=ttnn.bfloat16)
        ttnn.synchronize_device(DEV)
        out[name + "_" + arm] = ttnn.to_torch(y)[0].clone()
        ttnn.deallocate(y)
        for _ in range(8):
            ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                        compute_kernel_config=k, dtype=ttnn.bfloat16))
        inner = 6 if K > 512 else 12
        ts = []
        for _ in range(5):
            ttnn.synchronize_device(DEV)
            t0 = time.perf_counter()
            for _ in range(inner):
                ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                            compute_kernel_config=k, dtype=ttnn.bfloat16))
            ttnn.synchronize_device(DEV)
            ts.append((time.perf_counter() - t0) / inner * 1e6)
        r["t_%s_us" % arm] = round(st.median(ts), 2)
    r["delta_us"] = round(r["t_dram_us"] - r["t_l1_us"], 2)
    r["write_gbps_vs_l1arm"] = round(25.165824e6 / (r["delta_us"] * 1e-6) / 1e9, 1)
    res[name] = r
    ttnn.deallocate(a); ttnn.deallocate(w)

torch.save(out, "/tmp/w3_%s.pt" % TAG)
print("W3 " + json.dumps({"tag": TAG, "rungs": res}))
ttnn.close_device(DEV)
