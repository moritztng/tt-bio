"""Time + PCC the production rung. Run once against the stock writer and once against the
deferred-barrier writer; the only thing that changes between the two is the kernel source."""
import json, time, statistics as st
import torch, ttnn

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

def pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))

res = {}
for K, obh in [(256, 4), (256, 1), (4096, 1)]:
    ta = torch.randn(1, M, K); tw = torch.randn(K, N)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(tw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ref = (ta.to(torch.bfloat16).float() @ tw.to(torch.bfloat16).float())
    r = {"K": K, "out_block_h": obh}
    for tag, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
        pc, k = cfg(obh), ckc()
        y = ttnn.linear(a, w, program_config=pc, memory_config=mem, compute_kernel_config=k,
                        dtype=ttnn.bfloat16)
        r["pcc_" + tag] = pcc(ttnn.to_torch(y)[0], ref)
        ttnn.deallocate(y)
        for _ in range(8):
            ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                        compute_kernel_config=k, dtype=ttnn.bfloat16))
        ts = []
        inner = 6 if K > 512 else 12
        for _ in range(5):
            ttnn.synchronize_device(DEV)
            t0 = time.perf_counter()
            for _ in range(inner):
                ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                            compute_kernel_config=k, dtype=ttnn.bfloat16))
            ttnn.synchronize_device(DEV)
            ts.append((time.perf_counter() - t0) / inner * 1e3)
        r["t_%s_ms" % tag] = st.median(ts)
    r["delta_us"] = (r["t_dram_ms"] - r["t_l1_ms"]) * 1e3
    r["dram_write_gbps_vs_l1arm"] = 25.166e6 / (r["delta_us"] * 1e-6) / 1e9
    print("BF " + json.dumps(r), flush=True)
    res["K%d_obh%d" % (K, obh)] = r
    ttnn.deallocate(a); ttnn.deallocate(w)

print("RESULT_JSON " + json.dumps(res))
ttnn.close_device(DEV)
