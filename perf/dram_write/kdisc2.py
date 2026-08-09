import json, time, statistics as st
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(13, 10)
M, N = 16384, 768
PER_CORE_M, PER_CORE_N = 4, 24
def ckc(fp32=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=fp32, packer_l1_acc=True)

def cfg(in0_block_w, sbh=1, sbw=4, obh=PER_CORE_M, obw=PER_CORE_N):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID, in0_block_w=in0_block_w,
        out_subblock_h=sbh, out_subblock_w=sbw, out_block_h=obh, out_block_w=obw,
        per_core_M=PER_CORE_M, per_core_N=PER_CORE_N,
        fuse_batch=True, fused_activation=None, mcast_in0=False)

def time_it(a, w, pc, mem, k, warm=8, reps=5, inner=12):
    for _ in range(warm):
        ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                    compute_kernel_config=k, dtype=ttnn.bfloat16))
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(inner):
            ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                        compute_kernel_config=k, dtype=ttnn.bfloat16))
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) / inner * 1e3)
    return st.median(ts)

def mk(K):
    a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return a, w

res = {"outblock": [], "ksweep": [], "nofp32": []}

# --- A. out_block_h sweep at K=256: does a shallower out CB change anything? -----------------
a, w = mk(256)
for obh in [4, 2, 1]:
    r = {"out_block_h": obh, "h_blocks": PER_CORE_M // obh, "out_CB_tiles": obh * PER_CORE_N}
    for tag, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
        try:
            r["t_%s_ms" % tag] = time_it(a, w, cfg(8, 1, 4, obh, PER_CORE_N), mem, ckc())
        except Exception as e:
            r["t_%s_ms" % tag] = None; r["err_" + tag] = str(e)[:150]
    if r.get("t_dram_ms") and r.get("t_l1_ms"):
        r["delta_us"] = (r["t_dram_ms"] - r["t_l1_ms"]) * 1e3
        r["implied_write_gbps"] = 25.166e6 / (r["delta_us"] * 1e-6) / 1e9
    res["outblock"].append(r); print("OB " + json.dumps(r), flush=True)
ttnn.deallocate(a); ttnn.deallocate(w)

# --- B. THE DISCRIMINATOR: grow compute at fixed 25.166 MB output ----------------------------
for K in [256, 512, 1024, 2048, 4096]:
    a, w = mk(K)
    kt = K // 32
    for obh in [1]:
        r = {"K": K, "k_tiles": kt, "in0_block_w": 8, "out_block_h": obh}
        for tag, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
            try:
                r["t_%s_ms" % tag] = time_it(a, w, cfg(8, 1, 4, obh, PER_CORE_N), mem, ckc(), inner=6)
            except Exception as e:
                r["t_%s_ms" % tag] = None; r["err_" + tag] = str(e)[:150]
        if r.get("t_dram_ms") and r.get("t_l1_ms"):
            r["delta_us"] = (r["t_dram_ms"] - r["t_l1_ms"]) * 1e3
            r["implied_write_gbps"] = 25.166e6 / (r["delta_us"] * 1e-6) / 1e9
        res["ksweep"].append(r); print("KS " + json.dumps(r), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(w)

# --- C. barrier granularity, fp32_dest_acc off so 8-tile subblocks are legal ------------------
a, w = mk(256)
for fp32 in [True, False]:
    for sbh, sbw in [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4)]:
        if sbh * sbw > (4 if fp32 else 8):
            continue
        r = {"fp32_dest_acc": fp32, "sb": [sbh, sbw],
             "barriers_per_core": (PER_CORE_M // sbh) * (PER_CORE_N // sbw)}
        for tag, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
            try:
                r["t_%s_ms" % tag] = time_it(a, w, cfg(8, sbh, sbw, 1, PER_CORE_N), mem, ckc(fp32))
            except Exception as e:
                r["t_%s_ms" % tag] = None; r["err_" + tag] = str(e)[:150]
        if r.get("t_dram_ms") and r.get("t_l1_ms"):
            r["delta_us"] = (r["t_dram_ms"] - r["t_l1_ms"]) * 1e3
        res["nofp32"].append(r); print("SB " + json.dumps(r), flush=True)

print("RESULT_JSON " + json.dumps(res))
ttnn.close_device(DEV)
