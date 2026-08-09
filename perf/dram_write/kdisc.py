"""Discriminate two models that BOTH fit the single measured point 174.3 us.

  model S (serialized): T_dram(K) = T_l1(K) + W,  W = 25.17MB / 262 GB/s = 96 us, constant in K
  model O (overlapped): T_dram(K) = max(T_l1(K), 25.17MB / BW_eff), BW_eff = 144 GB/s

At K=256 both predict 174.3. They diverge as compute grows at fixed output bytes:
S keeps delta = T_dram - T_l1 pinned near 96 us; O drives delta to 0 once compute exceeds the write.
"""
import json, time, statistics as st
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(13, 10)
M, N = 16384, 768
M_T, N_T = M // 32, N // 32
PER_CORE_M, PER_CORE_N = 4, 24
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

def cfg(k_tiles, in0_block_w, sbh=1, sbw=4):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID, in0_block_w=in0_block_w,
        out_subblock_h=sbh, out_subblock_w=sbw,
        per_core_M=PER_CORE_M, per_core_N=PER_CORE_N,
        fuse_batch=True, fused_activation=None, mcast_in0=False)

def time_it(a, w, pc, mem, warm=8, reps=5, inner=12):
    for _ in range(warm):
        ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                    compute_kernel_config=CKC, dtype=ttnn.bfloat16))
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        outs = [ttnn.linear(a, w, program_config=pc, memory_config=mem,
                            compute_kernel_config=CKC, dtype=ttnn.bfloat16) for _ in range(inner)]
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) / inner * 1e3)
        for o in outs:
            ttnn.deallocate(o)
    return st.median(ts)

res = {"note": "M=16384 N=768 fixed; output is always 25.166 MB bf16", "ksweep": [], "subblock": []}
for K in [256, 512, 1024, 2048]:
    kt = K // 32
    a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bw = 8 if kt >= 8 else kt
    pc = cfg(kt, bw)
    row = {"K": K, "k_tiles": kt, "in0_block_w": bw, "num_k_blocks": kt // bw}
    try:
        row["t_dram_ms"] = time_it(a, w, pc, ttnn.DRAM_MEMORY_CONFIG)
    except Exception as e:
        row["t_dram_ms"] = None; row["err_dram"] = str(e)[:200]
    try:
        row["t_l1_ms"] = time_it(a, w, pc, ttnn.L1_MEMORY_CONFIG)
    except Exception as e:
        row["t_l1_ms"] = None; row["err_l1"] = str(e)[:200]
    if row.get("t_dram_ms") and row.get("t_l1_ms"):
        row["delta_us"] = (row["t_dram_ms"] - row["t_l1_ms"]) * 1e3
        row["implied_write_gbps"] = 25.166e6 / (row["delta_us"] * 1e-6) / 1e9
    res["ksweep"].append(row)
    print(json.dumps(row), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(w)

# barrier-granularity probe: writer does noc_async_write_barrier() once per out_subblock
K = 256
a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
for sbh, sbw in [(1, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 1), (1, 8), (4, 6)]:
    row = {"out_subblock_h": sbh, "out_subblock_w": sbw, "tiles": sbh * sbw,
           "barriers_per_core": (PER_CORE_M // sbh) * (PER_CORE_N // sbw) if (PER_CORE_M % sbh == 0 and PER_CORE_N % sbw == 0) else None}
    try:
        pc = cfg(8, 8, sbh, sbw)
        row["t_dram_ms"] = time_it(a, w, pc, ttnn.DRAM_MEMORY_CONFIG)
        row["t_l1_ms"] = time_it(a, w, pc, ttnn.L1_MEMORY_CONFIG)
        row["delta_us"] = (row["t_dram_ms"] - row["t_l1_ms"]) * 1e3
    except Exception as e:
        row["err"] = str(e)[:200]
    res["subblock"].append(row)
    print(json.dumps(row), flush=True)

print("RESULT_JSON " + json.dumps(res))
ttnn.close_device(DEV)
