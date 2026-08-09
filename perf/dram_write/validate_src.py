"""Validate the tt-metal source checkout (v0.71.0-dev) reproduces the 0.67.4 wheel's behaviour
before we trust its profiler. Two arms of the baseline, plus the K-discriminator at its ends."""
import json, time, statistics as st
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(13, 10)
M, N = 16384, 768
PER_CORE_M, PER_CORE_N = 4, 24

def ckc(fp32=True):
    for name in ("WormholeComputeKernelConfig", "BlackholeComputeKernelConfig"):
        if hasattr(ttnn, name):
            try:
                return getattr(ttnn, name)(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                    fp32_dest_acc_en=fp32, packer_l1_acc=True)
            except Exception:
                continue
    raise RuntimeError("no compute kernel config")

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

out = {"ttnn_version": getattr(ttnn, "__version__", "unknown"), "rungs": []}
K_OBH = [(256, 4), (256, 1), (2048, 1), (4096, 1)]
for K, obh in K_OBH:
    a, w = mk(K)
    r = {"K": K, "out_block_h": obh}
    for tag, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
        try:
            r["t_%s_ms" % tag] = time_it(a, w, cfg(8, 1, 4, obh, PER_CORE_N), mem, ckc(),
                                         inner=6 if K > 512 else 12)
        except Exception as e:
            r["t_%s_ms" % tag] = None; r["err_" + tag] = str(e)[:200]
    if r.get("t_dram_ms") and r.get("t_l1_ms"):
        r["delta_us"] = (r["t_dram_ms"] - r["t_l1_ms"]) * 1e3
        r["implied_write_gbps"] = 25.166e6 / (r["delta_us"] * 1e-6) / 1e9
    out["rungs"].append(r); print("VAL " + json.dumps(r), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(w)

print("RESULT_JSON " + json.dumps(out))
ttnn.close_device(DEV)
