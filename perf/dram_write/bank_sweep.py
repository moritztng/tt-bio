"""Bank-alignment discriminator, no kernel change.

DRAM page = 1 tile; interleaved pages round-robin over the 8 banks, so bank = tile_id % 8.
Core c owns tile ids (per_core_M*c + h) * N_t + w, so
    bank = (per_core_M * N_t * c + N_t * h + w) % 8.
With N_t = 24 both per_core_M*N_t = 96 and N_t = 24 are 0 mod 8: the core index and the row index
both drop out and every core is writing bank (w % 8) at the same moment -- 4 of 8 banks live.
Pick N_t not a multiple of 8 and the cores spread out. If bank lockstep is what caps the write,
this shows up as a large change in us per MB written.
"""
import json, time, statistics as st
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(13, 10)
M, K = 16384, 256
PER_CORE_M = 4

def ckc():
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                            math_approx_mode=False, fp32_dest_acc_en=True,
                                            packer_l1_acc=True)

def time_it(a, w, pc, mem, warm=8, reps=5, inner=12):
    for _ in range(warm):
        ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                    compute_kernel_config=ckc(), dtype=ttnn.bfloat16))
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(inner):
            ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                        compute_kernel_config=ckc(), dtype=ttnn.bfloat16))
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) / inner * 1e3)
    return st.median(ts)

ta = torch.randn(1, M, K)
a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
res = []
for N_t in [24, 25, 26, 27, 28, 32]:
    N = N_t * 32
    sbw = 4 if N_t % 4 == 0 else (1 if N_t % 2 else 2)
    w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID, in0_block_w=8, out_subblock_h=1, out_subblock_w=sbw,
        out_block_h=PER_CORE_M, out_block_w=N_t, per_core_M=PER_CORE_M, per_core_N=N_t,
        fuse_batch=True, fused_activation=None, mcast_in0=False)
    mb = M * N * 2 / 1e6
    r = {"N_t": N_t, "N": N, "out_subblock_w": sbw, "out_MB": round(mb, 3),
         "core_stride_mod8": (PER_CORE_M * N_t) % 8, "row_stride_mod8": N_t % 8}
    try:
        r["t_dram_ms"] = time_it(a, w, pc, ttnn.DRAM_MEMORY_CONFIG)
        r["t_l1_ms"] = time_it(a, w, pc, ttnn.L1_MEMORY_CONFIG)
        # the op is write-bound; ~14.3 us of in1 setup precedes the first write (profiled)
        r["us_per_MB"] = (r["t_dram_ms"] * 1e3 - 14.3) / mb
        r["est_write_gbps"] = mb * 1e6 / (r["t_dram_ms"] * 1e3 - 14.3) / 1e3
    except Exception as e:
        r["err"] = str(e)[:200]
    res.append(r); print("BANK " + json.dumps(r), flush=True)
    ttnn.deallocate(w)

print("RESULT_JSON " + json.dumps(res))
ttnn.close_device(DEV)
