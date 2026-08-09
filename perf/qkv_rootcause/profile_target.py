#!/usr/bin/env python3
"""Device-profiler target for the qkv projection: the same matmul, three result placements.

Driven with an explicit program config in every case so the kernel is identical across the three
and the only variable is where the result lands. Run under:

    python3 -m tracy -r -o <dir> --op-support-count 200 -- perf/qkv_rootcause/profile_target.py

Falsifiable prediction being tested: the DRAM-out runs are dominated by the writer RISC (NCRISC),
which collapses when the result stays in L1, while the math RISC (TRISC1) barely moves. That is the
signature of an op limited by writing its result down rather than by computing it.
"""
import torch
import ttnn

N, C_Z, H, D = 128, 256, 8, 32
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(C_Z, 3 * H * D), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    m_t, k_t, n_t = (N * N) // 32, C_Z // 32, (3 * H * D) // 32
    per_core_M = next(p for p in range(-(-m_t // (gx * gy)), m_t + 1) if m_t % p == 0)
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=k_t,
        out_subblock_h=1, out_subblock_w=4, out_block_h=per_core_M, out_block_w=n_t,
        per_core_M=per_core_M, per_core_N=n_t, fuse_batch=True, fused_activation=None, mcast_in0=False)
    print(f"grid={gx}x{gy} per_core_M={per_core_M} per_core_N={n_t} in0_block_w={k_t}", flush=True)

    def run(mem, reps):
        for _ in range(reps):
            ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                        memory_config=mem, program_config=cfg))
        ttnn.synchronize_device(dev)

    # warm both program variants first so no compile lands inside the profiled region
    run(DRAM, 2)
    run(L1, 2)

    print("PROFILE_MARK result_to_DRAM", flush=True)
    run(DRAM, 5)
    print("PROFILE_MARK result_to_L1", flush=True)
    run(L1, 5)
    print("PROFILE_MARK done", flush=True)
finally:
    ttnn.close_device(dev)
