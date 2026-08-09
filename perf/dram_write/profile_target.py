"""Profiler target: four rungs, dispatched in a fixed order so they can be identified in the
device CSV by position. Rung order:
  0  K=256  obh=4  DRAM-out   (production shape)
  1  K=256  obh=4  L1-out
  2  K=4096 obh=1  DRAM-out   (compute 8x the write; any overlap would be unmissable)
  3  K=4096 obh=1  L1-out
"""
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

def mk(K):
    a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return a, w

RUNGS = [(256, 4, "dram"), (256, 4, "l1"), (4096, 1, "dram"), (4096, 1, "l1")]
NPROF = 4

for i, (K, obh, tag) in enumerate(RUNGS):
    a, w = mk(K)
    mem = ttnn.DRAM_MEMORY_CONFIG if tag == "dram" else ttnn.L1_MEMORY_CONFIG
    pc, k = cfg(obh), ckc()
    for _ in range(3):   # warm: compile + cache
        ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                    compute_kernel_config=k, dtype=ttnn.bfloat16))
    ttnn.synchronize_device(DEV)
    print("RUNG %d K=%d obh=%d %s" % (i, K, obh, tag), flush=True)
    for _ in range(NPROF):
        ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=mem,
                                    compute_kernel_config=k, dtype=ttnn.bfloat16))
    ttnn.synchronize_device(DEV)
    ttnn.deallocate(a); ttnn.deallocate(w)

ttnn.close_device(DEV)
