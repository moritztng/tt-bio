"""Device-side roofs at the size that matters. Everything here is read by DEVICE KERNEL DURATION
from the profiler, the same instrument the matmul writeback was measured with -- the doc's roofs
were host wall-clock, and comparing a host-timed roof to a device-timed op is not a comparison.

Op order (identified by position in the device CSV):
  A  clone DRAM->L1   25.166 MB   read-only
  B  clone L1->DRAM   25.166 MB   write-only
  C  clone DRAM->L1   50.33 MB    read-only
  D  clone L1->DRAM   50.33 MB    write-only
  E  matmul production rung, DRAM out
"""
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(13, 10)

def clone_rung(rows, cols, src_mem, dst_mem, n=4):
    x = ttnn.from_torch(torch.randn(rows, cols), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=DEV, memory_config=src_mem)
    for _ in range(3):
        ttnn.deallocate(ttnn.clone(x, memory_config=dst_mem))
    ttnn.synchronize_device(DEV)
    for _ in range(n):
        ttnn.deallocate(ttnn.clone(x, memory_config=dst_mem))
    ttnn.synchronize_device(DEV)
    ttnn.deallocate(x)

D, L = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
print("A read 25.166MB", flush=True);  clone_rung(16384, 768, D, L)
print("B write 25.166MB", flush=True); clone_rung(16384, 768, L, D)
print("C read 50.33MB", flush=True);   clone_rung(32768, 768, D, L)
print("D write 50.33MB", flush=True);  clone_rung(32768, 768, L, D)

print("E matmul", flush=True)
M, N, K = 16384, 768, 256
a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                    memory_config=D)
w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV,
                    memory_config=D)
pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=GRID, in0_block_w=8, out_subblock_h=1, out_subblock_w=4,
    out_block_h=4, out_block_w=24, per_core_M=4, per_core_N=24,
    fuse_batch=True, fused_activation=None, mcast_in0=False)
k = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                     fp32_dest_acc_en=True, packer_l1_acc=True)
for _ in range(3):
    ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=D, compute_kernel_config=k,
                                dtype=ttnn.bfloat16))
ttnn.synchronize_device(DEV)
for _ in range(4):
    ttnn.deallocate(ttnn.linear(a, w, program_config=pc, memory_config=D, compute_kernel_config=k,
                                dtype=ttnn.bfloat16))
ttnn.synchronize_device(DEV)
ttnn.close_device(DEV)
