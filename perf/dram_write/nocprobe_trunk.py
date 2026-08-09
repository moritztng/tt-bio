"""Which matmul kernel, and which NOC, does a real trunk-shaped ttnn.linear land on?

Shape is the trimul output projection at 298 aa: [1, 320*320, 256] @ [256, 256], c_z=256, 52.4 MB
bf16 result to DRAM -- the same call W6 measured at 0.678 ms / 154.6 GB/s against minimal_matmul's
0.408 ms / 256.9 GB/s. Every kernel in the matmul dataflow directory carries an #error that reports
its NOC_INDEX, so whichever kernels this op compiles announce themselves and their NOC.
"""
import sys
import torch, ttnn

DEV = ttnn.open_device(device_id=0)
g = DEV.compute_with_storage_grid_size()
torch.manual_seed(0)
N = 320
a = ttnn.from_torch(torch.randn(1, N * N, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
w = ttnn.from_torch(torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
y = ttnn.linear(a, w, core_grid=ttnn.CoreGrid(y=g.y, x=g.x), memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ckc, dtype=ttnn.bfloat16)
ttnn.synchronize_device(DEV)
print("PROBE_DID_NOT_FIRE")
ttnn.close_device(DEV)
