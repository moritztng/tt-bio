"""Read a kernel's assigned NOC off its own JIT compile, without a rebuild and without the factory source.

A kernel's NOC arrives as the host-set compile define NOC_INDEX. So an overlay that replaces the
kernel with `#if NOC_INDEX == 1 / #error .../ #else / #error ... / #endif` makes the compiler announce
the answer and abort. Same trick as the overlay positive control, used as a probe.

  python3 nocprobe.py linear          -> probes the 1D-mcast matmul writer
  python3 nocprobe.py minimal_matmul  -> probes minimal_matmul's in1-sender-writer and in0 sender
"""
import sys
import torch, ttnn

WHICH = sys.argv[1]
DEV = ttnn.open_device(device_id=0)
g = DEV.compute_with_storage_grid_size()
torch.manual_seed(0)

a = ttnn.from_torch(torch.randn(1, 16384, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
w = ttnn.from_torch(torch.randn(256, 768), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)

if WHICH == "linear":
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), in0_block_w=8,
        out_subblock_h=1, out_subblock_w=4, out_block_h=4, out_block_w=24,
        per_core_M=4, per_core_N=24, fuse_batch=True, fused_activation=None, mcast_in0=False)
    y = ttnn.linear(a, w, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckc, dtype=ttnn.bfloat16)
else:
    y = ttnn.experimental.minimal_matmul(input_tensor=a, weight_tensor=w,
                                         compute_kernel_config=ckc, dtype=ttnn.bfloat16)
ttnn.synchronize_device(DEV)
print("PROBE_DID_NOT_FIRE")
ttnn.close_device(DEV)
