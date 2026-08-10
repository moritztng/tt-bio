"""Minimal fused-silu matmul: proves whether ckernel_sfpu_silu.h is JIT-compiled from TT_METAL_HOME."""
import os, sys, torch, ttnn
dev = ttnn.open_device(device_id=0)
try:
    a = ttnn.from_torch(torch.randn(1, 2, 298, 256), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(256, 1024), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=False)
    out = ttnn.linear(a, b, activation="silu", compute_kernel_config=cfg,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    t = ttnn.to_torch(out)
    print("PROBE_OK", tuple(t.shape), float(t.float().abs().mean()))
finally:
    ttnn.close_device(dev)
