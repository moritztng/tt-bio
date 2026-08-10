import torch, ttnn, sys
torch.manual_seed(0)
ta = torch.randn(1, 2, 298, 256); tb = torch.randn(256, 1024)
dev = ttnn.open_device(device_id=0)
try:
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)
    fused = ttnn.to_torch(ttnn.linear(a, b, activation="silu", compute_kernel_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG))
    bare  = ttnn.to_torch(ttnn.linear(a, b, compute_kernel_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG))
    ttnn.synchronize_device(dev)
    ident = torch.equal(fused, bare)
    print("RESULT fused_mean=%.6f bare_mean=%.6f fused_equals_bare=%s" % (
        float(fused.float().abs().mean()), float(bare.float().abs().mean()), ident))
finally:
    ttnn.close_device(dev)
