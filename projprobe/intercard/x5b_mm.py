"""Batched matmul roof: 20 back-to-back calls in one timed region, so dispatch and
sync are amortised instead of charged to every call (the oversync trap)."""
import json, os, time
import torch, ttnn
dev = ttnn.open_device(device_id=0)
m, k, n = 7744, 13312, 2304
a = ttnn.from_torch(torch.randn(1,1,m,k, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
b = ttnn.from_torch(torch.randn(1,1,k,n, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
for _ in range(3):
    r = ttnn.matmul(a, b, compute_kernel_config=cfg); ttnn.synchronize_device(dev); ttnn.deallocate(r)
best = 1e9
for trial in range(3):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    rs = [ttnn.matmul(a, b, compute_kernel_config=cfg) for _ in range(20)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / 20
    for r in rs: ttnn.deallocate(r)
    best = min(best, dt)
row = {"visible": os.environ.get("TT_VISIBLE_DEVICES",""), "per_call_ms": best*1e3,
       "TFLOPs": 2.0*m*k*n/best/1e12}
print(json.dumps(row), flush=True)
ttnn.close_device(dev)
