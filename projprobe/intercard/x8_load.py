"""Sustained matmul load, for measuring power under compute (not under a collective)."""
import json, os, sys, time
import torch, ttnn
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
dev = ttnn.open_device(device_id=0)
m, k, n = 7744, 13312, 2304
a = ttnn.from_torch(torch.randn(1,1,m,k, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
b = ttnn.from_torch(torch.randn(1,1,k,n, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)
r = ttnn.matmul(a, b, compute_kernel_config=cfg); ttnn.synchronize_device(dev); ttnn.deallocate(r)
calls = 0
t0 = time.perf_counter()
while time.perf_counter() - t0 < secs:
    rs = [ttnn.matmul(a, b, compute_kernel_config=cfg) for _ in range(20)]
    ttnn.synchronize_device(dev)
    for x in rs: ttnn.deallocate(x)
    calls += 20
dt = time.perf_counter() - t0
print(json.dumps({"card": os.environ.get("TT_VISIBLE_DEVICES",""), "calls": calls,
                  "secs": dt, "TFLOPs": 2.0*m*k*n*calls/dt/1e12}), flush=True)
ttnn.close_device(dev)
