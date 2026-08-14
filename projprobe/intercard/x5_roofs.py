"""The two roofs every byte/FLOP model in this task is checked against, measured on THIS chip.

DRAM copy roof (read+write, ttnn.clone) and the bf16 matmul roof at the shape the perf lineage
quotes 254.5 TFLOP/s for. Median of 9 synced calls after 2 warm.
usage: x5_roofs.py            (TT_VISIBLE_DEVICES picks the chip)
"""
import json, os, statistics as st, time
import torch
import ttnn

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
out = {"host": os.uname().nodename, "visible": os.environ.get("TT_VISIBLE_DEVICES", ""),
       "grid": [g.x, g.y], "rows": []}


def bench(fn, n=9, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return min(ts), st.median(ts)


# --- DRAM copy roof: read S + write S ---------------------------------------
for mb in (256, 821):
    rows = int(mb * 1024 * 1024 / 4 / 1024)
    t = torch.zeros(1, 1, rows // 32 * 32, 1024, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    nbytes = x.volume() * 4
    best, med = bench(lambda: ttnn.clone(x))
    row = {"kind": "dram_copy_fp32", "mb": nbytes / 1048576, "best_ms": best * 1e3,
           "GBs_rw": 2 * nbytes / best / 1e9}
    out["rows"].append(row); print(json.dumps(row), flush=True)
    ttnn.deallocate(x)

# --- bf16 matmul roof -------------------------------------------------------
for (m, k, n) in ((7744, 13312, 2304), (4096, 4096, 4096)):
    a = ttnn.from_torch(torch.randn(1, 1, m, k, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(1, 1, k, n, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                           fp32_dest_acc_en=False, packer_l1_acc=True)
    best, med = bench(lambda: ttnn.matmul(a, b, compute_kernel_config=cfg))
    row = {"kind": "matmul_bf16_lofi", "m": m, "k": k, "n": n, "best_ms": best * 1e3,
           "TFLOPs": 2.0 * m * k * n / best / 1e12}
    out["rows"].append(row); print(json.dumps(row), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)

ttnn.close_device(dev)
name = "x5_roofs_%s.json" % (out["visible"] or "auto")
with open(os.path.expanduser("~/mthuening/relion-intercard/" + name), "w") as f:
    json.dump(out, f, indent=1)
print("WROTE", name)
