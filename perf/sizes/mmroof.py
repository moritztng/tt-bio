"""Practical bf16 matmul roof on this card: a large square DRAM matmul, median of 9 synced calls.
Named explicitly so nothing in this task quotes a '% of peak' against a number nobody measured."""
import json, os, statistics as st, sys, time, torch, ttnn
dev = ttnn.open_device(device_id=0)
out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "loadavg": open("/proc/loadavg").read().split()[:3], "matmul": {}}
for M in (4096, 8192):
    a = ttnn.from_torch(torch.zeros(1, 1, M, M, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.zeros(1, 1, M, M, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    f = lambda: ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for _ in range(2):
        f(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(9):
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); o = f()
        ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0); ttnn.deallocate(o)
    t = st.median(ts)
    out["matmul"][M] = {"ms": round(t * 1e3, 3), "tflops": round(2 * M ** 3 / t / 1e12, 1)}
    print(M, out["matmul"][M], flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)
# and the batched shape TriMul actually runs: 384 independent NxN @ NxN
for N in (512, 1024):
    C = 384
    a = ttnn.from_torch(torch.zeros(1, C, N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.zeros(1, C, N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    f = lambda: ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        for _ in range(1):
            f(); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(3):
            ttnn.synchronize_device(dev); t0 = time.perf_counter(); o = f()
            ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0); ttnn.deallocate(o)
        t = st.median(ts)
        out["matmul"][f"batched_{N}x{C}"] = {"ms": round(t * 1e3, 3),
                                             "tflops": round(2 * C * N ** 3 / t / 1e12, 1)}
    except Exception as e:
        out["matmul"][f"batched_{N}x{C}"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    print(N, out["matmul"][f"batched_{N}x{C}"], flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)
open(sys.argv[1], "w").write(json.dumps(out, indent=1))
ttnn.close_device(dev)
print("wrote", sys.argv[1])
