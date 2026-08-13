"""DRAM->DRAM roofs on this card, at OpenDDE's real pair shapes.

Three roofs, because a component is only ever against ONE of them and naming the wrong one is how
this lineage made an op at 96 % of its read roof look like 13 %:
  copy      : ttnn.clone, reads N bytes and writes N  -> 2N/t
  unary_w   : ttnn.relu out-of-place, same traffic     -> 2N/t   (writer-bound eltwise)
  matmul_w  : ttnn.linear at the pair projection shape -> bytes actually moved / t
Median of 9 synced calls after 2 warm.
"""
import json, os, statistics as st, sys, time
import ttnn

dev = ttnn.open_device(device_id=0)
res = {"host": os.uname().nodename, "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "loadavg": open("/proc/loadavg").read().split()[:3],
       "grid": [dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y],
       "roofs": {}}


def timeit(fn, n=9):
    for _ in range(2):
        fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    return st.median(ts)


for N in (128, 256, 512, 768, 1024):
    shp = (1, N, N, 384)
    nbytes = N * N * 384 * 2
    x = ttnn.from_torch(__import__("torch").zeros(*shp, dtype=__import__("torch").bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t_copy = timeit(lambda: ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    t_un = timeit(lambda: ttnn.relu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    res["roofs"][N] = {"bytes_one_way": nbytes,
                       "copy_ms": round(t_copy * 1e3, 3),
                       "copy_gbps": round(2 * nbytes / t_copy / 1e9, 1),
                       "unary_w_ms": round(t_un * 1e3, 3),
                       "unary_w_gbps": round(2 * nbytes / t_un / 1e9, 1)}
    ttnn.deallocate(x)
    print(N, res["roofs"][N], flush=True)

open(sys.argv[1], "w").write(json.dumps(res, indent=1))
ttnn.close_device(dev)
print("wrote", sys.argv[1])
