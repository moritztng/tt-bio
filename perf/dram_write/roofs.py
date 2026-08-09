"""Pure-direction DRAM roofs on qb1 card 3.

read roof : source DRAM interleaved -> dest L1. DRAM sees reads only.
write roof: source L1               -> dest DRAM. DRAM sees writes only.
Neither inherits 435 (mixed R+W) or 512 (datasheet).
"""
import json, time, statistics as st
import torch, ttnn

DEV = ttnn.open_device(device_id=0)

def bench(x, mem, reps=7, inner=4):
    for _ in range(3):
        ttnn.deallocate(ttnn.clone(x, memory_config=mem))
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(inner):
            ttnn.deallocate(ttnn.clone(x, memory_config=mem))
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) / inner)
    return st.median(ts)

out = []
for mb in [8, 16, 32, 48, 64]:
    n = int(mb * 1e6 / 2)
    rows = n // 4096
    shape = (rows, 4096)
    nbytes = rows * 4096 * 2
    r = {"MB": nbytes / 1e6, "shape": shape}
    try:
        xd = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t = bench(xd, ttnn.L1_MEMORY_CONFIG)
        r["read_roof_gbps"] = nbytes / t / 1e9; r["read_ms"] = t * 1e3
        ttnn.deallocate(xd)
    except Exception as e:
        r["read_err"] = str(e)[:120]
    try:
        xl = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=DEV, memory_config=ttnn.L1_MEMORY_CONFIG)
        t = bench(xl, ttnn.DRAM_MEMORY_CONFIG)
        r["write_roof_gbps"] = nbytes / t / 1e9; r["write_ms"] = t * 1e3
        ttnn.deallocate(xl)
    except Exception as e:
        r["write_err"] = str(e)[:120]
    try:
        xd = ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t = bench(xd, ttnn.DRAM_MEMORY_CONFIG)
        r["dram2dram_rw_gbps"] = 2 * nbytes / t / 1e9
        ttnn.deallocate(xd)
    except Exception as e:
        r["rw_err"] = str(e)[:120]
    out.append(r); print("R " + json.dumps(r), flush=True)

print("RESULT_JSON " + json.dumps(out))
ttnn.close_device(DEV)
