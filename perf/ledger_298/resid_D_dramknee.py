"""D: DRAM-read knee via to_memory_config (G1 calibration), + watcher core-count parse."""
import json, time, sys, os, re
import torch, ttnn
T = 32
def sh(rows, cols, gy, gx):
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(gx-1, gy-1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(crs, [rows//gy, cols//gx], ttnn.ShardOrientation.ROW_MAJOR))
dev = ttnn.open_device(device_id=0)
def eager(fn, it):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(it): fn()
    ttnn.synchronize_device(dev)
    return round((time.perf_counter()-t0)/it*1e6, 3)
res = {"knee_dram_read": []}
R2, C2_ = 2560, 2816                     # 14.4 MB, divides for the grids below
xd = ttnn.from_torch(torch.randn(R2, C2_), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
nb = R2*C2_*2
for gy, gx in [(1,1),(2,2),(2,4),(4,4),(4,8),(8,4),(8,8),(10,8),(10,11)]:
    if (R2//gy) % T or (C2_//gx) % T: print(json.dumps({"cores": gy*gx, "skip": 1}), flush=True); continue
    m = sh(R2, C2_, gy, gx)
    r = {"cores": gy*gx, "gy": gy, "gx": gx, "MB": round(nb/1e6, 2)}
    try:
        us = eager(lambda: ttnn.deallocate(ttnn.to_memory_config(xd, m)), 15)
        r["read_us"] = us; r["read_GBs"] = round(nb/(us*1e-6)/1e9, 1)
    except Exception as e:
        r["err"] = str(e)[:100]
    res["knee_dram_read"].append(r); print(json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=2)
ttnn.close_device(dev); print("DONE", flush=True)
