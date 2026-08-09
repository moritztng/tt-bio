"""C2: knee sweep, no watcher. L1 op with known grid + DRAM read (G1 calibration ~32 cores)."""
import json, time, sys
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
res = {"knee_l1_mul": [], "knee_dram_read": []}
GRIDS = [(1,1),(1,2),(2,2),(2,4),(4,4),(4,8),(8,4),(8,8),(10,8),(10,11)]

# L1 op, 1024x1408 so every grid in GRIDS divides it into tile multiples
R, C = 1280, 1408
ta = torch.randn(R, C)
for gy, gx in GRIDS:
    if (R//gy) % T or (C//gx) % T: print(json.dumps({"cores": gy*gx, "skip": 1}), flush=True); continue
    m = sh(R, C, gy, gx)
    try:
        a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=m)
        c = ttnn.from_torch(torch.zeros(R, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=m)
    except Exception as e:
        print(json.dumps({"cores": gy*gx, "err": str(e)[:80]}), flush=True); continue
    r = {"cores": gy*gx, "gy": gy, "gx": gx, "tiles_per_core": (R//T)*(C//T)//(gy*gx),
         "mul_us": eager(lambda: ttnn.mul(a, 1.0001, memory_config=m, output_tensor=c), 60),
         "add_us": eager(lambda: ttnn.add(a, a, memory_config=m, output_tensor=c), 60)}
    res["knee_l1_mul"].append(r); print(json.dumps(r), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(c)

# DRAM read: interleaved source -> L1 block-sharded dest on a chosen grid
R2, C2_ = 8192, 2816            # 46.1 MB; 8192/gy and 2816/gx tile-aligned for GRIDS
xd = ttnn.from_torch(torch.randn(R2, C2_), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
nbytes = R2*C2_*2
for gy, gx in GRIDS:
    if (R2//gy) % T or (C2_//gx) % T: print(json.dumps({"cores": gy*gx, "skip": 1}), flush=True); continue
    m = sh(R2, C2_, gy, gx)
    r = {"cores": gy*gx, "MB": round(nbytes/1e6, 2)}
    try:
        us = eager(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=m)), 12)
        r["read_us"] = us; r["read_GBs"] = round(nbytes/(us*1e-6)/1e9, 1)
    except Exception as e:
        r["err"] = str(e)[:80]
    res["knee_dram_read"].append(r); print(json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=2)
ttnn.close_device(dev); print("DONE", flush=True)
