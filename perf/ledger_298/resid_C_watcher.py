"""C: core-utilisation instrument. (1) does TT_METAL_WATCHER give per-core k_ids?
(2) knee sweep, calibrated on a DRAM read (G1 says ~32 cores) and an L1 op at a known grid."""
import json, time, sys, os, glob, re
import torch, ttnn

T = 32
def sh(rows, cols, gy, gx, buf=ttnn.BufferType.L1):
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(gx-1, gy-1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, buf,
                             ttnn.ShardSpec(crs, [rows//gy, cols//gx], ttnn.ShardOrientation.ROW_MAJOR))

dev = ttnn.open_device(device_id=0)
def eager(fn, it):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(it): fn()
    ttnn.synchronize_device(dev)
    return round((time.perf_counter()-t0)/it*1e6, 3)

res = {"watcher": {}, "knee_l1_mul": [], "knee_dram_read": []}

# --- (1) watcher: run a long op loop and see if a dump lands with per-core k_ids
R = C = T*24
mc = sh(R, C, 10, 11) if (R % 10 == 0 and C % 11 == 0) else sh(R, C, 6, 8)
a = ttnn.from_torch(torch.randn(R, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=sh(R,C,6,8))
c = ttnn.from_torch(torch.zeros(R, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=sh(R,C,6,8))
t_end = time.time() + 6
n = 0
while time.time() < t_end:
    for _ in range(500):
        ttnn.mul(a, 1.0001, memory_config=sh(R,C,6,8), output_tensor=c)
    ttnn.synchronize_device(dev); n += 500
res["watcher"]["iters"] = n
logs = glob.glob("generated/watcher/*.log") + glob.glob("/tmp/ttnn/**/watcher*.log", recursive=True) + glob.glob(os.path.expanduser("~/generated/watcher/*.log"))
res["watcher"]["logs"] = logs[:5]
for L in logs[:1]:
    txt = open(L).read()
    res["watcher"]["bytes"] = len(txt)
    kl = [l for l in txt.splitlines() if "k_ids" in l]
    res["watcher"]["k_id_lines"] = len(kl)
    res["watcher"]["sample"] = kl[-3:] if kl else txt.splitlines()[-3:]
print(json.dumps(res["watcher"])[:900], flush=True)
ttnn.deallocate(a); ttnn.deallocate(c)

# --- (2a) knee sweep on an L1 op whose grid we set (ground truth = the grid)
R = C = T*24                                    # 768x768, 1.180 MB/tensor
ta = torch.randn(R, C)
for gy, gx in [(1,1),(2,2),(3,3),(4,4),(6,4),(6,6),(8,6),(8,8),(12,8),(12,16)]:
    if (R//gy) % T or (C//gx) % T: continue
    m = sh(R, C, gy, gx)
    try:
        a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=m)
        c = ttnn.from_torch(torch.zeros(R, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=m)
    except Exception as e:
        print(json.dumps({"cores": gy*gx, "err": str(e)[:80]}), flush=True); continue
    r = {"cores": gy*gx, "gy": gy, "gx": gx,
         "mul_us": eager(lambda: ttnn.mul(a, 1.0001, memory_config=m, output_tensor=c), 100)}
    res["knee_l1_mul"].append(r); print(json.dumps(r), flush=True)
    ttnn.deallocate(a); ttnn.deallocate(c)

# --- (2b) DRAM read: interleaved source, L1 block-sharded destination on a chosen grid.
# G1 calibration point: DRAM read saturates at ~32 of 130 cores.
MB = 32
nrow = int(MB*1e6/2)//4096
xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
nbytes = nrow*4096*2
for gy, gx in [(1,1),(2,2),(4,2),(4,4),(8,4),(8,8),(10,8),(10,11)]:
    if nrow % gy or 4096 % gx or (nrow//gy) % T or (4096//gx) % T: 
        print(json.dumps({"cores": gy*gx, "skip": "not divisible"}), flush=True); continue
    m = sh(nrow, 4096, gy, gx)
    try:
        r = {"cores": gy*gx, "MB": round(nbytes/1e6, 2)}
        us = eager(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=m)), 20)
        r["read_us"] = us; r["read_GBs"] = round(nbytes/(us*1e-6)/1e9, 1)
    except Exception as e:
        r["err"] = str(e)[:80]
    res["knee_dram_read"].append(r); print(json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=2)
ttnn.close_device(dev); print("DONE", flush=True)
