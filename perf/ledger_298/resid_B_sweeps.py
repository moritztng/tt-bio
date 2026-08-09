"""B: (1) add size sweep to the L1 ceiling, (2) op-class matrix at fixed shape/grid,
(3) launch floor vs core count and operand count."""
import json, time, sys
import torch, ttnn

T = 32
def sh(rows, cols, gy, gx):
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(gx-1, gy-1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(crs, [rows//gy, cols//gx], ttnn.ShardOrientation.ROW_MAJOR))

dev = ttnn.open_device(device_id=0, trace_region_size=200_000_000)
GY, GX = 10, 11

def traced(fn, chain, reps=8):
    fn(); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(chain): fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    us = (time.perf_counter()-t0)/reps/chain*1e6
    ttnn.release_trace(dev, tid)
    return round(us, 3)

def eager(fn, it):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(it): fn()
    ttnn.synchronize_device(dev)
    return round((time.perf_counter()-t0)/it*1e6, 3)

res = {"size_sweep": [], "op_matrix": [], "launch_floor": []}

# ---- B1: add size sweep, 110 cores, block-sharded L1
for m in (1, 2, 3, 4, 6, 8, 10, 12):
    rows, cols = GY*T*m, GX*T*m
    mc = sh(rows, cols, GY, GX)
    try:
        mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
        a, b, c = mk(torch.randn(rows, cols)), mk(torch.randn(rows, cols)), mk(torch.zeros(rows, cols))
    except Exception as e:
        res["size_sweep"].append({"m": m, "err": str(e)[:90]}); print("ERR", m, str(e)[:90], flush=True); continue
    tiles = (rows//T)*(cols//T)
    r = {"m": m, "rows": rows, "cols": cols, "tiles_per_core": tiles//(GY*GX),
         "MB_per_tensor": round(rows*cols*2/1e6, 3), "traffic_MB": round(3*rows*cols*2/1e6, 3)}
    ch = 32 if m <= 6 else 8
    r["add_traced_us"] = traced(lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), ch)
    r["add_eager_us"] = eager(lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), 100)
    r["add_traced_TBs"] = round(3*rows*cols*2/(r["add_traced_us"]*1e-6)/1e12, 3)
    res["size_sweep"].append(r); print(json.dumps(r), flush=True)
    for t in (a, b, c): ttnn.deallocate(t)

# ---- B2: op class matrix, fixed 110 cores, two work sizes
for m in (2, 8):
    rows, cols = GY*T*m, GX*T*m
    mc = sh(rows, cols, GY, GX)
    mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    a, b, c = mk(torch.randn(rows, cols)), mk(torch.randn(rows, cols)), mk(torch.zeros(rows, cols))
    tiles = (rows//T)*(cols//T); tpc = tiles//(GY*GX)
    ops = [
        ("add_tt",   lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), 3),
        ("mul_tt",   lambda: ttnn.mul(a, b, memory_config=mc, output_tensor=c), 3),
        ("sub_tt",   lambda: ttnn.sub(a, b, memory_config=mc, output_tensor=c), 3),
        ("mul_ts",   lambda: ttnn.mul(a, 1.0001, memory_config=mc, output_tensor=c), 2),
        ("add_ts",   lambda: ttnn.add(a, 1.0001, memory_config=mc, output_tensor=c), 2),
        ("exp",      lambda: ttnn.exp(a, memory_config=mc, output_tensor=c), 2),
        ("relu",     lambda: ttnn.relu(a, memory_config=mc, output_tensor=c), 2),
        ("sigmoid",  lambda: ttnn.sigmoid(a, memory_config=mc, output_tensor=c), 2),
        ("clone_l1", lambda: ttnn.deallocate(ttnn.clone(a, memory_config=mc)), 2),
        ("copy",     lambda: ttnn.copy(a, c), 2),
    ]
    for name, fn, bufs in ops:
        r = {"m": m, "tiles_per_core": tpc, "op": name, "bufs": bufs,
             "traffic_MB": round(bufs*rows*cols*2/1e6, 3)}
        try:
            r["traced_us"] = traced(fn, 16 if m <= 4 else 8)
            r["eager_us"] = eager(fn, 60)
            r["traced_GBs"] = round(bufs*rows*cols*2/(r["traced_us"]*1e-6)/1e9, 1)
        except Exception as e:
            r["err"] = str(e)[:110]
        res["op_matrix"].append(r); print(json.dumps(r), flush=True)
    for t in (a, b, c): ttnn.deallocate(t)

# ---- B3: launch floor vs core count (1 tile/core) and vs operand count
for gy, gx in [(1,1),(1,2),(2,2),(2,4),(4,4),(4,8),(8,8),(10,8),(10,11)]:
    rows, cols = gy*T, gx*T
    mc = sh(rows, cols, gy, gx)
    mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    a, b, c = mk(torch.randn(rows, cols)), mk(torch.randn(rows, cols)), mk(torch.zeros(rows, cols))
    r = {"gy": gy, "gx": gx, "cores": gy*gx}
    r["add_traced_us"] = traced(lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), 64)
    r["relu_traced_us"] = traced(lambda: ttnn.relu(a, memory_config=mc, output_tensor=c), 64)
    r["add_eager_us"] = eager(lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), 300)
    res["launch_floor"].append(r); print(json.dumps(r), flush=True)
    for t in (a, b, c): ttnn.deallocate(t)

# DRAM-interleaved 1-tile control (no sharding at all)
a = ttnn.from_torch(torch.randn(32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
b = ttnn.from_torch(torch.randn(32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
c = ttnn.from_torch(torch.zeros(32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
r = {"gy": 0, "gx": 0, "cores": 1, "note": "DRAM interleaved 1 tile",
     "add_traced_us": traced(lambda: ttnn.add(a, b, output_tensor=c), 64),
     "add_eager_us": eager(lambda: ttnn.add(a, b, output_tensor=c), 300)}
res["launch_floor"].append(r); print(json.dumps(r), flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
ttnn.close_device(dev); print("DONE", flush=True)
