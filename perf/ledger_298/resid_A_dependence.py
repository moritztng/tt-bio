"""A: is the flat block-sharded binary add real work? Data-dependence + correctness."""
import json, time, sys
import torch, ttnn

T = 32
def sh(rows, cols, gy, gx):
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(gx-1, gy-1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(crs, [rows//gy, cols//gx], ttnn.ShardOrientation.ROW_MAJOR))

dev = ttnn.open_device(device_id=0, trace_region_size=200_000_000)
GY, GX = 10, 11
out = []
for m in (1, 8):
    rows, cols = GY*T*m, GX*T*m
    tiles = (rows//T)*(cols//T)
    mc = sh(rows, cols, GY, GX)
    mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    a = mk(torch.zeros(rows, cols)); b = mk(torch.ones(rows, cols)); c = mk(torch.zeros(rows, cols))
    rec = {"m": m, "rows": rows, "cols": cols, "tiles": tiles, "tiles_per_core": tiles//(GY*GX),
           "MB_per_tensor": round(rows*cols*2/1e6, 3)}

    # 1. one add, correctness vs torch
    ttnn.add(a, b, memory_config=mc, output_tensor=c); ttnn.synchronize_device(dev)
    got = ttnn.to_torch(c).float()
    rec["add_once_correct"] = bool(torch.equal(got, torch.ones(rows, cols)))

    # 2. DATA-DEPENDENT chain: c += b, N times. Result must be exactly N everywhere.
    N = 64
    ttnn.copy(a, c); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(N):
        ttnn.add(c, b, memory_config=mc, output_tensor=c)
    ttnn.synchronize_device(dev)
    rec["chain_eager_us_per_op"] = round((time.perf_counter()-t0)/N*1e6, 2)
    got = ttnn.to_torch(c).float()
    rec["chain_value_max"] = float(got.max()); rec["chain_value_min"] = float(got.min())
    rec["chain_correct_N"] = bool(torch.equal(got, torch.full((rows, cols), float(N))))

    # 3. independent (non-data-dependent) add, eager + traced
    def bench(fn, it, chain):
        fn(); ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(it): fn()
        ttnn.synchronize_device(dev)
        eag = (time.perf_counter()-t0)/it*1e6
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(chain): fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); ttnn.synchronize_device(dev)
        reps = max(4, 128//chain)
        t0 = time.perf_counter()
        for _ in range(reps): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        tr = (time.perf_counter()-t0)/reps/chain*1e6
        ttnn.release_trace(dev, tid)
        return round(eag,2), round(tr,2)
    rec["indep_add_eager_us"], rec["indep_add_traced_us"] = bench(
        lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), 200, 32)
    rec["chain_dep_traced_us"] = None
    # traced data-dependent chain
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(32): ttnn.add(c, b, memory_config=mc, output_tensor=c)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(8): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    rec["chain_dep_traced_us"] = round((time.perf_counter()-t0)/8/32*1e6, 2)
    ttnn.release_trace(dev, tid)

    # 4. unary scalar mul, and a no-output-arg add (allocating)
    rec["mul_s_eager_us"], rec["mul_s_traced_us"] = bench(
        lambda: ttnn.mul(a, 1.0001, memory_config=mc, output_tensor=c), 100, 16)
    def alloc_add():
        z = ttnn.add(a, b, memory_config=mc); ttnn.deallocate(z)
    alloc_add(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(100): alloc_add()
    ttnn.synchronize_device(dev)
    rec["alloc_add_eager_us"] = round((time.perf_counter()-t0)/100*1e6, 2)

    out.append(rec); print(json.dumps(rec), flush=True)
    for t in (a, b, c): ttnn.deallocate(t)
json.dump(out, open(sys.argv[1], "w"), indent=2)
ttnn.close_device(dev); print("DONE", flush=True)
