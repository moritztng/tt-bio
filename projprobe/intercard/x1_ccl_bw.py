"""Inter-card all-reduce bandwidth at the RELION volume-reduction shapes.

usage: x1_ccl_bw.py <n_devices> <mb1,mb2,...> [reps]
Reports, per size: wall ms, algorithmic bytes/s (payload/time) and bus bytes/s
(ring model: 2*(N-1)/N * payload / time).
"""
import json, os, sys, time
import torch
import ttnn

N = int(sys.argv[1])
sizes_mb = [float(x) for x in sys.argv[2].split(",")]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
print("mesh open:", md.shape, md.get_num_devices(), flush=True)

out = {"n_devices": N, "reps": REPS, "rows": []}
try:
    for mb in sizes_mb:
        nbytes = int(mb * 1024 * 1024)
        cols = 1024
        rows = max(32, (nbytes // (cols * 4)) // 32 * 32)
        real_bytes = rows * cols * 4
        t = torch.randn(1, 1, rows, cols, dtype=torch.float32)
        try:
            mapper = ttnn.ReplicateTensorToMesh(md)
        except AttributeError:
            mapper = ttnn.replicate_tensor_to_mesh_mapper(md)
        x = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                            device=md, mesh_mapper=mapper,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # warm up (compile)
        y = ttnn.all_reduce(x, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(md)
        ttnn.deallocate(y)
        walls = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            y = ttnn.all_reduce(x, topology=ttnn.Topology.Linear)
            ttnn.synchronize_device(md)
            walls.append(time.perf_counter() - t0)
            ttnn.deallocate(y)
        ttnn.deallocate(x)
        best = min(walls); med = sorted(walls)[len(walls)//2]
        alg = real_bytes / best / 1e9
        bus = 2.0 * (N - 1) / N * real_bytes / best / 1e9
        row = {"mb": real_bytes / 1024 / 1024, "best_ms": best * 1e3, "med_ms": med * 1e3,
               "spread_pct": (max(walls) - min(walls)) / min(walls) * 100,
               "alg_GBs": alg, "bus_GBs": bus}
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
finally:
    ttnn.close_mesh_device(md)

with open(os.path.expanduser("~/mthuening/relion-intercard/x1_ccl_bw_n%d.json" % N), "w") as f:
    json.dump(out, f, indent=1)
print("WROTE json")
