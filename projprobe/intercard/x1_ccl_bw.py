"""Inter-card all-reduce bandwidth at the RELION volume-reduction shapes.

usage: x1_ccl_bw.py <n_devices> <mb1,mb2,...> [reps] [--num-links K] [--tag NAME] [--verify]

Reports, per size: wall ms, algorithmic bytes/s (payload/time) and bus bytes/s
(ring model: 2*(N-1)/N * payload / time).  --verify additionally checks the
reduced volume against a host reference (replicated input, so the exact sum is
N*x) and prints the sha256 of the device result.
"""
import hashlib, json, os, sys, time
import torch
import ttnn

N = int(sys.argv[1])
sizes_mb = [float(x) for x in sys.argv[2].split(",")]
rest = sys.argv[3:]
REPS = 10
NUM_LINKS = None
TAG = ""
VERIFY = False
i = 0
while i < len(rest):
    a = rest[i]
    if a == "--num-links":
        NUM_LINKS = int(rest[i + 1]); i += 2
    elif a == "--tag":
        TAG = rest[i + 1]; i += 2
    elif a == "--verify":
        VERIFY = True; i += 1
    else:
        REPS = int(a); i += 1

kw = {"topology": ttnn.Topology.Linear}
if NUM_LINKS is not None:
    kw["num_links"] = NUM_LINKS

print("loadavg at start:", open("/proc/loadavg").read().strip(), flush=True)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
print("mesh open:", md.shape, md.get_num_devices(), flush=True)

out = {"n_devices": N, "reps": REPS, "num_links": NUM_LINKS, "tag": TAG,
       "visible": os.environ.get("TT_VISIBLE_DEVICES", ""), "rows": []}
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
        y = ttnn.all_reduce(x, **kw)
        ttnn.synchronize_device(md)
        row_extra = {}
        if VERIFY:
            got = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1]
            ref = t * float(N)
            row_extra["bit_exact"] = bool(torch.equal(got, ref))
            row_extra["max_abs_err"] = float((got - ref).abs().max())
            denom = float(ref.abs().max()) or 1.0
            row_extra["max_rel_err"] = row_extra["max_abs_err"] / denom
            ref_bf16 = ref.to(torch.bfloat16).float()
            row_extra["equals_bf16_rounded_ref"] = bool(torch.equal(got, ref_bf16))
            row_extra["max_err_vs_bf16_ref"] = float((got - ref_bf16).abs().max())
            row_extra["sha256"] = hashlib.sha256(got.numpy().tobytes()).hexdigest()[:16]
            row_extra["sha256_ref"] = hashlib.sha256(ref.numpy().tobytes()).hexdigest()[:16]
        ttnn.deallocate(y)
        walls = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            y = ttnn.all_reduce(x, **kw)
            ttnn.synchronize_device(md)
            walls.append(time.perf_counter() - t0)
            ttnn.deallocate(y)
        ttnn.deallocate(x)
        best = min(walls); med = sorted(walls)[len(walls) // 2]
        alg = real_bytes / best / 1e9
        bus = 2.0 * (N - 1) / N * real_bytes / best / 1e9
        row = {"mb": real_bytes / 1024 / 1024, "best_ms": best * 1e3, "med_ms": med * 1e3,
               "spread_pct": (max(walls) - min(walls)) / min(walls) * 100,
               "alg_GBs": alg, "bus_GBs": bus}
        row.update(row_extra)
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
finally:
    ttnn.close_mesh_device(md)

out["loadavg_end"] = open("/proc/loadavg").read().strip()
name = "x1_ccl_bw_n%d%s.json" % (N, ("_" + TAG) if TAG else "")
with open(os.path.expanduser("~/mthuening/relion-intercard/" + name), "w") as f:
    json.dump(out, f, indent=1)
print("WROTE", name, "loadavg_end", out["loadavg_end"])
