"""The composed 15-iteration reduction trajectory, one timed region — the fold-level A/B
equivalent for this task. Not a per-call rate times a call census.

Sizes are RELION's exact per-iteration backprojector volumes for Refine3D/job019, from the
CurrentImageSize read out of run_it0NN_half1_model.star: bytes = 12 * (2cs+3)^2 * (cs+2).

usage: x7_trajectory.py <n_devices> [arms]
"""
import hashlib, json, os, sys, time
import torch
import ttnn

CS = [32, 76, 80, 128, 144, 146, 146, 150, 154, 154, 160, 160, 196, 196, 196]
N = int(sys.argv[1])
ARMS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def vol_bytes(cs):
    pad = 2 * (2 * (cs // 2)) + 3          # 2*(round(2*(cs/2))+1)+1 with padding_factor 2
    return 12 * pad * pad * (cs + 2)


sizes = [vol_bytes(cs) for cs in CS]
print("trajectory: %d iterations, %.3f GB per card" % (len(sizes), sum(sizes) / 1e9), flush=True)
print("loadavg at start:", open("/proc/loadavg").read().strip(), flush=True)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, N))
print("mesh open:", md.get_num_devices(), flush=True)
out = {"n_devices": N, "cs": CS, "bytes": sizes, "traj_GB": sum(sizes) / 1e9,
       "visible": os.environ.get("TT_VISIBLE_DEVICES", ""), "arms": []}
try:
    try:
        mapper = ttnn.ReplicateTensorToMesh(md)
    except AttributeError:
        mapper = ttnn.replicate_tensor_to_mesh_mapper(md)
    xs = []
    for nb in sizes:
        cols = 1024
        rows = max(32, (nb // (cols * 4)) // 32 * 32)
        t = torch.randn(1, 1, rows, cols, dtype=torch.float32)
        xs.append(ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=md,
                                  mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    real = sum(x.volume() * 4 for x in xs)
    out["real_bytes"] = real
    print("allocated, real trajectory %.3f GB" % (real / 1e9), flush=True)

    # compile every shape once, outside the timed region
    for x in xs:
        y = ttnn.all_reduce(x, topology=ttnn.Topology.Linear, num_links=2)
        ttnn.synchronize_device(md); ttnn.deallocate(y)

    for arm in range(ARMS):
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        ys = [ttnn.all_reduce(x, topology=ttnn.Topology.Linear, num_links=2) for x in xs]
        ttnn.synchronize_device(md)
        wall = time.perf_counter() - t0
        sha = hashlib.sha256(ttnn.to_torch(
            ys[-1], mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1].numpy().tobytes()
        ).hexdigest()[:16]
        for y in ys:
            ttnn.deallocate(y)
        row = {"arm": arm, "wall_ms": wall * 1e3, "GBs_alg": real / wall / 1e9,
               "GBs_bus": 2.0 * (N - 1) / N * real / wall / 1e9, "sha256_last": sha}
        out["arms"].append(row)
        print(json.dumps(row), flush=True)
    for x in xs:
        ttnn.deallocate(x)
finally:
    ttnn.close_mesh_device(md)

out["loadavg_end"] = open("/proc/loadavg").read().strip()
with open(os.path.expanduser("~/mthuening/relion-intercard/x7_traj_n%d.json" % N), "w") as f:
    json.dump(out, f, indent=1)
print("WROTE x7_traj_n%d.json" % N, out["loadavg_end"])
