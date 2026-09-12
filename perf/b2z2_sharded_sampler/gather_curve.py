"""Price the sampler shard's collective BEFORE building it.

`b2z2-dual-chip-fold` fitted the p300c on-card link over 4.2-134 MB and got
t = 10.7 us + bytes / 20.1 GB/s. The diffusion sampler's token track is 0.786 MB,
5x BELOW the bottom of that fit, so the fixed term is the whole risk and it has to
be measured, not extrapolated. Every size here is a tensor the token-axis shard
would actually move, named after the op that produces it.

Also answers the question that decides whether the shard can ever pay at fold
level: can `all_gather` be captured inside a ttnn trace on this build? The fold
pays a 1.0538x mesh tax untraced and 1.0129x traced (b2z2-dual-chip-fold), so a
collective that cannot be traced makes the whole row a NO-GO.
"""

import json, os, time
import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("GC_OUT", "perf/b2z2_sharded_sampler/gather_curve.json")
REPS = int(os.environ.get("GC_REPS", "21"))
BURST = int(os.environ.get("GC_BURST", "24"))

_lease = CardSetLease(); _lease.acquire()
print(f"lease: {_lease.cards}", flush=True)

import ttnn  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
print(f"mesh {mesh.shape} ids={[int(d) for d in mesh.get_device_ids()]}", flush=True)

# (name, full shape, shard/gather dim) -- every one is a real tensor of the shard.
CASES = [
    ("a_token_track[1,512,768]",     (1, 512, 768),      1),
    ("kv_heads_one[1,16,512,64]",    (1, 16, 512, 64),   2),
    ("kv_fused[1,512,2048]",         (1, 512, 2048),     1),
    ("qkv_full[1,512,3072]",         (1, 512, 3072),     1),
    ("atom_track[1,7168,128]",       (1, 7168, 128),     1),
    ("r_update[1,7168,32]",          (1, 7168, 32),      1),
]

res = {"lease_cards": _lease.cards, "reps": REPS, "burst": BURST, "cases": [], "trace": None}

for name, shape, dim in CASES:
    n = 1
    for s in shape:
        n *= s
    full = torch.arange(n, dtype=torch.int32).remainder(1021).to(torch.bfloat16).reshape(shape)
    nbytes = n * 2
    tt = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                         mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, dim))
    g = ttnn.all_gather(tt, dim=dim)
    ttnn.synchronize_device(mesh)
    per_chip = ttnn.to_torch(g, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, 0))
    chip_a, chip_b = per_chip[0:1], per_chip[1:2]
    ok_a = torch.equal(chip_a.float().reshape(shape), full.float())
    ok_b = torch.equal(chip_b.float().reshape(shape), full.float())
    # negative control: what a local-shard-only no-op would have produced
    half = full.clone()
    idx = [slice(None)] * len(shape)
    lo = [slice(None)] * len(shape)
    idx[dim] = slice(shape[dim] // 2, None)
    lo[dim] = slice(0, shape[dim] // 2)
    half[tuple(idx)] = full[tuple(lo)]
    ctrl_rejects = not torch.equal(chip_a.float().reshape(shape), half.float())
    ttnn.deallocate(g)

    calls = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        g = ttnn.all_gather(tt, dim=dim)
        ttnn.synchronize_device(mesh)
        calls.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    calls.sort()
    gs = []
    t0 = time.perf_counter()
    for _ in range(BURST):
        gs.append(ttnn.all_gather(tt, dim=dim))
    ttnn.synchronize_device(mesh)
    burst = (time.perf_counter() - t0) / BURST
    for x in gs:
        ttnn.deallocate(x)
    med = calls[len(calls) // 2]
    row = {"name": name, "shape": list(shape), "dim": dim, "bytes": nbytes,
           "median_us": med * 1e6, "min_us": calls[0] * 1e6, "max_us": calls[-1] * 1e6,
           "burst_us": burst * 1e6, "bitexact_chip0": ok_a, "bitexact_chip1": ok_b,
           "negctrl_rejected": ctrl_rejects}
    res["cases"].append(row)
    print(f"{name:30s} {nbytes/1e6:7.3f} MB  med {med*1e6:8.1f} us  burst {burst*1e6:8.1f} us  "
          f"exact={ok_a and ok_b} ctrl_rejects={ctrl_rejects}", flush=True)
    ttnn.deallocate(tt)

# Can a collective live inside a trace?
try:
    shape, dim = (1, 512, 768), 1
    full = torch.randn(shape).to(torch.bfloat16)
    tt = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                         mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, dim))
    _ = ttnn.all_gather(tt, dim=dim); ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = ttnn.all_gather(tt, dim=dim)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    tr_us = (time.perf_counter() - t0) / 20 * 1e6
    got = ttnn.to_torch(out, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, 0))[0:1]
    res["trace"] = {"captured": True, "replay_us": tr_us,
                    "bitexact": bool(torch.equal(got.float().reshape(shape), full.float()))}
    print("TRACE: captured, replay %.1f us, exact=%s" % (tr_us, res["trace"]["bitexact"]), flush=True)
    ttnn.release_trace(mesh, tid)
except Exception as e:
    res["trace"] = {"captured": False, "error": f"{type(e).__name__}: {e}"}
    print(f"TRACE: FAILED {type(e).__name__}: {e}", flush=True)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("wrote", OUT, flush=True)
ttnn.close_mesh_device(mesh)
