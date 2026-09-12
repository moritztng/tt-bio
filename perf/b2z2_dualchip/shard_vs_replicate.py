"""Does sharding the i-axis across the p300c pair beat replicating it? Both arms on ONE mesh.

This replaces the mesh-tax screen, which was broken at the root. That screen compared a 1x1
SUBMESH against the parent 1x2 mesh, and `ttnn.synchronize_device(parent)` does not drain a
submesh's queue -- so the 1-chip arm timed enqueue while the 2-chip arm timed execution, and the
"14.6x mesh tax" was the gap between two instruments. The tell was physical: 67 MB of layer_norm in
0.028 ms is 2.4 TB/s against this part's measured 444.9 GB/s DRAM roof.

No submesh here. Every arm runs on the same 1x2 mesh, so one `synchronize_device(mesh)` drains all
of them and the arms are directly comparable:

  REPLICATED  z is replicated, each chip computes the WHOLE output. Both chips do identical work in
              parallel, so the wall clock is what one chip takes -- this is the incumbent.
  SHARDED     z is sharded on the i-axis, each chip computes HALF the output rows.
  SHARDED+AG  the same, plus the all_gather that restores the full tensor for the next op.

SHARDED+AG / REPLICATED is the honest per-op value of the shard, communication included.

Every row prints the implied DRAM bandwidth. A number above the measured 444.9 GB/s roof is not a
fast arm, it is an unsynchronised one, and that is exactly how the previous instrument lied.
"""

import json
import os
import statistics
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("SHARD_OUT", "/tmp/b2z2_shard.json")
BURST = int(os.environ.get("SHARD_BURST", "8"))
REPS = int(os.environ.get("SHARD_REPS", "5"))
DRAM_ROOF_GBPS = 444.9

CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
res = {"burst": BURST, "reps": REPS, "ops": {}}

I, L, C = 512, 512, 128
repl = ttnn.replicate_tensor_to_mesh_mapper(mesh)
shard1 = ttnn.shard_tensor_to_mesh_mapper(mesh, 1)


def dev(t, mapper):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh, mesh_mapper=mapper)


def timeit(fn, *a):
    outs = []
    t0 = time.perf_counter()
    for _ in range(BURST):
        outs.append(fn(*a))
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / BURST
    for o in outs:
        ttnn.deallocate(o)
    return dt


try:
    z = torch.empty(1, I, L, C, dtype=torch.bfloat16).uniform_(-1, 1)
    w = torch.ones(C, dtype=torch.bfloat16)
    z_rep, z_shd = dev(z, repl), dev(z, shard1)
    w_rep = dev(w, repl)

    # trimul's per-channel contraction, chunked to 32 channels: out[c,i,j] = sum_k a[c,i,k] b[c,j,k].
    # The a-side carries the i-axis and is what a shard splits; the b-side stays whole on both chips.
    a = torch.empty(1, 32, I, L, dtype=torch.bfloat16).uniform_(-1, 1)
    b = torch.empty(1, 32, L, L, dtype=torch.bfloat16).uniform_(-1, 1)
    a_rep, a_shd, b_rep = dev(a, repl), dev(a, ttnn.shard_tensor_to_mesh_mapper(mesh, 2)), dev(b, repl)

    CASES = {
        "layernorm_pair[1,I,512,128]": (
            lambda: ttnn.layer_norm(z_rep, weight=w_rep, epsilon=1e-5, compute_kernel_config=KC),
            lambda: ttnn.layer_norm(z_shd, weight=w_rep, epsilon=1e-5, compute_kernel_config=KC),
            z.numel() * 2 * 2),
        "add_pair[1,I,512,128]": (
            lambda: ttnn.add(z_rep, z_rep), lambda: ttnn.add(z_shd, z_shd), z.numel() * 2 * 3),
        "trimul_contract[1,32,I,512]x[1,32,512,512]": (
            lambda: ttnn.matmul(a_rep, b_rep, compute_kernel_config=KC),
            lambda: ttnn.matmul(a_shd, b_rep, compute_kernel_config=KC),
            (a.numel() + b.numel() + a.numel()) * 2),
    }

    for name, (f_rep, f_shd, traffic_bytes) in CASES.items():
        ttnn.deallocate(f_rep()); ttnn.deallocate(f_shd())
        ttnn.synchronize_device(mesh)

        def shd_ag():
            o = f_shd()
            g = ttnn.all_gather(o, dim=1 if "matmul" not in name else 2)
            ttnn.deallocate(o)
            return g

        ttnn.deallocate(shd_ag()); ttnn.synchronize_device(mesh)

        s = {"replicated": [], "sharded": [], "sharded_ag": []}
        for _ in range(REPS):
            s["replicated"].append(timeit(f_rep))
            s["sharded"].append(timeit(f_shd))
            s["sharded_ag"].append(timeit(shd_ag))
        m = {k: statistics.median(v) for k, v in s.items()}
        gbps = traffic_bytes / m["replicated"] / 1e9
        row = {**{f"median_{k}_s": v for k, v in m.items()},
               "speedup_shard_only": m["replicated"] / m["sharded"],
               "speedup_shard_with_gather": m["replicated"] / m["sharded_ag"],
               "implied_gbps_replicated": gbps,
               "physically_possible": gbps <= DRAM_ROOF_GBPS,
               "samples": s}
        res["ops"][name] = row
        log(f"{name}")
        log(f"   replicated(=1chip) {m['replicated']*1e3:8.3f} ms   implied {gbps:7.1f} GB/s "
            f"{'OK' if row['physically_possible'] else '*** ABOVE DRAM ROOF, ARM NOT SYNCED ***'}")
        log(f"   sharded            {m['sharded']*1e3:8.3f} ms   speedup {row['speedup_shard_only']:.3f}x")
        log(f"   sharded+allgather  {m['sharded_ag']*1e3:8.3f} ms   speedup {row['speedup_shard_with_gather']:.3f}x")
finally:
    try:
        ttnn.close_mesh_device(mesh)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")

json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
