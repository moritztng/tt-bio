"""Can the all_gather hide behind compute on a second command queue?

This is the deciding question for the dual-chip shard. Measured so far: sharding the i-axis across
the p300c pair halves row-local pair-track ops (1.953x, 1.985x) and SPMD costs only 1.04x, but the
gather that restores the full tensor costs 1.680 ms for the 67 MB pair track against ops costing
~0.45 ms, because the link is 20.1 GB/s and DRAM is 444.9. At 3 gathers per Pairformer block that
is 5.040 ms sitting in series inside a 27 ms block.

The gather's cost is pure bandwidth (t = 10.7 us + bytes / 20.1 GB/s), so it cannot be tuned. It can
only be overlapped. If it overlaps, the block goes from 1.343x/1.520x to 1.61x/1.92x and this row
stops being a contribution and becomes the campaign's largest single lever.

SERIAL   gather then compute, one queue. Cost must be gather + compute.
OVERLAP  gather on cq1, compute on cq0, joined with events. If the link and the Tensix run
         independently, cost approaches max(gather, compute).

The honest read is OVERLAP vs SERIAL on the SAME pair of operations in the same process. Anything
faster than the slower of the two halves alone would be non-physical and is flagged.
"""

import json
import os
import statistics
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("OVL_OUT", "/tmp/b2z2_ovl.json")
BURST, REPS = 6, 5
CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768, num_command_queues=2)
log(f"mesh open with {mesh.num_program_cache_entries and ''}2 command queues")
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
repl = ttnn.replicate_tensor_to_mesh_mapper(mesh)
shard1 = ttnn.shard_tensor_to_mesh_mapper(mesh, 1)
res = {}

try:
    I, L, C = 512, 512, 128
    z = torch.empty(1, I, L, C, dtype=torch.bfloat16).uniform_(-1, 1)
    z_shd = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=shard1)
    # An independent compute stream that does not depend on the gather, which is what a real
    # sharded block has: the next op's work on the LOCAL half while the far half is in flight.
    a = ttnn.from_torch(torch.empty(1, 32, 256, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    b = ttnn.from_torch(torch.empty(1, 32, 512, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)

    def gather(qid=0):
        try:
            return ttnn.all_gather(z_shd, dim=1, queue_id=qid)
        except TypeError:
            return ttnn.all_gather(z_shd, dim=1)

    def compute(qid=0):
        try:
            return ttnn.matmul(a, b, compute_kernel_config=KC, queue_id=qid)
        except TypeError:
            return ttnn.matmul(a, b, compute_kernel_config=KC)

    # does queue_id even reach the op
    res["queue_id_accepted"] = {}
    for nm, fn in (("all_gather", gather), ("matmul", compute)):
        try:
            o = fn(1)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(o)
            res["queue_id_accepted"][nm] = True
        except TypeError:
            res["queue_id_accepted"][nm] = False
        except Exception as e:
            res["queue_id_accepted"][nm] = f"error: {type(e).__name__}"
    log(f"queue_id accepted: {res['queue_id_accepted']}")

    def bench(fn):
        s = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            outs = [fn() for _ in range(BURST)]
            ttnn.synchronize_device(mesh)
            s.append((time.perf_counter() - t0) / BURST)
            for o in outs:
                for x in (o if isinstance(o, tuple) else (o,)):
                    ttnn.deallocate(x)
        return statistics.median(s)

    bench(lambda: (gather(0), compute(0)))   # warm
    g_only = bench(lambda: gather(0))
    c_only = bench(lambda: compute(0))
    serial = bench(lambda: (gather(0), compute(0)))
    overlap = bench(lambda: (gather(1), compute(0)))

    res.update({
        "gather_only_s": g_only, "compute_only_s": c_only,
        "serial_s": serial, "overlap_s": overlap,
        "serial_over_sum": serial / (g_only + c_only),
        "overlap_over_serial": serial / overlap,
        "overlap_over_max_half": overlap / max(g_only, c_only),
        "non_physical": overlap < max(g_only, c_only) * 0.95,
    })
    log(f"gather alone   {g_only*1e3:8.3f} ms")
    log(f"compute alone  {c_only*1e3:8.3f} ms   (sum would be {(g_only+c_only)*1e3:.3f} ms)")
    log(f"SERIAL  cq0+cq0 {serial*1e3:8.3f} ms")
    log(f"OVERLAP cq1+cq0 {overlap*1e3:8.3f} ms   speedup vs serial {serial/overlap:.3f}x")
    log(f"  overlap / max(gather,compute) = {overlap/max(g_only,c_only):.3f}  "
        f"(1.00 = perfect hiding, 2.00 = none)"
        + ("   *** NON-PHYSICAL, INSTRUMENT SUSPECT ***" if res["non_physical"] else ""))
finally:
    try:
        ttnn.close_mesh_device(mesh)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")
json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
