"""Overlap the gather with compute by giving each its own sub-device.

gather_overlap.py established that both ops take `queue_id` but tt-metal refuses a cross-queue
enqueue on the same sub-device ("Sub device id 0 currently in use by cq 1"). So overlap is not free:
the core grid has to be partitioned, and `ttnn.all_gather` takes a `subdevice_id` for exactly this.

The trade this measures is the real one. Handing the CCL its own worker row costs the compute side
that row, so the question is not "does it overlap" but "does hiding 1.680 ms of link cost less than
the compute those cores were doing". Both numbers are below.
"""

import json
import os
import statistics
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("OVL2_OUT", "/tmp/b2z2_ovl2.json")
BURST, REPS = 6, 5
CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768, num_command_queues=2)
g = mesh.compute_with_storage_grid_size()
GX, GY = int(g.x), int(g.y)
log(f"compute grid {GX}x{GY} = {GX*GY} cores")
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
repl = ttnn.replicate_tensor_to_mesh_mapper(mesh)
shard1 = ttnn.shard_tensor_to_mesh_mapper(mesh, 1)
res = {"grid": [GX, GY]}

try:
    cc = ttnn.CoreCoord
    compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(cc(0, 0), cc(GX - 1, GY - 3))])   # all but the last two rows
    ccl_set = ttnn.CoreRangeSet([ttnn.CoreRange(cc(0, GY - 2), cc(GX - 1, GY - 1))])  # last two rows, handed to the CCL
    sub_compute, sub_ccl = ttnn.SubDevice([compute_set]), ttnn.SubDevice([ccl_set])
    mgr = mesh.create_sub_device_manager([sub_compute, sub_ccl], 0)
    mesh.load_sub_device_manager(mgr)
    sd_compute, sd_ccl = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
    log(f"sub-devices loaded: compute {GX}x{GY-2}={GX*(GY-2)} cores, ccl {GX}x2={GX*2} cores")
    res["cores_compute"], res["cores_ccl"] = GX * (GY - 2), GX * 2

    z = torch.empty(1, 512, 512, 128, dtype=torch.bfloat16).uniform_(-1, 1)
    z_shd = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=shard1)
    a = ttnn.from_torch(torch.empty(1, 32, 256, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    b = ttnn.from_torch(torch.empty(1, 32, 512, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)

    gath = lambda q: ttnn.all_gather(z_shd, dim=1, queue_id=q, subdevice_id=sd_ccl)
    comp = lambda q: ttnn.matmul(a, b, compute_kernel_config=KC, queue_id=q)

    def bench(fn, stall):
        mesh.set_sub_device_stall_group(stall)
        s = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            outs = [fn() for _ in range(BURST)]
            ttnn.synchronize_device(mesh)
            s.append((time.perf_counter() - t0) / BURST)
            for o in outs:
                for x in (o if isinstance(o, tuple) else (o,)):
                    ttnn.deallocate(x)
        mesh.reset_sub_device_stall_group()
        return statistics.median(s)

    bench(lambda: (gath(1), comp(0)), [sd_compute, sd_ccl])          # warm
    g_only = bench(lambda: gath(1), [sd_ccl])
    c_only = bench(lambda: comp(0), [sd_compute])
    serial = bench(lambda: (gath(0), comp(0)), [sd_compute, sd_ccl])
    overlap = bench(lambda: (gath(1), comp(0)), [sd_compute, sd_ccl])

    res.update({"gather_only_s": g_only, "compute_only_s": c_only,
                "serial_s": serial, "overlap_s": overlap,
                "overlap_speedup_vs_serial": serial / overlap,
                "hiding_efficiency": (serial - overlap) / min(g_only, c_only),
                "non_physical": overlap < max(g_only, c_only) * 0.95})
    log(f"gather alone (ccl row)      {g_only*1e3:8.3f} ms")
    log(f"compute alone ({GX*(GY-2)} cores) {c_only*1e3:8.3f} ms")
    log(f"SERIAL  both on cq0         {serial*1e3:8.3f} ms")
    log(f"OVERLAP gather cq1/comp cq0 {overlap*1e3:8.3f} ms   {serial/overlap:.3f}x vs serial")
    log(f"  hid {(serial-overlap)*1e3:.3f} ms of the {min(g_only,c_only)*1e3:.3f} ms shorter leg "
        f"= {(serial-overlap)/min(g_only,c_only)*100:.0f}% hiding"
        + ("   *** NON-PHYSICAL ***" if res["non_physical"] else ""))
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    log(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    try:
        ttnn.close_mesh_device(mesh)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")
json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
