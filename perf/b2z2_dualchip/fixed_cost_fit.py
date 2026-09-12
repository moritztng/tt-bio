"""Is the trimul contraction's poor 1.213x shard speedup a fixed per-op cost?

Sharding halves the contraction's FLOPs and two of its three tensors, so it should approach 2x,
and it gave 1.213x. If there is a fixed per-op cost F, then t(I) = F + k*I and the speedup at a
half-shard is (F + k*I) / (F + k*I/2), which tends to 1 for large F and to 2 for small F. At the
measured 0.489 / 0.403 ms, F ~ 0.2 ms would explain the whole shortfall and would mean the WORK
halves correctly -- in which case the real fused trimul kernel, which is 36x this microbenchmark's
duration per block, would barely notice F.

So: sweep I, fit F and k, and say which it is. Replicated only -- one arm, no shard, no gather.
"""

import json
import os
import statistics
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("FIXED_OUT", "/tmp/b2z2_fixed.json")
BURST, REPS = 8, 5
CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
repl = ttnn.replicate_tensor_to_mesh_mapper(mesh)
rows = []
try:
    b = ttnn.from_torch(torch.empty(1, 32, 512, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
    for I in (64, 128, 256, 512, 1024):
        a = ttnn.from_torch(torch.empty(1, 32, I, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
        ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC))
        ttnn.synchronize_device(mesh)
        s = []
        for _ in range(REPS):
            outs = []
            t0 = time.perf_counter()
            for _ in range(BURST):
                outs.append(ttnn.matmul(a, b, compute_kernel_config=KC))
            ttnn.synchronize_device(mesh)
            s.append((time.perf_counter() - t0) / BURST)
            for o in outs:
                ttnn.deallocate(o)
        m = statistics.median(s)
        rows.append({"i_axis": I, "median_s": m})
        log(f"I={I:5d}  {m*1e3:8.4f} ms")
        ttnn.deallocate(a)

    # least squares on t = F + k*I
    n = len(rows)
    sx = sum(r["i_axis"] for r in rows); sy = sum(r["median_s"] for r in rows)
    sxx = sum(r["i_axis"] ** 2 for r in rows); sxy = sum(r["i_axis"] * r["median_s"] for r in rows)
    k = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    F = (sy - k * sx) / n
    pred = lambda I: F + k * I
    log(f"FIT  fixed F = {F*1e3:.4f} ms   slope k = {k*1e6:.4f} us per i-row")
    log(f"     at I=512 fixed is {F/pred(512)*100:.1f} % of the call")
    log(f"     predicted half-shard speedup at I=512: {pred(512)/pred(256):.3f}x "
        f"(measured 1.213x)")
    json.dump({"rows": rows, "fixed_s": F, "slope_s_per_row": k,
               "predicted_half_shard_speedup": pred(512) / pred(256)}, open(OUT, "w"), indent=2)
finally:
    try:
        ttnn.close_mesh_device(mesh)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")
log(f"wrote {OUT}")
os._exit(0)
