"""The same matmul sweep on ONE chip, no mesh. What is the per-op floor without the pair?

fixed_cost_fit.py found a ~0.373 ms floor per op on the 1x2 mesh: below i=128 the trimul
contraction costs the same regardless of size, and at i=512 only ~23 % of the call is i-dependent.
That floor is what caps the shard speedup at 1.213x instead of ~2x.

Everything turns on whether the floor is a MESH cost or just what this op costs on this part. A
Pairformer block runs ~272 programs in 36.5 ms, i.e. 0.134 ms/program on average -- already well
UNDER the mesh floor. If the floor is mesh-specific, putting the trunk on a mesh multiplies the
block instead of halving it and this row is dead. If one chip shows the same floor, the floor is
the microbenchmark's own overhead and says nothing about the shard.

Separate process on purpose: a 1x1 SUBMESH cannot be timed (the parent's synchronize does not drain
it, which is what produced this pass's phantom 14.6x "mesh tax"), so the one-chip arm has to be a
top-level device in its own process. Routed through tt_bio.get_device so the lone-p300 p150 mesh
graph descriptor is handled.
"""

import json
import os
import statistics
import time

import torch

OUT = os.environ.get("FLOOR1_OUT", "/tmp/b2z2_floor1.json")
BURST, REPS = 8, 5

from tt_bio import tenstorrent as tt  # noqa: E402
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


dev = tt.get_device()
log(f"device open: {dev}")
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
rows = []
b = ttnn.from_torch(torch.empty(1, 32, 512, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for I in (64, 128, 256, 512, 1024):
    a = ttnn.from_torch(torch.empty(1, 32, I, 512, dtype=torch.bfloat16).uniform_(-1, 1),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC))
    ttnn.synchronize_device(dev)
    s = []
    for _ in range(REPS):
        outs = []
        t0 = time.perf_counter()
        for _ in range(BURST):
            outs.append(ttnn.matmul(a, b, compute_kernel_config=KC))
        ttnn.synchronize_device(dev)
        s.append((time.perf_counter() - t0) / BURST)
        for o in outs:
            ttnn.deallocate(o)
    m = statistics.median(s)
    # traffic = a + b + out, all bf16
    traffic = (32 * I * 512 + 32 * 512 * 512 + 32 * I * 512) * 2
    rows.append({"i_axis": I, "median_s": m, "implied_gbps": traffic / m / 1e9})
    log(f"I={I:5d}  {m*1e3:8.4f} ms   implied {traffic/m/1e9:7.1f} GB/s")
    ttnn.deallocate(a)

json.dump({"rows": rows}, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
os._exit(0)
