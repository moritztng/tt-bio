"""Step 1c (v2): the mesh tax, with an instrument that actually synchronises.

v1 reported an 18.4x "mesh tax" and a 0.0114 ms 1x16x512x512 @ 512x128 matmul. 0.0114 ms for a
1.07 GFLOP matmul is 94 TFLOP/s -- not impossible on this part, which is exactly why it slipped
past -- but over a burst of 10 it is 1.14 us per call, which is enqueue latency, not device time.
`ttnn.synchronize_device(submesh)` does not drain the parent's queue, so arm A was timing dispatch
and arm B was timing execution, and the ratio was the difference between the two instruments.

Fixed here two ways, and the fix is CHECKED rather than assumed:
  * both arms synchronise on the PARENT mesh, which owns the command queues;
  * every burst ends by reading one element back to host, which cannot return before the device
    has actually produced it.
The guard below fails the run if a per-call time is under 5 us, the enqueue-latency floor that
told v1's lie -- a silent repeat of that bug is worse than no number.
"""

import json
import os
import statistics
import sys
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("MESHTAX_OUT", "/tmp/b2z2_meshtax2.json")
REPS = int(os.environ.get("MESHTAX_REPS", "5"))
BURST = int(os.environ.get("MESHTAX_BURST", "8"))
ENQUEUE_FLOOR_S = 5e-6

_lease = CardSetLease()
_lease.acquire()
print(f"lease: {_lease.cards}", flush=True)

import ttnn  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh2 = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
mesh1 = mesh2.create_submesh(ttnn.MeshShape(1, 1))
print(f"mesh2={mesh2.shape} mesh1={mesh1.shape}", flush=True)

KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True,
)

CASES = {
    "matmul_1x16x512x512_at_512x128": (
        lambda d: (
            ttnn.from_torch(torch.randn(1, 16, 512, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=d),
            ttnn.from_torch(torch.randn(1, 16, 512, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=d),
        ),
        lambda a, b: ttnn.matmul(a, b, compute_kernel_config=KC),
    ),
    "layernorm_pair_track_512": (
        lambda d: (
            ttnn.from_torch(torch.randn(1, 512, 512, 128, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=d),
            ttnn.from_torch(torch.randn(128, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=d),
        ),
        lambda a, w: ttnn.layer_norm(a, weight=w, epsilon=1e-5, compute_kernel_config=KC),
    ),
    "softmax_triatt_1x4x512x512": (
        lambda d: (
            ttnn.from_torch(torch.randn(1, 4, 512, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=d),
            None,
        ),
        lambda a, _: ttnn.softmax(a, dim=-1, compute_kernel_config=KC),
    ),
}


def burst_time(operands, fn):
    outs = []
    t0 = time.perf_counter()
    for _ in range(BURST):
        outs.append(fn(*operands))
    ttnn.synchronize_device(mesh2)          # the parent owns the queues for BOTH arms
    _ = ttnn.to_torch(outs[-1])[(0,) * len(outs[-1].shape)]  # a readback cannot beat the device
    dt = (time.perf_counter() - t0) / BURST
    for o in outs:
        ttnn.deallocate(o)
    return dt


result = {"lease_cards": _lease.cards, "reps": REPS, "burst": BURST, "cases": {}}
failed = []

for name, (build, fn) in CASES.items():
    ops = {"1x1": build(mesh1), "1x2": build(mesh2)}
    for k in ops:
        ttnn.deallocate(fn(*ops[k]))
    ttnn.synchronize_device(mesh2)

    s = {"1x1": [], "1x2": [], "1x1b": []}
    for _ in range(REPS):
        s["1x1"].append(burst_time(ops["1x1"], fn))
        s["1x2"].append(burst_time(ops["1x2"], fn))
        s["1x1b"].append(burst_time(ops["1x1"], fn))

    med = {k: statistics.median(v) for k, v in s.items()}
    suspect = [k for k, v in med.items() if v < ENQUEUE_FLOOR_S]
    row = {
        "median_1chip_s": med["1x1"],
        "median_2chip_replicated_s": med["1x2"],
        "mesh_tax_ratio": med["1x2"] / med["1x1"],
        "aa_floor_ratio": med["1x1b"] / med["1x1"],
        "below_enqueue_floor": suspect,
        "samples": s,
    }
    result["cases"][name] = row
    if suspect:
        failed.append(f"{name}: arms {suspect} under {ENQUEUE_FLOOR_S*1e6:.0f} us/call")
    print(
        f"{name:34s} 1chip={med['1x1']*1e3:8.4f} ms  2chip={med['1x2']*1e3:8.4f} ms  "
        f"MESH-TAX={row['mesh_tax_ratio']:.4f}  A/A={row['aa_floor_ratio']:.4f}"
        + (f"  SUSPECT:{suspect}" if suspect else ""),
        flush=True,
    )
    for k in ops:
        for t in ops[k]:
            if t is not None:
                ttnn.deallocate(t)

ttnn.close_mesh_device(mesh2)
result["instrument_ok"] = not failed
json.dump(result, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
if failed:
    print("INSTRUMENT FAILED, numbers not usable: " + "; ".join(failed), flush=True)
    sys.exit(4)
