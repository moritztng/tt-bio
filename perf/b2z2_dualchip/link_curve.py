"""Step 1b: prove the transfer is real, then get the bandwidth CURVE, not one point.

link_probe.py showed a 1x2 mesh opens on a p300c board pair and a 67.1 MB all-gather takes
1.81 ms. Two things that number cannot tell you on its own:

  * whether any byte actually crossed the ethernet -- an all_gather that silently returned the
    local shard would look exactly this fast. So every size here is verified bit-exact against
    the torch tensor it came from, on BOTH chips.
  * what a gather of some OTHER size costs. The shard design needs to price gathers from 1 MB
    (a bias) to 134 MB (a pair track in fp32), and a single point cannot separate the fixed
    per-call cost from the per-byte cost.

Also times a back-to-back burst between two syncs, so the per-call host dispatch is amortised
and what is left is what a device-resident sharded trunk would actually pay.
"""

import json
import os
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("PROBE2_OUT", "/tmp/b2z2_probe2.json")
REPS = int(os.environ.get("PROBE2_REPS", "20"))
BURST = int(os.environ.get("PROBE2_BURST", "20"))

_lease = CardSetLease()
_lease.acquire()
print(f"lease: {_lease.cards}", flush=True)

import ttnn  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
print(f"mesh: {mesh.shape} ids={[int(d) for d in mesh.get_device_ids()]}", flush=True)

result = {"lease_cards": _lease.cards, "sizes": [], "point_to_point": None}

# i-axis sizes of a [1, I, 512, 128] bf16 pair track. I=512 is the production pair track
# (67.1 MB); the others bracket it so the fixed cost and the per-byte cost separate.
for I in (32, 64, 128, 256, 512, 1024):
    full = torch.arange(I * 512 * 128, dtype=torch.int32).remainder(1021).to(torch.bfloat16)
    full = full.reshape(1, I, 512, 128)
    nbytes = full.numel() * 2
    tt = ttnn.from_torch(
        full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, 1),
    )

    g = ttnn.all_gather(tt, dim=1)
    ttnn.synchronize_device(mesh)

    # VERIFY: pull the gathered tensor back from EACH chip separately and demand it equal the
    # whole input. If the gather were a local no-op, chip (0,0) would hold rows [0:I/2] twice
    # and this comparison would fail on the second half -- which is exactly what it must catch.
    per_chip = ttnn.to_torch(
        g, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, 0)
    )
    chip_a, chip_b = per_chip[0:1], per_chip[1:2]
    ok_a = torch.equal(chip_a.float(), full.float())
    ok_b = torch.equal(chip_b.float(), full.float())
    # Negative control: the local-shard-only result the check must be able to reject.
    half = full.clone()
    half[:, I // 2 :] = full[:, : I // 2]
    ctrl_rejects = not torch.equal(chip_a.float(), half.float())
    ttnn.deallocate(g)

    per_call = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        g = ttnn.all_gather(tt, dim=1)
        ttnn.synchronize_device(mesh)
        per_call.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    per_call.sort()

    # Burst: BURST gathers enqueued back to back, one sync at the end. Divides out host dispatch.
    gs = []
    t0 = time.perf_counter()
    for _ in range(BURST):
        gs.append(ttnn.all_gather(tt, dim=1))
    ttnn.synchronize_device(mesh)
    burst_s = (time.perf_counter() - t0) / BURST
    for x in gs:
        ttnn.deallocate(x)

    med = per_call[len(per_call) // 2]
    moved = nbytes / 2  # each chip receives the half it did not own
    row = {
        "i_axis": I,
        "tensor_bytes": nbytes,
        "bytes_across_link_per_direction": moved,
        "verified_chip_a": ok_a,
        "verified_chip_b": ok_b,
        "negative_control_rejects_local_only": ctrl_rejects,
        "percall_median_s": med,
        "percall_min_s": per_call[0],
        "burst_per_op_s": burst_s,
        "gbps_percall": moved / med / 1e9,
        "gbps_burst": moved / burst_s / 1e9,
    }
    result["sizes"].append(row)
    print(
        f"I={I:5d} {nbytes/1e6:7.2f} MB  verified={ok_a and ok_b}  negctrl={ctrl_rejects}  "
        f"percall={med*1e3:7.3f} ms ({row['gbps_percall']:6.2f} GB/s)  "
        f"burst={burst_s*1e3:7.3f} ms ({row['gbps_burst']:6.2f} GB/s)",
        flush=True,
    )
    ttnn.deallocate(tt)

ttnn.close_mesh_device(mesh)
json.dump(result, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
