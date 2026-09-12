"""Is `from_torch(device=mesh)` with no mesh_mapper actually REPLICATING?

The mesh-tax screen reported 14.6x on a matmul and then spent 30+ minutes on 120 layer_norms,
implying ~20 s per call on the 1x2 arm. A tax that ranges from 14x to ~10,000x across two ops is
not a hardware property, so the suspect is the benchmark: every multi-device tensor in it was built
with `ttnn.from_torch(t, device=mesh2)` and NO mesh_mapper, on the assumption that this replicates.
This checks that assumption instead of carrying it, and prices the explicit replicate mapper
against it.

Walks a size ladder and ABANDONS the ladder as soon as a per-call time crosses CAP_S, so a
pathology costs seconds instead of half an hour.
"""

import json
import os
import statistics
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("REPL_OUT", "/tmp/b2z2_repl.json")
BURST = 4
REPS = 3
CAP_S = 0.25          # per-call ceiling; past this the ladder stops

CardSetLease().acquire()
import ttnn  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh2 = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
mesh1 = mesh2.create_submesh(ttnn.MeshShape(1, 1))
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True)
res = {"shape_probe": {}, "ladder": []}

try:
    # ---- 1. what does each construction actually produce -----------------------------------
    probe = torch.arange(1 * 32 * 64 * 128, dtype=torch.int32).remainder(97).to(torch.bfloat16)
    probe = probe.reshape(1, 32, 64, 128)
    builds = {
        "no_mapper": dict(device=mesh2),
        "replicate": dict(device=mesh2, mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh2)),
    }
    for nm, kw in builds.items():
        t = ttnn.from_torch(probe, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw)
        comp = ttnn.to_torch(t, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh2, 0))
        # Replicated => the two per-device copies are identical AND each equals the input.
        a, b = comp[0:1], comp[1:2]
        res["shape_probe"][nm] = {
            "device_shape": [int(d) for d in t.shape],
            "composed_shape": list(comp.shape),
            "two_copies_identical": bool(torch.equal(a.float(), b.float())),
            "copy0_equals_input": bool(torch.equal(a.float(), probe.float())),
        }
        log(f"{nm}: dev_shape={res['shape_probe'][nm]['device_shape']} "
            f"composed={res['shape_probe'][nm]['composed_shape']} "
            f"identical={res['shape_probe'][nm]['two_copies_identical']} "
            f"eq_input={res['shape_probe'][nm]['copy0_equals_input']}")
        ttnn.deallocate(t)

    # ---- 2. size ladder: 1 chip vs both, both constructions --------------------------------
    def timed(dev, ten, w):
        outs = []
        t0 = time.perf_counter()
        for _ in range(BURST):
            outs.append(ttnn.layer_norm(ten, weight=w, epsilon=1e-5, compute_kernel_config=KC))
        ttnn.synchronize_device(mesh2)
        dt = (time.perf_counter() - t0) / BURST
        for o in outs:
            ttnn.deallocate(o)
        return dt

    stop = False
    for I in (32, 64, 128, 256, 512):
        if stop:
            break
        x = torch.empty(1, I, 512, 128, dtype=torch.bfloat16).uniform_(-1, 1)
        arms = {
            "1chip": (mesh1, dict(device=mesh1)),
            "2chip_no_mapper": (mesh2, dict(device=mesh2)),
            "2chip_replicate": (mesh2, dict(device=mesh2,
                                            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh2))),
        }
        row = {"i_axis": I, "bytes": x.numel() * 2}
        tens = {}
        for nm, (dev, kw) in arms.items():
            tens[nm] = (ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **kw),
                        ttnn.from_torch(torch.ones(128, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, **kw), dev)
            ttnn.deallocate(ttnn.layer_norm(tens[nm][0], weight=tens[nm][1], epsilon=1e-5,
                                            compute_kernel_config=KC))
        ttnn.synchronize_device(mesh2)
        for nm in arms:
            t, w, dev = tens[nm]
            s = sorted(timed(dev, t, w) for _ in range(REPS))
            row[nm] = s[len(s) // 2]
            if row[nm] > CAP_S:
                stop = True
        row["tax_no_mapper"] = row["2chip_no_mapper"] / row["1chip"]
        row["tax_replicate"] = row["2chip_replicate"] / row["1chip"]
        res["ladder"].append(row)
        log(f"I={I:4d} {row['bytes']/1e6:6.2f}MB  1chip={row['1chip']*1e3:9.3f}ms  "
            f"no_mapper={row['2chip_no_mapper']*1e3:9.3f}ms ({row['tax_no_mapper']:8.2f}x)  "
            f"replicate={row['2chip_replicate']*1e3:9.3f}ms ({row['tax_replicate']:8.2f}x)"
            + ("  [CAP HIT, stopping ladder]" if stop else ""))
        for nm in tens:
            ttnn.deallocate(tens[nm][0]); ttnn.deallocate(tens[nm][1])
finally:
    import gc
    mesh1 = None
    gc.collect()
    try:
        ttnn.close_mesh_device(mesh2)
        log("closed cleanly")
    except Exception as e:
        log(f"close failed: {e}")

json.dump(res, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")

# Skip interpreter finalisation. tt-metal's MeshDevice destructor runs at Py_FinalizeEx and
# throws there even after a clean close, and an abort inside a live fabric context wedges an
# active ethernet core -- which costs a tt-smi -r on BOTH chips of the board. The results are
# already on disk by this line, so there is nothing left for a clean shutdown to protect.
os._exit(0)
