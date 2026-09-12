"""Pass 2, one device open on a quiet box: the mesh tax, then the link curve re-taken.

Two questions, in this order, because the second only matters if the first is small:

  MESH TAX -- does SPMD over the p300c pair cost anything per op BEFORE any sharding? A tax is
  charged 280 times in the trunk, so 10 % per op would eat a third of the whole lever. Arms:
    1x1  a 1x1 submesh of the pair, one chip, fabric up
    1x2  the full pair, every tensor REPLICATED, so both chips do identical full work in parallel
  A tax-free mesh gives the SAME wall clock for both, not double. (What this pair CANNOT see is
  the cost of turning fabric on at all -- both arms run under fabric. fabric_tax.py measures that
  separately against a plain no-fabric device open.)

  LINK CURVE -- re-taken. The first table was measured before 14:41Z, when a wave-1 parity gate
  was still folding on all four chips of this host, so it had an unannounced co-tenant.

Bounded by its own rep counts and never by an external `timeout`: killing a fabric job mid-flight
wedges an active ethernet core and costs a board reset.
"""

import json
import os
import statistics
import sys
import time

import torch

from tt_bio.device_lease import CardSetLease

OUT = os.environ.get("PASS2_OUT", "/tmp/b2z2_pass2.json")
REPS = int(os.environ.get("PASS2_REPS", "5"))
BURST = int(os.environ.get("PASS2_BURST", "8"))
LINK_REPS = int(os.environ.get("PASS2_LINK_REPS", "20"))
ENQUEUE_FLOOR_S = 5e-6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_lease = CardSetLease()
_lease.acquire()
log(f"lease {_lease.cards}")

import ttnn  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh2 = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768)
mesh1 = mesh2.create_submesh(ttnn.MeshShape(1, 1))
log(f"mesh2={mesh2.shape} mesh1={mesh1.shape} ids={[int(d) for d in mesh2.get_device_ids()]}")

KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
    fp32_dest_acc_en=False, packer_l1_acc=True,
)
result = {"lease_cards": _lease.cards, "reps": REPS, "burst": BURST, "mesh_tax": {}, "link": []}


def rnd(*shape):
    """Cheap bf16 fill. torch.randn on 33.5M elements costs seconds and we build many of these."""
    return torch.empty(*shape, dtype=torch.bfloat16).uniform_(-1, 1)


CASES = {
    "matmul_1x16x512x512_at_512x128": (
        lambda d: (ttnn.from_torch(rnd(1, 16, 512, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d),
                   ttnn.from_torch(rnd(1, 16, 512, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)),
        lambda a, b: ttnn.matmul(a, b, compute_kernel_config=KC),
    ),
    "layernorm_pair_1x512x512x128": (
        lambda d: (ttnn.from_torch(rnd(1, 512, 512, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d),
                   ttnn.from_torch(rnd(128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)),
        lambda a, w: ttnn.layer_norm(a, weight=w, epsilon=1e-5, compute_kernel_config=KC),
    ),
    "softmax_triatt_1x4x512x512": (
        lambda d: (ttnn.from_torch(rnd(1, 4, 512, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d), None),
        lambda a, _: ttnn.softmax(a, dim=-1, compute_kernel_config=KC),
    ),
    "add_pair_1x512x512x128": (
        lambda d: (ttnn.from_torch(rnd(1, 512, 512, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d),
                   ttnn.from_torch(rnd(1, 512, 512, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)),
        lambda a, b: ttnn.add(a, b),
    ),
}


def burst_time(operands, fn):
    """BURST calls between two syncs on the PARENT mesh, which owns the queues for both arms.

    Deliberately NO host readback inside the timed region. A readback does prove the device
    finished, but composing a 67 MB pair-track output back to host costs tens of milliseconds of
    PCIe against a 1-2 ms op, so the guard would have become most of the measurement. The
    enqueue-floor check after the loop catches the same failure -- a sync that does not drain
    collapses per-call time to ~1 us, which no real op here can reach -- without paying for it.
    """
    outs = []
    t0 = time.perf_counter()
    for _ in range(BURST):
        outs.append(fn(*operands))
    ttnn.synchronize_device(mesh2)
    dt = (time.perf_counter() - t0) / BURST
    for o in outs:
        ttnn.deallocate(o)
    return dt


def produces_data(out, dev):
    """One composed readback per case, OUTSIDE the timed region: does this arm compute at all."""
    composer = (ttnn.concat_mesh_to_tensor_composer(dev, 0)
                if dev.get_num_devices() > 1 else None)
    t = ttnn.to_torch(out, mesh_composer=composer) if composer else ttnn.to_torch(out)
    return int(t.numel()) > 0


suspect_any = []
try:
    for name, (build, fn) in CASES.items():
        log(f"mesh-tax {name}: building")
        ops = {"1x1": build(mesh1), "1x2": build(mesh2)}
        live = {}
        for k, dev in (("1x1", mesh1), ("1x2", mesh2)):
            warm = fn(*ops[k])
            ttnn.synchronize_device(mesh2)
            live[k] = produces_data(warm, dev)
            ttnn.deallocate(warm)
        assert all(live.values()), f"{name}: an arm produced nothing: {live}"
        log(f"mesh-tax {name}: timing (both arms produce data: {live})")

        s = {"1x1": [], "1x2": [], "1x1b": []}
        for _ in range(REPS):
            s["1x1"].append(burst_time(ops["1x1"], fn))
            s["1x2"].append(burst_time(ops["1x2"], fn))
            s["1x1b"].append(burst_time(ops["1x1"], fn))
        med = {k: statistics.median(v) for k, v in s.items()}
        suspect = [k for k, v in med.items() if v < ENQUEUE_FLOOR_S]
        suspect_any += [f"{name}:{k}" for k in suspect]
        result["mesh_tax"][name] = {
            "median_1chip_s": med["1x1"], "median_2chip_replicated_s": med["1x2"],
            "mesh_tax_ratio": med["1x2"] / med["1x1"], "aa_floor_ratio": med["1x1b"] / med["1x1"],
            "below_enqueue_floor": suspect, "samples": s,
        }
        print(f"  {name:32s} 1chip={med['1x1']*1e3:8.4f} ms  2chip={med['1x2']*1e3:8.4f} ms  "
              f"MESH-TAX={med['1x2']/med['1x1']:.4f}  A/A={med['1x1b']/med['1x1']:.4f}"
              + (f"  SUSPECT:{suspect}" if suspect else ""), flush=True)
        for k in ops:
            for t in ops[k]:
                if t is not None:
                    ttnn.deallocate(t)

    # ---- link curve, re-taken on the quiet box ------------------------------------------------
    for I in (32, 64, 128, 256, 512, 1024):
        full = torch.arange(I * 512 * 128, dtype=torch.int32).remainder(1021).to(torch.bfloat16)
        full = full.reshape(1, I, 512, 128)
        nbytes = full.numel() * 2
        tt = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh2,
                             mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh2, 1))
        g = ttnn.all_gather(tt, dim=1)
        ttnn.synchronize_device(mesh2)
        per_chip = ttnn.to_torch(g, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh2, 0))
        ok_a = torch.equal(per_chip[0:1].float(), full.float())
        ok_b = torch.equal(per_chip[1:2].float(), full.float())
        half = full.clone()
        half[:, I // 2:] = full[:, : I // 2]
        ctrl = not torch.equal(per_chip[0:1].float(), half.float())
        ttnn.deallocate(g)

        gs = []
        t0 = time.perf_counter()
        for _ in range(LINK_REPS):
            gs.append(ttnn.all_gather(tt, dim=1))
        ttnn.synchronize_device(mesh2)
        burst_s = (time.perf_counter() - t0) / LINK_REPS
        for x in gs:
            ttnn.deallocate(x)
        moved = nbytes / 2
        row = {"i_axis": I, "tensor_bytes": nbytes, "verified_chip_a": ok_a, "verified_chip_b": ok_b,
               "negative_control_rejects_local_only": ctrl, "burst_per_op_s": burst_s,
               "gbps_burst": moved / burst_s / 1e9}
        result["link"].append(row)
        print(f"  link I={I:5d} {nbytes/1e6:7.2f} MB verified={ok_a and ok_b} negctrl={ctrl} "
              f"burst={burst_s*1e3:7.3f} ms ({row['gbps_burst']:6.2f} GB/s)", flush=True)
        ttnn.deallocate(tt)

finally:
    # Close the submesh first: it borrows the parent's command queue, and closing the parent
    # while it is open throws 'cq ID 0 is in use by parent mesh' from a destructor.
    import gc
    mesh1 = None
    gc.collect()
    try:
        ttnn.close_mesh_device(mesh2)
        log('meshes closed cleanly')
    except Exception as e:
        log(f'parent close failed, letting the process exit: {e}')
result["instrument_ok"] = not suspect_any
json.dump(result, open(OUT, "w"), indent=2)
log(f"wrote {OUT}")
if suspect_any:
    print("INSTRUMENT FAILED, mesh-tax numbers not usable: " + ", ".join(suspect_any), flush=True)
    sys.exit(4)
