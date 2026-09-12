"""Does the p300c on-card chip-to-chip link come up, and how fast is it?

Step 1 of b2z2-dual-chip-fold. Opens a 1x2 mesh over ONE p300c board pair, reports what
tt-metal thinks the ethernet topology is, then times a pair-track-sized transfer
([1,512,512,128] bf16 = 67.1 MB) across the link so the shard-axis decision is made on a
measured number instead of on taste.

Run pinned to a board pair, e.g.
    TT_VISIBLE_DEVICES=2,3 TT_BIO_LEASE_CARDS=2,3 TT_BIO_LEASE_HOLDER=worker:b2z2-dual-chip-fold \
        python3 perf/b2z2_dualchip/link_probe.py
"""

import json
import os
import sys
import time

OUT = os.environ.get("LINK_PROBE_OUT", "/tmp/b2z2_link_probe.json")
REPS = int(os.environ.get("LINK_PROBE_REPS", "20"))
result = {"argv": sys.argv, "visible": os.environ.get("TT_VISIBLE_DEVICES"), "stages": {}}


def stage(name, fn):
    t0 = time.perf_counter()
    try:
        v = fn()
        result["stages"][name] = {"ok": True, "value": v, "wall_s": time.perf_counter() - t0}
        print(f"[ok]   {name}: {v}", flush=True)
        return v
    except Exception as e:  # noqa: BLE001 - a probe reports every failure, it does not raise
        result["stages"][name] = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "wall_s": time.perf_counter() - t0,
        }
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return None


# Hold the fleet lease on EVERY visible chip for the probe's whole life. A raw-ttnn probe
# bypasses tt_bio.get_device(), so nothing else would stop a co-tenant landing on the pair
# mid-measurement, and a 1x2 mesh open brings up both chips whether or not it computes on both.
from tt_bio.device_lease import CardSetLease  # noqa: E402

_lease = CardSetLease()
_lease.acquire()
print(f"[ok]   lease: holding cards {_lease.cards}", flush=True)
result["lease_cards"] = _lease.cards

import ttnn  # noqa: E402


def _cluster_eth():
    """What tt-metal itself says about the ethernet topology of the visible chips."""
    import ttnn._ttnn as _t

    out = {}
    for modname in ("cluster", "device"):
        mod = getattr(_t, modname, None)
        if mod is None:
            continue
        out[modname] = sorted(
            n for n in dir(mod) if "eth" in n.lower() or "cluster" in n.lower()
        )
    return out


stage("ttnn_import", lambda: ttnn.get_num_devices())
stage("cluster_eth_api", _cluster_eth)

FABRIC = os.environ.get("LINK_PROBE_FABRIC", "FABRIC_1D")
if FABRIC != "none":
    stage("set_fabric_config", lambda: str(ttnn.set_fabric_config(getattr(ttnn.FabricConfig, FABRIC))))

mesh = stage(
    "open_mesh_1x2",
    lambda: ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=32768),
)
if mesh is None:
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    sys.exit(3)

stage("mesh_shape", lambda: str(mesh.shape))
stage("mesh_num_devices", lambda: mesh.get_num_devices())
stage("mesh_device_ids", lambda: [int(d) for d in mesh.get_device_ids()])
stage("visualize", lambda: str(ttnn.visualize_mesh_device(mesh)))


def _bandwidth():
    """All-gather a 67.1 MB pair-track tensor across the pair and time the steady state.

    Each chip starts with half the i-axis ([1,256,512,128]); the all-gather leaves both chips
    holding the full [1,512,512,128]. That is exactly the primitive an i-axis trunk shard needs
    per axis-flip, so its measured cost is the number the falsifier is priced against.
    """
    import torch

    full = torch.randn(1, 512, 512, 128, dtype=torch.bfloat16)
    bytes_full = full.numel() * 2
    tt = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, 1),
    )
    # warm up: first call compiles the CCL program and brings the routers to steady state
    g = ttnn.all_gather(tt, dim=1)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(g)

    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        g = ttnn.all_gather(tt, dim=1)
        ttnn.synchronize_device(mesh)
        samples.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    samples.sort()
    med = samples[len(samples) // 2]
    # Each chip RECEIVES the half it did not own: bytes_full/2 crosses the link per direction.
    moved = bytes_full / 2
    return {
        "tensor_bytes": bytes_full,
        "bytes_across_link_per_direction": moved,
        "reps": REPS,
        "median_s": med,
        "min_s": samples[0],
        "max_s": samples[-1],
        "gbps_effective": moved / med / 1e9,
        "gbps_full_tensor_equiv": bytes_full / med / 1e9,
        "all_samples_s": samples,
    }


def _latency():
    """Small-tensor all-gather: the fixed cost an axis-flip pays before any bytes move."""
    import torch

    small = torch.randn(1, 64, 32, 32, dtype=torch.bfloat16)
    tt = ttnn.from_torch(
        small,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh, 1),
    )
    g = ttnn.all_gather(tt, dim=1)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(g)
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        g = ttnn.all_gather(tt, dim=1)
        ttnn.synchronize_device(mesh)
        samples.append(time.perf_counter() - t0)
        ttnn.deallocate(g)
    samples.sort()
    return {"reps": REPS, "median_s": samples[len(samples) // 2], "min_s": samples[0]}


stage("allgather_bandwidth_67MB", _bandwidth)
stage("allgather_latency_small", _latency)
stage("close", lambda: ttnn.close_mesh_device(mesh))

json.dump(result, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
