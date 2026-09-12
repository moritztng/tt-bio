"""Measure Wormhole chip-to-chip collective cost at the shapes a row-sharded Boltz-2 pair
tensor implies, so the break-even shard count can be computed before any model code exists.

Usage:  link_bench.py --devices 2 [--out results.json]

For each payload we report both:
  * latency  -- one collective, device synchronised on each side (what a serial trunk pays)
  * burst    -- N collectives back to back, one sync at the end (the pipelined floor)
Bandwidth is quoted as bytes ARRIVING at one device / time, which is the number that has to be
compared against the compute the shard buys.
"""
import argparse, json, statistics, sys, time

import torch
import ttnn

# z at 512 aa is (N, N, c) = (512, 512, 128) bf16 = 67.108864 MB.
PAIR_C = 128


def payloads(n_dev):
    """(label, global_shape) with the gather dim divisible by n_dev and by 32 (tile)."""
    out = []
    for n in (64, 128, 256, 512, 1024):
        if n % n_dev or (n // n_dev) % 32:
            continue
        out.append((f"z_{n}aa", (1, n, n, PAIR_C)))
    # a pure size sweep on a fixed row width, to separate fixed cost from bandwidth
    for rows in (32, 64, 128, 256, 512):
        r = rows * n_dev
        out.append((f"sweep_{r}x512x128", (1, r, 512, PAIR_C)))
    return out


def bytes_of(shape, dtype_bytes=2):
    n = 1
    for s in shape:
        n *= s
    return n * dtype_bytes


def run_one(mesh, shape, n_dev, reps, burst):
    torch_t = torch.zeros(shape, dtype=torch.bfloat16)
    mapper = ttnn.shard_tensor_to_mesh_mapper(mesh, dim=1)
    t = ttnn.from_torch(torch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
    res = {}
    for opname in ("all_gather", "reduce_scatter"):
        try:
            if opname == "all_gather":
                call = lambda: ttnn.all_gather(t, dim=1, topology=ttnn.Topology.Linear)
            else:
                # reduce_scatter consumes the FULL tensor replicated; build it once
                call = None
            if call is None:
                full = ttnn.from_torch(torch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh))
                call = lambda: ttnn.reduce_scatter(full, dim=1, topology=ttnn.Topology.Linear)
            for _ in range(3):
                o = call(); ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)

            lat = []
            for _ in range(reps):
                t0 = time.perf_counter()
                o = call()
                ttnn.synchronize_device(mesh)
                lat.append(time.perf_counter() - t0)
                ttnn.deallocate(o)

            outs = []
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(burst):
                outs.append(call())
            ttnn.synchronize_device(mesh)
            bt = (time.perf_counter() - t0) / burst
            for o in outs:
                ttnn.deallocate(o)

            res[opname] = {"latency_s": statistics.median(lat),
                           "latency_min_s": min(lat),
                           "burst_s": bt}
        except Exception as e:                                    # noqa: BLE001
            res[opname] = {"error": f"{type(e).__name__}: {e}"[:300]}
    ttnn.deallocate(t)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, required=True)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--burst", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dispatch", default="worker", choices=("worker", "eth", "default"))
    ap.add_argument("--fabric", default="1d", choices=("1d", "2d", "off"))
    a = ap.parse_args()

    fab = {"1d": ttnn.FabricConfig.FABRIC_1D, "2d": ttnn.FabricConfig.FABRIC_2D,
           "off": ttnn.FabricConfig.DISABLED}[a.fabric]
    ttnn.set_fabric_config(fab)
    kw = {}
    if a.dispatch == "worker":
        kw["dispatch_core_config"] = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
    elif a.dispatch == "eth":
        kw["dispatch_core_config"] = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, a.devices), **kw)
    print(f"mesh opened: {mesh.shape}, {mesh.get_num_devices()} devices", flush=True)
    rows = []
    try:
        for label, shape in payloads(a.devices):
            gb = bytes_of(shape)
            r = run_one(mesh, shape, a.devices, a.reps, a.burst)
            row = {"label": label, "shape": list(shape), "n_dev": a.devices,
                   "global_bytes": gb, "results": r}
            rows.append(row)
            for op, v in r.items():
                if "error" in v:
                    print(f"{label:24s} {gb/1e6:8.2f} MB  {op:14s} ERR {v['error'][:120]}", flush=True)
                    continue
                # bytes arriving at one device
                arr = gb * (a.devices - 1) / a.devices if op == "all_gather" else gb * (a.devices - 1) / a.devices
                print(f"{label:24s} {gb/1e6:8.2f} MB  {op:14s} "
                      f"lat {v['latency_s']*1e3:8.3f} ms ({arr/v['latency_s']/1e9:6.2f} GB/s)  "
                      f"burst {v['burst_s']*1e3:8.3f} ms ({arr/v['burst_s']/1e9:6.2f} GB/s)", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
