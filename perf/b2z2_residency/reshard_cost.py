"""What does it cost to put a tensor into a shard, and how many ops must the shard then feed?

The residency measurement says a core reading its own shard saves ~110 ns/tile on Blackhole and
~190 ns/tile on Wormhole against the same tile read out of interleaved L1, and that the saving
survives compute. That is the credit side. This is the debit side: `ttnn.to_memory_config` into
a shard is itself a full pass over every tile, so a layout that needs one between every pair of
ops pays back everything it saved. The break-even is

    ops a shard must feed = reshard cost per tile / saving per op per tile

measured, not assumed. Three conversions are timed, all at fixed tiles-per-core on a fixed core
set, so they are directly comparable with `arrival_rate.py`'s arms:

    DRAM  -> shard     the cold case, a tensor that lives in DRAM being made resident
    L1    -> shard     the warm case, an interleaved L1 tensor being made resident
    shard -> L1        the exit, what it costs to hand a shard back to an interleaved consumer

Same estimator discipline: one process, conversions interleaved, median over reps, A/A floor.
"""
import argparse, json, statistics, time

import torch
import ttnn

TILE = 32


def core_range_set(gx, gy, ncores):
    full_rows, rem = divmod(ncores, gx)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(set(ranges))


def sharded(crs, sh, sw):
    spec = ttnn.ShardSpec(crs, [sh, sw], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def time_convert(dev, host, src_cfg, dst_cfg, iters):
    x = None
    outs = []
    try:
        x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=src_cfg)
        y = ttnn.to_memory_config(x, dst_cfg)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(y)
        t0 = time.perf_counter()
        for _ in range(iters):
            outs.append(ttnn.to_memory_config(x, dst_cfg))
            if len(outs) > 2:
                ttnn.deallocate(outs.pop(0))
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / iters
    finally:
        for o in outs:
            try:
                ttnn.deallocate(o)
            except Exception:  # noqa: BLE001
                pass
        if x is not None:
            try:
                ttnn.deallocate(x)
            except Exception:  # noqa: BLE001
                pass


def run(args):
    dev = ttnn.open_device(device_id=0)
    try:
        g = dev.compute_with_storage_grid_size()
        gx, gy = g.x, g.y
        tpc, tw = args.tiles_per_core, args.width_tiles
        th = tpc // tw
        out = {"arch": str(dev.arch()), "grid": [gx, gy], "tiles_per_core": tpc,
               "iters": args.iters, "reps": args.reps, "by_cores": {}}
        for ncores in [int(c) for c in args.cores.split(",")]:
            crs = core_range_set(gx, gy, ncores)
            sh, sw = th * TILE, tw * TILE
            S = sharded(crs, sh, sw)
            host = torch.randn(1, 1, ncores * th * TILE, tw * TILE)
            jobs = {
                "DRAM->shard": (ttnn.DRAM_MEMORY_CONFIG, S),
                "L1->shard": (ttnn.L1_MEMORY_CONFIG, S),
                "shard->L1": (S, ttnn.L1_MEMORY_CONFIG),
                "shard->DRAM": (S, ttnn.DRAM_MEMORY_CONFIG),
                "L1->L1_AA": (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
            }
            samples = {k: [] for k in jobs}
            for _ in range(args.reps):
                for k, (a, b) in jobs.items():
                    try:
                        samples[k].append(time_convert(dev, host, a, b, args.iters))
                    except Exception as e:  # noqa: BLE001
                        print(f"[{ncores}c] {k}: FAILED {type(e).__name__}: {str(e)[:140]}",
                              flush=True)
            entry = {}
            for k, v in samples.items():
                if v:
                    med = statistics.median(v)
                    entry[k] = {"ns_per_tile_per_core": med / tpc * 1e9,
                                "us_per_op": med * 1e6,
                                "spread_pct": (max(v) - min(v)) / med * 100}
            out["by_cores"][str(ncores)] = entry
            line = "  ".join(f"{k} {entry[k]['ns_per_tile_per_core']:7.1f}"
                             for k in jobs if k in entry)
            print(f"[{ncores}c] {line}", flush=True)
        print(json.dumps(out, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cores", default="1,8,32")
    p.add_argument("--tiles-per-core", type=int, default=128)
    p.add_argument("--width-tiles", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default="")
    run(p.parse_args())
