"""b2z2-l1-sharded-residency, step 1: what does a tile cost as a function of where it comes from?

Same op, same dtype, same tile count, same core grid, three operand sources:

  DRAM   input and output in DRAM_INTERLEAVED            -- tiles arrive over the NOC from DRAM
  L1I    input and output in L1_INTERLEAVED              -- pages round-robin over every core's L1
                                                            bank, so ~(1 - 1/cores) of a core's
                                                            reads are a neighbour's L1
  L1S    input and output height-sharded on the same     -- the tiles a core consumes are already
         grid                                               in that core's own L1

Reported as ns per tile per core, which is the unit the campaign's 71.3 ns/tile is in.

Estimator discipline (CONTEXT rule 3): one process, arms interleaved round-robin, n>=5 reps each,
median of per-rep values, and an A/A floor from two independently labelled copies of the DRAM arm.
A linearity gate runs first: if wall/op does not scale with tiles-per-core the loop is host-bound
and no arm number means anything.
"""
import argparse, json, os, statistics, sys, time

import ttnn
import torch

TILE = 32
BYTES_BF16 = 2


def core_range_set(gx, gy):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def sharded_config(gx, gy, shard_h, shard_w):
    spec = ttnn.ShardSpec(core_range_set(gx, gy), [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


MEMCFG = {
    "DRAM": lambda gx, gy, sh, sw: ttnn.DRAM_MEMORY_CONFIG,
    "L1I": lambda gx, gy, sh, sw: ttnn.L1_MEMORY_CONFIG,
    "L1S": sharded_config,
}


def build_op(name):
    if name == "exp":
        return lambda x, mc: ttnn.exp(x, memory_config=mc)
    if name == "mul":
        return lambda x, mc: ttnn.multiply(x, 1.0000001, memory_config=mc)
    if name == "relu":
        return lambda x, mc: ttnn.relu(x, memory_config=mc)
    if name == "clone":
        return lambda x, mc: ttnn.clone(x, memory_config=mc)
    raise SystemExit("unknown op " + name)


def time_arm(dev, op, host, arm, gx, gy, sh, sw, iters):
    """Enqueue `iters` back-to-back copies of the op and return seconds per op."""
    mc = MEMCFG[arm](gx, gy, sh, sw)
    x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    # warm: first call compiles the program
    y = op(x, mc)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(y)
    t0 = time.perf_counter()
    outs = []
    for _ in range(iters):
        outs.append(op(x, mc))
        if len(outs) > 2:
            ttnn.deallocate(outs.pop(0))
    ttnn.synchronize_device(dev)
    t1 = time.perf_counter()
    for o in outs:
        ttnn.deallocate(o)
    ttnn.deallocate(x)
    return (t1 - t0) / iters


def run(args):
    dev = ttnn.open_device(device_id=0, l1_small_size=args.l1_small)
    try:
        g = dev.compute_with_storage_grid_size()
        gx, gy = g.x, g.y
        cores = gx * gy
        op = build_op(args.op)

        results = {
            "arch": str(dev.arch()), "grid": [gx, gy], "cores": cores,
            "op": args.op, "iters": args.iters, "reps": args.reps,
            "linearity": {}, "arms": {},
        }

        def make(tiles_per_core):
            tw = args.width_tiles
            assert tiles_per_core % tw == 0, "tiles_per_core must be a multiple of width_tiles"
            th = tiles_per_core // tw
            M = cores * th * TILE
            N = tw * TILE
            host = torch.randn(1, 1, M, N)
            return host, th * TILE, N

        # ---- linearity gate: device-bound or host-bound? ----
        for tpc in (args.tiles_per_core // 2, args.tiles_per_core):
            host, sh, sw = make(tpc)
            s = time_arm(dev, op, host, "DRAM", gx, gy, sh, sw, args.iters)
            results["linearity"][str(tpc)] = s
            del host
        lo = results["linearity"][str(args.tiles_per_core // 2)]
        hi = results["linearity"][str(args.tiles_per_core)]
        results["linearity_ratio"] = hi / lo
        print(f"[gate] tiles/core {args.tiles_per_core//2} -> {lo*1e6:.2f} us, "
              f"{args.tiles_per_core} -> {hi*1e6:.2f} us, ratio {hi/lo:.3f} (want ~2.0)", flush=True)

        # ---- the three arms, interleaved, plus an A/A control ----
        host, sh, sw = make(args.tiles_per_core)
        tpc = args.tiles_per_core
        labels = ["DRAM", "L1I", "L1S", "DRAM_AA"]
        samples = {k: [] for k in labels}
        for rep in range(args.reps):
            for lab in labels:
                arm = "DRAM" if lab == "DRAM_AA" else lab
                try:
                    s = time_arm(dev, op, host, arm, gx, gy, sh, sw, args.iters)
                except Exception as e:  # noqa: BLE001
                    samples[lab].append(None)
                    print(f"[rep {rep}] {lab}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                    continue
                samples[lab].append(s)
                print(f"[rep {rep}] {lab}: {s*1e6:8.2f} us/op  {s/tpc*1e9:7.2f} ns/tile/core", flush=True)

        for lab in labels:
            vals = [v for v in samples[lab] if v is not None]
            if not vals:
                results["arms"][lab] = None
                continue
            med = statistics.median(vals)
            results["arms"][lab] = {
                "us_per_op": med * 1e6,
                "ns_per_tile_per_core": med / tpc * 1e9,
                "n": len(vals),
                "spread_pct": (max(vals) - min(vals)) / med * 100.0,
                "samples_us": [v * 1e6 for v in vals],
            }
        results["tiles_per_core"] = tpc
        results["bytes_per_tile"] = TILE * TILE * BYTES_BF16
        print(json.dumps(results, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="exp")
    p.add_argument("--tiles-per-core", type=int, default=192)
    p.add_argument("--width-tiles", type=int, default=16)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--l1-small", type=int, default=0)
    p.add_argument("--out", default="")
    run(p.parse_args())
