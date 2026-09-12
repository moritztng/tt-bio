"""b2z2-l1-sharded-residency: what does a tile cost as a function of where it comes from?

Same op, same dtype, same tiles-per-core, same ACTIVE CORE COUNT, three placements of the operand:

  DRAM   input and output DRAM_INTERLEAVED   tiles arrive over the NOC from a DRAM channel
  L1I    input and output L1_INTERLEAVED     pages round-robin over every core's L1 bank, so
                                             ~(1 - 1/cores) of a core's reads are a neighbour's L1
  L1S    input and output height-sharded on  the tiles a core consumes are already in that core's
         exactly the active cores            own L1 and no NOC read happens at all

The active core count is pinned with `sub_core_grids`, which every ttnn unary op accepts, so the
same core set is used whether the operand lives in DRAM, in interleaved L1 or in a shard. Sweeping
it is the whole point: an aggregate-bandwidth limit grows with core count, a per-tile arrival cost
does not. At `--cores 1` this is literally the brief's "one kernel, one core, three sources".

`neg` is the probe op. It is the cheapest unary that still runs the full unpack/compute/pack chain,
and it is exactly invertible, so the same call doubles as a bit-exact parity check across arms.

Estimator discipline (b2z2 CONTEXT rule 3): one process, arms interleaved round-robin, median over
reps, and an A/A floor from two independently labelled copies of the DRAM arm. A linearity gate
runs first -- wall/op at `tiles_per_core` vs half that must be ~2.0, or the loop is host-bound and
no arm number means anything.
"""
import argparse, json, statistics, time

import torch
import ttnn

TILE = 32
BYTES_BF16 = 2


def core_range_set(gx, gy, ncores):
    """The first `ncores` cores in row-major order, as contiguous CoreRanges."""
    assert 1 <= ncores <= gx * gy, f"{ncores} cores requested of {gx * gy}"
    full_rows, rem = divmod(ncores, gx)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(set(ranges))


def memcfg(arm, crs, shard_h, shard_w):
    if arm == "DRAM":
        return ttnn.DRAM_MEMORY_CONFIG
    if arm == "L1I":
        return ttnn.L1_MEMORY_CONFIG
    if arm == "L1S":
        spec = ttnn.ShardSpec(crs, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    raise SystemExit("unknown arm " + arm)


OPS = {
    "neg": ttnn.neg,
    "abs": ttnn.abs,
    "relu": ttnn.relu,
    "exp": ttnn.exp,
}


def time_arm(dev, opfn, host, arm, crs, sh, sw, iters, want_out=False):
    """Enqueue `iters` back-to-back copies of the op; return (seconds per op, optional output)."""
    mc = memcfg(arm, crs, sh, sw)
    x = None
    outs = []
    keep = None
    try:
        x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=mc)
        y = opfn(x, memory_config=mc, sub_core_grids=crs)  # first call compiles the program
        ttnn.synchronize_device(dev)
        if want_out:
            keep = ttnn.to_torch(y)
        ttnn.deallocate(y)
        t0 = time.perf_counter()
        for _ in range(iters):
            outs.append(opfn(x, memory_config=mc, sub_core_grids=crs))
            if len(outs) > 2:
                ttnn.deallocate(outs.pop(0))
        ttnn.synchronize_device(dev)
        t1 = time.perf_counter()
        return (t1 - t0) / iters, keep
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
        opfn = OPS[args.op]
        tw = args.width_tiles
        assert args.tiles_per_core % tw == 0, "tiles-per-core must be a multiple of width-tiles"
        th = args.tiles_per_core // tw

        out = {
            "arch": str(dev.arch()), "grid": [gx, gy], "max_cores": gx * gy,
            "op": args.op, "dtype": "bfloat16", "tiles_per_core": args.tiles_per_core,
            "width_tiles": tw, "bytes_per_tile": TILE * TILE * BYTES_BF16,
            "iters": args.iters, "reps": args.reps, "by_cores": {},
        }

        def host_for(ncores, tpc):
            return torch.randn(1, 1, ncores * (tpc // tw) * TILE, tw * TILE)

        for ncores in [int(c) for c in args.cores.split(",")]:
            crs = core_range_set(gx, gy, ncores)
            sh, sw = th * TILE, tw * TILE
            entry = {"cores": ncores}

            # linearity gate: is the loop device-bound?
            g_lo, _ = time_arm(dev, opfn, host_for(ncores, args.tiles_per_core // 2),
                               "DRAM", crs, (th // 2) * TILE, sw, args.iters)
            g_hi, _ = time_arm(dev, opfn, host_for(ncores, args.tiles_per_core),
                               "DRAM", crs, sh, sw, args.iters)
            entry["linearity_ratio"] = g_hi / g_lo
            entry["linearity_us"] = [g_lo * 1e6, g_hi * 1e6]

            host = host_for(ncores, args.tiles_per_core)
            ref = -host.to(torch.bfloat16).to(torch.float32) if args.op == "neg" else None
            labels = ["DRAM", "L1I", "L1S", "DRAM_AA"]
            samples = {k: [] for k in labels}
            parity = {}
            for rep in range(args.reps):
                for lab in labels:
                    arm = "DRAM" if lab == "DRAM_AA" else lab
                    try:
                        s, y = time_arm(dev, opfn, host, arm, crs, sh, sw, args.iters,
                                        want_out=(rep == 0 and ref is not None))
                    except Exception as e:  # noqa: BLE001
                        samples[lab].append(None)
                        print(f"[{ncores}c rep{rep}] {lab}: FAILED {type(e).__name__}: "
                              f"{str(e)[:160]}", flush=True)
                        continue
                    samples[lab].append(s)
                    if y is not None:
                        parity[lab] = float((y.to(torch.float32) - ref).abs().max())
                    print(f"[{ncores}c rep{rep}] {lab:8s} {s*1e6:9.2f} us/op  "
                          f"{s/args.tiles_per_core*1e9:8.2f} ns/tile/core", flush=True)

            for lab in labels:
                vals = [v for v in samples[lab] if v is not None]
                if not vals:
                    entry[lab] = None
                    continue
                med = statistics.median(vals)
                entry[lab] = {
                    "us_per_op": med * 1e6,
                    "ns_per_tile_per_core": med / args.tiles_per_core * 1e9,
                    "n": len(vals),
                    "spread_pct": (max(vals) - min(vals)) / med * 100.0,
                }
            entry["parity_max_abs"] = parity
            d, l1s, l1i, aa = entry["DRAM"], entry["L1S"], entry["L1I"], entry["DRAM_AA"]
            if d and aa:
                entry["aa_floor"] = aa["us_per_op"] / d["us_per_op"]
            if d and l1s:
                entry["ratio_L1S_over_DRAM"] = l1s["us_per_op"] / d["us_per_op"]
            if l1s and l1i:
                entry["ratio_L1I_over_L1S"] = l1i["us_per_op"] / l1s["us_per_op"]
            # aggregate DRAM bytes/s on the DRAM arm: read once, write once
            if d:
                total_bytes = 2 * ncores * args.tiles_per_core * TILE * TILE * BYTES_BF16
                entry["dram_arm_GBps"] = total_bytes / (d["us_per_op"] * 1e-6) / 1e9
            out["by_cores"][str(ncores)] = entry
            print(f"  -> {ncores} cores: L1S/DRAM {entry.get('ratio_L1S_over_DRAM')}, "
                  f"A/A {entry.get('aa_floor')}, DRAM arm {entry.get('dram_arm_GBps')} GB/s",
                  flush=True)

        print(json.dumps(out, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--op", default="neg", choices=sorted(OPS))
    p.add_argument("--cores", default="1,2,4,8,16,32,64")
    p.add_argument("--tiles-per-core", type=int, default=128)
    p.add_argument("--width-tiles", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default="")
    run(p.parse_args())
