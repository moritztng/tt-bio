"""Separate the READ source from the WRITE destination, and ask whether arrival hides behind math.

`arrival_rate.py` moves the operand and the result together, so its L1S/DRAM ratio mixes two
things: where the tile comes from, and where it goes. Wave 1 already banked part of the write side
(`b2z-op-knob-sweep`, matmul `outbuf=L1`, 1.124x-2.159x, bit-exact), so a claim about *arrival*
has to be the read side on its own.

Part 1, the 2x2. The interleaved unary kernel takes its input memory config and its output memory
config independently, so all four combinations of {DRAM, L1_INTERLEAVED} run the SAME kernel with
only the buffer homes changed. Differencing the rows gives the read-side cost and the write-side
cost separately.

Part 2, overlap. An eltwise op unpacks each tile once and packs it once, so it has the least math
per tile of anything on the part and the least opportunity to hide an arrival behind compute.
Repeating the 2x2 for ops of rising per-tile cost says whether the read-side gap is a true serial
addition or something the pipeline absorbs when there is work to absorb it behind.

Same estimator discipline as `arrival_rate.py`: one process, arms interleaved, median over reps,
A/A floor from two independently labelled copies of the DRAM->DRAM arm.
"""
import argparse, json, statistics, time

import torch
import ttnn

TILE = 32

HOMES = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}
OPS = {"neg": ttnn.neg, "abs": ttnn.abs, "relu": ttnn.relu, "exp": ttnn.exp,
       "sigmoid": ttnn.sigmoid, "erf": ttnn.erf}


def core_range_set(gx, gy, ncores):
    full_rows, rem = divmod(ncores, gx)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(set(ranges))


def time_pair(dev, opfn, host, src, dst, crs, iters):
    x = None
    outs = []
    try:
        x = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=HOMES[src])
        y = opfn(x, memory_config=HOMES[dst], sub_core_grids=crs)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(y)
        t0 = time.perf_counter()
        for _ in range(iters):
            outs.append(opfn(x, memory_config=HOMES[dst], sub_core_grids=crs))
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
        tw = args.width_tiles
        tpc = args.tiles_per_core
        out = {"arch": str(dev.arch()), "grid": [gx, gy], "tiles_per_core": tpc,
               "width_tiles": tw, "iters": args.iters, "reps": args.reps, "cells": {}}

        for ncores in [int(c) for c in args.cores.split(",")]:
            crs = core_range_set(gx, gy, ncores)
            host = torch.randn(1, 1, ncores * (tpc // tw) * TILE, tw * TILE)
            for opname in args.ops.split(","):
                opfn = OPS[opname]
                labels = [("DRAM", "DRAM"), ("DRAM", "L1"), ("L1", "DRAM"), ("L1", "L1"),
                          ("DRAM", "DRAM")]  # last is the A/A copy
                keys = ["D>D", "D>L", "L>D", "L>L", "AA"]
                samples = {k: [] for k in keys}
                for _ in range(args.reps):
                    for key, (src, dst) in zip(keys, labels):
                        try:
                            samples[key].append(time_pair(dev, opfn, host, src, dst, crs, args.iters))
                        except Exception as e:  # noqa: BLE001
                            print(f"[{ncores}c {opname}] {key}: FAILED {type(e).__name__}: "
                                  f"{str(e)[:140]}", flush=True)
                cell = {}
                for k in keys:
                    if samples[k]:
                        med = statistics.median(samples[k])
                        cell[k] = {"ns_per_tile_per_core": med / tpc * 1e9,
                                   "us_per_op": med * 1e6,
                                   "spread_pct": (max(samples[k]) - min(samples[k])) / med * 100}
                # A cell whose program failed to build is reported as a hole, not as a crash:
                # one unavailable combination must not cost the whole sweep its other rows.
                def n(k):
                    return cell[k]["ns_per_tile_per_core"] if k in cell else float("nan")
                if "AA" in cell and "D>D" in cell:
                    cell["aa_floor"] = cell["AA"]["us_per_op"] / cell["D>D"]["us_per_op"]
                # read side: hold the destination fixed, move the source
                cell["read_cost_ns_at_dram_dst"] = n("D>D") - n("L>D")
                cell["read_cost_ns_at_l1_dst"] = n("D>L") - n("L>L")
                # write side: hold the source fixed, move the destination
                cell["write_cost_ns_at_dram_src"] = n("D>D") - n("D>L")
                cell["write_cost_ns_at_l1_src"] = n("L>D") - n("L>L")
                cell["missing"] = [k for k in keys if k not in cell]
                out["cells"][f"{ncores}c/{opname}"] = cell
                print(f"[{ncores}c {opname}] D>D {n('D>D'):7.1f}  D>L {n('D>L'):7.1f}  "
                      f"L>D {n('L>D'):7.1f}  L>L {n('L>L'):7.1f}  | read "
                      f"{cell['read_cost_ns_at_dram_dst']:6.1f}/{cell['read_cost_ns_at_l1_dst']:6.1f}"
                      f"  write {cell['write_cost_ns_at_dram_src']:6.1f}/"
                      f"{cell['write_cost_ns_at_l1_src']:6.1f}  A/A "
                      f"{cell.get('aa_floor', float('nan')):.4f}", flush=True)
        print(json.dumps(out, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ops", default="neg,relu,exp,erf")
    p.add_argument("--cores", default="1,8,32")
    p.add_argument("--tiles-per-core", type=int, default=128)
    p.add_argument("--width-tiles", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default="")
    run(p.parse_args())
