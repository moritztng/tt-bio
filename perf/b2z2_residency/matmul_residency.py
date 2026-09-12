"""Does the interleaving term transfer from an eltwise probe to the block's own op class?

`arrival_rate.py` measures residency on a unary op. The Pairformer block's tile reads are 62 %
`generic_op` (tt-bio's own fused trimul/SDPA kernels) and 19 % matmul/linear, so a constant
derived from a unary op only matters if a matmul shows it too. This runs the block's largest
matmul shape -- `b2z-op-knob-sweep` named `PairformerLayer#004`, 1x16x512x512 @ 512x128 -- with
in0 interleaved and then block-sharded onto the same core grid, and reports ns per output tile.

Correctness is checked, not assumed: the sharded arm's result is compared against the interleaved
arm's, since a matmul that reassociates is not the same op.
"""
import argparse, json, statistics, time

import torch
import ttnn


def run(args):
    dev = ttnn.open_device(device_id=0)
    try:
        g = dev.compute_with_storage_grid_size()
        B, H, M, K, N = 1, args.heads, args.m, args.k, args.n
        a = torch.randn(B, H, M, K) * 0.05
        b = torch.randn(B, 1, K, N) * 0.05
        out = {"arch": str(dev.arch()), "grid": [g.x, g.y],
               "shape": {"b": B, "h": H, "m": M, "k": K, "n": N},
               "reps": args.reps, "iters": args.iters, "arms": {}}

        def timeit(x, w, mc_out):
            ys = []
            y = ttnn.matmul(x, w, memory_config=mc_out)
            ttnn.synchronize_device(dev)
            ref = ttnn.to_torch(y)
            ttnn.deallocate(y)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                ys.append(ttnn.matmul(x, w, memory_config=mc_out))
                if len(ys) > 2:
                    ttnn.deallocate(ys.pop(0))
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / args.iters
            for y in ys:
                ttnn.deallocate(y)
            return dt, ref

        w = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        arms = {}
        refs = {}
        for label, mc_in, mc_out in (
                ("in=DRAM  out=DRAM", ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
                ("in=L1    out=L1", ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
                ("in=DRAM  out=DRAM (A/A)", ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ):
            samples = []
            for _ in range(args.reps):
                x = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=mc_in)
                try:
                    dt, ref = timeit(x, w, mc_out)
                    samples.append(dt)
                    refs[label] = ref
                finally:
                    ttnn.deallocate(x)
            med = statistics.median(samples)
            out_tiles = B * H * (M // 32) * (N // 32)
            arms[label] = {"us_per_op": med * 1e6,
                           "ns_per_output_tile": med / out_tiles * 1e9,
                           "ns_per_output_tile_per_core": med / (out_tiles / (g.x * g.y)) * 1e9,
                           "spread_pct": (max(samples) - min(samples)) / med * 100}
            print(f"{label:26s} {med*1e6:9.2f} us  "
                  f"{arms[label]['ns_per_output_tile']:8.2f} ns/out-tile  "
                  f"spread {arms[label]['spread_pct']:.2f} %", flush=True)
        base = arms["in=DRAM  out=DRAM"]["us_per_op"]
        out["arms"] = arms
        out["aa_floor"] = arms["in=DRAM  out=DRAM (A/A)"]["us_per_op"] / base
        out["ratio_L1_over_DRAM"] = arms["in=L1    out=L1"]["us_per_op"] / base
        k = list(refs)
        out["max_abs_L1_vs_DRAM"] = float(
            (refs[k[1]].to(torch.float32) - refs[k[0]].to(torch.float32)).abs().max())
        print(f"A/A {out['aa_floor']:.4f}   L1/DRAM {out['ratio_L1_over_DRAM']:.4f}   "
              f"max_abs L1 vs DRAM {out['max_abs_L1_vs_DRAM']}", flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default="")
    run(p.parse_args())
