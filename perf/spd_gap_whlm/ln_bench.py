"""The protein LMs' LayerNorm at the WH page shape: interleaved (shipped) against the op's sharded programs.

The 6B census on Wormhole (spd-gap-whlm cen4) spends 42.8 ms of a 298 ms page batch in 321 layer norms over [544, 2560],
0.133 ms a call, ~42 GB/s. The interleaved program splits work by tile rows, and 544 rows are 17 of them, so most of the
grid should sit idle. This times the shipped call against block- and width-sharded programs over many more cores, both
alone (input already sharded) and as the round trip a caller would pay (DRAM -> shard -> layer_norm -> DRAM). Per arm:
device ms (median of --repeat, --inner calls per synchronize), cores, max abs error against float64.

    python3 perf/spd_gap_whlm/ln_bench.py --model esmc-6b --out ln.jsonl
"""
import argparse
import json
import time

import torch
import ttnn

# (d_model, rows of the page batch)
MODELS = {"esmc-300m": (960, 7 * 544), "esmc-600m": (1152, 7 * 544), "esmc-6b": (2560, 544), "saprot-650m": (1280, 7 * 544)}


def _divisors(n, top):
    return [d for d in range(1, min(n, top) + 1) if n % d == 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-6b", choices=sorted(MODELS))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--min-cores", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    d, m = MODELS[args.model]
    mt, wt = m // 32, d // 32
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, m, d, generator=gen).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16)
    xf = x.double()
    ref = (xf - xf.mean(-1, keepdim=True)) / torch.sqrt(xf.var(-1, unbiased=False, keepdim=True) + 1e-5) * w.double()
    xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # as Module.torch_to_tt
    out = open(args.out, "w")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()

    def timed(fn):
        got = fn()
        err = float((ttnn.to_torch(got).double() - ref).abs().max())
        ttnn.deallocate(got)
        ts = []
        for _ in range(args.repeat):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.inner):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / args.inner * 1e3)
        return round(sorted(ts)[len(ts) // 2], 4), err

    base = {"model": args.model, "shape": [m, d]}
    ms, err = timed(lambda: ttnn.layer_norm(xd, weight=wd, epsilon=1e-5, compute_kernel_config=ck))
    emit({**base, "arm": "dram", "cores": mt, "ms": ms, "max_err": err})

    # (layout, cores across x, cores across y, block_h tiles, block_w tiles)
    plans = [("block", gx, gy, mt // gy, wt // gx) for gy in _divisors(mt, g.y) for gx in _divisors(wt, g.x)]
    plans += [("width", gx, n // gx, mt, wt // n) for n in _divisors(wt, g.x * g.y)
              for gx in [min(g.x, n)] if n % gx == 0 and n // gx <= g.y]
    for kind, gx, gy, bh, bw in plans:
        if gx * gy < args.min_cores:
            continue
        rec = {**base, "arm": f"{kind}{gx}x{gy}", "cores": gx * gy, "block": [bh, bw]}
        try:
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
            layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED if kind == "block" else ttnn.TensorMemoryLayout.WIDTH_SHARDED
            mc = ttnn.MemoryConfig(layout, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(grid, [bh * 32, bw * 32], ttnn.ShardOrientation.ROW_MAJOR))
            sub = max(s for s in (4, 2, 1) if bw % s == 0)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                                             subblock_w=sub, block_h=bh, block_w=bw, inplace=False)
            xs = ttnn.to_memory_config(xd, mc)

            def ln(t):
                return ttnn.layer_norm(t, weight=wd, epsilon=1e-5, compute_kernel_config=ck, program_config=pc,
                                       memory_config=mc)

            rec["ms"], rec["max_err"] = timed(lambda: ln(xs))

            def trip():
                s = ttnn.to_memory_config(xd, mc)
                o = ln(s)
                ttnn.deallocate(s)
                r = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(o)
                return r

            rec["trip_ms"], _ = timed(trip)
            ttnn.deallocate(xs)
        except Exception as e:  # a plan the op refuses is a result, not a crash
            rec["error"] = str(e).splitlines()[0][:200]
        emit(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
