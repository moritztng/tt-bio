"""SaProt block LayerNorm at the page shape: interleaved (DRAM, L1) against block-sharded, with the reshard legs priced.

Each arm is timed on device (inner loop, synchronize) and scored against float64 LayerNorm of the same bf16 input.
One JSON line per arm: ms per call, rel_l2, and for the sharded arms the cost of getting in and out of the shard.
"""
import argparse
import json
import time

import torch
import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=7 * 544)
    ap.add_argument("--d", type=int, nargs="+", default=[480, 1280, 2560])
    ap.add_argument("--inner", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    gx, gy = dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=False)
    f = open(a.out, "a")

    def timed(fn):
        o = fn(); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter()
            for _ in range(a.inner):
                ttnn.deallocate(o); o = fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t) / a.inner * 1e3)
        return o, min(ts)

    for d in a.d:
        torch.manual_seed(d)
        x = (torch.randn(a.rows, d) * 2 + 0.3).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(d)).to(torch.bfloat16)
        b = (0.1 * torch.randn(d)).to(torch.bfloat16)
        ref = torch.nn.functional.layer_norm(x.double(), (d,), w.double(), b.double(), 1e-5)
        wt = ttnn.from_torch(w.reshape(1, d), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        bt = ttnn.from_torch(b.reshape(1, d), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wr = ttnn.from_torch(w.reshape(d // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        br = ttnn.from_torch(b.reshape(d // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        def emit(arm, o, ms, **meta):
            got = ttnn.to_torch(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG) if o.is_sharded() else o)
            got = got.double().reshape(ref.shape)
            rel = float((got - ref).norm() / ref.norm())
            row = dict(d=d, rows=a.rows, arm=arm, ms=round(ms, 4), rel_l2=rel, gbs=round(2 * a.rows * d * 2 / ms / 1e6, 1),
                       **meta)
            print(json.dumps(row), flush=True); f.write(json.dumps(row) + "\n"); f.flush()

        for mem, name in ((ttnn.DRAM_MEMORY_CONFIG, "dram"), (ttnn.L1_MEMORY_CONFIG, "l1")):
            xi = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mem)
            try:
                o, ms = timed(lambda: ttnn.layer_norm(xi, weight=wt, bias=bt, epsilon=1e-5, compute_kernel_config=ck,
                                                      memory_config=mem))
                emit(name, o, ms); ttnn.deallocate(o)
            except RuntimeError as e:
                print(json.dumps(dict(d=d, arm=name, error=str(e)[:200])), flush=True)
            ttnn.deallocate(xi)

        mt, kt = a.rows // 32, d // 32
        grids = [(r, c) for r in range(1, gy + 1) for c in range(1, gx + 1) if mt % r == 0 and kt % c == 0 and r * c >= 16]
        grids = sorted(grids, key=lambda rc: -rc[0] * rc[1])[:4]
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for r, c in grids:
            smc = ttnn.create_sharded_memory_config(shape=(a.rows, d), core_grid=ttnn.CoreGrid(y=r, x=c),
                                                    strategy=ttnn.ShardStrategy.BLOCK,
                                                    orientation=ttnn.ShardOrientation.ROW_MAJOR)
            bw = kt // c
            sub = max(s for s in range(1, 5) if bw % s == 0)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(c, r), subblock_w=sub,
                                                             block_h=mt // r, block_w=bw, inplace=False)
            try:
                xs, ms_in = timed(lambda: ttnn.to_memory_config(xd, smc))
                for gname, g, be in (("tile", wt, bt), ("rm", wr, br)):
                    try:
                        o, ms = timed(lambda: ttnn.layer_norm(xs, weight=g, bias=be, epsilon=1e-5, compute_kernel_config=ck,
                                                              program_config=pc, memory_config=smc))
                    except RuntimeError as e:
                        print(json.dumps(dict(d=d, arm=f"shard{r}x{c}-{gname}", error=str(e)[:200])), flush=True)
                        continue
                    o2, ms_out = timed(lambda: ttnn.to_memory_config(o, ttnn.L1_MEMORY_CONFIG))
                    ttnn.deallocate(o2)
                    emit(f"shard{r}x{c}", o, ms, gamma=gname, in_ms=round(ms_in, 4), out_l1_ms=round(ms_out, 4))
                    ttnn.deallocate(o)
                    break
                ttnn.deallocate(xs)
            except RuntimeError as e:
                print(json.dumps(dict(d=d, arm=f"shard{r}x{c}", error=str(e)[:200])), flush=True)
        ttnn.deallocate(xd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
