#!/usr/bin/env python3
"""W7: does a memory config carried across a whole op chain beat round-tripping DRAM?

The Pairformer block's `transition_z` is the cleanest place in the block to ask that.
It is 12.6% of the block, it is a pure row-local chain (layer_norm -> fc1(silu) ->
fc2 -> multiply -> fc3), and it is the one phase whose entire working set can be
row-blocked to any size we like. Everything here changes only WHERE the carried
tensors live and how wide the row block is; the arithmetic and its order are fixed,
so the placement arms must be bit-exact against the production module.

Arms (one per process -- a variant that overflows L1 leaves the allocator fragmented
and poisons every later arm in the same process, which is how the first W7 A/B lost
four of five legs):

  module      production `Transition.__call__`, h_chunk 32                REFERENCE
  module_h16  same, h_chunk 16
  module_h64  same, h_chunk 64
  dram        explicit chain, every carried tensor DRAM-interleaved
  l1          explicit chain, every carried tensor L1-interleaved  (= production placement)
  shard       explicit chain, carried tensors L1 HEIGHT-SHARDED across the chain
  mm          `l1` with ttnn.linear(core_grid=...) swapped for minimal_matmul

  roof        this card's compute roof + the shape-matched achieved rate (no chain)

    ~/tt-bio/env/bin/python3 perf/sram_strategy/chain_ab.py --arm l1 --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def sharded_cfg(m, k, grid):
    """Height-shard [m, k] row-major over `grid` cores. m/tiles must divide the core count."""
    return ttnn.create_sharded_memory_config(
        shape=(m, k), core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def build_chain(tz, ckc, mode, grid):
    """One row-block through the swiglu, with every carried tensor placed by `mode`."""
    W = tz.fc1_weight.shape[-2]      # c_z
    F = tz.fc1_weight.shape[-1]      # hidden

    def run(chunk):
        h, w = chunk.shape[1], chunk.shape[2]
        m = h * w
        if mode == "shardm":
            # layer_norm stays interleaved (ttnn 0.68 refuses a height-sharded layer_norm
            # input outright), then the whole matmul segment is carried height-sharded.
            flat = ttnn.reshape(chunk, (1, 1, m, W))
            x_norm = ttnn.layer_norm(flat, weight=tz.norm_weight, bias=tz.norm_bias,
                                     epsilon=1e-5, compute_kernel_config=ckc, memory_config=L1)
            xs = ttnn.to_memory_config(x_norm, sharded_cfg(m, W, grid))
            ttnn.deallocate(x_norm)
            cmid = sharded_cfg(m, F, grid)
            # a sharded matmul refuses a fused activation from the linear() arg, so silu is
            # a separate sharded unary here. Same arithmetic, one extra full-tensor pass.
            x_1 = ttnn.linear(xs, tz.fc1_weight, compute_kernel_config=ckc, memory_config=cmid)
            x_1 = ttnn.silu(x_1, memory_config=cmid)
            x_2 = ttnn.linear(xs, tz.fc2_weight, compute_kernel_config=ckc, memory_config=cmid)
            ttnn.deallocate(xs)
            x = ttnn.multiply_(x_1, x_2)
            ttnn.deallocate(x_2)
            out = ttnn.linear(x, tz.fc3_weight, compute_kernel_config=ckc,
                              memory_config=sharded_cfg(m, W, grid))
            ttnn.deallocate(x)
            out = ttnn.to_memory_config(out, DRAM)
            return ttnn.reshape(out, (1, h, w, W))
        if mode in ("shard", "shardb", "shardw"):
            strat = {"shard": ttnn.ShardStrategy.HEIGHT, "shardb": ttnn.ShardStrategy.BLOCK,
                     "shardw": ttnn.ShardStrategy.WIDTH}[mode]

            def cfg(mm, kk):
                return ttnn.create_sharded_memory_config(
                    shape=(mm, kk), core_grid=grid, strategy=strat,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR)
            flat = ttnn.reshape(chunk, (1, 1, m, W))
            cin = ttnn.to_memory_config(flat, cfg(m, W))
            cmid = cfg(m, F)
            cout = cfg(m, W)
        else:
            cin = chunk
            cmid = cout = (DRAM if mode == "dram" else L1)
        x_norm = ttnn.layer_norm(cin, weight=tz.norm_weight, bias=tz.norm_bias,
                                 epsilon=1e-5, compute_kernel_config=ckc,
                                 memory_config=(cin.memory_config()
                                                if mode in ("shard", "shardb", "shardw")
                                                else (DRAM if mode == "dram" else L1)))
        if mode == "mm":
            x_1 = ttnn.experimental.minimal_matmul(x_norm, tz.fc1_weight,
                                                   compute_kernel_config=ckc, memory_config=L1)
            x_1 = ttnn.silu(x_1, memory_config=L1)
            x_2 = ttnn.experimental.minimal_matmul(x_norm, tz.fc2_weight,
                                                   compute_kernel_config=ckc, memory_config=L1)
        else:
            if mode.startswith("shard"):
                x_1 = ttnn.linear(x_norm, tz.fc1_weight, compute_kernel_config=ckc,
                                  memory_config=cmid)
                x_1 = ttnn.silu(x_1, memory_config=cmid)
            else:
                x_1 = ttnn.linear(x_norm, tz.fc1_weight, activation="silu",
                                  compute_kernel_config=ckc, memory_config=cmid,
                                  core_grid=T.CORE_GRID_MAIN)
            x_2 = ttnn.linear(x_norm, tz.fc2_weight, compute_kernel_config=ckc,
                              memory_config=cmid,
                              core_grid=(None if mode.startswith("shard") else T.CORE_GRID_MAIN))
        ttnn.deallocate(x_norm)
        x = ttnn.multiply_(x_1, x_2)
        ttnn.deallocate(x_2)
        if mode == "mm":
            out = ttnn.experimental.minimal_matmul(x, tz.fc3_weight,
                                                   compute_kernel_config=ckc, memory_config=DRAM)
        else:
            out = ttnn.linear(x, tz.fc3_weight, compute_kernel_config=ckc, memory_config=cout,
                              core_grid=(None if mode.startswith("shard") else T.CORE_GRID_MAIN))
        ttnn.deallocate(x)
        if mode.startswith("shard"):
            out = ttnn.to_memory_config(out, DRAM)
            out = ttnn.reshape(out, (1, h, w, W))
            ttnn.deallocate(cin)
        return out

    return run


def chain_call(tz, ckc, mode, grid, h_chunk):
    run = build_chain(tz, ckc, mode, grid)

    def call(z):
        H = z.shape[1]
        parts = []
        for s in range(0, H, h_chunk):
            c = z[:, s:min(s + h_chunk, H)]
            parts.append(run(c))
            ttnn.deallocate(c)
        out = ttnn.concat(parts, dim=1, memory_config=DRAM)
        for p in parts:
            ttnn.deallocate(p)
        return out

    return call


def roof(dev, ckc, iters=20, warm=5):
    """Dense bf16 peak on THIS card, and the rate the chain's own matmul shape gets."""
    out = {}

    def timed(fn):
        for _ in range(warm):
            fn()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / iters

    for n in (4096,):
        a = ttnn.ones((1, 1, n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                      memory_config=DRAM)
        b = ttnn.ones((1, 1, n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                      memory_config=DRAM)
        t = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=DRAM))
        out[f"dense_{n}_TFLOPs"] = 2 * n ** 3 / t / 1e12
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    # the shapes transition_z actually runs, h_chunk=32, W=320, c_z=256, F=1024
    for (m, k, n, tag) in [(10240, 256, 1024, "fc1"), (10240, 1024, 256, "fc3")]:
        a = ttnn.ones((1, 1, m, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                      memory_config=L1)
        b = ttnn.ones((1, 1, k, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                      memory_config=DRAM)
        t = timed(lambda: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=L1,
                                      core_grid=T.CORE_GRID_MAIN))
        out[f"{tag}_linear_TFLOPs"] = 2 * m * k * n / t / 1e12
        t = timed(lambda: ttnn.experimental.minimal_matmul(a, b, compute_kernel_config=ckc,
                                                           memory_config=L1))
        out[f"{tag}_minmm_TFLOPs"] = 2 * m * k * n / t / 1e12
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--grid", default="10x8", help="core grid for the sharded arm")
    ap.add_argument("--dump", default=None, help="save the output tensor for a bit-exact check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    res = {"arm": args.arm, "n": args.n, "grid_main": list(T.COMPUTE_GRID_MAIN)}

    if args.arm == "roof":
        res.update(roof(dev, ckc))
        print(json.dumps(res, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return

    layer, c_z = build_layer(ckc)
    tz = layer.transition_z
    N = args.n
    gx, gy = (int(v) for v in args.grid.split("x"))
    grid = ttnn.CoreGrid(x=gx, y=gy)
    res["c_z"] = c_z

    if args.arm.startswith("module"):
        h = {"module": 32, "module_h16": 16, "module_h64": 64}[args.arm]
        T.TRANSITION_H_CHUNK_SIZE_BIG = h
        call = lambda z: layer.transition_z(z)  # noqa: E731
        res["h_chunk"] = h
    else:
        mode, _, hs = args.arm.partition("_h")
        h = int(hs) if hs else 32
        call = chain_call(tz, ckc, mode, grid, h)
        res["h_chunk"] = h
        res["mode"] = mode

    torch.manual_seed(0)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    try:
        for _ in range(args.warm):
            o = call(z)
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(args.iters):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = call(z)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            if len(ts) < args.iters:
                ttnn.deallocate(o)
        res["median_ms"] = sorted(ts)[len(ts) // 2]
        res["min_ms"] = min(ts)
        res["ms"] = [round(t, 3) for t in ts]
        if args.dump:
            torch.save(ttnn.to_torch(o), args.dump)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {' '.join(str(e).split())[:400]}"

    print(json.dumps(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
