#!/usr/bin/env python3
"""The diffusion step's LayerNorm runs on 16 of 110 cores. Is a width split worth anything?

`LayerNormDefaultProgramConfig` parallelises over tile ROWS only, so a (1,1,512,768) normalise
gets 16 tile rows and 16 cores whatever the grid. The alternative is a block-sharded layout, which
splits the width too and reduces across cores. That trade has to pay for its own two layout
conversions, so all three legs are timed: the shipped default, the sharded kernel alone, and the
sharded kernel plus the conversions a caller would actually pay.

Run under `python -m tracy -r`; the device kernel duration is the number, the wall is host-bound
at this op size.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
SHAPE = (1, 1, 512, 768)
REPS = 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    device = ttnn.open_device(device_id=0)
    res = {}
    try:
        burn0 = ttnn.from_torch(torch.randn(1, 256, 1024, 1024) * 0.1, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        burn1 = ttnn.from_torch(torch.randn(1, 1, 1024, 1024) * 0.1, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

        def boost(n=3):
            for _ in range(n):
                ttnn.deallocate(ttnn.matmul(burn0, burn1, memory_config=DRAM,
                                            compute_kernel_config=HIFI4))
            ttnn.synchronize_device(device)

        xt = torch.randn(*SHAPE)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=DRAM)
        w = ttnn.from_torch(torch.ones(1, 1, 32, SHAPE[3]), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

        Mt, Nt = SHAPE[2] // 32, SHAPE[3] // 32          # 16 x 24 tiles
        legs = {}

        def default():
            return ttnn.layer_norm(x, weight=w, epsilon=1e-5, memory_config=DRAM,
                                   compute_kernel_config=HIFI4)
        legs["default"] = (default, None)

        for rows, cols in ((8, 8), (8, 6), (4, 11), (8, 11), (16, 6)):
            if Mt % rows or rows > 10 and rows != 16:
                continue
            bh, bw = Mt // rows, -(-Nt // cols)
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                     ttnn.CoreCoord(cols - 1, rows - 1))})
            shard = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [SHAPE[2] // rows, -(-SHAPE[3] // cols)],
                               ttnn.ShardOrientation.ROW_MAJOR))
            sw = max(d for d in range(1, bw + 1) if bw % d == 0 and d <= 4)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cols, rows), subblock_w=sw,
                block_h=bh, block_w=bw, inplace=False)

            def sharded(shard=shard, pc=pc, cols=cols, rows=rows):
                xs = ttnn.to_memory_config(x, shard)
                o = ttnn.layer_norm(xs, weight=w, epsilon=1e-5, memory_config=shard,
                                    program_config=pc, compute_kernel_config=HIFI4)
                oi = ttnn.to_memory_config(o, DRAM)
                ttnn.deallocate(xs)
                ttnn.deallocate(o)
                return oi
            legs[f"sharded-{rows}x{cols}"] = (sharded, rows * cols)

        for name, (fn, cores) in legs.items():
            try:
                for _ in range(3):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(device)
            except Exception as e:                                # noqa: BLE001
                res[name] = {"refused": str(e).replace("\n", " ")[:180]}
                print(f"  {name:<18} REFUSED {str(e).splitlines()[0][:100]}")
                continue
            boost()
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(device)
            us = (time.perf_counter() - t0) * 1e6 / REPS
            res[name] = {"wall_us": us, "cores_asked": cores}
            print(f"  {name:<18} {us:8.2f} us/call wall, cores asked {cores}")

        # parity of the sharded path against the default, on the same input
        ref = ttnn.to_torch(default()).float()
        for name, (fn, _) in legs.items():
            if name == "default":
                continue
            if "refused" in res.get(name, {}):
                continue
            got = ttnn.to_torch(fn()).float()
            res[name]["max_abs_vs_default"] = float((got - ref).abs().max())
            res[name]["bit_exact_vs_default"] = bool(torch.equal(got, ref))
            print(f"  {name:<18} max|delta| vs default "
                  f"{res[name]['max_abs_vs_default']:.3e}  bit-exact="
                  f"{res[name]['bit_exact_vs_default']}")
    finally:
        ttnn.close_device(device)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
