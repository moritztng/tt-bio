#!/usr/bin/env python3
"""Routes for the pair tensor's dim0/dim1 transpose, at the sizes where RF3 pays for it.

`tri_att_end` is the only trunk component whose share grows with N (16.4 % of the trunk at
512 aa, 22.1 % at 768, 23.0 % at 1024), and the whole excess over `tri_att_start` is two
`_pair_transpose` calls on the full [S, S, c_z] pair tensor: 0.76 ms each into L1 at 512 aa,
4.42 ms each into DRAM at 768 and 7.84 ms at 1024. The DRAM route moves 68 GB/s against the
440 GB/s this card demonstrably has, so the question this answers is whether any composition of
stock ttnn ops gets closer to the roof.

Every arm is checked with torch.equal against ttnn.permute, so a faster arm is a faster arm and
not a different answer.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,768,1024")
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--widths", default="64,128,192,256,384")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T

    device = T.get_device()
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    grid = T.COMPUTE_GRID_MAIN
    l1_total = per_core * grid[0] * grid[1]
    print(f"compute grid {grid}, {per_core} B/core unreserved, L1 budget "
          f"{l1_total / 2**20:.2f} MB", flush=True)

    def timed(fn, reps):
        fn()                                       # warm: kernel compile, config pick
        ttnn.synchronize_device(device)
        ts = []
        for _ in range(reps):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(device)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        return statistics.median(ts) * 1e3

    report = []
    for S in [int(s) for s in args.sizes.split(",")]:
        xt = torch.randn(S, S, args.c, dtype=torch.float32).bfloat16()
        ref = xt.permute(1, 0, 2).contiguous()
        x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=device,
                            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mb = S * S * args.c * 2 / 2**20
        print(f"\n--- {S}x{S}x{args.c} bf16 = {mb:.1f} MB "
              f"({mb * 2**20 / l1_total * 100:.0f} % of the L1 budget)", flush=True)

        def arm_tiled():
            return ttnn.permute(x, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def arm_rm():
            rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            p = ttnn.permute(rm, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(rm)
            o = ttnn.to_layout(p, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(p)
            return o

        def arm_l1_whole():
            t = ttnn.permute(x, (1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
            o = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t)
            return o

        def make_chunk_col(W, dest_l1):
            def f():
                parts = []
                mc = ttnn.L1_MEMORY_CONFIG if dest_l1 else ttnn.DRAM_MEMORY_CONFIG
                for s in range(0, S, W):
                    e = min(s + W, S)
                    strip = x[:, s:e, :]
                    t = ttnn.permute(strip, (1, 0, 2), memory_config=mc)
                    ttnn.deallocate(strip)
                    if dest_l1:
                        d = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(t)
                        t = d
                    parts.append(t)
                o = ttnn.concat(parts, dim=0)
                for p in parts:
                    ttnn.deallocate(p)
                return o
            return f

        def make_chunk_rm(W):
            def f():
                parts = []
                for s in range(0, S, W):
                    e = min(s + W, S)
                    strip = x[:, s:e, :]
                    rm = ttnn.to_layout(strip, ttnn.ROW_MAJOR_LAYOUT,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(strip)
                    p = ttnn.permute(rm, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(rm)
                    t = ttnn.to_layout(p, ttnn.TILE_LAYOUT,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(p)
                    parts.append(t)
                o = ttnn.concat(parts, dim=0)
                for p in parts:
                    ttnn.deallocate(p)
                return o
            return f

        arms = [("tiled_dram", arm_tiled), ("rm_dram", arm_rm), ("l1_whole", arm_l1_whole)]
        for W in [int(w) for w in args.widths.split(",")]:
            if W >= S:
                continue
            arms.append((f"chunk_col_l1_{W}", make_chunk_col(W, True)))
            arms.append((f"chunk_col_dram_{W}", make_chunk_col(W, False)))
            arms.append((f"chunk_rm_{W}", make_chunk_rm(W)))

        base = None
        for name, fn in arms:
            try:
                o = fn()
                ok = torch.equal(ttnn.to_torch(o), ref)
                ttnn.deallocate(o)
                ms = timed(fn, args.reps)
            except Exception as exc:                                          # noqa: BLE001
                print(f"  {name:22s} FAILED {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:110]}", flush=True)
                report.append({"S": S, "c": args.c, "arm": name, "ms": None,
                               "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"})
                continue
            if base is None:
                base = ms
            gbs = 2 * mb / 2**10 / (ms / 1e3)
            print(f"  {name:22s} {ms:8.3f} ms  {gbs:6.1f} GB/s  "
                  f"{base / ms:5.3f}x vs tiled  bit-exact={ok}", flush=True)
            report.append({"S": S, "c": args.c, "arm": name, "ms": round(ms, 4),
                           "eff_gbs": round(gbs, 2), "bit_exact": ok})
        ttnn.deallocate(x)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
