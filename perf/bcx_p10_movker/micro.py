#!/usr/bin/env python3
"""The pair transpose priced leg by leg, and the ONE-PASS floor a kernel could reach.

Leg 1 says `z -> z^T` on `[288,288,128]` bf16 costs three stock dispatches and moves the tensor
three times. Before building a kernel this asks the only question that decides whether the build
is worth it: **what does ONE pass over this tensor cost on this card?**

`ttnn.clone` DRAM -> DRAM at the same shape and dtype is that floor. It reads every tile once and
writes every tile once with no gather at all, which is the kernel's best case and nothing the
kernel can beat. If the three-call round trip is not ~3x that number, the accounting in leg 1 is
wrong; if a kernel cannot get near it, the byte cut does not become a second cut.

Also prices the tiled permute the RM route exists to avoid, so the alternative is on the record.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROOF = 442.3e9
OUT = ROOT / 'perf' / 'bcx_p10_movker' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--c', type=int, default=128)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--warm', type=int, default=5)
    ap.add_argument('--out', default='micro.json')
    args = ap.parse_args()

    import ttnn
    from perf.bcx_stack import stack as S
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    clock = S.Clock()
    N, C = args.n, args.c
    mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(torch.randn(N, N, C).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    nbytes = N * N * C * 2

    # `stack.Clock` samples on a Python daemon thread, and a tight ttnn micro-benchmark loop
    # starves it: a 60-iteration run of this file collected ONE sample. The block and round
    # censuses are fine because their windows are seconds long with plenty of GIL yields; this
    # one is not. So the clock is read from the MAIN thread, once per timed iteration, right
    # after the synchronize that ends it -- in the window by construction.
    clk_path = clock.path

    def aiclk_now():
        try:
            return int(open(clk_path).read().split()[0])
        except Exception:                                                   # noqa: BLE001
            return 0

    def timed(fn, passes):
        """Median wall of `fn`, synced, with the traffic it implies. `passes` counts how many
        times the op reads-and-writes the tensor, so GB/s is comparable across legs."""
        t0s = time.time()
        for _ in range(args.warm):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        t, clks = [], []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            t.append(time.perf_counter() - t0)
            clks.append(aiclk_now())
            ttnn.deallocate(o)
        ms = statistics.median(t) * 1e3
        clks.sort()
        traffic = 2 * nbytes * passes
        return {'ms': round(ms, 4), 'MB': round(traffic / 1e6, 1),
                'GB_s': round(traffic / (ms / 1e3) / 1e9, 1),
                'pct_roof': round(traffic / (ms / 1e3) / ROOF * 100, 1),
                'aiclk_med': clks[len(clks) // 2], 'aiclk_min': clks[0], 'aiclk_n': len(clks)}

    def round_trip():
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        p = ttnn.permute(rm, (1, 0, 2), memory_config=mc)
        ttnn.deallocate(rm)
        o = ttnn.to_layout(p, ttnn.TILE_LAYOUT, memory_config=mc)
        ttnn.deallocate(p)
        return o

    rows = {}
    t_start = time.time()
    # The floor: one pass, no gather. Nothing a kernel does can beat this.
    rows['clone (ONE PASS, the kernel floor)'] = timed(lambda: ttnn.clone(x), 1)
    rows['to_layout TILE->ROW_MAJOR'] = timed(
        lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc), 1)
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
    rows['permute (1,0,2) row-major'] = timed(
        lambda: ttnn.permute(rm, (1, 0, 2), memory_config=mc), 1)
    rows['to_layout ROW_MAJOR->TILE'] = timed(
        lambda: ttnn.to_layout(rm, ttnn.TILE_LAYOUT, memory_config=mc), 1)
    ttnn.deallocate(rm)
    rows['THE SHIPPED ROUND TRIP (3 calls)'] = timed(round_trip, 3)
    # The route `_PT_ROW_MAJOR` exists to avoid, on the record so nobody re-measures it.
    try:
        rows['permute (1,0,2) TILED (the route the RM one replaced)'] = timed(
            lambda: ttnn.permute(x, (1, 0, 2), memory_config=mc), 1)
    except Exception as e:                                                  # noqa: BLE001
        rows['permute (1,0,2) TILED'] = {'error': str(e)[:120]}

    clock.stop()
    aiclk = {'per_leg': {k: (v.get('aiclk_med'), v.get('aiclk_min'), v.get('aiclk_n'))
                         for k, v in rows.items() if 'aiclk_med' in v}}
    w = max(len(k) for k in rows)
    print(f'\n[{N},{N},{C}] bf16 DRAM, tensor {nbytes/1e6:.1f} MB, '
          f'{args.iters} warm synced iterations, median')
    print(f'loadavg1 {os.getloadavg()[0]:.2f}; AICLK sampled DURING every iteration, '
          f'from the main thread, per leg below')
    print(f"{'leg':<{w}}{'ms':>9}{'MB moved':>10}{'GB/s':>9}{'% roof':>9}"
          f"{'AICLK med/min/n':>18}")
    for k, v in rows.items():
        if 'error' in v:
            print(f'{k:<{w}}  {v["error"]}')
        else:
            print(f'{k:<{w}}{v["ms"]:>9.4f}{v["MB"]:>10.1f}{v["GB_s"]:>9.1f}'
                  f'{v["pct_roof"]:>9.1f}'
                  f'{str(v["aiclk_med"]) + "/" + str(v["aiclk_min"]) + "/" + str(v["aiclk_n"]):>18}')
    rt = rows['THE SHIPPED ROUND TRIP (3 calls)']['ms']
    fl = rows['clone (ONE PASS, the kernel floor)']['ms']
    print(f'\nround trip / one-pass floor = {rt/fl:.2f}x  '
          f'(the byte model says 3.0x; a kernel at the floor saves {rt-fl:.4f} ms a call)')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(
        {'n': N, 'c': C, 'bytes': nbytes, 'iters': args.iters, 'aiclk': aiclk,
         'load': os.getloadavg(), 'rows': rows}, indent=1, default=str))
    print('wrote ' + str(OUT / args.out))


if __name__ == '__main__':
    main()
