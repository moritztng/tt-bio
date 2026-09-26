#!/usr/bin/env python3
"""Where the shipped ROW_MAJOR pair transpose wins and where it loses, by shape.

`micro.py` measures `[288,288,128]` bf16 and reads the shipped three-call round trip at 1.20x
SLOWER than the single tiled `ttnn.permute` it exists to replace. That is not a contradiction of
`233137232` ("the pair transpose is 1.614 s/fold faster through ROW_MAJOR"): that commit measured
the FOLD's shapes, this campaign runs a training round at n=288. **The route has no shape gate**
(`_pair_transpose_impl` takes it for every 3-D bf16 DRAM TILE tensor), so if the crossover sits
inside the range tt-bio runs, the gate is missing rather than the route being wrong.

Three legs per shape, all DRAM -> DRAM, AICLK read from the MAIN thread once per iteration
(`stack.Clock`'s daemon thread is starved by a tight ttnn loop -- it collected 1 sample in 60):

    clone         one pass, no gather. The floor any kernel is measured against.
    round trip    to_layout(ROW_MAJOR) + permute(1,0,2) + to_layout(TILE), what ships.
    tiled         ttnn.permute(x, (1,0,2)) straight, what the route replaced.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument('--shapes', default='128x128,256x128,288x128,288x256,384x128,512x128')
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--warm', type=int, default=4)
    ap.add_argument('--out', default='ladder.json')
    args = ap.parse_args()

    import ttnn
    from perf.bcx_stack import stack as S
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    clk_path = S.Clock().path
    mc = ttnn.DRAM_MEMORY_CONFIG

    def aiclk_now():
        try:
            return int(open(clk_path).read().split()[0])
        except Exception:                                                   # noqa: BLE001
            return 0

    def timed(fn, nbytes, passes):
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
        return {'ms': round(ms, 4), 'pct_roof': round(traffic / (ms / 1e3) / ROOF * 100, 1),
                'aiclk_med': clks[len(clks) // 2], 'aiclk_min': clks[0]}

    rows = {}
    for spec in args.shapes.split(','):
        N, C = (int(v) for v in spec.split('x'))
        nbytes = N * N * C * 2
        x = ttnn.from_torch(torch.randn(N, N, C).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=mc)

        def round_trip(_x=x):
            rm = ttnn.to_layout(_x, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            p = ttnn.permute(rm, (1, 0, 2), memory_config=mc)
            ttnn.deallocate(rm)
            o = ttnn.to_layout(p, ttnn.TILE_LAYOUT, memory_config=mc)
            ttnn.deallocate(p)
            return o

        r = {'MB': round(nbytes / 1e6, 1)}
        try:
            r['clone'] = timed(lambda _x=x: ttnn.clone(_x), nbytes, 1)
            r['round_trip'] = timed(round_trip, nbytes, 3)
            r['tiled'] = timed(lambda _x=x: ttnn.permute(_x, (1, 0, 2), memory_config=mc),
                               nbytes, 1)
            # Both routes compute the same function; check it rather than assume it.
            a = ttnn.to_torch(round_trip())
            b = ttnn.to_torch(ttnn.permute(x, (1, 0, 2), memory_config=mc))
            r['equal'] = bool(torch.equal(torch.Tensor(a), torch.Tensor(b)))
        except Exception as e:                                              # noqa: BLE001
            r['error'] = str(e)[:140]
        ttnn.deallocate(x)
        rows[spec] = r
        print(json.dumps({spec: r}), flush=True)

    print(f"\n{'shape':<12}{'MB':>7}{'clone ms':>10}{'round trip':>12}{'tiled':>10}"
          f"{'rt/tiled':>10}{'rt/clone':>10}{'equal':>7}{'AICLK':>8}")
    for spec, r in rows.items():
        if 'error' in r:
            print(f'{spec:<12}  {r["error"]}')
            continue
        rt, tl, cl = r['round_trip']['ms'], r['tiled']['ms'], r['clone']['ms']
        print(f'{spec:<12}{r["MB"]:>7.1f}{cl:>10.4f}{rt:>12.4f}{tl:>10.4f}'
              f'{rt/tl:>9.2f}x{rt/cl:>9.2f}x{str(r["equal"]):>7}'
              f'{r["round_trip"]["aiclk_med"]:>8}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps({'iters': args.iters, 'rows': rows}, indent=1))
    print('wrote ' + str(OUT / args.out))


if __name__ == '__main__':
    main()
