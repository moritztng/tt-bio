#!/usr/bin/env python3
"""Is `reblock_permute_back` really 3.66x slower when the other kernel runs beside it?

The A/B census read the SAME kernel at the SAME shape and the same call count at 373.0 GB/s on
the `off` arm's backward and 102.0 GB/s on the `on` arm's. The lever does not touch that call,
so either the census is mis-attributing or the two kernels interfere. This asks the kernels
directly, warm, synced, on one card, with the two orders interleaved in one process.
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
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_stack'))
from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_p10_devmap import devmap as D      # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_trimove' / 'out'
ROOF = 442.3e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--c', type=int, default=128)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--out', default='micro_n288.json')
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    from tt_bio import reblock_permute as R
    # qb2 is 2x p300c, so `ttnn.open_device` under a single-card TT_VISIBLE_DEVICES hard-fatals
    # on the CUSTOM cluster type. Open it the way every other instrument in this campaign does.
    _lv, _dev, _ref = S.open_all(args)
    dev = _dev.device
    clock = S.Clock()
    N, C = args.n, args.c
    mc = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t: ttnn.from_torch(                                       # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    chan = up(torch.randn(1, N, N, C))        # [1,N,N,C] -> permute(0,3,1,2)
    back = up(torch.randn(1, C, N, N))        # [1,C,N,N] -> permute(0,2,3,1)
    by_f = 1 * N * N * C * 2 * 2              # read + write
    by_b = by_f

    def timed(fn, iters):
        for _ in range(4):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        xs = []
        for _ in range(iters):
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            xs.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        return xs

    F = lambda: R.reblock_permute(chan, mc)                               # noqa: E731
    B = lambda: R.reblock_permute_back(back, mc)                          # noqa: E731
    PF = lambda: ttnn.permute(chan, (0, 3, 1, 2), memory_config=mc)       # noqa: E731
    PB = lambda: ttnn.permute(back, (0, 2, 3, 1), memory_config=mc)       # noqa: E731

    def TT():
        o = ttnn.transpose(back, 1, 2, memory_config=mc)
        r = ttnn.transpose(o, 2, 3, memory_config=mc)
        ttnn.deallocate(o)
        return r

    res = {}
    plan = [('back_alone', B, by_b), ('fwd_alone', F, by_f),
            ('back_alone_r2', B, by_b), ('fwd_alone_r2', F, by_f),
            ('stock_permute_fwd', PF, by_f), ('stock_permute_back', PB, by_b),
            ('two_transposes_back', TT, by_b * 2)]
    for name, fn, by in plan:
        xs = sorted(timed(fn, args.iters))
        med = statistics.median(xs)
        res[name] = {'median_ms': round(med * 1e3, 4), 'min_ms': round(xs[0] * 1e3, 4),
                     'gbs': round(by / med / 1e9, 1),
                     'pct_roof': round(by / med / ROOF * 100, 1),
                     'aiclk': clock.window([(time.time() - 60, time.time())]),
                     'load': round(os.getloadavg()[0], 2)}
        print(name, json.dumps(res[name]), flush=True)

    # Interleaved: the backward's own order on the `on` arm is back, back, fwd, back, back.
    def inter():
        outs = [B(), B(), F(), B(), B()]
        for o in outs[:-1]:
            ttnn.deallocate(o)
        return outs[-1]
    xs = sorted(timed(inter, args.iters))
    med = statistics.median(xs)
    res['interleaved_4back_1fwd'] = {
        'median_ms': round(med * 1e3, 4), 'per_back_ms': round((med - res['fwd_alone']
                                                                ['median_ms'] / 1e3) / 4 * 1e3, 4),
        'load': round(os.getloadavg()[0], 2)}
    print('interleaved', json.dumps(res['interleaved_4back_1fwd']), flush=True)
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps({'n': N, 'c': C, 'iters': args.iters, **res},
                                           indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
