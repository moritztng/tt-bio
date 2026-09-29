#!/usr/bin/env python3
"""What a `generic_op` dispatch costs with the cheap path armed, against the stock op beside it.

The claim of this row is ms per dispatch, so it is taken the way a ms-per-dispatch claim has to
be: both arms and the stock baselines in ONE process, on one card, alternated `off, on, on, off`
inside every round so a drift over the sitting cancels, each call clocked against a drained
queue. `decomp.py` takes the same numbers one arm per process and is the decomposition; this is
the A/B, and it is the one to quote. Two processes at loadavg 7.9 and 9.3 read `ttnn.add` at
0.0267 and 0.0260 ms, which is how much a cross-process comparison would have been worth.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_stack'))
from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_p10_devmap import devmap as D      # noqa: E402
from perf.bcx_p10_genq import bench as B         # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_genq' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--c', type=int, default=128)
    ap.add_argument('--reps', type=int, default=120)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--out', default='dispatch_ab.json')
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    from tt_bio import rne_add as RA
    from tt_bio import genq
    _lv, _dev, _ref = S.open_all(args)
    dev = _dev.device
    clock = S.Clock()
    N, C = args.n, args.c
    mc = ttnn.DRAM_MEMORY_CONFIG
    RA.set_enabled(True)
    up = lambda t: ttnn.from_torch(                                       # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    a, b = up(torch.randn(1, N, N, C)), up(torch.randn(1, N, N, C))
    dest = ttnn.allocate_tensor_on_device(a.shape, RA.OUT_DTYPE, ttnn.TILE_LAYOUT, dev, mc)

    # Build and JIT both arms before anything is timed, so neither pays the other's compile.
    entries = {}
    for arm in (False, True):
        genq.set_compact(arm)
        RA.rne_add(a, b, mc, out=dest)
        ttnn.synchronize_device(dev)
        entries[arm] = RA._prepare(a, b, dest, dev)['compact']
    assert entries == {False: False, True: True}, entries

    def arm_fn(compact):
        def run():
            genq.set_compact(compact)
            RA.rne_add(a, b, mc, out=dest)      # the caller owns `dest`
        return run

    spans = []
    series = {k: [] for k in ('off', 'on', 'stock_add', 'stock_permute')}
    chan = up(torch.randn(1, N, N, C))
    plan = (('off', arm_fn(False)), ('on', arm_fn(True)),
            ('stock_add', lambda: ttnn.add(a, b, memory_config=mc)),
            ('stock_permute', lambda: ttnn.permute(chan, (0, 3, 1, 2), memory_config=mc)),
            ('on', arm_fn(True)), ('off', arm_fn(False)))
    for r in range(args.rounds):
        for name, fn in plan:
            keep = name in ('off', 'on')
            series[name].append(B.timed(f'r{r} {name}', fn, dev, ttnn, args.reps, spans,
                                        keep=keep)['median_ms'])

    res = {k: {'medians': v, 'median_ms': round(statistics.median(v), 5),
               'min_ms': round(min(v), 5), 'max_ms': round(max(v), 5)}
           for k, v in series.items()}
    # Pair round against round: the two `on` windows of round r against the two `off` windows of
    # round r, so a load excursion that spans a round moves both arms.
    paired = [(off - on) for off, on in zip(series['off'], series['on'])]
    res['paired'] = {'delta_ms': [round(d, 5) for d in paired],
                     'median_delta_ms': round(statistics.median(paired), 5),
                     'on_wins': sum(1 for d in paired if d > 0), 'n': len(paired)}
    res['ratio_on_over_stock_add'] = round(res['on']['median_ms']
                                           / res['stock_add']['median_ms'], 3)
    res['ratio_off_over_stock_add'] = round(res['off']['median_ms']
                                            / res['stock_add']['median_ms'], 3)
    res['speedup_off_over_on'] = round(res['off']['median_ms'] / res['on']['median_ms'], 3)
    print(json.dumps({k: v for k, v in res.items() if k != 'medians'}, indent=1), flush=True)

    blob = {'stamp': S.stamp(args, clock), 'n': N, 'c': C, 'reps': args.reps,
            'rounds': args.rounds, 'aiclk': clock.window(spans),
            'loadavg1': round(os.getloadavg()[0], 2), 'results': res}
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
