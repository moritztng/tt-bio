#!/usr/bin/env python3
"""The block A/B with NO verb timer installed.

`census.py` reads the same `reblock_permute_back` call at 0.171 ms on the `off` arm and 0.625 ms
on the `on` arm, while `micro.py` measures that kernel at 0.239 ms alone and 0.171 ms interleaved
with the other one on the same card in the same process. The lever does not touch that call, so
the per-op backward column is contaminated and the arm has to be priced without the instrument
that contaminates it.

So: `block_step`'s own synced forward and backward walls, arms alternated at the rep boundary
(`off, on, on, off` over a pair of reps, which cancels a linear drift over the sitting), AICLK
sampled DURING each leg, loadavg1 beside every point. Paired by rep index, because the two arms
of a rep see the same host.
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

OUT = ROOT / 'perf' / 'bcx_p10_trimove' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--pad', type=int, default=288)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--stack', default='evo')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', default='block_ab_n288.json')
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    lv, dev, ref = S.open_all(args)
    lv.mask = True
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    from tt_bio import reblock_permute as R

    for _ in range(3):
        S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)

    pts = []
    for rep in range(args.reps):
        # off, on | on, off | off, on | ... : a linear drift cancels over each pair of reps.
        order = ('off', 'on') if rep % 2 == 0 else ('on', 'off')
        for arm in order:
            R.set_taped_channel_move(arm == 'on')
            R.STATS[:] = [0, 0]
            R.STATS_BACK[:] = [0, 0]
            r, _ = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
            p = {'rep': rep, 'arm': arm, 'fwd': round(r['fwd'], 4), 'bwd': round(r['bwd'], 4),
                 'block': round(2 * r['fwd'] + r['bwd'], 4),
                 'aiclk': clock.window(r['spans']), 'load': round(os.getloadavg()[0], 2),
                 'reblock_fwd': R.STATS[0], 'reblock_back': R.STATS_BACK[0]}
            pts.append(p)
            print(json.dumps(p), flush=True)
    clock.stop()

    def col(arm, f):
        return [p[f] for p in pts if p['arm'] == arm]

    summary = {}
    for f in ('fwd', 'bwd', 'block'):
        x, y = col('off', f), col('on', f)
        ratios = [a / b for a, b in zip(x, y)]
        summary[f] = {
            'off_median': round(statistics.median(x), 4),
            'on_median': round(statistics.median(y), 4),
            'median_ratio': round(statistics.median(x) / statistics.median(y), 4),
            'paired_median_ratio': round(statistics.median(ratios), 4),
            'paired_wins_on': sum(1 for a, b in zip(x, y) if b < a),
            'n_pairs': len(ratios),
            'off_range': [round(min(x), 4), round(max(x), 4)],
            'on_range': [round(min(y), 4), round(max(y), 4)],
        }
        print(f, json.dumps(summary[f]), flush=True)
    clks = [p['aiclk'].get('median') for p in pts if p['aiclk'].get('n')]
    mins = [p['aiclk'].get('min') for p in pts if p['aiclk'].get('n')]
    print(f'AICLK median {statistics.median(clks)}  min {min(mins)}  '
          f'loadavg1 {min(p["load"] for p in pts)}-{max(p["load"] for p in pts)}', flush=True)
    reach = {a: {'fwd': sorted({p['reblock_fwd'] for p in pts if p['arm'] == a}),
                 'back': sorted({p['reblock_back'] for p in pts if p['arm'] == a})}
             for a in ('off', 'on')}
    print('reach', json.dumps(reach), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(
        {'stamp': S.stamp(args, clock), 'n': args.n, 'k': args.k, 'reps': args.reps,
         'points': pts, 'summary': summary, 'reach': reach,
         'aiclk_median': statistics.median(clks), 'aiclk_min': min(mins)},
        indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
