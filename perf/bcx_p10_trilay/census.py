#!/usr/bin/env python3
"""bcx-p10-trilay: the triangle multiplication's verbs at the token axis the fold runs.

`bcx-p10-trimul`'s census answered the same question at n=275, which `perf/bcx_stack` used to
build and BindCraft 2 never runs. This one runs it at the bucketed 288 and alternates the chunk
width the taped loop takes, so the per-verb table and the A/B come out of one process on one
card. Same subtraction devmap and the trimul census use,

    device_i = synced_i - free_i - lambda * calls_i

with the sync floor measured in this process. The timer is trimul's, imported rather than
copied; the only thing added here is the arm loop and a reach counter on the chunk width, which
lives in the harness so production code carries no census state.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_stack'))
from perf.bcx_stack import stack as S                  # noqa: E402
from perf.bcx_p10_devmap import devmap as D            # noqa: E402
from perf.bcx_p10_trimul.census import VerbTimer       # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_trilay' / 'out'


def count_chunk_widths(T, widths):
    """Wrap `_trimul_chunk_size` so a reading that claims the arm fired has to show the width."""
    real = T._trimul_chunk_size

    def w(seq_len, hidden, batch=1):
        c = real(seq_len, hidden, batch)
        widths["%dx%d->c%d" % (int(seq_len), int(hidden), int(c))] += 1
        return c
    T._trimul_chunk_size = w
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--pad', type=int, default=288)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--stack', default='evo')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--arms', default='', help='alternate the taped chunk width in one process')
    ap.add_argument('--out', default='census_n288.json')
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    lv, dev, ref = S.open_all(args)
    lv.mask = True
    D.install_labels()
    timer = VerbTimer(ttnn, dev.device).install()
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)

    from tt_bio import tenstorrent as T
    widths = collections.Counter()
    count_chunk_widths(T, widths)

    for _ in range(2):
        S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
    floor = D.sync_floor(ttnn, dev.device)
    print(json.dumps({'sync_floor_median_s': floor['median']}), flush=True)

    arms = args.arms.split(',') if args.arms else ['-']
    acc = {(a, m): {'wall': collections.Counter(), 'calls': collections.Counter(),
                    'read': collections.Counter(), 'written': collections.Counter()}
           for a in arms for m in ('sync', 'free')}
    reach = {a: collections.Counter() for a in arms}
    aiclks, loads = [], []
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            if args.arms:
                T.set_trimul_taped_full_chunk(arm == 'on')
            for mode in (['free', 'sync'] if rep % 2 == 0 else ['sync', 'free']):
                timer.sync = (mode == 'sync')
                widths.clear()
                timer.on = True
                r, _ = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
                timer.on = False
                snap = timer.take()
                reach[arm].update(widths)
                aiclks.append(clock.window(r['spans']))
                loads.append(round(os.getloadavg()[0], 2))
                for f in ('wall', 'calls', 'read', 'written'):
                    acc[(arm, mode)][f].update(snap[f])
                print(json.dumps({'rep': rep, 'arm': arm, 'mode': mode,
                                  'fwd': round(r['fwd'], 3), 'bwd': round(r['bwd'], 3),
                                  'aiclk': aiclks[-1], 'load': loads[-1],
                                  'widths': dict(widths)}), flush=True)
    clock.stop()
    timer.uninstall()
    blob = {'stamp': S.stamp(args, clock), 'n': args.n, 'pad': args.pad, 'stack': args.stack,
            'k': args.k, 'reps': args.reps, 'sync_floor_s': floor, 'aiclk': aiclks,
            'load': loads, 'pair_masks': False, 'arms': arms,
            'reach': {a: dict(c) for a, c in reach.items()},
            'by_arm': {a: {m: {f: dict(acc[(a, m)][f]) for f in acc[(a, m)]}
                           for m in ('sync', 'free')} for a in arms}}
    if not args.arms:
        blob['sync'] = blob['by_arm']['-']['sync']
        blob['free'] = blob['by_arm']['-']['free']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
