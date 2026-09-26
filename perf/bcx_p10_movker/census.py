#!/usr/bin/env python3
"""bcx-p10-movker leg 1: the round's four movement verbs, by CALL SITE, with the GEOMETRY named.

`bcx-p10-calls` ranked movement by op name: `permute` 0.638 s, `slice` 0.366, `transpose` 0.311,
`to_layout` 0.185, all at 13-33 % of a DRAM roof an eltwise op on the same card reaches 76-85 %
of. `bcx-p10-trimove` censused `permute` and `transpose` inside the triangle multiplication and
named their consumers. **`slice` (99 % triangle attention) and `to_layout` (81 %) have never been
censused, and this extends trimove's instrument to them unchanged in its arithmetic.**

What it adds is the one column that decides whether a kernel can help: the MOVE GEOMETRY.

    contig-run   the output is a contiguous run of the input. Only leading axes are cut, so the
                 copy is one burst per run and stock ttnn should already be near the roof.
    strided      a cut or a reorder on a non-leading axis. The walk visits the input in a stride
                 that is not the page order, which is the shape of the 13-33 % readings.
    relayout     TILE <-> ROW_MAJOR. Inherently a gather: one 32x32 tile is 32 discrete row
                 segments, so it is strided even when neither side is permuted.

A contiguous copy at 47.7 % of roof is a different bug from a strided one at 13 %, and the brief
asks which is which before a kernel is designed. That is this file's output.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_stack'))
from perf.bcx_stack import stack as S                          # noqa: E402
from perf.bcx_p10_devmap import devmap as D                    # noqa: E402
from perf.bcx_p10_trimul.census import VerbTimer, _sig         # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_movker' / 'out'

#: Every family, not trimove's two: `slice` and `to_layout` live in triangle attention and the
#: outer-product mean, which trimove's filter excluded by construction.
MOVES = ('permute', 'transpose', 'slice', 'to_layout', 'chunk', 'concat', 'clone',
         'to_memory_config', 'reallocate', 'typecast', 'generic_op', 'reshape')


def _geom(path, args, kwargs, ins, outs, ttnn):
    """The move's geometry: what a kernel would have to walk. See the module docstring."""
    try:
        if path == 'to_layout':
            src = str(ins[0].layout).rsplit('.', 1)[-1] if ins else '?'
            dst = str(outs[0].layout).rsplit('.', 1)[-1] if outs else '?'
            return f'relayout[{src}->{dst}]' if src != dst else 'relayout[noop]'
        if path == 'slice':
            shape = [int(d) for d in ins[0].shape]
            starts = [int(v) for v in (kwargs.get('slice_start') or args[1])]
            ends = [int(v) for v in (kwargs.get('slice_end') or args[2])]
            cut = [i for i in range(len(shape))
                   if starts[i] != 0 or ends[i] != shape[i]]
            if not cut:
                return 'contig-run[whole]'
            # Only a prefix of the axes is cut -> the output is one contiguous input range.
            tail_whole = all(starts[i] == 0 and ends[i] == shape[i]
                             for i in range(max(cut) + 1, len(shape)))
            axes = ','.join(str(i) for i in cut)
            return (f'contig-run[ax{axes}]' if tail_whole and len(cut) == 1 and cut[0] == 0
                    else f'strided[cut ax{axes} of {len(shape)}]')
        if path in ('permute', 'transpose'):
            if path == 'permute':
                dims = tuple(int(v) for v in (kwargs.get('dims') or args[1]))
            else:
                d0, d1 = int(args[1]), int(args[2])
                dims = list(range(len(ins[0].shape)))
                dims[d0], dims[d1] = dims[d1], dims[d0]
                dims = tuple(dims)
            ident = tuple(range(len(dims)))
            if dims == ident:
                return 'contig-run[identity]'
            # A permutation that leaves the LAST axis alone still reorders whole rows, which is
            # a page-granular gather, not a burst. Only a permutation of leading axes above a
            # whole contiguous tail is a run copy.
            first_moved = next(i for i in range(len(dims)) if dims[i] != i)
            tail_fixed = all(dims[i] == i for i in range(first_moved + 1, len(dims)))
            return (f'contig-run[perm{dims}]' if tail_fixed
                    else f'strided[perm{dims}]')
    except Exception:
        return 'geom?'
    return ''


class MoveTimer(VerbTimer):
    """trimove's `MoveTimer`, over EVERY family, with the move geometry in the key."""

    def __init__(self, ttnn, device):
        super().__init__(ttnn, device)
        self.edge = collections.Counter()
        self.alias_misses = collections.Counter()
        self._prod: dict[int, tuple[str, str]] = {}

    def clear_edges(self):
        self._prod.clear()

    def _wrap(self, path, real):
        def w(*args, **kwargs):
            if not self.on:
                return real(*args, **kwargs)
            ins = []
            self._walk(args, ins)
            self._walk(kwargs, ins)
            tag = D._tag()
            interesting = path in MOVES or D.CUR['family'] != 'outside'
            rb = sum(D._nbytes(t) for t in ins)
            insig = ','.join(_sig(t, self.ttnn) for t in ins) if interesting else ''
            if interesting and self._prod:
                for t in ins:
                    hit = self._prod.get(id(t))
                    if hit is None:
                        continue
                    if hit[1] != _sig(t, self.ttnn):
                        self.alias_misses[hit[0]] += 1
                        continue
                    self.edge[(hit[0], path)] += 1
            sync = self.sync
            t0 = time.perf_counter()
            out = real(*args, **kwargs)
            if sync:
                self.ttnn.synchronize_device(self.device)
            t1 = time.perf_counter()
            outs = []
            self._walk(out, outs)
            wb = sum(D._nbytes(t) for t in outs)
            outsig = ','.join(_sig(t, self.ttnn) for t in outs) if interesting else ''
            if interesting:
                geom = _geom(path, args, kwargs, ins, outs, self.ttnn) if path in MOVES else ''
                site = ';'.join(
                    f'{f.filename.rsplit("/", 1)[-1]}:{f.lineno}:{f.name}'
                    for f in traceback.extract_stack(limit=9)[:-1]
                    if 'tt_bio/' in f.filename or 'af2' in f.filename)
                key = (tag, path, geom, insig + ' -> ' + outsig + ' @ ' + site)
            else:
                key = (tag, path, '', '')
            self.wall[key] += t1 - t0
            self.calls[key] += 1
            self.read[key] += rb
            self.written[key] += wb
            if interesting and path in MOVES:
                ks = '||'.join(key)
                for t in outs:
                    self._prod[id(t)] = (ks, _sig(t, self.ttnn))
            return out
        w.__name__ = getattr(real, '__name__', path)
        return w

    def take(self):
        snap = super().take()
        snap['edge'] = {'||'.join(k) if isinstance(k, tuple) else k: v
                        for k, v in self.edge.items()}
        snap['alias_misses'] = dict(self.alias_misses)
        self.edge.clear()
        self.alias_misses.clear()
        return snap


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
    ap.add_argument('--out', default='census_n288.json')
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    args.arms = ''
    torch.set_num_threads(args.threads)

    import ttnn
    lv, dev, ref = S.open_all(args)
    lv.mask = True
    D.install_labels()
    timer = MoveTimer(ttnn, dev.device).install()
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)

    for _ in range(2):
        S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
    floor = D.sync_floor(ttnn, dev.device)
    print(json.dumps({'sync_floor_median_s': floor['median']}), flush=True)

    acc = {m: {f: collections.Counter() for f in
               ('wall', 'calls', 'read', 'written', 'edge', 'alias_misses')}
           for m in ('sync', 'free')}
    aiclks, loads = [], []
    for rep in range(args.reps):
        for mode in (['free', 'sync'] if rep % 2 == 0 else ['sync', 'free']):
            timer.sync = (mode == 'sync')
            timer.clear_edges()
            timer.on = True
            r, _ = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
            timer.on = False
            snap = timer.take()
            aiclks.append(clock.window(r['spans']))
            loads.append(round(os.getloadavg()[0], 2))
            for f in acc[mode]:
                acc[mode][f].update(snap[f])
            print(json.dumps({'rep': rep, 'mode': mode,
                              'fwd': round(r['fwd'], 3), 'bwd': round(r['bwd'], 3),
                              'aiclk': aiclks[-1], 'load': loads[-1],
                              'alias_misses': sum(snap['alias_misses'].values())}),
                  flush=True)
    clock.stop()
    timer.uninstall()
    blob = {'stamp': S.stamp(args, clock), 'n': args.n, 'pad': args.pad, 'stack': args.stack,
            'k': args.k, 'reps': args.reps, 'sync_floor_s': floor, 'aiclk': aiclks,
            'load': loads,
            'by_mode': {m: {f: dict(acc[m][f]) for f in acc[m]} for m in ('sync', 'free')}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
