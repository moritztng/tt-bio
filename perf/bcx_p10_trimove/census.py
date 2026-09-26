#!/usr/bin/env python3
"""bcx-p10-trimove leg 1: the triangle multiplication's moves, keyed by CALL SITE and CONSUMER.

`bcx-p10-calls` ranked the round's movement by ttnn op name. That says `permute` costs 0.638 s
and 59 % of it is the triangle multiplication; it does not say which of the five `permute` call
sites inside the trimul that is, nor what reads the result. The removability of a move is a
property of its consumer: a move feeding a matmul is removable by `transpose_a`/`transpose_b`,
a move feeding an eltwise op is not.

So this extends `bcx-p10-trimul`'s `VerbTimer` (same subtraction, same sync floor,

    device_i = synced_i - free_i - lambda * calls_i

) with two things production code does not carry:

* a producer->consumer edge. Every move's output tensors are registered by object id with the
  signature they had; the next op that READS one of them records the edge. `id()` is reused
  after a deallocate, so the signature is checked on the way out and a mismatch is counted
  rather than attributed -- `alias_misses` is the instrument's own error bar, and a table with
  a non-trivial one is not quotable.
* the matmul's `transpose_a` / `transpose_b` flags in its key, so a call site whose transpose
  has ALREADY been folded into the matmul is visible as such instead of being scoped again.

`TRIMUL_MM_TRANSPOSE_STATS` is snapshotted per window beside it: that is production's own
per-branch census of which channel-move branch ran and whether it deferred.
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

OUT = ROOT / 'perf' / 'bcx_p10_trimove' / 'out'

FAMILIES = ('tri_mul_out', 'tri_mul_in')
#: The ops whose output is tracked to its consumer. Everything the census calls "movement"
#: inside this family, plus the two that produce the operands a move consumes.
MOVES = ('permute', 'transpose', 'chunk', 'to_memory_config', 'reallocate', 'clone',
         'concat', 'slice', 'typecast', 'generic_op')


class MoveTimer(VerbTimer):
    """`VerbTimer` plus the producer->consumer edge and the matmul's transpose flags."""

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
            interesting = D.CUR['family'] in FAMILIES
            rb = sum(D._nbytes(t) for t in ins)
            insig = ','.join(_sig(t, self.ttnn) for t in ins) if interesting else ''
            # The consumer edge, read BEFORE the call: these are the tensors this op reads.
            if interesting and self._prod:
                for t in ins:
                    hit = self._prod.get(id(t))
                    if hit is None:
                        continue
                    if hit[1] != _sig(t, self.ttnn):
                        self.alias_misses[hit[0]] += 1
                        continue
                    self.edge[(hit[0], path + self._mmflags(path, kwargs))] += 1
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
                site = ';'.join(
                    f'{f.filename.rsplit("/", 1)[-1]}:{f.lineno}:{f.name}'
                    for f in traceback.extract_stack(limit=9)[:-1]
                    if 'tt_bio/' in f.filename or 'af2' in f.filename)
                key = (tag, path + self._mmflags(path, kwargs),
                       insig + ' -> ' + outsig + ' @ ' + site)
            else:
                key = (tag, path, '')
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

    @staticmethod
    def _mmflags(path, kwargs):
        if path not in ('matmul', 'linear', 'experimental.minimal_matmul'):
            return ''
        ta, tb = bool(kwargs.get('transpose_a')), bool(kwargs.get('transpose_b'))
        return f'[ta={int(ta)},tb={int(tb)}]'

    def take(self):
        snap = super().take()
        snap['edge'] = {'||'.join(k): v for k, v in self.edge.items()}
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
    ap.add_argument('--arms', default='', help='comma list; see LEVERS below')
    ap.add_argument('--out', default='census_n288.json')
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    lv, dev, ref = S.open_all(args)
    lv.mask = True
    D.install_labels()
    timer = MoveTimer(ttnn, dev.device).install()
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)

    from tt_bio import tenstorrent as T
    from tt_bio import reblock_permute as R

    def set_arm(arm):
        if arm == '-':
            return
        LEVERS[arm](T)

    for _ in range(2):
        S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
    floor = D.sync_floor(ttnn, dev.device)
    print(json.dumps({'sync_floor_median_s': floor['median']}), flush=True)

    arms = args.arms.split(',') if args.arms else ['-']
    acc = {(a, m): {f: collections.Counter() for f in
                    ('wall', 'calls', 'read', 'written', 'edge', 'alias_misses')}
           for a in arms for m in ('sync', 'free')}
    branch = {a: collections.Counter() for a in arms}
    aiclks, loads = [], []
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            set_arm(arm)
            for mode in (['free', 'sync'] if rep % 2 == 0 else ['sync', 'free']):
                timer.sync = (mode == 'sync')
                timer.clear_edges()
                T.TRIMUL_MM_TRANSPOSE_STATS.clear()
                R.STATS[:] = [0, 0]
                R.STATS_BACK[:] = [0, 0]
                timer.on = True
                r, _ = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
                timer.on = False
                snap = timer.take()
                branch[arm].update({'|'.join(k): v
                                    for k, v in T.TRIMUL_MM_TRANSPOSE_STATS.items()})
                branch[arm].update({'reblock_fwd': R.STATS[0],
                                    'reblock_back': R.STATS_BACK[0]})
                aiclks.append(clock.window(r['spans']))
                loads.append(round(os.getloadavg()[0], 2))
                for f in acc[(arm, mode)]:
                    acc[(arm, mode)][f].update(snap[f])
                print(json.dumps({'rep': rep, 'arm': arm, 'mode': mode,
                                  'fwd': round(r['fwd'], 3), 'bwd': round(r['bwd'], 3),
                                  'aiclk': aiclks[-1], 'load': loads[-1],
                                  'branch': dict(branch[arm]),
                                  'alias_misses': sum(snap['alias_misses'].values())}),
                      flush=True)
    clock.stop()
    timer.uninstall()
    blob = {'stamp': S.stamp(args, clock), 'n': args.n, 'pad': args.pad, 'stack': args.stack,
            'k': args.k, 'reps': args.reps, 'sync_floor_s': floor, 'aiclk': aiclks,
            'load': loads, 'arms': arms, 'branch': {a: dict(c) for a, c in branch.items()},
            'by_arm': {a: {m: {f: dict(acc[(a, m)][f]) for f in acc[(a, m)]}
                           for m in ('sync', 'free')} for a in arms}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


def _taped_move(on):
    from tt_bio import reblock_permute as R
    R.set_taped_channel_move(on)


#: Arms this row alternates at the window boundary, in one process on one card. `off` / `on`
#: are the lever. `nodefer` turns OFF the SHIPPED deferral of the operand transpose into
#: `ttnn.matmul`, which prices what leg 2 would have bought had the campaign not already had it.
LEVERS: dict = {
    'off': lambda T: _taped_move(False),
    'on': lambda T: _taped_move(True),
    'nodefer': lambda T: (_taped_move(False), setattr(T, '_TRIMUL_MM_TRANSPOSE', False)),
}


if __name__ == '__main__':
    main()
