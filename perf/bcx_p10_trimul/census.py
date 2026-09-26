#!/usr/bin/env python3
"""bcx-p10-trimul: every ttnn verb the triangle multiplication issues, with its shapes.

devmap attributed the op family. This attributes the verbs inside it: for each call site it
records the verb, the padded shapes and dtypes it read and wrote, its synced wall and its
enqueue-only wall, so a verb's device seconds and its bytes stand next to each other.

    device_i = synced_i - free_i - lambda * calls_i

the same subtraction devmap uses, with the same measured sync floor. Nothing here changes what
the model computes; every wrapper calls through.
"""
from __future__ import annotations

import argparse
import collections
import traceback
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_stack'))
from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_p10_devmap import devmap as D      # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_trimul'

FAMILIES = ('tri_mul_out', 'tri_mul_in')


def _sig(t, ttnn):
    try:
        shape = 'x'.join(str(int(d)) for d in t.padded_shape)
    except Exception:
        return '?'
    dt = str(t.dtype).rsplit('.', 1)[-1].upper()
    try:
        buf = 'L1' if t.memory_config().buffer_type == ttnn.BufferType.L1 else 'DR'
    except Exception:
        buf = '??'
    return f'{shape}:{dt}:{buf}'


class VerbTimer:
    """Per (tag, verb, call-signature): wall, calls, bytes read, bytes written."""

    def __init__(self, ttnn, device):
        self.ttnn, self.device = ttnn, device
        self.sync = False
        self.on = False
        self.wall = collections.Counter()
        self.calls = collections.Counter()
        self.read = collections.Counter()
        self.written = collections.Counter()
        self._saved = []

    def _walk(self, obj, out):
        T = self.ttnn.Tensor
        if isinstance(obj, T):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self._walk(o, out)
        elif isinstance(obj, dict):
            for o in obj.values():
                self._walk(o, out)

    def install(self):
        for path in D.VERBS:
            parent, _, leaf = path.rpartition('.')
            mod = self.ttnn
            ok = True
            for part in parent.split('.') if parent else []:
                mod = getattr(mod, part, None)
                if mod is None:
                    ok = False
                    break
            if not ok or not hasattr(mod, leaf):
                continue
            real = getattr(mod, leaf)
            if not callable(real):
                continue
            self._saved.append((mod, leaf, real))
            setattr(mod, leaf, self._wrap(path, real))
        from tt_bio import taped_ttnn as T
        T.forget_shim_bindings()
        return self

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
                key = (tag, path, insig + ' -> ' + outsig + ' @ ' + site)
            else:
                key = (tag, path, '')
            self.wall[key] += t1 - t0
            self.calls[key] += 1
            self.read[key] += rb
            self.written[key] += wb
            return out
        w.__name__ = getattr(real, '__name__', path)
        return w

    def uninstall(self):
        for mod, leaf, real in self._saved:
            setattr(mod, leaf, real)
        from tt_bio import taped_ttnn as T
        T.forget_shim_bindings()

    def take(self):
        snap = {'wall': {'||'.join(k): v for k, v in self.wall.items()},
                'calls': {'||'.join(k): v for k, v in self.calls.items()},
                'read': {'||'.join(k): v for k, v in self.read.items()},
                'written': {'||'.join(k): v for k, v in self.written.items()}}
        for c in (self.wall, self.calls, self.read, self.written):
            c.clear()
        return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--n', type=int, default=275)
    ap.add_argument('--pad', type=int, default=288)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--stack', default='evo')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--arms', default='',
                    help='alternate autograd.DGRAD_2D_MINIMAL arms in one process')
    ap.add_argument('--out', default='census.json')
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

    for _ in range(2):
        S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
    floor = D.sync_floor(ttnn, dev.device)
    print(json.dumps({'sync_floor_median_s': floor['median']}), flush=True)

    from tt_bio import autograd as AG
    arms = args.arms.split(',') if args.arms else ['-']
    acc = {(a, m): {'wall': collections.Counter(), 'calls': collections.Counter(),
                    'read': collections.Counter(), 'written': collections.Counter()}
           for a in arms for m in ('sync', 'free')}
    served = {a: collections.Counter() for a in arms}
    aiclks = []
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            if args.arms:
                AG.DGRAD_2D_MINIMAL = (arm == 'on')
            for mode in (['free', 'sync'] if rep % 2 == 0 else ['sync', 'free']):
                timer.sync = (mode == 'sync')
                before = dict(AG.DGRAD_2D_STATS)
                timer.on = True
                r, _ = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
                timer.on = False
                snap = timer.take()
                served[arm].update({k: AG.DGRAD_2D_STATS[k] - before[k] for k in before})
                aiclks.append(clock.window(r['spans']))
                for f in ('wall', 'calls', 'read', 'written'):
                    acc[(arm, mode)][f].update(snap[f])
                print(json.dumps({'rep': rep, 'arm': arm, 'mode': mode,
                                  'fwd': round(r['fwd'], 3), 'bwd': round(r['bwd'], 3),
                                  'aiclk': aiclks[-1],
                                  'load': round(os.getloadavg()[0], 2)}), flush=True)
    clock.stop()
    timer.uninstall()
    blob = {'stamp': S.stamp(args, clock), 'n': args.n, 'pad': args.pad, 'stack': args.stack,
            'k': args.k, 'reps': args.reps, 'sync_floor_s': floor, 'aiclk': aiclks,
            'pair_masks': False,  # block_step runs the unmasked pair track, as devmap did
            'arms': arms, 'served': {a: dict(c) for a, c in served.items()},
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
