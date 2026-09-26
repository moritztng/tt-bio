#!/usr/bin/env python3
"""What inside `ttnn.generic_op` costs the 0.035 ms, once the four named suspects are excluded?

`decomp.py` put the whole tax inside the `generic_op` call itself: the allocation is 0.003 ms,
the address reads 0.0002 and the runtime-arg mutation 0.0009, against 0.035 for the dispatch and
0.080 for the full call. This asks the dispatch three questions:

  A. does the cost scale with the SIZE of the program descriptor (core count, runtime-arg words)?
     If it does, the dispatch is hashing or serialising the descriptor on every call.
  B. does CHANGING an address between calls cost more than repeating one? If it does, ttnn's
     program cache is keyed on the runtime args and a new destination buffer is a cache miss.
  C. what does a fresh destination allocation add on top of B?
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
from perf.bcx_p10_genq import bench as B         # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_genq' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--drain', type=int, default=20)
    ap.add_argument('--out', default='probe.json')
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    from tt_bio import rne_add as RA
    _lv, _dev, _ref = S.open_all(args)
    dev = _dev.device
    clock = S.Clock()
    mc = ttnn.DRAM_MEMORY_CONFIG
    RA.set_enabled(True)
    up = lambda t: ttnn.from_torch(                                       # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    spans = []

    def timed(name, fn, **kw):
        return B.timed(name, fn, dev, ttnn, args.reps, spans, **kw)

    res = {}

    # ---- A. does the dispatch cost scale with descriptor size? ----
    # One shape per core count. The kernel splits the tile count over cores, so a tensor with
    # fewer tiles than cores places fewer cores and writes fewer runtime-arg words.
    for tag, (N, C) in (('1t', (32, 32)), ('8t', (32, 256)), ('64t', (256, 256)),
                        ('2592t', (288, 32)), ('10368t', (288, 128))):
        a = up(torch.randn(1, N, N, C)) if N != 32 else up(torch.randn(1, 1, N, C))
        b = ttnn.clone(a)
        out = ttnn.allocate_tensor_on_device(a.shape, RA.OUT_DTYPE, ttnn.TILE_LAYOUT, dev, mc)
        e = RA._prepare(a, b, out, dev)
        pd = e['pd']
        pd.kernels[0].common_runtime_args = [a.buffer_address(), b.buffer_address()]
        pd.kernels[1].common_runtime_args = [out.buffer_address()]
        ttnn.generic_op([a, b, out], pd)
        ttnn.synchronize_device(dev)
        r = timed(f'A dispatch, {tag} ({e["num_cores"]} cores)',
                  lambda pd=pd, a=a, b=b, out=out: ttnn.generic_op([a, b, out], pd))
        r['num_cores'] = e['num_cores']
        r['num_tiles'] = e['num_tiles']
        res[f'A_{tag}'] = r
        for t in (a, b, out):
            ttnn.deallocate(t)

    # ---- B/C. address churn and allocation, at the production shape ----
    N, C = 288, 128
    a = up(torch.randn(1, N, N, C))
    b = up(torch.randn(1, N, N, C))
    dests = [ttnn.allocate_tensor_on_device(a.shape, RA.OUT_DTYPE, ttnn.TILE_LAYOUT, dev, mc)
             for _ in range(4)]
    assert len({d.buffer_address() for d in dests}) == 4, 'destinations share an address'
    e = RA._prepare(a, b, dests[0], dev)
    pd = e['pd']
    pd.kernels[0].common_runtime_args = [a.buffer_address(), b.buffer_address()]

    def fixed():
        pd.kernels[1].common_runtime_args = [dests[0].buffer_address()]
        ttnn.generic_op([a, b, dests[0]], pd)

    ctr = [0]

    def rotating():
        d = dests[ctr[0] & 3]
        ctr[0] += 1
        pd.kernels[1].common_runtime_args = [d.buffer_address()]
        ttnn.generic_op([a, b, d], pd)

    res['B_fixed_dest'] = timed('B dispatch, one dest repeated', fixed)
    res['B_rotating_dest'] = timed('B dispatch, 4 dests rotating', rotating)
    def prealloc_full():
        RA.rne_add(a, b, mc, out=dests[0])     # the caller owns `out`; nothing to free

    res['C_preallocated_full'] = timed('C rne_add(out=preallocated)', prealloc_full)
    res['C_fresh_alloc_full'] = timed('C rne_add(fresh alloc)', lambda: RA.rne_add(a, b, mc))
    res['stock_add'] = timed('  ttnn.add', lambda: ttnn.add(a, b, memory_config=mc))

    blob = {'stamp': S.stamp(args, clock), 'reps': args.reps, 'drain': args.drain,
            'aiclk': clock.window(spans), 'probes': res}
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
