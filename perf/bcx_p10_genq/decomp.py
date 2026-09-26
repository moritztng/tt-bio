#!/usr/bin/env python3
"""Where does a `generic_op` dispatch's 0.167 ms of host actually go?

`bcx-TRIMOVE.md` leg 3 §2 measured the tax and named four suspects: the destination allocation,
the two `buffer_address()` reads, the runtime-arg mutation and the dispatch proper. This times
each stage separately on one fixed shape, warm, host-side only, against the same measurement for
the stock ttnn ops the kernels replace. If the cost is somewhere other than those four, this
table says so before a line of the fix is written.

Host-side only means the clock covers one enqueue and nothing else. The queue is drained BEFORE
every timed call, outside the clock, so no enqueue is ever pushed back on by the device -- see
`bench.py` for what that correction is worth (`ttnn.add` read 0.018 and 0.030 ms in two
processes at the same loadavg before it was applied).
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
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--c', type=int, default=128)
    ap.add_argument('--reps', type=int, default=300)
    ap.add_argument('--drain', type=int, default=20)
    ap.add_argument('--out', default='decomp_n288.json')
    ap.add_argument('--compact', type=int, default=0,
                    help='arm tt_bio.genq\'s cheap dispatch path for this process')
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    import ttnn
    from tt_bio import rne_add as RA
    from tt_bio import reblock_permute as RP
    from tt_bio import genq
    genq.set_compact(bool(args.compact))
    _lv, _dev, _ref = S.open_all(args)
    dev = _dev.device
    clock = S.Clock()
    N, C = args.n, args.c
    mc = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t: ttnn.from_torch(                                       # noqa: E731
        t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
    a = up(torch.randn(1, N, N, C))
    b = up(torch.randn(1, N, N, C))
    chan = up(torch.randn(1, N, N, C))

    RA.set_enabled(True)
    # Warm: JIT-build both programs and fill the descriptor caches, so nothing below builds.
    for _ in range(3):
        ttnn.deallocate(RA.rne_add(a, b, mc))
        ttnn.deallocate(RP.reblock_permute(chan, mc))
        ttnn.deallocate(ttnn.add(a, b, memory_config=mc))
        ttnn.deallocate(ttnn.permute(chan, (0, 3, 1, 2), memory_config=mc))
    ttnn.synchronize_device(dev)

    spans = []

    def timed(name, fn, **kw):
        return B.timed(name, fn, dev, ttnn, args.reps, spans, **kw)

    res = {}

    # ---- the four stages trimove named, plus the one it did not: _prepare ----
    out_t = ttnn.allocate_tensor_on_device(a.shape, RA.OUT_DTYPE, ttnn.TILE_LAYOUT, dev, mc)
    entry = RA._prepare(a, b, out_t, dev)
    pd = entry['pd']

    def stage_alloc():
        return ttnn.allocate_tensor_on_device(a.shape, RA.OUT_DTYPE, ttnn.TILE_LAYOUT, dev, mc)

    def stage_prepare():
        RA._prepare(a, b, out_t, dev)

    def stage_accessor_args():
        ct = [RA._gran()]
        ct.extend(ttnn.TensorAccessorArgs(a).get_compile_time_args())
        ct.extend(ttnn.TensorAccessorArgs(b).get_compile_time_args())
        w = [RA._gran()]
        w.extend(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())

    def stage_cache_key():
        RA._cache_key(a, out_t, dev, (), ())

    def stage_addr():
        return None if (a.buffer_address(), b.buffer_address(), out_t.buffer_address()) else None

    def stage_rtargs():
        pd.kernels[0].common_runtime_args = [1, 2]
        pd.kernels[1].common_runtime_args = [3]

    def stage_dispatch():
        ttnn.generic_op([a, b, out_t], pd)

    res['alloc_dest'] = timed('1 allocate dest', stage_alloc)
    res['prepare'] = timed('2 _prepare (cache hit)', stage_prepare)
    res['prepare_accessor_args'] = timed('2a   TensorAccessorArgs x3', stage_accessor_args)
    res['prepare_cache_key'] = timed('2b   _cache_key', stage_cache_key)
    res['buffer_address_x3'] = timed('3 buffer_address x3', stage_addr)
    res['runtime_arg_mutation'] = timed('4 runtime-arg mutation', stage_rtargs)
    # restore the real addresses before any dispatch runs with them
    pd.kernels[0].common_runtime_args = [a.buffer_address(), b.buffer_address()]
    pd.kernels[1].common_runtime_args = [out_t.buffer_address()]
    res['dispatch_only'] = timed('5 generic_op dispatch', stage_dispatch)

    # ---- the whole call, and the stock ops it is measured against ----
    res['rne_add_full'] = timed('rne_add full (5 stages)', lambda: RA.rne_add(a, b, mc))

    def prealloc_full():
        RA.rne_add(a, b, mc, out=out_t)    # the caller owns `out`; the allocator is off the path

    res['rne_add_preallocated'] = timed('rne_add full, dest preallocated', prealloc_full)
    res['reblock_permute_full'] = timed('reblock_permute full',
                                        lambda: RP.reblock_permute(chan, mc))
    res['stock_add'] = timed('ttnn.add', lambda: ttnn.add(a, b, memory_config=mc))
    res['stock_permute'] = timed('ttnn.permute (0,3,1,2)',
                                 lambda: ttnn.permute(chan, (0, 3, 1, 2), memory_config=mc))
    res['stock_transpose'] = timed('ttnn.transpose(1,2)',
                                   lambda: ttnn.transpose(chan, 1, 2, memory_config=mc))

    ttnn.deallocate(out_t)
    blob = {'stamp': S.stamp(args, clock), 'n': N, 'c': C, 'reps': args.reps,
            'compact': bool(args.compact), 'compact_entry': entry['compact'],
            'genq_refused': dict(genq.REFUSED), 'num_cores': entry['num_cores'],
            'drain': args.drain, 'addr_write_mode': RA.ADDR_WRITE_MODE,
            'aiclk': clock.window(spans), 'stages': res}
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
