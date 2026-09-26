#!/usr/bin/env python3
"""The pair-transpose shape gate: bit-exactness, reach, and the A/B, through the real entry.

`ladder.py` measured the two routes as standalone op sequences. This drives `_pair_transpose`
itself -- the function the model calls -- so the number is the one the round would see, and it
checks the two things a dispatch lever has to prove before anyone quotes a speed:

* **`torch.equal` between arms.** The routes compute the same permutation, so anything short of
  equal means the gate changed the function rather than the dispatch.
* **Reach, counted both ways.** `LATCH_STATS['pt_row_major']` separates calls that took the
  ROW_MAJOR route from calls a narrow C sent down the tiled permute. An arm that silently never
  fired reads exactly like an arm with nothing to do.

AICLK is read from the MAIN thread once per iteration: `stack.Clock`'s daemon thread is starved
by a tight ttnn loop and collected one sample in a 60-iteration run of `micro.py`.
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
    ap.add_argument('--shapes', default='288x128,512x128,288x256')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warm', type=int, default=4)
    ap.add_argument('--min-c', type=int, default=256, help='the armed gate')
    ap.add_argument('--out', default='lever.json')
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

    def latch():
        st = T.LATCH_STATS['pt_row_major']
        return {'served': st['served'], 'declined': st['declined']}

    def timed(x, nbytes):
        fn = lambda: T._pair_transpose(x, mc)                               # noqa: E731
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
        return {'ms': round(ms, 4),
                'pct_roof': round(2 * nbytes / (ms / 1e3) / ROOF * 100, 1),
                'aiclk_med': clks[len(clks) // 2], 'aiclk_min': clks[0], 'n': len(clks)}

    rows = {}
    for spec in args.shapes.split(','):
        N, C = (int(v) for v in spec.split('x'))
        nbytes = N * N * C * 2
        x = ttnn.from_torch(torch.randn(N, N, C).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=mc)
        r = {'MB': round(nbytes / 1e6, 1)}
        outs = {}
        for arm, min_c in (('off', 0), ('on', args.min_c), ('off2', 0)):
            T._PT_ROW_MAJOR_MIN_C = min_c
            b0 = latch()
            outs[arm] = torch.Tensor(ttnn.to_torch(T._pair_transpose(x, mc))).clone()
            r[arm] = timed(x, nbytes)
            b1 = latch()
            r[arm]['reach'] = {k: b1[k] - b0[k] for k in b0}
        # The A/A control sits beside the A/B so "equal" is not just a property of one run.
        r['equal_off_on'] = bool(torch.equal(outs['off'], outs['on']))
        r['equal_off_off2'] = bool(torch.equal(outs['off'], outs['off2']))
        ref = torch.Tensor(ttnn.to_torch(x)).permute(1, 0, 2).contiguous()
        r['equal_off_torch'] = bool(torch.equal(outs['off'], ref))
        r['equal_on_torch'] = bool(torch.equal(outs['on'], ref))
        ttnn.deallocate(x)
        rows[spec] = r
        print(json.dumps({spec: r}), flush=True)

    T._PT_ROW_MAJOR_MIN_C = 0
    print(f"\n{'shape':<10}{'off ms':>9}{'on ms':>9}{'off/on':>9}"
          f"{'off reach':>20}{'on reach':>20}{'equal':>8}{'AICLK':>7}")
    for spec, r in rows.items():
        eq = r['equal_off_on'] and r['equal_off_off2'] and r['equal_on_torch']
        print(f"{spec:<10}{r['off']['ms']:>9.4f}{r['on']['ms']:>9.4f}"
              f"{r['off']['ms']/r['on']['ms']:>8.2f}x"
              f"{str(r['off']['reach']):>20}{str(r['on']['reach']):>20}"
              f"{str(eq):>8}{r['on']['aiclk_med']:>7}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(
        {'iters': args.iters, 'min_c': args.min_c, 'rows': rows}, indent=1))
    print('wrote ' + str(OUT / args.out))


if __name__ == '__main__':
    main()
