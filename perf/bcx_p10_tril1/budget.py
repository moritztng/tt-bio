#!/usr/bin/env python3
"""bcx-p10-tril1: what the taped triangle multiplication's L1 residency costs, and where it stops.

`bcx-p10-trilay` priced the refusal: at a 288 token axis every one of the 58 buffers the family
touches is in DRAM, because `_triangle_mul_memory_config` refuses L1 outright while a tape is
open, and the family then spends 331.3 GB a round on the DRAM bus. `_TRIMUL_TAPED_L1` prices the
residency instead. Two subcommands, both on one card in one process:

  table  the budget itself, per token axis: the width the L1 budget buys, the six chunk-multiples
         the loop holds live at its peak, the share of the banks that is, and the verdict. This is
         the thing the next shape reads instead of re-deriving it.
  clash  the failure mode the gate exists for, made to happen. `--share` inflates the budget past
         what the grid can hold, so the channel loop throws "statically allocated circular buffers
         ... clash with L1 buffers" at program validation. The run must SURVIVE it: the retry
         demotes the shape to DRAM, the block completes, and its gradient is identical to the
         gradient the DRAM arm produces. A budget with no failure test is a budget nobody checked.
"""
from __future__ import annotations

import argparse
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

OUT = ROOT / 'perf' / 'bcx_p10_tril1' / 'out'
MB = 1 << 20


def row(T, seq_len, hidden, batch, elem=2):
    """One line of the budget, computed from the engine's own helpers, never re-derived here."""
    gx, gy = T.COMPUTE_GRID_MAIN
    banks = T._l1_bank_bytes() * gx * gy
    area_budget = T._trimul_l1_chunk_budget()
    chunk = T.TRIANGLE_MULT_CHUNK_SIZE
    while (hidden % (chunk * 2) == 0
           and T._trimul_chunk_l1_area(seq_len, chunk * 2, batch) <= area_budget):
        chunk *= 2
    one = T._trimul_chunk_l1_area(seq_len, chunk, batch) * elem
    ht = -(-seq_len // 32) * 32
    return {
        'seq_len': seq_len, 'hidden': hidden, 'batch': batch, 'chunk': chunk,
        'padded_seq': ht,
        # the head is the fused in-projection, 4 chunk-multiples at group 1; the tail is the two
        # transformed operands live beside it.
        'head_MB': round(4 * one / MB, 2),
        'tail_MB': round(2 * one / MB, 2),
        'peak_MB': round(T._TRIMUL_TAPED_L1_LIVE * one / MB, 2),
        'banks_MB': round(banks / MB, 2),
        'share_of_banks': round(T._TRIMUL_TAPED_L1_LIVE * one / banks, 4),
        'budget_share': T._TRIMUL_TAPED_L1_SHARE,
        'below_max_seq': seq_len <= T._trimul_l1_max_seq(),
        'fits': T._trimul_taped_l1_fits(seq_len, hidden, batch),
    }


def cmd_table(args):
    lv, dev, ref = S.open_all(args)
    from tt_bio import tenstorrent as T
    was = T.set_trimul_taped_l1(True)
    rows = []
    for n in [int(x) for x in args.seqs.split(',')]:
        for hidden in [int(x) for x in args.hidden.split(',')]:
            r = row(T, n, hidden, args.batch)
            # the verdict the engine actually returns, asked through the shipped gate rather
            # than inferred from the row above it
            with dev.tt.tape():
                r['memory_config'] = str(
                    T._triangle_mul_memory_config(n, hidden, args.batch).buffer_type)
            r['verdict'] = 'L1' if 'L1' in r['memory_config'] else 'DRAM'
            rows.append(r)
            print(json.dumps(r), flush=True)
    T.set_trimul_taped_l1(was)
    blob = {'stamp': S.stamp(args), 'grid': list(T.COMPUTE_GRID_MAIN),
            'l1_bank_bytes': T._l1_bank_bytes(), 'live': T._TRIMUL_TAPED_L1_LIVE,
            'share': T._TRIMUL_TAPED_L1_SHARE, 'max_seq': T._trimul_l1_max_seq(), 'rows': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


def cmd_clash(args):
    """Inflate the budget until the loop cannot lay out, and require the round to survive it."""
    lv, dev, ref = S.open_all(args)
    lv.mask = True
    from tt_bio import tenstorrent as T
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)

    budget0 = T.TRIANGLE_MULT_L1_CHUNK_BUDGET

    def arm(taped_l1, share, chunk_budget):
        T.set_trimul_taped_l1(taped_l1)
        T._TRIMUL_TAPED_L1_SHARE = share
        T.TRIANGLE_MULT_L1_CHUNK_BUDGET = chunk_budget
        T._TRIMUL_TAPED_L1_CLASH.clear()
        T.TRIMUL_TAPED_L1_STATS.update({'l1': 0, 'dram': 0, 'clash': 0})
        _, g = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
        return g, dict(T.TRIMUL_TAPED_L1_STATS)

    S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)     # warm, dropped
    base, base_stats = arm(False, args.share, budget0)
    # A width the grid is MEASURED not to hold: `TRIANGLE_MULT_L1_CHUNK_BUDGET`'s own comment has
    # chunk 128 at seq 256 throwing the clash, so chunk 128 at 288 is past it by construction.
    over, over_stats = arm(True, args.share, args.chunk_budget)
    T.set_trimul_taped_l1(False)
    T.TRIANGLE_MULT_L1_CHUNK_BUDGET = budget0

    out = {'stamp': S.stamp(args), 'n': args.n, 'share': args.share,
           'chunk_budget': args.chunk_budget,
           'dram_stats': base_stats, 'inflated_stats': over_stats,
           'clashed': over_stats['clash'] > 0, 'grads': []}
    for name, a, b in zip(('m', 'z'), base, over):
        d = (a.double() - b.double())
        out['grads'].append({'which': name, 'equal': bool(torch.equal(a, b)),
                             'max_abs': float(d.abs().max()),
                             'rel_l2': float(d.norm() / a.double().norm())})
    out['survived'] = all(g['equal'] for g in out['grads'])
    print(json.dumps(out, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(out, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)
    if not out['clashed']:
        print('NO CLASH: raise --share until the loop cannot lay out, or this proves nothing',
              flush=True)
    return 0 if out['survived'] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('table', 'clash'))
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--seqs', default='288,352,384,512')
    ap.add_argument('--hidden', default='128')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--stack', default='evo')
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--share', type=float, default=1.0)
    ap.add_argument('--chunk-budget', type=int, default=16_000_000,
                    dest='chunk_budget', help='override TRIANGLE_MULT_L1_CHUNK_BUDGET (clash)')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    args.out = args.out or f'{args.cmd}.json'
    torch.set_num_threads(args.threads)
    sys.exit(cmd_table(args) if args.cmd == 'table' else cmd_clash(args))


if __name__ == '__main__':
    main()
