#!/usr/bin/env python3
"""Does the taped channel move reach, and does it move a bit?

One process, one card, both arms, the same leaves and the same seeds. The arm is a pure index
reordering on both legs, so the bar is `torch.equal` and not a tolerance. Reach is counted off
`reblock_permute.STATS` / `STATS_BACK`, because an arm that declined every call would come back
bit-exact too and prove nothing.
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
from perf.bcx_stack import stack as S                 # noqa: E402
from perf.bcx_p10_devmap import devmap as D           # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_trimove' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=D.A.DEFAULT_PARAMS)
    ap.add_argument('--n', type=int, default=288)
    ap.add_argument('--pad', type=int, default=288)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--stack', default='evo')
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', default='bits_n288.json')
    args = ap.parse_args()
    args.card = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))
    torch.set_num_threads(args.threads)

    lv, dev, ref = S.open_all(args)
    lv.mask = True
    clock = S.Clock()
    m0, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    from tt_bio import reblock_permute as R
    from tt_bio import tenstorrent as T

    S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)

    res, grads = {}, {}
    for arm in ('off', 'on', 'off2'):
        R.set_taped_channel_move(arm == 'on')
        R.STATS[:] = [0, 0]
        R.STATS_BACK[:] = [0, 0]
        R.STATS_GATED[:] = [0, 0]
        T.TRIMUL_MM_TRANSPOSE_STATS.clear()
        r, g = S.block_step(dev, lv, m0, z0, wm, wz, args.stack, k=args.k)
        grads[arm] = [t.clone() for t in g]
        res[arm] = {'fwd': round(r['fwd'], 4), 'bwd': round(r['bwd'], 4),
                    'aiclk': clock.window(r['spans']),
                    'load': round(os.getloadavg()[0], 2),
                    'reblock_fwd_calls': R.STATS[0], 'reblock_back_calls': R.STATS_BACK[0],
                    'reblock_gated_calls': R.STATS_GATED[0],
                    'branch': {'|'.join(k): v
                               for k, v in T.TRIMUL_MM_TRANSPOSE_STATS.items()},
                    'rejects': {str(k): v for k, v in R.REJECTS.items()}}
        print(json.dumps({arm: res[arm]}), flush=True)
    clock.stop()

    def cmp(a, b):
        out = []
        for x, y in zip(grads[a], grads[b]):
            x, y = x.float(), y.float()
            d = (x - y).abs()
            n = x.norm()
            out.append({'equal': bool(torch.equal(grads[a][len(out)], grads[b][len(out)])),
                        'max_abs': float(d.max()),
                        'rel_l2': float((d.norm() / n) if n > 0 else 0.0)})
        return out

    res['on_vs_off'] = cmp('off', 'on')
    res['off_vs_off2'] = cmp('off', 'off2')
    print(json.dumps({'on_vs_off': res['on_vs_off'],
                      'off_vs_off2 (A/A)': res['off_vs_off2']}, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(
        {'stamp': S.stamp(args, clock), 'n': args.n, 'k': args.k, **res}, indent=1, default=str))
    print('wrote ' + str(OUT / args.out), flush=True)


if __name__ == '__main__':
    main()
