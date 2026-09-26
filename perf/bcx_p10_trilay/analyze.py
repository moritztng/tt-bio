#!/usr/bin/env python3
"""Per-verb device seconds and bytes for the triangle multiplication, per arm.

`bcx-p10-trimul`'s `canalyze.py` reads a one-arm blob. This reads the arm blob and prints each
arm's table plus the difference, ranked by absolute headroom rather than by ratio: at 47.1 % of
the roof a 20 % verb worth 8 ms is not the target.
"""
import collections
import json
import sys

ROOF = 442.3e9
FAMILIES = ('tri_mul_out', 'tri_mul_in')


def rows_for(b, arm, block='1'):
    lam = b['sync_floor_s']['median']
    src = b['by_arm'][arm]
    rows = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for mode, idx in (('sync', 0), ('free', 1)):
        for k, v in src[mode]['wall'].items():
            tag, verb, sig = k.split('||')
            st, blk, dr, fam = tag.split('|')
            if fam not in FAMILIES or st != 'evo' or blk != block:
                continue
            r = rows[(dr, fam, verb, sig)]
            r[idx] += v
            if idx == 0:
                r[2] += src[mode]['calls'][k]
                r[3] += src[mode]['read'][k]
                r[4] += src[mode]['written'][k]
    return rows, lam


def dev(v, lam):
    return max(v[0] - v[1] - lam * v[2], 0.0)


def table(b, arm, reps, top=14):
    rows, lam = rows_for(b, arm)
    out = {}
    for dr in ('fwd', 'bwd'):
        sel = [(k, v) for k, v in rows.items() if k[0] == dr]
        tot = sum(dev(v, lam) for _, v in sel) / reps
        totb = sum(v[3] + v[4] for _, v in sel) / reps
        calls = sum(v[2] for _, v in sel) / reps
        out[dr] = (tot, totb, calls)
        gbs = totb / tot / 1e9 if tot else 0
        print(f'=== {arm} {dr}: device {tot * 1e3:.3f} ms/block, {totb / 1e6:.1f} MB, '
              f'{calls:.0f} calls, {gbs:.1f} GB/s = {gbs * 1e9 / ROOF * 100:.1f}% of roof')
        ranked = sorted(sel, key=lambda x: -dev(x[1], lam))
        for (d, fam, verb, sig) in [k for k, _ in ranked[:top]]:
            v = rows[(d, fam, verb, sig)]
            d_s, by = dev(v, lam) / reps, (v[3] + v[4]) / reps
            g = by / d_s / 1e9 if d_s > 0 else 0
            head = d_s - by / ROOF
            print(f'  {d_s * 1e3:8.3f} ms {v[2] / reps:5.1f}c {by / 1e6:8.2f} MB '
                  f'{g:7.1f} GB/s {g * 1e9 / ROOF * 100:5.1f}%  hd {head * 1e3:7.3f} ms  '
                  f'{fam[8:]:4s} {verb:26s} {sig[:150]}')
    return out


def main():
    b = json.load(open(sys.argv[1]))
    reps = b['reps']
    clk = [a['median'] for a in b['aiclk']]
    print(f'##### {sys.argv[1]}  n={b["n"]}  reps={reps}  arms={b["arms"]}  '
          f'sync_floor={b["sync_floor_s"]["median"] * 1e6:.1f} us')
    print(f'AICLK median per window {clk}  min {min(clk)}   load {b.get("load")}')
    for a, c in b.get('reach', {}).items():
        print(f'reach {a}: {dict(c)}')
    print()
    per = {}
    for arm in b['arms']:
        per[arm] = table(b, arm, reps)
        print()
    if len(b['arms']) == 2:
        a0, a1 = b['arms']
        print('=== A/B, device ms per block (both triangle multiplications)')
        for dr in ('fwd', 'bwd'):
            x, y = per[a0][dr], per[a1][dr]
            print(f'  {dr}: {x[0] * 1e3:8.3f} -> {y[0] * 1e3:8.3f} ms  '
                  f'{x[0] / y[0] if y[0] else 0:.4f}x   '
                  f'bytes {x[1] / 1e6:8.1f} -> {y[1] / 1e6:8.1f} MB   '
                  f'calls {x[2]:.0f} -> {y[2]:.0f}')
        tx = per[a0]['fwd'][0] * 2 + per[a0]['bwd'][0]
        ty = per[a1]['fwd'][0] * 2 + per[a1]['bwd'][0]
        print(f'  block (2 fwd + 1 bwd): {tx * 1e3:.3f} -> {ty * 1e3:.3f} ms, '
              f'{tx / ty if ty else 0:.4f}x')
        print(f'  x 52 blocks a round:   {tx * 52:.3f} -> {ty * 52:.3f} s, '
              f'{(tx - ty) * 52:+.3f} s')


if __name__ == '__main__':
    main()
