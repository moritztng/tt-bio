#!/usr/bin/env python3
"""Per-verb device seconds and bytes for the triangle multiplication, out of a census run.

device_i = synced_i - free_i - lambda * calls_i, the subtraction devmap uses, with the sync
floor this run measured. Rows are (direction, family, verb, call signature), where the
signature carries the padded shapes, the dtypes, the buffer types and the tt_bio call site.
"""
import collections
import json
import sys

ROOF = 442.3e9


def load(path, block='1'):
    b = json.load(open(path))
    reps, lam = b['reps'], b['sync_floor_s']['median']
    S, F = b['sync'], b['free']
    rows = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for src, idx in ((S, 0), (F, 1)):
        for k, v in src['wall'].items():
            tag, verb, sig = k.split('||')
            st, blk, dr, fam = tag.split('|')
            if fam not in ('tri_mul_out', 'tri_mul_in') or st != 'evo' or blk != block:
                continue
            r = rows[(dr, fam, verb, sig)]
            r[idx] += v
            if idx == 0:
                r[2] += src['calls'][k]
                r[3] += src['read'][k]
                r[4] += src['written'][k]
    return b, reps, rows


def main():
    for path in sys.argv[1:]:
        b, reps, rows = load(path)
        print(f'##### {path}  n={b["n"]}  reps={reps}  '
              f'sync_floor={b["sync_floor_s"]["median"] * 1e6:.1f} us  '
              f'aiclk={[a["median"] for a in b["aiclk"]]}')
        for dr in ('fwd', 'bwd'):
            sel = [(k, v) for k, v in rows.items() if k[0] == dr]
            dev = lambda v: max(v[0] - v[1] - b['sync_floor_s']['median'] * v[2], 0.0)  # noqa: E731
            tot = sum(dev(v) for _, v in sel)
            totb = sum(v[3] + v[4] for _, v in sel)
            print(f'=== {dr}: device {tot / reps * 1e3:.3f} ms/block, '
                  f'{totb / reps / 1e6:.1f} MB, {sum(v[2] for _, v in sel) / reps:.0f} calls')
            for (d, fam, verb, sig), v in sorted(sel, key=lambda x: -dev(x[1])):
                d_s, by = dev(v) / reps, (v[3] + v[4]) / reps
                gbs = by / d_s / 1e9 if d_s > 0 else 0
                print(f'  {d_s * 1e3:8.3f} ms {v[2] / reps:5.1f}c {by / 1e6:8.2f} MB '
                      f'{gbs:7.1f} GB/s {gbs * 1e9 / ROOF * 100:5.1f}%  {fam[8:]:4s} '
                      f'{verb:32s} {sig}')
        print()


if __name__ == '__main__':
    main()
