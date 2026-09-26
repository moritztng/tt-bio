#!/usr/bin/env python3
"""Leg 1's tables out of `census.py`'s json: the four movement verbs by geometry and call site.

`device = synced - free - lambda * calls`, trimove's subtraction unchanged, with the sync floor
this process measured. Per key it reports device seconds, bytes moved (read + written) and the
achieved fraction of the 442.3 GB/s DRAM roof this campaign measures on this card.
"""
from __future__ import annotations

import collections
import json
import sys

ROOF = 442.3e9          # B/s, the campaign's measured DRAM roof on this card
VERBS = ('permute', 'slice', 'transpose', 'to_layout')


def load(path):
    b = json.loads(open(path).read())
    lam = b['sync_floor_s']['median']
    reps = b['reps']
    sync, free = b['by_mode']['sync'], b['by_mode']['free']
    rows = {}
    for k, w in sync['wall'].items():
        c = sync['calls'][k]
        dev = w - free['wall'].get(k, 0.0) - lam * c
        tag, path_, geom, sig = k.split('||')
        rows[k] = {'tag': tag, 'verb': path_, 'geom': geom, 'sig': sig,
                   'calls': c / reps, 'dev': max(dev, 0.0) / reps,
                   # A move writes its output once and reads exactly the bytes it
                   # writes. `read` is the whole operand, which for a `slice` charges
                   # bytes the call never touches: at ax0 on [288,4,288,288] that is
                   # 1.5x the real traffic and it reads the op OVER the roof. So the
                   # traffic of a move is 2 x written, and only `slice` changes.
                   'bytes': ((2 * sync['written'][k]) if path_ in VERBS
                             else (sync['read'][k] + sync['written'][k])) / reps}
    return b, rows


def pct(dev, by):
    return f'{by / dev / ROOF * 100:6.1f} %' if dev > 0 else '     - '


def main():
    b, rows = load(sys.argv[1])
    n = b['n']
    print(f"n={n} stack={b['stack']} k={b['k']} reps={b['reps']} "
          f"aiclk={b['aiclk'][0]['median']} load={b['load']}")
    print(f"sync floor {b['sync_floor_s']['median']*1e6:.2f} us\n")

    # ---- the four verbs, by geometry class
    print('=== the four movement verbs by GEOMETRY (per block step, both directions) ===')
    print(f"{'verb':<11}{'class':<10}{'calls':>7}{'ms':>9}{'MB':>9}{'% roof':>9}")
    agg = collections.Counter()
    aggb = collections.Counter()
    aggc = collections.Counter()
    for r in rows.values():
        if r['verb'] not in VERBS:
            continue
        cls = r['geom'].split('[')[0] or '?'
        agg[(r['verb'], cls)] += r['dev']
        aggb[(r['verb'], cls)] += r['bytes']
        aggc[(r['verb'], cls)] += r['calls']
    for verb in VERBS:
        for (v, cls), dev in sorted(agg.items(), key=lambda kv: -kv[1]):
            if v != verb:
                continue
            by = aggb[(v, cls)]
            print(f'{v:<11}{cls:<10}{aggc[(v, cls)]:>7.0f}{dev*1e3:>9.3f}'
                  f'{by/1e6:>9.1f}{pct(dev, by):>9}')
    print()

    # ---- per call site
    for verb in VERBS:
        sel = [r for r in rows.values() if r['verb'] == verb and r['dev'] > 0]
        sel.sort(key=lambda r: -r['dev'])
        tot = sum(r['dev'] for r in sel)
        print(f'=== {verb}: {tot*1e3:.3f} ms a block step, top call sites ===')
        for r in sel[:10]:
            fam = r['tag'].split('|')[-1]
            d = r['tag'].split('|')[2]
            sig = r['sig'].split(' @ ')
            shapes = sig[0]
            site = sig[1].split(';')[-1] if len(sig) > 1 else '?'
            print(f"  {r['dev']*1e3:7.3f} ms  {r['calls']:5.0f} call  "
                  f"{r['bytes']/1e6:7.1f} MB  {pct(r['dev'], r['bytes'])}  "
                  f"{d:>3} {fam:<14} {r['geom']:<26} {shapes}")
            print(f"{'':>13}{site}")
        print()

    # ---- consumers of the moves
    print('=== producer -> consumer, the moves that matter ===')
    edge = collections.Counter()
    for k, v in b['by_mode']['sync']['edge'].items():
        parts = k.split('||')
        prod, cons = parts[:4], parts[4] if len(parts) > 4 else '?'
        if prod[1] in VERBS:
            edge[(prod[1], prod[2], cons)] += v
    for (v, geom, cons), c in sorted(edge.items(), key=lambda kv: -kv[1])[:25]:
        print(f'  {c:6d}  {v:<11}{geom:<26} -> {cons}')
    misses = sum(b['by_mode']['sync']['alias_misses'].values())
    print(f'\nalias misses (this table\'s error bar): {misses}')


if __name__ == '__main__':
    main()
