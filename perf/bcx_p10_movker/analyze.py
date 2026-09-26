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


def load(path, arm=None):
    b = json.loads(open(path).read())
    lam = b['sync_floor_s']['median']
    reps = b['reps']
    per = b['by_arm'] if 'by_arm' in b else {'-': b['by_mode']}
    arm = arm if arm is not None else list(per)[0]
    sync, free = per[arm]['sync'], per[arm]['free']
    rows = {}
    for k, w in sync['wall'].items():
        c = sync['calls'][k]
        dev = w - free['wall'].get(k, 0.0) - lam * c
        tag, path_, geom, sig = k.split('||')
        rows[k] = {'tag': tag, 'verb': path_, 'geom': geom, 'sig': sig,
                   'calls': c / reps, 'dev': max(dev, 0.0) / reps,
                   # A move writes its output once and reads exactly the bytes it writes.
                   # `read` is the whole operand, which for a `slice` charges bytes the call
                   # never touches: at ax0 on [288,4,288,288] that is 1.5x the real traffic
                   # and it reads the op OVER the roof. Traffic is 2 x written; only `slice`
                   # changes, the other three write what they read.
                   'bytes': ((2 * sync['written'][k]) if path_ in VERBS
                             else (sync['read'][k] + sync['written'][k])) / reps}
    return b, rows


def family(rows):
    """The four verbs' device seconds and bytes, by verb."""
    dev, by, ca = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows.values():
        if r['verb'] in VERBS:
            dev[r['verb']] += r['dev']
            by[r['verb']] += r['bytes']
            ca[r['verb']] += r['calls']
    return dev, by, ca


def pct(dev, by):
    return f'{by / dev / ROOF * 100:6.1f} %' if dev > 0 else '     - '


#: The three verbs one `_pair_transpose_impl` round trip issues, in order.
PT_LEGS = (('to_layout', 'TILE->ROW_MAJOR', 'to_layout TILE->RM'),
           ('permute', 'perm(1, 0, 2)', 'permute(1,0,2)'),
           ('to_layout', 'ROW_MAJOR->TILE', 'to_layout RM->TILE'))


def pair_transpose(rows):
    """`z -> z^T` on `[N, N, C]`, rolled up over its three stock dispatches.

    The round trip moves the tensor THREE times for a job whose floor is once in and once
    out, so the kernel's win is a byte cut before it is a bandwidth win. Reported per leg,
    then per round trip: 3 dispatches make one transpose, not 6 -- forward and backward are
    separate transposes that each pay all three legs.
    """
    dev, byt, cal = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows.values():
        if '288x288x128' not in r['sig'] or not r['tag'].split('|')[-1].startswith('tri_att'):
            continue
        leg = next((n for v, g, n in PT_LEGS if r['verb'] == v and g in r['geom']), None)
        if leg is None:
            continue
        k = (r['tag'].split('|')[2], leg)
        dev[k] += r['dev']; byt[k] += r['bytes']; cal[k] += r['calls']
    return dev, byt, cal


def main():
    b0 = json.loads(open(sys.argv[1]).read())
    arms = b0.get('arms', ['-'])
    print(f"n={b0['n']} stack={b0['stack']} k={b0['k']} reps={b0['reps']} "
          f"arms={','.join(arms)}")
    print(f"aiclk={[a['median'] for a in b0['aiclk']]} min={min(a['min'] for a in b0['aiclk'])} "
          f"load={b0['load']}")
    print(f"sync floor {b0['sync_floor_s']['median']*1e6:.2f} us\n")

    # ---- the first number the brief asks for: the family's size on each arm
    if len(arms) > 1:
        print('=== the movement family, ANCHOR vs COMPOSED (per block step, ms) ===')
        print(f"{'verb':<12}" + ''.join(f'{a:>22}' for a in arms) + f"{'ratio':>9}")
        tot = collections.Counter()
        for verb in VERBS:
            cells, devs = [], []
            for a in arms:
                _, rows = load(sys.argv[1], a)
                dev, by, ca = family(rows)
                devs.append(dev[verb])
                tot[a] += dev[verb]
                cells.append(f"{dev[verb]*1e3:8.3f} ({pct(dev[verb], by[verb]).strip():>7})")
            ratio = (devs[0] / devs[1]) if devs[1] > 0 else float('inf')
            print(f'{verb:<12}' + ''.join(f'{c:>22}' for c in cells) + f'{ratio:>9.2f}x')
        cells = [f'{tot[a]*1e3:8.3f}' for a in arms]
        rr = tot[arms[0]] / tot[arms[1]] if tot[arms[1]] > 0 else float('inf')
        print(f"{'TOTAL':<12}" + ''.join(f'{c:>22}' for c in cells) + f'{rr:>9.2f}x')
        print()

    print('=== the pair transpose `z -> z^T` on [288,288,128], per block step ===')
    for a in arms:
        _, rows = load(sys.argv[1], a)
        dev, byt, cal = pair_transpose(rows)
        if not dev:
            print(f'  {a}: none'); continue
        print(f'  --- {a} ---')
        for k in sorted(dev):
            print(f'    {k[0]:>3} {k[1]:<20}{cal[k]:5.0f} call{dev[k]*1e3:8.3f} ms'
                  f'{byt[k]/1e6:8.1f} MB{pct(dev[k], byt[k]):>9}')
        td, tb, tc = sum(dev.values()), sum(byt.values()), sum(cal.values())
        n = tc / 3                      # 3 dispatches = 1 round trip
        floor = tb / 3                  # one pass in and out, instead of three
        print(f'    {"":>3} {"TOTAL":<20}{tc:5.0f} call{td*1e3:8.3f} ms{tb/1e6:8.1f} MB'
              f'{pct(td, tb):>9}')
        print(f'    {n:.0f} round trips, {td*1e3/n:.3f} ms each, and the kernel\'s byte '
              f'floor is {floor/1e6:.1f} MB against {tb/1e6:.1f} MB moved today (3.0x)')
    print()

    for arm in arms:
        b, rows = load(sys.argv[1], arm)
        print(f'################ ARM {arm} ################')
        print('=== the four movement verbs by GEOMETRY (per block step, both directions) ===')
        print(f"{'verb':<11}{'class':<10}{'calls':>7}{'ms':>9}{'MB':>9}{'% roof':>9}")
        agg, aggb, aggc = collections.Counter(), collections.Counter(), collections.Counter()
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
        for verb in VERBS:
            sel = [r for r in rows.values() if r['verb'] == verb and r['dev'] > 0]
            sel.sort(key=lambda r: -r['dev'])
            tot = sum(r['dev'] for r in sel)
            print(f'--- {verb}: {tot*1e3:.3f} ms a block step, top call sites ---')
            for r in sel[:8]:
                fam = r['tag'].split('|')[-1]
                d = r['tag'].split('|')[2]
                sig = r['sig'].split(' @ ')
                site = sig[1].split(';')[-1] if len(sig) > 1 else '?'
                print(f"  {r['dev']*1e3:7.3f} ms {r['calls']:5.0f} call "
                      f"{r['bytes']/1e6:7.1f} MB {pct(r['dev'], r['bytes'])}  "
                      f"{d:>3} {fam:<14} {r['geom']:<24} {sig[0]}")
                print(f"{'':>12}{site}")
            print()
    print('=== reach, last window of each arm ===')
    for a in arms:
        rs = b0.get('reach', {}).get(a, [])
        if rs:
            print(f'  {a}: {json.dumps(rs[-1]["after"])[:400]}')


if __name__ == '__main__':
    main()
