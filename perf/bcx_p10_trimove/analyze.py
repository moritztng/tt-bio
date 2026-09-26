#!/usr/bin/env python3
"""Leg 1's table: the triangle multiplication's moves by CALL SITE, with the consumer named.

Same subtraction and the same roof as `bcx-p10-trilay`'s analyze. What is added is the
`consumers` column, read off the producer->consumer edges the census recorded, and the
`alias_misses` line that says how much of the edge map is untrustworthy.
"""
import collections
import json
import sys

ROOF = 442.3e9
FAMILIES = ('tri_mul_out', 'tri_mul_in')
MOVEVERBS = ('permute', 'transpose', 'chunk', 'to_memory_config', 'reallocate', 'clone',
             'concat', 'slice', 'typecast')


def rows_for(b, arm, block=None):
    lam = b['sync_floor_s']['median']
    src = b['by_arm'][arm]
    rows = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for mode, idx in (('sync', 0), ('free', 1)):
        for k, v in src[mode]['wall'].items():
            tag, verb, sig = k.split('||')
            st, blk, dr, fam = tag.split('|')
            if fam not in FAMILIES or st != 'evo':
                continue
            if block is not None and blk != block:
                continue
            r = rows[(dr, fam, verb, sig, blk)]
            r[idx] += v
            if idx == 0:
                r[2] += src[mode]['calls'][k]
                r[3] += src[mode]['read'][k]
                r[4] += src[mode]['written'][k]
    return rows, lam


def edges_for(b, arm):
    """producer key string -> Counter(consumer verb)."""
    out = collections.defaultdict(collections.Counter)
    for mode in ('sync', 'free'):
        for k, v in b['by_arm'][arm][mode]['edge'].items():
            prod, cons = k.rsplit('||', 1)
            out[prod][cons] += v
    return out


def short(sig):
    body, _, site = sig.partition(' @ ')
    frames = [f for f in site.split(';') if ':' in f]
    return body, (frames[-1] if frames else '?')


def main():
    b = json.load(open(sys.argv[1]))
    reps = b['reps']
    clk = [a['median'] for a in b['aiclk']]
    mn = [a['min'] for a in b['aiclk']]
    print(f'##### {sys.argv[1]}  n={b["n"]}  reps={reps}  arms={b["arms"]}  '
          f'lambda={b["sync_floor_s"]["median"] * 1e6:.2f} us')
    print(f'AICLK median/window {clk}  min {min(mn)}   loadavg1 {b.get("load")}')
    for a, c in b.get('branch', {}).items():
        print(f'branch {a}: {dict(c)}')
    for arm in b['arms']:
        am = sum(sum(b['by_arm'][arm][m]['alias_misses'].values()) for m in ('sync', 'free'))
        ed = sum(sum(b['by_arm'][arm][m]['edge'].values()) for m in ('sync', 'free'))
        print(f'edges {arm}: {ed} attributed, {am} alias misses '
              f'({100.0 * am / (am + ed) if am + ed else 0:.2f} %)')
    print()
    for arm in b['arms']:
        rows, lam = rows_for(b, arm)
        edges = edges_for(b, arm)
        dev = lambda v: max(v[0] - v[1] - lam * v[2], 0.0)          # noqa: E731
        for dr in ('fwd', 'bwd'):
            sel = [(k, v) for k, v in rows.items() if k[0] == dr]
            tot = sum(dev(v) for _, v in sel) / reps
            mv = [(k, v) for k, v in sel if k[2] in MOVEVERBS]
            mtot = sum(dev(v) for _, v in mv) / reps
            print(f'=== arm {arm} {dr}: family device {tot * 1e3:.3f} ms/window, '
                  f'moves {mtot * 1e3:.3f} ms ({100 * mtot / tot if tot else 0:.1f} %)')
            print(f'{"ms":>8} {"calls":>6} {"MB":>8} {"GB/s":>7} {"%roof":>6}  '
                  f'{"hd ms":>7}  fam  verb / shapes @ site -> consumers')
            for k in sorted(sel, key=lambda x: -dev(x[1]))[:22]:
                (d, fam, verb, sig, blk), v = k
                d_s, by = dev(v) / reps, (v[3] + v[4]) / reps
                g = by / d_s / 1e9 if d_s > 0 else 0
                body, site = short(sig)
                ks = '||'.join((f'{"evo"}|{blk}|{d}|{fam}', verb, sig))
                cons = edges.get(ks)
                cs = ', '.join(f'{c}x{n}' for c, n in cons.most_common(3)) if cons else '-'
                print(f'{d_s * 1e3:8.3f} {v[2] / reps:6.1f} {by / 1e6:8.2f} {g:7.1f} '
                      f'{g * 1e9 / ROOF * 100:5.1f}% {(d_s - by / ROOF) * 1e3:7.3f}  '
                      f'{fam[8:]:4s} {verb}')
                print(f'{"":>40}{body[:170]}')
                print(f'{"":>40}@ {site}   -> {cs}')
            print()


if __name__ == '__main__':
    main()
