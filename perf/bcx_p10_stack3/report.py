#!/usr/bin/env python3
"""The three-arm round table for `perf/bcx_p10_stack3/sit.sh`.

`perf/bcx_p10_genq/report.py` with three arms instead of two and the lever check made part of
the report rather than a thing a reader is trusted to do. The arm is the PROCESS TAG: every arm
runs the same composed round and they differ only in the two env-var levers, so the stamp alone
cannot name the arm.

Host is the round's wall minus its device seconds. `cmp` is host-only and `on` adds a device-only
kernel on top of it, so the two columns say which of the two carries a blended win.
"""
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_p10_rneker'))
from report import rounds                                              # noqa: E402

ARMS = ('off', 'cmp', 'on')
#: What each arm must read at every boundary of every one of its rounds. Anything else and the
#: arm is a blend of two arms, and its median is not a measurement of either.
EXPECT = {'off': {'genq_compact': False, 'taped_channel_move': False},
          'cmp': {'genq_compact': True, 'taped_channel_move': False},
          'on': {'genq_compact': True, 'taped_channel_move': True}}
ALWAYS = {'mm_layout': True, 'triatt_bw': True, 'triatt_hifi': True, 'rne_kernel': True}


def _levers(dump, arm):
    """Every boundary of this process, checked. Returns the rounds that disagreed."""
    want = {**ALWAYS, **EXPECT[arm]}
    bad = []
    for e in dump['events']:
        if e['kind'] not in ('round_start', 'round_stop'):
            continue
        got = {k[6:]: v for k, v in (e.get('reach') or {}).items() if k.startswith('lever_')}
        if not got:
            bad.append((e['round'], 'NO LEVER STAMP'))
        elif any(got.get(k) != v for k, v in want.items()):
            bad.append((e['round'], got))
    return bad


def main(*paths):
    rows, bad, clk = [], [], {}
    for i, path in enumerate(sorted(paths)):
        arm = pathlib.Path(path).parent.name.split('_')[1]
        assert arm in ARMS, f'{path}: tag does not name an arm'
        d, rs = rounds(path)
        bad += [(path, r, g) for r, g in _levers(d, arm)]
        for r in rs:
            r.update(proc=i, arm=arm, round=r['round'] + 100 * i,
                     host=round(r['wall'] - r['dev'], 3))
        rows += rs

    print(f"{'rnd':>4} {'arm':<4} {'wall':>8} {'host':>7} {'dev':>7} {'fwd':>7} {'bwd':>7} "
          f"{'clkmed':>7} {'clkmin':>7} {'load1':>6} {'served':>7}")
    for r in rows:
        print(f"{r['round']:>4} {r['arm']:<4} {r['wall']:>8.3f} {r['host']:>7.3f} "
              f"{r['dev']:>7.3f} {r['fwd']:>7.3f} {r['bwd']:>7.3f} {str(r['clkmed']):>7} "
              f"{str(r['clkmin']):>7} {str(r['load1']):>6} {str(r['fold']):>7}")

    warm, seen = [], set()
    for r in rows:
        if (r['arm'], r['proc']) in seen:
            warm.append(r)
        else:
            seen.add((r['arm'], r['proc']))     # this process's compile round, dropped
    arms = {a: [r for r in warm if r['arm'] == a] for a in ARMS}

    print('\nwarm medians, round 1 of every PROCESS dropped as its compile round')
    print(f"{'arm':<4} {'n':>3} {'host':>8} {'dev':>8} {'wall':>8} {'fwd':>8} {'bwd':>8} "
          f"{'clk med':>8} {'clk min':>8} {'load1':>7}")
    for a in ARMS:
        rs = arms[a]
        if not rs:
            continue
        m = lambda k: round(st.median([r[k] for r in rs]), 3)           # noqa: E731
        clk[a] = (st.median([r['clkmed'] for r in rs]), min(r['clkmin'] for r in rs))
        print(f"{a:<4} {len(rs):>3} {m('host'):>8.3f} {m('dev'):>8.3f} {m('wall'):>8.3f} "
              f"{m('fwd'):>8.3f} {m('bwd'):>8.3f} {clk[a][0]:>8} {clk[a][1]:>8} "
              f"{st.median([r['load1'] for r in rs]):>7.2f}")

    # Rank-paired: the k-th warm round of one arm against the k-th of the other. Rounds 2, 4 and
    # 7 of every process carry a heavy forward on EVERY arm, so raw ranges overlap for a reason
    # that is not noise and the paired statistic is the one to read.
    for base, arm in (('off', 'on'), ('off', 'cmp'), ('cmp', 'on')):
        if not (arms[base] and arms[arm]):
            continue
        print(f'\nrank-paired, {arm} against {base}')
        for col in ('host', 'dev', 'wall'):
            o = sorted(r[col] for r in arms[base])
            n = sorted(r[col] for r in arms[arm])
            k = min(len(o), len(n))
            d = [o[i] - n[i] for i in range(k)]
            print(f"  {col:<5} {base} {st.median(o):7.3f}  {arm} {st.median(n):7.3f}  "
                  f"delta {st.median(d):+7.3f} s  ratio {st.median(o)/st.median(n):.4f}x  "
                  f"{arm} faster in {sum(1 for x in d if x > 0)} of {k}")

    # POSITION PAIRING. Every process has the same shape -- rounds 2, 4 and 7 carry a forward
    # of 3.3-4.5 s against 2.1-2.5 elsewhere, on EVERY arm -- so the raw spread is BindCraft 2's
    # own per-round structure and not the lever. The sitting is a palindrome, so each arm has
    # two processes at mirrored positions; averaging them and pairing position against position
    # removes both the round structure and any drift over the sitting.
    by_pos = {}
    for r in rows:
        by_pos.setdefault((r['arm'], r['round'] % 100), []).append(r)
    positions = sorted({q for a, q in by_pos if q != 1})
    for base, arm in (('off', 'on'), ('off', 'cmp'), ('cmp', 'on')):
        print(f'\nposition-paired, {arm} against {base}, mean of the arm\'s two processes')
        print(f"{'pos':>4} " + ' '.join(f'{c + "_" + a:>10}'
                                        for c in ('host', 'dev', 'wall') for a in (base, arm)))
        deltas = {c: [] for c in ('host', 'dev', 'wall')}
        for q in positions:
            cells = []
            for c in ('host', 'dev', 'wall'):
                b = st.mean([r[c] for r in by_pos[(base, q)]])
                n = st.mean([r[c] for r in by_pos[(arm, q)]])
                deltas[c].append(b - n)
                cells += [f'{b:>10.3f}', f'{n:>10.3f}']
            print(f'{q:>4} ' + ' '.join(cells))
        for c in ('host', 'dev', 'wall'):
            d = deltas[c]
            b = st.mean([st.mean([r[c] for r in by_pos[(base, q)]]) for q in positions])
            n = st.mean([st.mean([r[c] for r in by_pos[(arm, q)]]) for q in positions])
            t = st.mean(d) / (st.stdev(d) / len(d) ** 0.5) if len(d) > 1 and st.stdev(d) else 0
            print(f"  {c:<5} mean delta {st.mean(d):+7.3f} s  ratio {b / n:.4f}x  "
                  f"{arm} faster at {sum(1 for x in d if x > 0)} of {len(d)} positions  "
                  f"t {t:.2f} on {len(d) - 1} df")

    print('\nlevers, every boundary of every process')
    print('  OK, all six read the armed value on every round' if not bad
          else json.dumps(bad, indent=1))
    if bad:
        raise SystemExit('a lever disagreed with its arm: no median here is a measurement')


if __name__ == '__main__':
    main(*sys.argv[1:])
