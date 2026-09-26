#!/usr/bin/env python3
"""The round table for `perf/bcx_p10_genq/ab.sh`, with the HOST column as the claim.

`perf/bcx_p10_rneker/report.py`'s bounds and warm-median rule unchanged, so the two sittings are
comparable, with two differences this row needs:

  * the arm is the PROCESS TAG, not the `rne_kernel` stamp. Both arms here run the kernel; the
    only difference between them is `TT_BIO_GENQ_COMPACT`, which is what this row moves.
  * host is reported, because that is where this lever lands. Host is the round's wall minus its
    device time: the lever removes host microseconds per dispatch and touches no device work, so
    a round column that did not move would mean the dispatches were not reached.
"""
import json
import statistics as st
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_p10_rneker'))
from report import rounds                                              # noqa: E402


def main(*paths):
    rows = []
    for i, path in enumerate(paths):
        arm = 'on' if '/p_on_' in path or path.split('/')[-2].startswith('p_on') else 'off'
        _, rs = rounds(path)
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
    arms = {a: [r for r in warm if r['arm'] == a] for a in ('off', 'on')}

    print('\nwarm medians, round 1 of every PROCESS dropped as its compile round')
    print(f"{'arm':<4} {'n':>3} {'host':>8} {'dev':>8} {'wall':>8} {'fwd':>8} {'bwd':>8} "
          f"{'clk min':>8}")
    for a, rs in arms.items():
        if not rs:
            continue
        m = lambda k: round(st.median([r[k] for r in rs]), 3)           # noqa: E731
        print(f"{a:<4} {len(rs):>3} {m('host'):>8.3f} {m('dev'):>8.3f} {m('wall'):>8.3f} "
              f"{m('fwd'):>8.3f} {m('bwd'):>8.3f} {min(r['clkmin'] for r in rs):>8}")

    # Rank-paired: the k-th warm round of `off` against the k-th of `on`. Rounds 2, 4 and 7 of
    # every process carry a heavy forward on BOTH arms, so raw ranges overlap for a reason that
    # is not noise and the paired statistic is the one to read.
    if arms['off'] and arms['on']:
        print('\nrank-paired, k-th warm round of each arm')
        for col in ('host', 'dev', 'wall'):
            o = sorted(r[col] for r in arms['off'])
            n = sorted(r[col] for r in arms['on'])
            k = min(len(o), len(n))
            d = [o[i] - n[i] for i in range(k)]
            print(f"  {col:<5} off {st.median(o):7.3f}  on {st.median(n):7.3f}  "
                  f"delta {st.median(d):+7.3f} s  ratio {st.median(o)/st.median(n):.4f}x  "
                  f"on faster in {sum(1 for x in d if x > 0)} of {k}")
        # Position-paired: round k of process p against round k of the process of the other arm
        # that ran next to it, which is what cancels a drift over the sitting.
        print('\nposition-paired, round k of process p against round k of the paired process')
        pairs = []
        for r in arms['off']:
            for s in arms['on']:
                if s['round'] % 100 == r['round'] % 100 and abs(s['proc'] - r['proc']) == 1:
                    pairs.append((r, s))
                    break
        for col in ('host', 'dev', 'wall'):
            d = [r[col] - s[col] for r, s in pairs]
            if d:
                print(f"  {col:<5} delta {st.median(d):+7.3f} s over {len(d)} pairs, "
                      f"on faster in {sum(1 for x in d if x > 0)}")


if __name__ == '__main__':
    main(*sys.argv[1:])
