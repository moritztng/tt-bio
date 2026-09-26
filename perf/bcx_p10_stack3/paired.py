#!/usr/bin/env python3
"""Round-level pairing for the palindrome sitting: 16 pairs an arm, t on 15 df.

`bcx-p10-stack2` and `bcx-p10-genq` both report a paired mean with a t, so this row reports the
same statistic on the same design. The sitting is `off cmp on on cmp off`, which has no adjacent
off/on process pair, so `off` is paired with the `on` process mirrored about the centre of the
sitting: p1<->p3 and p6<->p4. Both pairs span the same two process slots in opposite directions,
so a monotone drift cancels between them. `cmp` pairs are adjacent in both directions.
"""
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_p10_rneker'))
from report import rounds                                              # noqa: E402

#: (base process, arm process) pairs. Mirrored about the centre of the sitting.
PAIRS = {('off', 'on'): [('p1_off', 'p3_on'), ('p6_off', 'p4_on')],
         ('off', 'cmp'): [('p1_off', 'p2_cmp'), ('p6_off', 'p5_cmp')],
         ('cmp', 'on'): [('p2_cmp', 'p3_on'), ('p5_cmp', 'p4_on')]}


def main(root):
    root = pathlib.Path(root)
    warm = {}
    for tag in sorted(p.name for p in root.iterdir() if p.name.startswith('p')):
        _, rs = rounds(str(root / tag / 'round_events.json'))
        for r in rs:
            r['host'] = round(r['wall'] - r['dev'], 3)
        warm[tag] = {r['round']: r for r in rs if r['round'] > 1}   # round 1 is the compile

    for (base, arm), procs in PAIRS.items():
        print(f'\n{arm} against {base}, round k of one process against round k of its mirror')
        for col in ('host', 'dev', 'wall'):
            d, b, n = [], [], []
            for pb, pa in procs:
                for k in sorted(set(warm[pb]) & set(warm[pa])):
                    b.append(warm[pb][k][col])
                    n.append(warm[pa][k][col])
                    d.append(b[-1] - n[-1])
            t = st.mean(d) / (st.stdev(d) / len(d) ** 0.5)
            print(f'  {col:<5} {base} {st.median(b):7.3f}  {arm} {st.median(n):7.3f}  '
                  f'median delta {st.median(d):+7.3f} s  ratio {st.median(b) / st.median(n):.4f}x'
                  f'  {arm} faster in {sum(1 for x in d if x > 0)} of {len(d)}'
                  f'  mean {st.mean(d):+.3f} sd {st.stdev(d):.3f} t {t:.2f} on {len(d) - 1} df')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'perf/bcx_p10_stack3/out')
