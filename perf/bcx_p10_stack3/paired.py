#!/usr/bin/env python3
"""Round-level pairing for the palindrome sittings: t on the paired differences.

`bcx-p10-stack2` and `bcx-p10-genq` both report a paired mean with a t, so this row reports the
same statistic on the same design.

A sitting is a palindrome of six processes, so each arm holds two of them at mirrored positions.
Pairing the arms' processes in position order -- the earlier of one against the earlier of the
other, the later against the later -- gives two pairs that span the same slot distance in
opposite directions, so a monotone drift over the sitting cancels between them. Sitting 2 runs
the palindrome reversed, so pooling the two also cancels anything that depends on which end of a
sitting an arm sat at.

    paired.py <out-dir> [tag-prefix ...]        default prefixes: p q
"""
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'perf' / 'bcx_p10_rneker'))
from report import rounds                                              # noqa: E402

ARMS = ('off', 'cmp', 'on')
COMPARISONS = (('off', 'on'), ('off', 'cmp'), ('cmp', 'on'))


def load(root, prefixes):
    """{prefix: {arm: [(position, {round: row}), ...]}}, warm rounds only."""
    out = {}
    for tag in sorted(p.name for p in pathlib.Path(root).iterdir() if p.is_dir()):
        if tag[0] not in prefixes or '_' not in tag:
            continue
        pos, arm = int(tag[1:tag.index('_')]), tag[tag.index('_') + 1:]
        assert arm in ARMS, f'{tag}: tag does not name an arm'
        _, rs = rounds(str(pathlib.Path(root) / tag / 'round_events.json'))
        for r in rs:
            r['host'] = round(r['wall'] - r['dev'], 3)
        # Round 1 of every process is its compile round and is dropped.
        out.setdefault(tag[0], {}).setdefault(arm, []).append(
            (pos, {r['round']: r for r in rs if r['round'] > 1}))
    return out


def main(root='perf/bcx_p10_stack3/out', *prefixes):
    sittings = load(root, prefixes or ('p', 'q'))
    print(f"pooled over sittings {sorted(sittings)}")
    for base, arm in COMPARISONS:
        print(f'\n{arm} against {base}')
        cols = {c: ([], [], []) for c in ('host', 'dev', 'wall')}       # base, arm, delta
        for sit in sittings.values():
            if base not in sit or arm not in sit:
                continue
            for (_, rb), (_, ra) in zip(sorted(sit[base]), sorted(sit[arm])):
                for k in sorted(set(rb) & set(ra)):
                    for c, (b, n, d) in cols.items():
                        b.append(rb[k][c])
                        n.append(ra[k][c])
                        d.append(rb[k][c] - ra[k][c])
        for c, (b, n, d) in cols.items():
            t = st.mean(d) / (st.stdev(d) / len(d) ** 0.5)
            print(f'  {c:<5} {base} {st.median(b):7.3f}  {arm} {st.median(n):7.3f}  '
                  f'ratio {st.median(b) / st.median(n):.4f}x  '
                  f'median delta {st.median(d):+7.3f} s  '
                  f'{arm} faster in {sum(1 for x in d if x > 0)} of {len(d)}  '
                  f'mean {st.mean(d):+.3f} sd {st.stdev(d):.3f} t {t:.2f} on {len(d) - 1} df')


if __name__ == '__main__':
    main(*sys.argv[1:])
