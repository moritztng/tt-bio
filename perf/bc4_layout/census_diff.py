#!/usr/bin/env python3
"""Price a change from two `bc2_evo_devprof.py --census` runs: every op keyed by (phase, name, signature), the rows
whose call count differs between the arms are what the change added or removed, and their synced time is summed
per arm with the per-call floor taken off (`--floor-us`, as `dm_census.py`). Rows with equal counts are the same
work and are left out, so load noise on unchanged ops does not enter the difference.
    python perf/bc4_layout/census_diff.py main/devprof.json branch/devprof.json [more pairs ...]
Pairs are (main, branch) in order; with several pairs the per-pair differences and their mean are printed.
"""
import argparse
import collections
import json


def load(path):
    d = collections.defaultdict(lambda: [0, 0.0])
    for o in json.load(open(path))["census"]["ops"]:
        k = (o["phase"], o["name"], o["sig"][:160])
        d[k][0] += o["calls"]
        d[k][1] += o["synced_s"]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("devprofs", nargs="+")
    p.add_argument("--floor-us", type=float, default=30.0)
    p.add_argument("--top", type=int, default=14)
    a = p.parse_args()
    assert len(a.devprofs) % 2 == 0, "pairs of main, branch"
    diffs = []
    for m_path, b_path in zip(a.devprofs[::2], a.devprofs[1::2]):
        m, b = load(m_path), load(b_path)
        changed = [k for k in set(m) | set(b) if m.get(k, [0])[0] != b.get(k, [0])[0]]
        corr = lambda r: max(0.0, r[1] - r[0] * a.floor_us * 1e-6)
        sm = sum(corr(m[k]) for k in changed if k in m)
        sb = sum(corr(b[k]) for k in changed if k in b)
        tot = lambda d: sum(v[1] for v in d.values())
        print(f"{m_path} -> {b_path}")
        print(f"  changed rows {len(changed)}: main {sm:.4f} s, branch {sb:.4f} s, branch - main {sb - sm:+.4f} s/it "
              f"(corrected at {a.floor_us:.0f} us/call); whole census synced {tot(m):.3f} -> {tot(b):.3f}")
        rows = sorted(changed, key=lambda k: -abs(corr(b.get(k, [0, 0.0])) - corr(m.get(k, [0, 0.0]))))
        for k in rows[:a.top]:
            mc, bc = m.get(k, [0, 0.0]), b.get(k, [0, 0.0])
            print(f"   {corr(bc) - corr(mc):+.4f}  calls {mc[0]:4d} -> {bc[0]:4d}  {k[0]:<18s} {k[1]} {k[2][:80]}")
        diffs.append(sb - sm)
    if len(diffs) > 1:
        print(f"mean branch - main {sum(diffs) / len(diffs):+.4f} s/it over {len(diffs)} pairs: {[round(x, 4) for x in diffs]}")


if __name__ == "__main__":
    main()
