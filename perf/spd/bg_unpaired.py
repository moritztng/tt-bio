"""Unpaired seed-cluster bootstrap of BoltzGen design_iptm: test runs vs base runs.

A numeric lever re-rolls every design trajectory, so same-seed pairs are not pairs. This resamples whole seeds (runs)
with replacement on both sides and reports the mean difference with a 95 % interval.

    python perf/spd/bg_unpaired.py --base 'grade/bh-ee[0-7]' --test 'grade/bh-bD[0-7]' [--test ...]
"""
import argparse, glob
import numpy as np, pandas as pd


def runs(pattern, metric):
    out = {}
    for d in sorted(glob.glob(pattern)):
        m = sorted(glob.glob(d + "/bg/**/aggregate_metrics_analyze.csv", recursive=True))
        if m:
            out[d] = pd.read_csv(m[0])[metric].dropna().values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", action="append", required=True)
    ap.add_argument("--metric", default="design_iptm")
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()
    B = list(runs(a.base, a.metric).values())
    print(f"base {a.base}: {len(B)} runs, per-run means {[round(v.mean(), 3) for v in B]}")
    for pat in a.test:
        T = list(runs(pat, a.metric).values())
        rng = np.random.default_rng(0)
        ds = [np.concatenate([T[i] for i in rng.integers(0, len(T), len(T))]).mean()
              - np.concatenate([B[i] for i in rng.integers(0, len(B), len(B))]).mean() for _ in range(a.draws)]
        d = np.concatenate(T).mean() - np.concatenate(B).mean()
        print(f"{pat}: {len(T)} runs, {sum(map(len, T))} designs, {a.metric} {d:+.4f} "
              f"[{np.percentile(ds, 2.5):+.4f}, {np.percentile(ds, 97.5):+.4f}], per-run {[round(v.mean(), 3) for v in T]}")


if __name__ == "__main__":
    main()
