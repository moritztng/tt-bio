"""Unpaired seed-cluster bootstrap of BoltzGen design_iptm: test runs vs base runs.

A numeric lever re-rolls every design trajectory, so same-seed pairs are not pairs. This resamples whole seeds (runs)
with replacement on both sides and reports the mean difference with a 95 % interval.

    python perf/spd/bg_unpaired.py --base 'grade/bh-ee[0-7]' --test 'grade/bh-bD[0-7]' [--test ...]
"""
import argparse, glob
import numpy as np, pandas as pd


def runs(pattern, metric, expect):
    """Designs per run, by run. A run whose analyze CSV holds fewer than `expect` designs is DROPPED, not averaged:
    BoltzGen's analyze step can leave the CSV holding only one chunk of the run (seen 11-02 on wh-en1 = design 0 only
    and wh-en2 = designs 6,7 only, both rc=0 with all 8 inverse-folded structures on disk). Averaging such a run is
    averaging whichever designs survived the last write."""
    out = {}
    for d in sorted(glob.glob(pattern)):
        m = sorted(glob.glob(d + "/bg/**/aggregate_metrics_analyze.csv", recursive=True))
        if not m:
            continue
        v = pd.read_csv(m[0])[metric].dropna().values
        if expect and len(v) < expect:
            print(f"  DROPPED {d}: {len(v)} designs in the analyze CSV, expected {expect}")
            continue
        out[d] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", action="append", required=True)
    ap.add_argument("--metric", default="design_iptm")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--expect", type=int, default=8, help="designs per run; a run with fewer is dropped (0 = keep all)")
    a = ap.parse_args()
    B = list(runs(a.base, a.metric, a.expect).values())
    print(f"base {a.base}: {len(B)} runs, per-run means {[round(v.mean(), 3) for v in B]}")
    for pat in a.test:
        T = list(runs(pat, a.metric, a.expect).values())
        rng = np.random.default_rng(0)
        ds = [np.concatenate([T[i] for i in rng.integers(0, len(T), len(T))]).mean()
              - np.concatenate([B[i] for i in rng.integers(0, len(B), len(B))]).mean() for _ in range(a.draws)]
        d = np.concatenate(T).mean() - np.concatenate(B).mean()
        print(f"{pat}: {len(T)} runs, {sum(map(len, T))} designs, {a.metric} {d:+.4f} "
              f"[{np.percentile(ds, 2.5):+.4f}, {np.percentile(ds, 97.5):+.4f}], per-run {[round(v.mean(), 3) for v in T]}")


if __name__ == "__main__":
    main()
