"""Compare two models' sample-scaling curves on a paired target panel, with bootstrap CIs.

This regenerates the dataset's headline. It exists because that headline was twice stated ahead of
its uncertainty, and once computed over target sets that silently differed between the two models.

WHAT "PAIRED" HAS TO MEAN HERE
------------------------------
A target contributes only if **both** models folded it, it has a reference structure, and both
arms carry the full ``--depth`` samples for it. The third condition is the one that bites: an arm
that produced only 8 samples for a target cannot yield a best-of-16, so a naive "average over
targets where k <= n" quietly drops that target from one model's column and keeps it in the
other's. That is not a paired comparison, and it inflated one model's oracle gain by ~17% before
this script existed.

The bootstrap resamples *targets* (not samples) and applies the **same** resample to both models,
so the difference between them is estimated on a common panel and the models' correlated
per-target difficulty cancels.

WHAT THE TWO CURVES MEAN
------------------------
* ``oracle`` -- the best structure actually present in k samples, ranked by the reference metric.
  It is the ceiling sampling can buy.
* ``user``   -- the structure a user would choose, ranking the same k samples by the model's own
  ``confidence_score`` and reporting its true accuracy. It is what sampling actually delivers.

Both are exact order statistics over the N draws, not bootstrap estimates:

    E[max over a random k-subset] = sum_i v_i * C(N-i-1, k-1) / C(N, k)      (v sorted descending)

so the k-curve costs nothing beyond sorting. Only the CIs are resampled.

Usage:
    python3 scripts/abag_xm/paired_model_comparison.py \
        --arm opendde-abag:<opendde>_samples.parquet \
        --arm protenix-v2:<protenix>_samples.parquet
"""

from __future__ import annotations

import argparse
from math import comb

import numpy as np
import pandas as pd

METRIC = "global_dockq"          # the accuracy reference
RANK_BY = "confidence_score"     # what a user without a reference can rank on


def expected_best_of_k(values: np.ndarray, k: int) -> float:
    """E[max of a random k-subset] of `values`, exactly. `values` need not be sorted."""
    v = np.sort(values)[::-1]
    n = len(v)
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    denom = comb(n, k)
    return float(sum(v[i] * comb(n - i - 1, k - 1) for i in range(n - k + 1)) / denom)


def curves(group: pd.DataFrame, k: int) -> tuple[float, float]:
    """(oracle, user) expected best-of-k for one target."""
    oracle = expected_best_of_k(group[METRIC].to_numpy(), k)
    # Rank by confidence, carry the true metric: the k-subset's top-confidence member.
    by_conf = group.sort_values(RANK_BY, ascending=False)[METRIC].to_numpy()
    n = len(by_conf)
    denom = comb(n, k)
    user = float(sum(by_conf[i] * comb(n - i - 1, k - 1) for i in range(n - k + 1)) / denom)
    return oracle, user


def load(spec: str) -> tuple[str, dict[str, pd.DataFrame]]:
    name, path = spec.split(":", 1)
    df = pd.read_parquet(path)
    scored = df[df[METRIC].notna()]
    return name, {t: g for t, g in scored.groupby("target")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="name:path_to_samples.parquet (give exactly two)")
    ap.add_argument("--depth", type=int, default=16, help="k at which the gain is quoted")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if len(args.arm) != 2:
        raise SystemExit("give exactly two --arm specs")

    (n1, A), (n2, B) = (load(s) for s in args.arm)
    d = args.depth
    shared = sorted(set(A) & set(B))
    full = [t for t in shared if len(A[t]) >= d and len(B[t]) >= d]
    dropped = sorted(set(shared) - set(full))
    print(f"{n1}: {len(A)} scored targets | {n2}: {len(B)} scored targets")
    print(f"shared {len(shared)}; with {d} samples in BOTH arms: {len(full)}")
    if dropped:
        print(f"dropped (an arm has fewer than {d}): {' '.join(dropped)}")
    if not full:
        raise SystemExit("no paired targets")

    print(f"\npaired curves on {len(full)} targets")
    print(f"{'k':>4} {n1+' oracle':>20} {n1+' user':>18} {n2+' oracle':>20} {n2+' user':>18}")
    ks = [k for k in (1, 2, 4, 8, d) if k <= d]
    for k in sorted(set(ks)):
        a = [curves(A[t], k) for t in full]
        b = [curves(B[t], k) for t in full]
        print(f"{k:>4} {np.mean([x[0] for x in a]):>20.4f} {np.mean([x[1] for x in a]):>18.4f}"
              f" {np.mean([x[0] for x in b]):>20.4f} {np.mean([x[1] for x in b]):>18.4f}")

    # Per-target 1 -> d gains, then bootstrap over targets with a shared resample.
    g = {t: (curves(A[t], 1), curves(A[t], d), curves(B[t], 1), curves(B[t], d)) for t in full}
    a_or = np.array([g[t][1][0] - g[t][0][0] for t in full])
    a_us = np.array([g[t][1][1] - g[t][0][1] for t in full])
    b_or = np.array([g[t][3][0] - g[t][2][0] for t in full])
    b_us = np.array([g[t][3][1] - g[t][2][1] for t in full])

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(full), size=(args.boot, len(full)))   # SAME resample for both arms
    bo_a, bu_a = a_or[idx].mean(1), a_us[idx].mean(1)
    bo_b, bu_b = b_or[idx].mean(1), b_us[idx].mean(1)

    def line(label, obs, boot):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:<34} {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

    print(f"\npaired bootstrap, {args.boot} resamples of {len(full)} targets, gain 1 -> {d}")
    line(f"{n1} oracle", a_or.mean(), bo_a)
    line(f"{n2} oracle", b_or.mean(), bo_b)
    line(f"{n2} - {n1}  ORACLE", b_or.mean() - a_or.mean(), bo_b - bo_a)
    line(f"{n1} user", a_us.mean(), bu_a)
    line(f"{n2} user", b_us.mean(), bu_b)
    line(f"{n2} - {n1}  USER", b_us.mean() - a_us.mean(), bu_b - bu_a)

    print(f"\n  P({n2} oracle gain > {n1}) = {(bo_b - bo_a > 0).mean():.3f}")
    print(f"  P({n2} user   gain > {n1}) = {(bu_b - bu_a > 0).mean():.3f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = bo_b / bo_a
    lo, hi = np.nanpercentile(ratio, [2.5, 97.5])
    print(f"  oracle ratio {b_or.mean()/a_or.mean():.2f}x   95% CI [{lo:.2f}, {hi:.2f}]")
    print("\nA difference whose CI spans zero is not a result. Quote the interval, not the point.")


if __name__ == "__main__":
    main()
