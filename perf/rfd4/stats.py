"""Two-sample tests for the RFD4 design-level grade, without scipy.

Two tests are enough for the whole grade:

* `ks_2samp` for a continuous metric (refold RMSD, i_pAE, radius of gyration, ...). n is 30-ish
  per arm, where the asymptotic Kolmogorov p-value is only approximate, so the p-value that
  counts is an exact permutation p-value with a fixed seed. The asymptotic one is reported
  beside it as a cross-check, never as the verdict.
* `fisher_exact_greater` for a rate (a pass criterion, "how many designs cleared the bar"),
  one-sided on the direction that matters: the TT arm dropping below the reference arm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_PERMUTATIONS = 20000
DEFAULT_SEED = 20261021


@dataclass
class KSResult:
    d: float  # KS statistic
    p: float  # exact permutation p-value (the one the verdict uses)
    p_asymp: float  # asymptotic Kolmogorov p-value, cross-check only
    n_tt: int
    n_ref: int
    median_tt: float
    median_ref: float

    @property
    def delta_median(self) -> float:
        return self.median_tt - self.median_ref

    def as_dict(self) -> dict:
        return {
            "d": round(self.d, 6),
            "p": round(self.p, 6),
            "p_asymp": round(self.p_asymp, 6),
            "n_tt": self.n_tt,
            "n_ref": self.n_ref,
            "median_tt": _round(self.median_tt),
            "median_ref": _round(self.median_ref),
            "delta_median": _round(self.delta_median),
        }


def _round(x: float, nd: int = 6) -> float:
    return float("nan") if x != x else round(float(x), nd)


def kolmogorov_q(x: float) -> float:
    """Q_KS(x) = 2 sum_{k>=1} (-1)^(k-1) exp(-2 k^2 x^2); Q(0)=1, Q(1)=0.26999967."""
    if x <= 0:
        return 1.0
    if x < 0.2:  # the alternating series converges slowly here; the dual form does not
        y = math.exp(-1.233700550136170 / (x * x))
        return 1.0 - 2.506628274631001 / x * (y + y**9 + y**25 + y**49)
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * (-1) ** (k - 1) * math.exp(-2.0 * k * k * x * x)
        total += term
        if abs(term) < 1e-12:
            break
    return min(1.0, max(0.0, total))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic, ties handled by evaluating both ECDFs on the pooled sample."""
    pooled = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), pooled, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), pooled, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_2samp(
    tt: np.ndarray,
    ref: np.ndarray,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> KSResult:
    tt = np.asarray(tt, dtype=float)
    ref = np.asarray(ref, dtype=float)
    tt = tt[np.isfinite(tt)]
    ref = ref[np.isfinite(ref)]
    if tt.size == 0 or ref.size == 0:
        return KSResult(float("nan"), float("nan"), float("nan"), tt.size, ref.size, float("nan"), float("nan"))
    d = _ks_statistic(tt, ref)
    n1, n2 = tt.size, ref.size
    en = math.sqrt(n1 * n2 / (n1 + n2))
    p_asymp = kolmogorov_q((en + 0.12 + 0.11 / en) * d)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([tt, ref])
    n = pooled.size
    hits = 0
    for _ in range(permutations):
        perm = rng.permutation(pooled)
        if _ks_statistic(perm[:n1], perm[n1:]) >= d - 1e-12:
            hits += 1
    # (hits+1)/(permutations+1) keeps the p-value from ever being exactly 0, which would make a
    # bar of p >= 0.05 depend on how many permutations we happened to draw.
    p = (hits + 1) / (permutations + 1)
    return KSResult(d, p, p_asymp, n1, n2, float(np.median(tt)), float(np.median(ref)))


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_greater(tt_pass: int, tt_n: int, ref_pass: int, ref_n: int) -> float:
    """One-sided Fisher exact p for "the TT arm's pass rate is BELOW the reference arm's".

    Small p means the TT arm really did lose designs. A TT arm at or above the reference rate
    gets a large p and passes, which is what we want: a better arm is not a regression.
    """
    a, b = tt_pass, tt_n - tt_pass
    c, d = ref_pass, ref_n - ref_pass
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    total = row1 + row2
    if total == 0 or col1 == 0 or col2 == 0:
        return 1.0
    denom = _log_choose(total, col1)
    p = 0.0
    lo = max(0, col1 - row2)
    for x in range(lo, min(row1, col1) + 1):
        if x > a:  # only the tail at or below the observed TT pass count
            continue
        p += math.exp(_log_choose(row1, x) + _log_choose(row2, col1 - x) - denom)
    return min(1.0, p)
