#!/usr/bin/env python3
"""What a zero in the acceptance pair can and cannot mean, computed before the counts exist.

Three rows read a zero accept rate above 288 tokens and the campaign nearly wrote them up as a
regression. At the only rate anyone has -- 7 of 31 trajectories accepted at 288 -- a single
trajectory rejects 77.4 % of the time with nothing changed at all, so a short zero is the modal
outcome of a healthy pipeline. This prints the three numbers a verdict needs: how likely a zero is
under the baseline, the Clopper-Pearson interval a zero leaves open, and the Fisher power the pair
actually has to call an alarm.

The asymmetry it exposes is the point. At n = 10 an arm the pair is well powered for the GO branch
(P(arm B accepts at least one) = 0.92) and almost powerless for the alarm branch (0.053), because
the control itself only accepts about 2.3 of 10. Say that in advance or it reads as an excuse.
"""
from math import comb

P0_NUM, P0_DEN = 7, 31


def cp_upper(n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson upper bound for 0 of n: 1 - (alpha/2) ** (1/n), exactly."""
    return 1.0 - (alpha / 2) ** (1.0 / n)


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact: P(arm A accepts a or more | the two arms share a rate).

    The upper hypergeometric tail, which is what the RULE pre-committed to adjudicate the alarm
    branch on. Until 2026-10-02 this summed every table no more probable than the observed one,
    which is the TWO-sided test: it reproduced scipy two-sided to 4 dp on the whole 0..10 table.
    The error was conservative, since a two-sided p is never smaller than the one-sided p in the
    expected direction, so it could not have raised a false alarm and the alarm has never fired
    under either. But it mislabelled the statistic the verdict is read from and it understated the
    power. It moves the pre-registered threshold: arm B at 0 of 10 reaches p < 0.05 once arm A
    accepts 4 of 10, where the two-sided test needed 5.
    """
    n, r1, c1 = a + b + c + d, a + b, a + c
    tot = comb(n, c1)
    return sum(comb(r1, x) * comb(n - r1, c1 - x) / tot
               for x in range(a, min(r1, c1) + 1))


def power(n: int, p0: float, alpha: float = 0.05) -> float:
    """P(we call the alarm) when arm Bs true rate is 0 and arm As is p0."""
    return sum(comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
               for k in range(n + 1) if fisher_one_sided(k, n - k, 0, n) < alpha)


if __name__ == "__main__":
    p0 = P0_NUM / P0_DEN
    print(f"baseline p0 = {P0_NUM}/{P0_DEN} = {p0:.4f} (accepted trajectories at 288 tokens)")
    for n in (1, 4, 6, 10):
        print(f"n={n:2d}  P(0 of n | p0)={(1 - p0) ** n:.4f}  "
              f"P(>=1 | p0)={1 - (1 - p0) ** n:.4f}  "
              f"CP95 for 0 of n=[0, {cp_upper(n):.4f}]  "
              f"Fisher power(arm B truly 0)={power(n, p0):.3f}")
    print("\narm A accepted / 10 against arm B 0 of 10:")
    for k in range(11):
        print(f"  {k:2d}  Fisher p = {fisher_one_sided(k, 10 - k, 0, 10):.4f}")
