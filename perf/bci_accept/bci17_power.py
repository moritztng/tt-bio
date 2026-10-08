"""#17's statistics, which do not depend on either arm finishing.

The reporter grades on an acceptance count at 8 trajectories per arm. This says what that design
can and cannot settle, so the answer quotes a measured power rather than an adjective.
"""
from scipy.stats import binom, fisher_exact


def p(a, b, n):
    """Fisher exact, two-sided, for `a of n` accepted against `b of n`."""
    return fisher_exact([[a, n - a], [b, n - b]], alternative="two-sided")[1]


def power(n, pa=0.0, pb=0.375, alpha=0.05):
    """P(reject) at the reporter's own observed rates, by exact enumeration of both arms."""
    total = 0.0
    for a in range(n + 1):
        wa = binom.pmf(a, n, pa)
        if not wa:
            continue
        for b in range(n + 1):
            wb = binom.pmf(b, n, pb)
            if wb and p(a, b, n) < alpha:
                total += wa * wb
    return total


if __name__ == "__main__":
    print(f"reporter's 0 of 8 vs 3 of 8:      p = {p(0, 3, 8):.4f}")
    print(f"ours if card matches host, 3 v 3: p = {p(3, 3, 8):.4f}  (the test is silent here)")
    print(f"most extreme at n=8, 6 v 0:       p = {p(6, 0, 8):.4f}")
    for n in (8, 13, 16, 20):
        print(f"n={n:3d} per arm: power {power(n):.3f}")
