"""Smallest one-sided p a rank test can reach with m upstream draws against n device designs.

Distribution-free: under the null that both sides are draws from one distribution, every
interleaving of the m+n values is equally likely, so the most extreme outcome -- all m
upstream values below all n device values -- has probability 1/C(m+n, m). No effect size,
however large, can produce a smaller p. This is a property of the SAMPLE SIZES alone and can
be computed before any data lands.
"""
from math import comb
from scipy.stats import mannwhitneyu

n = 8  # device designs per cell, this row's standard
print(f"device n = {n}")
print(f"{'m':>3} {'C(m+n,m)':>10} {'min one-sided p':>16}  rejects at 0.05?")
for m in range(1, 5):
    c = comb(m + n, m)
    print(f"{m:>3} {c:>10} {1.0/c:>16.4f}  {'YES' if 1.0/c <= 0.05 else 'NO'}")

# Cross-check against scipy on a perfectly separated toy sample, m=1 and m=2.
for m in (1, 2):
    up = [1.0 * i for i in range(m)]          # all below
    dev = [100.0 + i for i in range(n)]       # all above
    u, p = mannwhitneyu(up, dev, alternative="less", method="exact")
    print(f"scipy exact, m={m}, perfect separation: U={u:.0f} p={p:.4f}  (1/C = {1.0/comb(m+n,m):.4f})")
