#!/usr/bin/env python3
"""Compare the per-sample DockQ distributions of two runs of the same target.

Written to answer one release-gated question: did the boltz2 diffusion chunk-width change
(b62301f5, max_parallel_samples 3 -> 8) shift what the sampler draws? Chunk-boundary float
non-associativity varies with the width, so "RNG order untouched" is an argument, not a
measurement. Two independent runs of the same target at different widths give the
measurement: if the quantiles agree through the body of the distribution and no threshold
separates the two runs beyond chance, the width is distribution-neutral.

  python3 scripts/abag_xm_label_dist_compare.py A/labels.json B/labels.json [--bars 0.23,0.49]

Reports quantiles side by side and an exact Fisher test at EVERY bar (all of them, so no
threshold is cherry-picked after the fact), plus the pooled-maximum exchangeability check.
"""
import argparse
import json
from math import comb


def load(path):
    d = json.loads(open(path).read())
    ss = d["samples"] if isinstance(d, dict) and "samples" in d else d
    v = [s["dockq"]["dockq"] for s in ss
         if isinstance(s.get("dockq"), dict) and s["dockq"].get("dockq") is not None]
    if not v:
        raise SystemExit(f"{path}: no scored samples")
    return sorted(v)


def quantile(v, p):
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]


def fisher_two_sided(a, n1, b, n2):
    """Exact two-sided p for a 2x2 table, by summing tables no more likely than observed."""
    k, N = a + b, n1 + n2
    if k == 0:
        return None
    denom = comb(N, k)
    def pr(i):
        return comb(n1, i) * comb(n2, k - i) / denom
    obs = pr(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(pr(i) for i in range(lo, hi + 1) if pr(i) <= obs * (1 + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_a")
    ap.add_argument("labels_b")
    ap.add_argument("--bars", default="0.05,0.075,0.10,0.15,0.20,0.23,0.49,0.80",
                    help="comma-separated DockQ thresholds; ALL are reported")
    a = ap.parse_args()

    va, vb = load(a.labels_a), load(a.labels_b)
    print(f"A = {a.labels_a}  n={len(va)}")
    print(f"B = {a.labels_b}  n={len(vb)}")
    print("\nquantile        A          B")
    for p in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0):
        print(f"  {p:5.2f}     {quantile(va, p):8.4f}   {quantile(vb, p):8.4f}")

    pooled_max = max(va[-1], vb[-1])
    holder, n_holder = ("A", len(va)) if va[-1] >= vb[-1] else ("B", len(vb))
    print(f"\npooled max = {pooled_max:.4f}, in run {holder}; under exchangeability that run "
          f"holds it with probability {n_holder}/{len(va) + len(vb)} = "
          f"{n_holder / (len(va) + len(vb)):.4f}")

    print("\nbar       A        B      Fisher two-sided p")
    for bar in [float(x) for x in a.bars.split(",")]:
        ca = sum(1 for x in va if x >= bar)
        cb = sum(1 for x in vb if x >= bar)
        p = fisher_two_sided(ca, len(va), cb, len(vb))
        tail = "(no successes in either run)" if p is None else f"p={p:.3f}"
        print(f"  {bar:5.3f}  {ca:4d}/{len(va):<5d} {cb:4d}/{len(vb):<5d}  {tail}")


if __name__ == "__main__":
    main()
