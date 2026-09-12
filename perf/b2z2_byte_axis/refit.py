#!/usr/bin/env python3
"""Re-fit the per-site wait model on `b2z2-bh-tile-census`'s own aggregates, with the read
side of the gated channel move corrected against its kernel source.

The census sizes a `ttnn.generic_op` read from every input slot's WHOLE tensor. At
`reblock_permute_gated` that slot is the four-way fused projection [1, N, N, 4*slice_c], and
`kernels/reblock_permute_gated/reader_reblock_permute_gated.cpp` reads exactly TWO of those four
channel slices (`p_off + ct` and `g_off + ct`, one page each per output tile). So the read is
2/4 of what is counted -- at the block's single largest byte site.

Leg 1 reproduces the published fit from the committed aggregates, so the correction is applied to
a model that is demonstrably theirs and not a re-derivation of my own.
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).with_name("results")


def lstsq(X, y):
    """Normal equations, no intercept -- the published fit has none (a site with no bytes waits 0)."""
    n, k = len(X), len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    v = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    for c in range(k):                      # gaussian elimination with partial pivot
        p = max(range(c, k), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        v[c], v[p] = v[p], v[c]
        for r in range(k):
            if r == c or A[c][c] == 0:
                continue
            f = A[r][c] / A[c][c]
            for j in range(c, k):
                A[r][j] -= f * A[c][j]
            v[r] -= f * v[c]
    return [v[i] / A[i][i] if A[i][i] else 0.0 for i in range(k)]


def r2(X, y, b):
    m = sum(y) / len(y)
    pred = [sum(x * c for x, c in zip(row, b)) for row in X]
    ss = sum((a - p) ** 2 for a, p in zip(y, pred))
    tt = sum((a - m) ** 2 for a in y)
    return 1 - ss / tt


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):                   # average ties
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** .5
    vb = sum((y - mb) ** 2 for y in rb) ** .5
    return cov / (va * vb)


def models(S):
    return {
        "tile arrivals only": lambda s: [s["arrivals_pc"]],
        "tile-pair MACs only": lambda s: [s["macs_pc"]],
        "DRAM bytes only": lambda s: [s["dram_rd"]],
        "DRAM + L1 bytes": lambda s: [s["dram_rd"], s["l1_rd"]],
        "DRAM + L1 + per-program": lambda s: [s["dram_rd"], s["l1_rd"], s["n"]],
    }


def report(tag, S):
    y = [s["wait_ns"] for s in S]
    out = [f"  {tag}: {len(S)} sites, wait {sum(y)/1e6:.4f} ms, "
           f"delivered {sum(s['dram_rd']+s['l1_rd'] for s in S)/1e6:,.1f} MB"]
    for name, fx in models(S).items():
        X = [fx(s) for s in S]
        b = lstsq(X, y)
        out.append(f"    R2 = {r2(X, y, b):8.4f}   {name:26s} " + "  ".join(f"{c:.6g}" for c in b))
    for name, key in (("bytes delivered (DRAM + L1 read)", lambda s: s["dram_rd"] + s["l1_rd"]),
                      ("CB tile arrivals per core", lambda s: s["arrivals_pc"])):
        out.append(f"    spearman {spearman([key(s) for s in S], y):+.3f}   {name}")
    return "\n".join(out)


def gated(s):
    """The gated channel move: one wide input, output is one slice, kernel reads two slices."""
    return (s["kernel"] == "compute_reblock_permute_gated"
            and len(s["inputs"]) == 1 and s["inputs"][0].count("x") == 3)


def main(path):
    d = json.load(open(path))["PairformerLayer"]
    S = [s for s in d["sites"] if s["wait_ns"] > 0 and s["arrivals_pc"] > 0]
    L = ["=== leg 1: the published model, on the published aggregates", report("as published", S)]

    hits = [s for s in S if gated(s)]
    L.append(f"\n=== leg 2: gated channel move read corrected 4 slices -> 2 ({len(hits)} site(s))")
    for s in hits:
        out_t = 1                            # one slice out per call
        old = s["dram_rd"] + s["l1_rd"]
        wide = int(s["inputs"][0].split("x")[-1])
        slice_c = int(s["output"].split("x")[1])
        f = 2 * slice_c / wide
        for k in ("dram_rd", "l1_rd"):
            s[k] *= f
        L.append(f"    {s['inputs'][0]} -> {s['output']}, n={s['n']:.0f}: "
                 f"{old/1e6:,.1f} MB -> {(s['dram_rd']+s['l1_rd'])/1e6:,.1f} MB  (x{f:.3f})")
    L.append(report("corrected", S))
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RES / "tile_census.json"))
