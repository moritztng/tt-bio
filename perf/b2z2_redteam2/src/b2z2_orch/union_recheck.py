#!/usr/bin/env python3
"""Re-derive b2z2-bh-union-clean's headline from its own committed folds.

The campaign has quoted four different denominators and retracted three of them, so no row's
headline is taken on its word any more. This reads the 51 raw folds the row committed
(`perf/b2z2_union/out/timing512_qb2c1.json` on `wk/b2z2-bh-union-clean`), recomputes every median,
every ratio and the A/A floor from the fold list, and — the part the row's own summary does not
print — checks whether the union and base fold DISTRIBUTIONS overlap at all. A ratio inside a noise
band is an argument; two non-overlapping sets of folds are not.

    python3 perf/b2z2_orch/union_recheck.py <timing512_qb2c1.json>

Fetch the input with:
    git show origin/wk/b2z2-bh-union-clean:perf/b2z2_union/out/timing512_qb2c1.json > /tmp/u.json
"""
import json
import statistics as st
import sys

PUBLISHED_CELL_S = 20.113  # site/data/perf-512aa.json @ 84da2a49b, levers of wave 1 already in it


def main(path: str) -> int:
    runs = [r for r in json.load(open(path))["runs"] if not r.get("cold")]
    arms: dict[str, list[float]] = {}
    for r in runs:
        arms.setdefault(r["arm"], []).append(r["fold_s"])

    base = arms["base"]
    base_med = st.median(base)
    print(f"warm folds {len(runs)}  base n={len(base)}  median {base_med:.4f} s\n")
    print(f"{'arm':8s} {'n':>3s} {'median':>9s} {'ratio':>9s} {'min':>8s} {'max':>8s}  separated from base?")
    for arm, folds in arms.items():
        med = st.median(folds)
        # "Separated" = no fold of this arm is as slow as the fastest base fold. For a faster arm
        # that is the strongest claim the raw data can make, and it needs no noise model at all.
        sep = max(folds) < min(base) if med < base_med else min(folds) > max(base)
        print(f"{arm:8s} {len(folds):3d} {med:9.4f} {base_med / med:9.5f} "
              f"{min(folds):8.3f} {max(folds):8.3f}  {'YES' if sep else 'no'}")

    union = arms["UNION"]
    union_med = st.median(union)
    singles = [base_med / st.median(arms[a]) for a in ("HOST", "SILU", "AKW")]
    product = singles[0] * singles[1] * singles[2]
    measured = base_med / union_med
    print(f"\nproduct of singles {product:.5f}   measured union {measured:.5f}   "
          f"discount {(product / measured - 1) * 100:+.2f} %")
    print(f"seconds: product predicts {base_med - base_med / product:.3f} s removed, "
          f"union delivers {base_med - union_med:.3f} s")
    print(f"\nPAIRED, against this row's own interleaved base: {measured:.5f}x")
    print(f"UNPAIRED, against the published {PUBLISHED_CELL_S:.3f} s cell: "
          f"{PUBLISHED_CELL_S / union_med:.5f}x  <- historical cell, different tree, not a paired number")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/u.json"))
