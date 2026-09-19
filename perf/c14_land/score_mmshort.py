#!/usr/bin/env python3
"""Score a bracketed fold A/B against the pre-registered stop, and refuse it when it does not clear.

The stop rule is `c14-matmul-ceiling`'s, inherited verbatim rather than restated: **no book at or
above the session's own A/A half-width, on a session whose A/A half-width is itself smaller than the
effect.** Three sessions of that lever were refused by it (f1 retracted, f3 +0.0470 s against a
0.0605 s half-width, f4 discarded), and the rule is not widened here because the fourth session is
ours.

Input is `apb_fold_ab.py --bracket` output: each block is base[0], on[1], base[2].

    per-block A/B delta   mean(base[0], base[2]) - on[1]     positive = the lever saves seconds
    per-block A/A delta   base[2] - base[0]                  same n, same arm separation

Both are paired per block, so a level shift from a stationary neighbour on the other board pair
cancels and a linear drift across a block is removed by bracketing. What does NOT cancel is a
non-stationary neighbour, which is why `pair_channel_quiet.py` keeps a live release_gate a hard
fail rather than a reported one.

Each arm's value is the MEDIAN of its own folds, not the mean: a fold that hits a host stall is a
one-sided outlier and the median is the estimator the rest of this campaign already uses.

    exit 0  BOOKS -- the delta clears its own A/A half-width
    exit 1  REFUSED -- and the reason, which is a finding
"""
import argparse
import json
import statistics as st
import sys

# two-sided 95 % t for small n, indexed by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t95(n: int) -> float:
    return T95.get(n - 1, 1.96)


def halfwidth(xs):
    if len(xs) < 2:
        return float("inf")
    return t95(len(xs)) * st.stdev(xs) / len(xs) ** 0.5


def folds_of(row):
    r = row.get("result") or {}
    v = [f["fold_s"] for f in r.get("folds", []) if f.get("fold_s")]
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--size", default="512")
    a = ap.parse_args()
    d = json.load(open(a.json))
    if d.get("aborted"):
        print("REFUSED: %s" % d["aborted"])
        return 1

    blocks = {}
    for row in d.get("blocks", []):
        if row["size"] != a.size or row.get("returncode") != 0:
            continue
        v = folds_of(row)
        if not v:
            continue
        blocks.setdefault(row["block"], {})[row.get("pos", 0)] = st.median(v)

    ab, aa, kept = [], [], []
    for b in sorted(blocks):
        p = blocks[b]
        if not {0, 1, 2} <= set(p):
            print("block %d incomplete (%s) -- dropped whole, no subset rescue"
                  % (b, sorted(p)))
            continue
        base = (p[0] + p[2]) / 2.0
        ab.append(base - p[1])
        aa.append(p[2] - p[0])
        kept.append(b)

    if len(ab) < 3:
        print("REFUSED: %d complete blocks, too few to estimate a floor" % len(ab))
        return 1

    d_ab, h_ab = st.mean(ab), halfwidth(ab)
    d_aa, h_aa = st.mean(aa), halfwidth(aa)
    print("blocks kept          %s (n=%d)" % (kept, len(ab)))
    print("A/B paired delta     %+.4f s   sd %.4f   95%% CI [%+.4f, %+.4f]"
          % (d_ab, st.stdev(ab), d_ab - h_ab, d_ab + h_ab))
    print("A/A paired delta     %+.4f s   sd %.4f   half-width %.4f"
          % (d_aa, st.stdev(aa), h_aa))

    if h_aa >= abs(d_ab):
        print("REFUSED: A/A half-width %.4f s is not smaller than the %+.4f s effect. The session "
              "cannot resolve its own lever; this is a finding about the box, not about the sign."
              % (h_aa, d_ab))
        return 1
    if d_ab - h_ab <= 0:
        print("REFUSED: the A/B 95%% CI includes zero (lower bound %+.4f s)." % (d_ab - h_ab))
        return 1
    print("BOOKS: %+.4f s, CI lower bound %+.4f s, above an A/A half-width of %.4f s"
          % (d_ab, d_ab - h_ab, h_aa))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
