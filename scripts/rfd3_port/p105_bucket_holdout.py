#!/usr/bin/env python3
"""p105 -- do the bucket edges chosen on one schedule still hold on a schedule they never saw?

p103 chose the key-axis buckets by minimising mean chain cost over ONE design's 199 atom indices,
one target, one seed. That is a fit to the sample, so the prize it reports is an in-sample prize
and the honest question is how much of it survives out of sample.

Takes two p102 harvests. For each, picks the best bucket set in-sample, then scores that same set
on the other one. The gap between the in-sample and out-of-sample prize is the overfit, and the
out-of-sample number is the one worth quoting.
"""
import itertools
import json
import pathlib
import sys

A = pathlib.Path(sys.argv[1])
B = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "perf/p105/holdout.json")
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
HOST_S = {320: 0.447, 608: 0.420, 1216: 0.404, 3040: 0.361}


def load(p):
    d = json.loads(p.read_text())
    meas = {int(q): {int(u): c for u, c in v.items()} for q, v in d["measured_chain"].items()}
    dist = {int(k[1:]): v["umax_tiled"] for k, v in d["report"].items()}
    return d, meas, dist


def priced(Q, meas, us, buckets, dense):
    lo, hi = min(meas[Q]), max(meas[Q])
    c_lo, c_hi = meas[Q][lo], meas[Q][hi]
    slope = (c_hi - c_lo) / (hi - lo)
    tot, nfall = 0.0, 0
    for u in us:
        fit = [b for b in buckets if b >= u]
        if fit:
            tot += c_lo + slope * (fit[0] - lo)
        else:
            tot += dense
            nfall += 1
    return tot / len(us), nfall


def best_set(Q, meas, us, dense, k):
    lo, hi = min(meas[Q]), max(meas[Q])
    cand = sorted({lo, hi} | {u for u in us if lo <= u <= hi})
    best = None
    for n in range(1, k + 1):
        for bs in itertools.combinations(cand, n):
            if bs[-1] != hi:
                continue
            m, _ = priced(Q, meas, us, bs, dense)
            if best is None or m < best[0]:
                best = (m, list(bs))
    return best[1]


def main():
    da, meas, dist_a = load(A)
    db, _, dist_b = load(B)
    dense, cps, steps = da["dense_ms"], da["calls_per_step"], da["steps"]
    prize = lambda m, Q: (dense - m) * cps * steps / 1000.0 - HOST_S.get(Q, 0.0)   # noqa: E731
    print("A = %s (seed %s)   B = %s (seed %s)   dense %.4f ms/call"
          % (A, da.get("seed"), B, db.get("seed"), dense), flush=True)
    out = {}
    for Q in sorted(set(dist_a) & set(dist_b)):
        ua, ub = dist_a[Q], dist_b[Q]
        row = {}
        for name, (fit_us, sco_us) in (("A_on_B", (ua, ub)), ("B_on_A", (ub, ua))):
            bs = best_set(Q, meas, fit_us, dense, K)
            m_in, f_in = priced(Q, meas, fit_us, bs, dense)
            m_out, f_out = priced(Q, meas, sco_us, bs, dense)
            row[name] = dict(buckets=bs,
                             in_sample_net=round(prize(m_in, Q), 3),
                             out_sample_net=round(prize(m_out, Q), 3),
                             overfit=round(prize(m_in, Q) - prize(m_out, Q), 3),
                             fallback_in=f_in, fallback_out=f_out)
        print("\nQ=%4d  u_max A %d / B %d   p50 A %d / B %d"
              % (Q, max(ua), max(ub), sorted(ua)[len(ua) // 2], sorted(ub)[len(ub) // 2]),
              flush=True)
        for name, r in row.items():
            print("   %s  edges %-24s  in-sample %+6.3f  out-of-sample %+6.3f  overfit %+6.3f"
                  % (name, str(r["buckets"]), r["in_sample_net"], r["out_sample_net"],
                     r["overfit"]), flush=True)
        out["Q%d" % Q] = row
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(a=str(A), b=str(B), seed_a=da.get("seed"),
                                   seed_b=db.get("seed"), dense_ms=dense, max_buckets=K,
                                   per_q=out), indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
