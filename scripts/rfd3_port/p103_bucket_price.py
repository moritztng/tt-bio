#!/usr/bin/env python3
"""p103 -- price a bucketed key axis using only widths that were actually measured.

p102's own bucket rows extrapolate p100/p101's measured chain(Q, U) line out to u_max, and at
Q=1216 u_max over the full schedule is 6048 of 6080 -- so 99 of 199 steps were priced by
extrapolating a two-point fit roughly twice as far as the bracket it was fitted on. Those rows
are not numbers, they are a straight line's opinion.

This prices the same distribution with two rules that remove the guessing:

  1. Every bucket is inside the measured bracket [U_lo, U_hi] for its Q, so every step that uses
     a bucket is priced by INTERPOLATION between two measured points.
  2. A step whose union is wider than the widest measured bucket runs the shipped dense chain.
     That caps the downside at dense by construction: the route can no longer lose, it can only
     fail to win on those steps.

Rule 2 is also the only honest thing to do with the tail. There are steps whose block union is
6048 of 6080 keys, i.e. steps with no block sparsity to exploit at all, and no choice of a single
compiled width makes those cheap.

Reports the best bucket set of size 1, 2 and 3 per Q, the fraction of steps that fall back, and
the resulting prize against the in-round dense control.
"""
import itertools
import json
import pathlib
import sys

IN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p102/umax_schedule.json")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "perf/p103/bucket_price.json")
MAX_BUCKETS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# host cost of the per-step index build, bitmap form, from perf/p99/index_host.json
HOST_S_PER_DESIGN = {320: 0.447, 608: 0.420, 1216: 0.404, 3040: 0.361}


def main():
    d = json.loads(IN.read_text())
    meas = {int(q): {int(u): c for u, c in v.items()} for q, v in d["measured_chain"].items()}
    dense, cps, steps = d["dense_ms"], d["calls_per_step"], d["steps"]
    print("dense control %.4f ms/call, %d calls/step, %d steps, %d atom indices"
          % (dense, cps, steps, d["n_atom_indices"]), flush=True)
    out = {}
    for qk, r in d["report"].items():
        Q = int(qk[1:])
        us = r["umax_tiled"]
        n = len(us)
        lo, hi = min(meas[Q]), max(meas[Q])
        c_lo, c_hi = meas[Q][lo], meas[Q][hi]
        slope = (c_hi - c_lo) / (hi - lo)

        def chain(U):
            return c_lo + slope * (U - lo)

        # candidate bucket edges: the measured endpoints plus every observed width inside them
        cand = sorted({lo, hi} | {u for u in us if lo <= u <= hi})
        rows = {}
        for k in range(1, MAX_BUCKETS + 1):
            best = None
            for bs in itertools.combinations(cand, k):
                if bs[-1] != hi:            # the widest bucket is the widest measured width
                    continue
                tot, nfall = 0.0, 0
                for u in us:
                    fit = [b for b in bs if b >= u]
                    if fit:
                        tot += chain(fit[0])
                    else:
                        tot += dense
                        nfall += 1
                m = tot / n
                if best is None or m < best[0]:
                    best = (m, list(bs), nfall)
            if best is None:
                continue
            m, bs, nfall = best
            gross = (dense - m) * cps * steps / 1000.0
            host = HOST_S_PER_DESIGN.get(Q, 0.0)
            rows["k%d" % k] = dict(buckets=bs, fallback_steps=nfall,
                                   fallback_frac=round(nfall / n, 3),
                                   mean_chain_ms=round(m, 4),
                                   gross_s_per_design=round(gross, 3),
                                   host_s_per_design=host,
                                   net_s_per_design=round(gross - host, 3))
        print("\nQ=%4d  measured bracket [%d, %d]  u_max over the schedule %d"
              % (Q, lo, hi, max(us)), flush=True)
        for k, v in rows.items():
            print("   %s buckets %-26s  %3d/%d fall back to dense (%2.0f%%)  "
                  "mean %7.4f ms  gross %+6.3f  net %+6.3f s/design"
                  % (k, str(v["buckets"]), v["fallback_steps"], n,
                     100 * v["fallback_frac"], v["mean_chain_ms"],
                     v["gross_s_per_design"], v["net_s_per_design"]), flush=True)
        out[qk] = dict(measured_bracket=[lo, hi], u_max_schedule=max(us),
                       n_indices=n, buckets=rows)
    best_q = max(out, key=lambda q: max(v["net_s_per_design"]
                                        for v in out[q]["buckets"].values()))
    best = max(out[best_q]["buckets"].values(), key=lambda v: v["net_s_per_design"])
    print("\nbest: %s, %s, %+.3f s/design net of the host index build"
          % (best_q, best["buckets"], best["net_s_per_design"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(source=str(IN), dense_ms=dense, calls_per_step=cps,
                                   steps=steps, host_s_per_design=HOST_S_PER_DESIGN,
                                   best_q=best_q, best=best, per_q=out), indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
