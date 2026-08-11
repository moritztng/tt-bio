#!/usr/bin/env python3
"""Split the Pairformer block into arithmetic and non-arithmetic, class-median priced.

Host-side only -- it reads a committed census artifact and opens no device.

Two prior numbers this replaces. Section 112 divided the block FLOP by the whole block WALL and
called the quotient the block rate (16.85 TF/s), then built an arithmetic floor by applying one
matmul best rate (40.40 TF/s) to all of it. More than half the block wall is not arithmetic, so
that ratio was never a rate, and applying one shape rate to every shape is the denominator
mismatch recorded in moonshot-4x-flop-rate-denominator-mismatch.

Pricing is class-median (op + site + input shapes + output shape), per section 116: bench()
re-runs one invocation per class beside the live block and holds its outputs alive, so summing a
site invocations carries that outlier 480 times. Coverage against the measured wall is printed as
the check -- median pricing must land under the wall, sum pricing does not.

FLOP is counted from the shapes the ops were called with: 2 * prod(out) * K for the matmul family,
4 * B * H * S * S * D for SDPA. Everything else is 0 by construction, which is the split.
"""
import argparse, json, statistics, sys
from collections import defaultdict

BLOCKS_PER_FOLD = 480


def prod(xs):
    p = 1
    for v in xs:
        p *= v
    return p


def flop(rec):
    op, ins = rec["op"], rec.get("in") or []
    out = (rec.get("out") or {}).get("shape")
    if op in ("linear", "matmul", "minimal_matmul") and out and len(ins) >= 2:
        return 2.0 * prod(out) * ins[1]["shape"][-2]
    if op == "scaled_dot_product_attention" and ins:
        b, h, s, d = ins[0]["shape"]
        return 4.0 * b * h * s * s * d
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--out")
    ap.add_argument("--target-rate", type=float, default=76.4,
                    help="TF/s the arithmetic headroom is scored against. Default is the rate "
                         ":2539 and :2550 reach in THIS fold, not a roof.")
    a = ap.parse_args()

    cen = json.load(open(a.census))
    cls = defaultdict(list)
    for r in cen["records"]:
        cls[(r["op"], r["site"],
             tuple(tuple(i["shape"]) for i in r.get("in", [])),
             tuple((r.get("out") or {}).get("shape") or ()))].append(r)

    sites, arith_s, arith_f, other_s, untimed_f = [], 0.0, 0.0, 0.0, 0.0
    for (op, site, _ish, _osh), v in cls.items():
        timed = [x["s"] for x in v if x.get("s")]
        med = statistics.median(timed) if timed else 0.0
        n, f = len(v), flop(v[0]) * len(v)
        out = v[0].get("out") or {}
        if f > 0:
            arith_s += n * med
            arith_f += f
            if not timed:
                untimed_f += f
            sites.append(dict(site=site, op=op, n=n, med_ms=med * 1e3,
                              s_per_fold=n * med * BLOCKS_PER_FOLD, tflop_per_block=f / 1e12,
                              tf_per_s=(f / (n * med) / 1e12) if med else None,
                              nt=(out["shape"][-1] + 31) // 32 if out.get("shape") else None,
                              buf=out.get("buf"), dtype=out.get("dtype")))
        else:
            other_s += n * med

    wall = cen["block_wall_s"]
    timed_f = arith_f - untimed_f
    # SDPA is scored separately: head_dim 32 is structural and the lever is a closed dead end.
    sdpa = [s for s in sites if s["op"] == "scaled_dot_product_attention"]
    sdpa_f = sum(s["tflop_per_block"] for s in sdpa) * 1e12
    sdpa_s = sum(s["s_per_fold"] for s in sdpa) / BLOCKS_PER_FOLD
    open_f, open_s = timed_f - sdpa_f, arith_s - sdpa_s

    res = dict(
        census=a.census, block_wall_ms=wall * 1e3, blocks_per_fold=BLOCKS_PER_FOLD,
        loadavg=cen.get("loadavg"), fast=cen.get("fast"), n=cen.get("n"),
        tflop_per_block=arith_f / 1e12, tflop_per_fold=arith_f * BLOCKS_PER_FOLD / 1e12,
        arith_ms_per_block=arith_s * 1e3, arith_s_per_fold=arith_s * BLOCKS_PER_FOLD,
        arith_tf_per_s=timed_f / arith_s / 1e12,
        nonarith_ms_per_block=other_s * 1e3, nonarith_s_per_fold=other_s * BLOCKS_PER_FOLD,
        coverage_pct=(arith_s + other_s) / wall * 100,
        residual_ms_per_block=(wall - arith_s - other_s) * 1e3,
        untimed_tflop_per_block=untimed_f / 1e12,
        untimed_implied_tf_per_s=(untimed_f / (wall - arith_s - other_s) / 1e12) if wall > arith_s + other_s else None,
        sdpa_s_per_fold=sdpa_s * BLOCKS_PER_FOLD,
        open_arith_s_per_fold=open_s * BLOCKS_PER_FOLD,
        open_arith_tf_per_s=open_f / open_s / 1e12,
        target_rate_tf_per_s=a.target_rate,
        arith_headroom_s_per_fold=(open_s - open_f / (a.target_rate * 1e12)) * BLOCKS_PER_FOLD,
        sites=sorted(sites, key=lambda s: -s["s_per_fold"]),
    )
    res["addressable_s_per_fold"] = res["nonarith_s_per_fold"] + res["arith_headroom_s_per_fold"]

    for k, v in res.items():
        if k != "sites":
            print(f"{k:32} {v}")
    print()
    hdr = ("site", "op", "n", "med ms", "s/fold", "TF/blk", "TF/s", "nt", "buf")
    print("%-22s %-30s %3s %8s %8s %8s %7s %4s %5s" % hdr)
    for s in res["sites"]:
        if s["s_per_fold"] < 0.02 and s["med_ms"]:
            continue
        print("%-22s %-30s %3d %8.4f %8.3f %8.4f %7s %4s %5s" % (
            s["site"], s["op"], s["n"], s["med_ms"], s["s_per_fold"], s["tflop_per_block"],
            ("%.2f" % s["tf_per_s"]) if s["tf_per_s"] else "untimed", s["nt"], s["buf"]))

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
