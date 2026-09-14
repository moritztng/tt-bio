#!/usr/bin/env python3
"""Where do two `op_trace.py` traces first disagree by an amount that matters?

`--diff` reports the first call whose output HASH differs, which on these two stacks is call 18
and is a rounding difference of 0.08 bf16 ULP. That reading cost this campaign a pass. The L2 norm
of each output is already in the trace, so scanning the cross-stack relative difference of that
scalar finds the first call that moves the answer instead of the first call that moves a bit.

    scan_l2.py <A.jsonl> <B.jsonl> [threshold, default 0.002]
"""
import json
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()][1:]


def main() -> int:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
    print(f"{len(a)} vs {len(b)} calls, threshold rel-l2 {thr}")
    first_sha = None
    for x, y in zip(a, b):
        if first_sha is None and x.get("sha") != y.get("sha"):
            first_sha = x["i"]
        la, lb = x.get("l2"), y.get("l2")
        if la is None or lb is None or la == 0:
            continue
        r = abs(la - lb) / abs(la)
        if r > thr:
            print("%4d %-42s out=%s l2 %.4f vs %.4f rel %.3e absmax %s vs %s" % (
                x["i"], x["op"], x["out_shape"], la, lb, r, x.get("absmax"), y.get("absmax")))
    print("first sha divergence:", first_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
