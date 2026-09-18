#!/usr/bin/env python3
"""Rank the fold's linear/matmul keys by the FLOPs they carry, from the executed call census.

Source: perf/roof_launch/op_census_512.json, the same 465,664-call census c10-fold-census weighted
its budget by. The key string already carries both operand shapes, so FLOPs are counted from the
shapes rather than from any recorded rate.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def parse(key):
    """-> (batch, M, K, N) or None when the key does not carry two 2D+ operands."""
    if "|in=" not in key:
        return None
    ins = key.split("|in=")[1].split(",")
    if len(ins) < 2:
        return None
    try:
        a = [int(x) for x in ins[0].split("x")]
        b = [int(x) for x in ins[1].split("x")]
    except ValueError:
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    m, k, n = a[-2], a[-1], b[-1]
    batch = 1
    for x in a[:-2]:
        batch *= x
    return batch, m, k, n


def main():
    d = json.loads((REPO / "perf/roof_launch/op_census_512.json").read_text())
    rows = []
    for key, v in d["top_shapes"].items():
        if not key.startswith(("ttnn.linear", "ttnn.matmul")):
            continue
        p = parse(key)
        calls = v["calls"]
        f = 0.0 if p is None else 2.0 * p[0] * p[1] * p[2] * p[3] * calls
        rows.append((f, calls, p, key, v))
    rows.sort(key=lambda r: -r[0])
    tot = sum(r[0] for r in rows)
    print("keys=%d  total FLOP=%.3f TF" % (len(rows), tot / 1e12))
    print("%9s %10s %7s %26s  %s" % ("calls", "TFLOP", "AI", "b,M,K,N", "key"))
    for f, calls, p, key, v in rows:
        if p is None:
            print("%9.0f %10s %7s %26s  %s" % (calls, "-", "-", "-", key))
            continue
        b, m, k, n = p
        bytes_min = 2.0 * (b * m * k + k * n + b * m * n)
        ai = (2.0 * b * m * k * n) / bytes_min
        print("%9.0f %10.3f %7.1f %26s  %s"
              % (calls, f / 1e12, ai, "%dx%dx%dx%d" % (b, m, k, n), key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
