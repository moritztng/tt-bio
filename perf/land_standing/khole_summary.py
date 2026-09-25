#!/usr/bin/env python3
"""What `TT_BIO_TRIATT_DIVIDING_K` does at every tile-aligned length, on Blackhole.

Reads the banked khole runs and answers the one question the flip waits on: at each length,
does the lever change which rung the fused-HiFi route takes, and if so, does it change it for
the better. A rung is (q_chunk, k_chunk, kv_buffer_factor, served), in offer order, so
"byte-identical" here is the whole offered sequence and not just the outcome.

    khole_summary.py out/khole_bh
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rungs(arm):
    return [(r["q_chunk"], r["k_chunk"], r["kv_bf"], r["served"]) for r in arm["rungs"]]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "out/khole_bh")
    rows = {}
    for f in sorted(root.glob("*.json")):
        for r in json.load(f.open())["rows"]:
            rows.setdefault(r["n"], (r, f.name))

    blast = Path(__file__).resolve().parents[1] / "bcx_tapedfwd" / "out" / "blast.json"
    changed = {c["n"] for c in json.load(blast.open())["changed"]}

    won, same, both_decline, regressed = [], [], [], []
    print(f"{'n':>6} {'in blast':>9} {'shipped':>8} {'dividing':>9}  verdict")
    for n in sorted(rows):
        r, _ = rows[n]
        s, d = r["arms"]["shipped"], r["arms"]["dividing"]
        ident = rungs(s) == rungs(d)
        if s["served"] and not d["served"]:
            v, bucket = "REGRESSION", regressed
        elif d["served"] and not s["served"]:
            v, bucket = "lever serves where shipped declined", won
        elif ident:
            v, bucket = "byte-identical rungs", same
        elif s["served"] and d["served"]:
            v, bucket = "both serve, DIFFERENT rung", regressed
        else:
            v, bucket = "both decline, lever's extra rungs all fail", both_decline
        bucket.append(n)
        print(f"{n:>6} {str(n in changed):>9} {str(s['served']):>8} {str(d['served']):>9}  {v}")

    print(f"\nmeasured {len(rows)} lengths")
    print(f"  lever serves where shipped declined : {len(won):>2}  {won}")
    print(f"  byte-identical rung sequence        : {len(same):>2}  {same}")
    print(f"  both decline, identical outcome     : {len(both_decline):>2}  {both_decline}")
    print(f"  REGRESSIONS                         : {len(regressed):>2}  {regressed}")
    assert not regressed, f"a length where the lever is worse: {regressed}"
    # Every length blast.py calls inert must be byte-identical; that is the claim it asserts
    # arithmetically and this is the same claim measured on a card.
    bad = [n for n in rows if n not in changed and n not in same]
    assert not bad, f"blast.py calls these inert but the rungs moved: {bad}"
    print("checked: no regression at any length, and every blast-inert length is byte-identical")


if __name__ == "__main__":
    main()
