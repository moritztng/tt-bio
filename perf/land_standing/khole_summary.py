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

    # The accuracy half, wherever a graded row exists. At a length the lever unlocks, the route
    # it REPLACES is `_fp32_softmax_attention` -- a decline falls through to that, not to the
    # stock bf16 op -- so fused-vs-fall-back against the same float64 reference is the change a
    # caller sees.
    graded = [(n, r) for n, (r, _) in sorted(rows.items())
              if r.get("graded") and r.get("reference_usable")
              and r["arms"].get("dividing", {}).get("rmsd_vs_f64")]
    if graded:
        print(f"\n{'n':>6} {'fused':>11} {'fall-back':>11} {'fused better by':>16}  "
              f"{'shipped arm':>12}")
        for n, r in graded:
            f = r["arms"]["dividing"]["rmsd_vs_f64"]["scale_after_bias"]
            b = r["fallback_rmsd_vs_f64"]["scale_after_bias"]
            print(f"{n:>6} {f:>11.7f} {b:>11.7f} {100 * (b - f) / b:>15.1f} %  "
                  f"{'serves' if r['arms']['shipped']['served'] else 'DECLINES':>12}")
        worse = [n for n, r in graded
                 if r["arms"]["dividing"]["rmsd_vs_f64"]["scale_after_bias"]
                 > r["fallback_rmsd_vs_f64"]["scale_after_bias"]]
        assert not worse, f"the lever is further from float64 than the route it replaces at {worse}"
        print("checked: at every graded length the fused route is closer to float64 than the "
              "fall-back it replaces")


if __name__ == "__main__":
    main()
