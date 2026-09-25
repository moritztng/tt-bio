#!/usr/bin/env python3
"""base against noexact: the break control for J0's attribution.

The closure histogram and `_EXACT_OPS`'s own dispatch table already say that
`exact_training`'s host float64 softmax and layer norm carry 53.7 % of the backward. This
is the confirmation by removal -- a suspect you cannot make worse on purpose is a suspect
you have not tested, and turning it off is how you make this one better on purpose.

Wall clock is the comparable column across the two arms. The per-verb columns are not:
`base` was recorded before the exclusive-time fix and sums its nested calls twice.
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def leg(d, k):
    return d.get(k) or {}


def main() -> int:
    b = json.loads((OUT / "hist_384_base.json").read_text())
    n = json.loads((OUT / "hist_384_noexact.json").read_text())
    if n.get("error"):
        print("noexact arm did not complete:\n" + n["error"][-600:])
        return 1
    print("crop 384, 1 taped cycle, pc card 0 (Blackhole p150a)")
    for d, nm in ((b, "base"), (n, "noexact")):
        print(f"  {nm:8s} {d['env'].get('aiclk_line', 'no clock')}")
        print(f"           exact ops active: {d['env'].get('exact_training_ops')}")
    print()
    print(f"{'leg':10s} {'base s':>10s} {'noexact s':>10s} {'delta s':>10s} {'x':>7s}")
    for k in ("forward", "backward"):
        fb, fn = leg(b, k).get("s"), leg(n, k).get("s")
        if fb and fn:
            print(f"{k:10s} {fb:10.2f} {fn:10.2f} {fb - fn:10.2f} {fb / fn:7.2f}")
    fb = leg(b, "forward").get("s", 0) + leg(b, "backward").get("s", 0)
    fn = leg(n, "forward").get("s", 0) + leg(n, "backward").get("s", 0)
    if fn:
        print(f"{'STEP':10s} {fb:10.2f} {fn:10.2f} {fb - fn:10.2f} {fb / fn:7.2f}")
    print()
    print("host round trips (autograd's own counters, delta over the leg):")
    for k in ("forward", "backward"):
        print(f"  {k:9s} base    {leg(b, k).get('host_roundtrips')}")
        print(f"  {k:9s} noexact {leg(n, k).get('host_roundtrips')}")
    print()
    print("noexact backward, top verbs by SELF time (children subtracted):")
    for r in (n.get("by_verb") or [])[:10]:
        print("  %-28s n=%-6d self %8.2fs  %8.1f us/call  %s GB/s"
              % (r["verb"], r["n"], r.get("self_s", 0), r["us_per_call"], r["gb_s"]))
    print()
    print("noexact backward, top closures:")
    for r in (n.get("by_closure") or [])[:8]:
        print("  %-46s fired=%-6d %8.2fs" % (r["closure"][:46], r["fired"], r["total_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
