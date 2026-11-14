"""Rank the host timeline's call sites by host time before the call (Python work the device may
idle through) and inside it. Usage: read.py hosttl.jsonl [N]. Reads the last fold (the warm one)."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])][-1]["rows"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print(f"total before {sum(r['before_s'] for r in rows):.2f} s, in-call {sum(r['in_s'] for r in rows):.2f} s, "
      f"{sum(r['calls'] for r in rows)} calls")
for key in ("before_s", "in_s"):
    print(f"\n-- by {key}")
    for r in sorted(rows, key=lambda r: -r[key])[:n]:
        print(f"{r[key]:8.3f} s {r['calls']:7d} x  max_before {r['max_before_s']:.3f}  {r['op']:<18} {r['site']}")
