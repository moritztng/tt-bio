#!/usr/bin/env python3
"""Count COMPLETE blocks in a bracketed fold A/B session.

Complete means all three bracket positions (base, on, base) landed with returncode 0 and the full
--folds count. A block whose third leg was truncated -- as block 4 was on 2026-09-19 when leg
512_base_4_2 host-spun at fold 7 of 12 -- is not a block, because the bracket's whole point is that
the on arm is compared against base arms on BOTH sides of it.
"""
import json
import sys

rows = json.load(open(sys.argv[1]))["blocks"]
folds = int(sys.argv[2]) if len(sys.argv) > 2 else 12
seen = {}
for r in rows:
    if r.get("returncode") == 0 and len(((r.get("result") or {}).get("folds")) or []) == folds:
        seen.setdefault(r["block"], set()).add(r["pos"])
print(sum(1 for v in seen.values() if v >= {0, 1, 2}))
