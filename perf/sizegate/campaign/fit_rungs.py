#!/usr/bin/env python3
"""Per-interval runtime exponents for one card's ladder, from the recorded fragments.

The question the new rungs exist to answer is whether the curve STEPS between two rungs (a
fast path falling off, which is a defect) or just steepens (the complexity). A single
exponent over the whole ladder cannot tell those apart, so this prints k for every
consecutive pair and marks the ones that jump.

k = ln(t2/t1) / ln(n2/n1). A jump is flagged when an interval's k exceeds the previous
interval's by more than JUMP, which is deliberately loose: a monotone rise as the pair term
comes to dominate a size-independent term is normal and is not what this is hunting.
"""
import json, pathlib, sys

CARD = sys.argv[1] if len(sys.argv) > 1 else "p150a"
FRAG = pathlib.Path(__file__).resolve().parents[3] / "docs" / "size_ladder_baseline.d"
JUMP = 1.0

import math
rows = []
for f in sorted(FRAG.glob("*.json")):
    blk = (json.loads(f.read_text()).get("cards") or {}).get(CARD)
    if not blk:
        continue
    for model, e in sorted((blk.get("models") or {}).items()):
        rt = {int(k): v for k, v in (e.get("runtime_s") or {}).items()}
        refused = sorted(int(k) for k in (e.get("refused") or {}))
        ns = sorted(rt)
        ks = []
        for a, b in zip(ns, ns[1:]):
            ks.append((a, b, math.log(rt[b] / rt[a]) / math.log(b / a)))
        rows.append((model, ns, refused, ks, e.get("commit") or blk.get("commit")))

if not rows:
    sys.exit(f"no cells for card {CARD} in {FRAG}")

for model, ns, refused, ks, commit in rows:
    top = max(ns) if ns else 0
    print(f"\n{model}  measured {ns}" + (f"  refused {refused}" if refused else "")
          + f"  top {top}")
    prev = None
    for a, b, k in ks:
        flag = ""
        if prev is not None and k - prev > JUMP:
            flag = f"   <-- JUMP +{k - prev:.2f} over the interval below"
        print(f"    {a:>5} -> {b:<5} k = {k:5.2f}{flag}")
        prev = k

bar = 1536
short = [m for m, ns, ref, _k, _c in rows if max(ns + ref, default=0) < bar]
print(f"\nrungs reaching {bar}: {len(rows) - len(short)}/{len(rows)}"
      + (f"; short: {', '.join(short)}" if short else ""))
