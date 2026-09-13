#!/usr/bin/env python3
"""The non-PASS legs of the current-main parity gate, extracted so the ship run is controlled leg
by leg against digits rather than against a summary.

`perf/b2z2_gate/gate_asg.json` was produced on `wk/b2z2-asg-ship` @ `2f5072d88`, and
`git diff 2f5072d88 origin/main -- tt_bio/ scripts/` is empty, so it IS main's report: a second
141-minute control run would reproduce it. A leg that is not PASS here must reproduce these digits
on the ship branch, or it is a regression this branch caused and not a gap it inherited.
"""
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
d = json.loads(src.read_text())
legs = d["legs"]
rows = {}
counts = {}
for leg in sorted(legs, key=lambda x: x["leg"]):
    v = leg.get("verdict", "?")
    counts[v] = counts.get(v, 0) + 1
    if v != "PASS":
        rows[leg["leg"]] = leg
out.write_text(json.dumps({"doc": __doc__, "source": str(src), "verdict_counts": counts,
                           "tally": d.get("tally"), "total_wall_s": d.get("total_wall_s"),
                           "non_pass": rows}, indent=1))
print(json.dumps(counts, indent=1))
print(f"{len(rows)} non-PASS legs -> {out}")
for n, leg in rows.items():
    print(f"   {n:24s} {leg['verdict']:26s} committed={leg.get('committed')}  {str(leg.get('detail'))[:70]}")
