#!/usr/bin/env python3
"""The size x depth screen as one table: per-cell trunk time, the lever ratio, and the
log-log exponent between consecutive rungs.

`one-size-tuning-is-a-standing-defect-class` wants the lever set scored at every rung and both
depths, and an N^2 -> N^3+ jump inside one interval is the only signal a dark gate gives.
"""
import json, math, sys
from pathlib import Path

rows = {}
for arm in ("def", "lev"):
    p = Path(f"perf/rf3/msa_depth/ladder_{arm}.jsonl")
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        r = json.loads(line)
        rows[(r["aa"], r["depth_req"], arm)] = r

sizes = sorted({k[0] for k in rows})
depths = sorted({k[1] for k in rows})

print(f"{'aa':>5} {'depth':>5} {'def s/rec':>10} {'lev s/rec':>10} {'lever x':>8} "
      f"{'triatt served/decl':>19} {'opm small/mat':>14}")
for aa in sizes:
    for d in depths:
        rd, rl = rows.get((aa, d, "def")), rows.get((aa, d, "lev"))
        def fmt(r):
            if r is None:
                return None
            return None if "error" in r else r["trunk_s_per_recycle"]
        td, tl = fmt(rd), fmt(rl)
        ratio = f"{td/tl:8.4f}" if td and tl else " " * 8
        ts = rl["triatt_stats"] if rl and "error" not in rl else {}
        os_ = rl["opm_stats"] if rl and "error" not in rl else []
        print(f"{aa:>5} {d:>5} "
              f"{(f'{td:10.4f}' if td else '   not run')} "
              f"{(f'{tl:10.4f}' if tl else '   not run')} {ratio} "
              f"{str(ts.get('served','-'))+'/'+str(ts.get('declined','-')):>19} "
              f"{('/'.join(str(x) for x in os_) if os_ else '-'):>14}")

print()
for arm in ("def", "lev"):
    for d in depths:
        pts = [(aa, rows[(aa, d, arm)]["trunk_s_per_recycle"]) for aa in sizes
               if (aa, d, arm) in rows and "error" not in rows[(aa, d, arm)]]
        if len(pts) < 2:
            continue
        segs = []
        for (a0, t0), (a1, t1) in zip(pts, pts[1:]):
            segs.append(f"{a0}->{a1}: {math.log(t1/t0)/math.log(a1/a0):.2f}")
        print(f"exponent {arm} d{d}: " + "  ".join(segs))
