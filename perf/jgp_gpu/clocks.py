#!/usr/bin/env python3
"""Per-rep GPU clock/power/temperature from a run's smi.csv, windowed by the runner's own item start/finish log lines.

    python clocks.py <run dir>
Throttle: clocks_event_reasons bitmask; 0x4 = SW power cap, 0x20/0x40 = SW/HW thermal slowdown, 0x8 = HW slowdown.
"""
import re, statistics as st, sys
from datetime import datetime
from pathlib import Path

d = Path(sys.argv[1])
rows = []
for ln in (d / "smi.csv").read_text().splitlines():
    f = [x.strip() for x in ln.split(",")]
    if len(f) < 9 or "MHz" not in f[1]:
        continue
    t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f")
    rows.append((t, int(f[1].split()[0]), float(f[3].split()[0]), int(f[5]), int(f[6].split()[0]), int(f[7], 16)))
start, end = {}, {}
for ln in (d / "stderr.log").read_text().splitlines():
    m = re.match(r"(\S+ \S+),\d+ .*\[Rank 0 \(\d+/\d+\)\] (r\d) \[seed:\d+\]: N_asym", ln)
    if m:
        start[m.group(2)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    m = re.match(r"(\S+ \S+),\d+ .*\[Rank 0\] (r\d) \[seed:\d+\] succeeded", ln)
    if m:
        end[m.group(2)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
for r in sorted(end):
    w = [x for x in rows if start[r] <= x[0] <= end[r] and x[4] >= 50]  # busy samples only
    ck = [x[1] for x in w]; pw = [x[2] for x in w]
    reasons = {}
    for x in w:
        reasons[x[5]] = reasons.get(x[5], 0) + 1
    print(f"{d.name} {r}: n={len(w)} sm MHz median {st.median(ck):.0f} min {min(ck)} max {max(ck)} | power W median {st.median(pw):.0f} "
          f"max {max(pw):.0f} | temp C max {max(x[3] for x in w)} | reasons {', '.join(f'{k:#x}:{v}' for k, v in sorted(reasons.items()))}")
