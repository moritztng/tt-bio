#!/usr/bin/env python3
"""How many linear/layer-norm call sites the attach point actually reaches. 11 of 205."""
import re
from pathlib import Path
R = Path('/home/ttuser/.coworker/wt/train-a1-defork/tt_bio')
tot_raw = tot_routed = 0
rows = []
for f in sorted(R.glob('*.py')):
    if f.name in ('ops.py', 'autograd.py'):
        continue
    s = f.read_text()
    raw = len(re.findall(r'ttnn\.linear\(', s)) + len(re.findall(r'ttnn\.layer_norm\(', s))
    routed = len(re.findall(r'ops\.linear\(', s)) + len(re.findall(r'ops\.layer_norm\(', s))
    if raw or routed:
        rows.append((f.name, raw, routed))
        tot_raw += raw
        tot_routed += routed
for n, a, b in sorted(rows, key=lambda r: -r[1]):
    print(f"{n:<42} raw={a:<4} routed={b}")
print(f"{'TOTAL':<42} raw={tot_raw:<4} routed={tot_routed}  "
      f"routed share={tot_routed/(tot_raw+tot_routed):.1%}")
