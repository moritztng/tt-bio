#!/usr/bin/env python3
"""Aggregate the box-256 precision campaign into mean, sd and a 95% CI per arm.

The predecessor's hardest lesson (spike section 31.2) is that one seed at one box produced
three different verdicts of the same experiment, so the claim needs an interval and not a
point. Seeds are paired across arms -- the same seed drives the same noise realisation in
every arm -- so the per-seed deltas are compared directly.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAT = re.compile(r"p3fsc_box(\d+)_snr([\d.]+)_s(\d+)_(\w+?)_(fp\d+|tex8|bf16\w*|pert[\w.-]+)(_f\d+)?\.json$")

rows: dict[tuple, dict[int, tuple[float, str]]] = {}
for p in sorted(HERE.glob("p3fsc_box*.json")):
    m = PAT.match(p.name)
    if not m:
        print(f"  (unparsed: {p.name})")
        continue
    box, _snr, seed, variant, prec, flush = m.groups()
    d = json.load(open(p))
    key = (int(box), variant, prec + (flush or ""))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    rows.setdefault(key, {})[int(seed)] = (d["delta_A"], sha, p.name)

print(f"{'box':>4} {'variant':<8} {'precision':<14} {'n':>2} "
      f"{'mean dA':>11} {'sd':>10} {'95% CI':>26}   seeds")
for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
    box, variant, prec = key
    by_seed = rows[key]
    seeds = sorted(by_seed)
    v = np.array([by_seed[s][0] for s in seeds], float)
    n = len(v)
    mean = v.mean()
    if n > 1:
        sd = v.std(ddof=1)
        # Student t, two-sided 95%, n-1 df. Table rather than scipy: no scipy on pc's env.
        T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086}
        t = T.get(n - 1, 1.96)
        h = t * sd / np.sqrt(n)
        ci = f"[{mean-h:+.6f}, {mean+h:+.6f}]"
        sds = f"{sd:.6f}"
    else:
        ci, sds = "--", "--"
    print(f"{box:>4} {variant:<8} {prec:<14} {n:>2} {mean:>+11.6f} {sds:>10} {ci:>26}   "
          f"{','.join(str(s) for s in seeds)}")

print("\nsha256 (leading 16 hex) of every file above:")
for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
    for s in sorted(rows[key]):
        d, sha, name = rows[key][s]
        print(f"    {sha}  {name:<52} {d:+.6f} A")
