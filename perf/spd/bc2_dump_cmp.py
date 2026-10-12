"""Compare two bc2_census arms' gradient dumps call by call: relative L2 error per leaf and overall.

usage: python perf/spd/bc2_dump_cmp.py <arm_dir_a> <arm_dir_b>
"""
import sys
from pathlib import Path

import numpy as np

a, b = (Path(x) for x in sys.argv[1:3])
for fa in sorted(a.glob("dump_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
    fb = b / fa.name
    if not fb.exists():
        continue
    da, db = np.load(fa), np.load(fb)
    num = den = 0.0
    worst = (0.0, None)
    for k in da.files:
        if k not in db.files or da[k].shape != db[k].shape:
            print(f"{fa.stem} {k}: shape {da[k].shape} vs {db[k].shape if k in db.files else 'missing'}")
            continue
        d = float(np.sum((da[k] - db[k]) ** 2))
        n = float(np.sum(da[k] ** 2))
        num, den = num + d, den + n
        rel = (d / n) ** 0.5 if n else (0.0 if d == 0 else float("inf"))
        if rel > worst[0]:
            worst = (rel, k)
    print(f"{fa.stem}: rel L2 {(num / den) ** 0.5 if den else 0.0:.3e} over {len(da.files)} leaves, "
          f"worst {worst[1]} {worst[0]:.3e}")
