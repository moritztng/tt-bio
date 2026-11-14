"""Whole-design wall leg of run2: per arm median trunk_s / design_s over warm designs, AICLK, and the ratio.
    python3 wall.py <run dir>"""
import json, statistics, sys
from pathlib import Path
rows = {"base": [], "lev": []}
for line in Path(sys.argv[1], "wall512.log").read_text().splitlines():
    if '"kind": "warm"' not in line:
        continue
    d = json.loads(line[line.index("{"):])
    arm = "base" if "/base" in d["cif"] else "lev"
    rows[arm].append(d)
out = {}
for arm, rs in rows.items():
    out[arm] = {k: statistics.median(r[k] for r in rs) for k in ("trunk_s", "design_s", "wall_s")}
    out[arm]["n"] = len(rs)
    out[arm]["aiclk_min"] = min(r["aiclk"]["0"]["min"] for r in rs)
    out[arm]["aiclk_median"] = statistics.median(r["aiclk_mhz"] for r in rs)
for k in ("trunk_s", "design_s"):
    out[k + "_ratio"] = round(out["base"][k] / out["lev"][k], 4)
print(json.dumps(out, indent=1))
