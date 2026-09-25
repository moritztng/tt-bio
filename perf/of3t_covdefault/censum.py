#!/usr/bin/env python3
"""Print the per-process call census and region timings a run left behind."""
import json
import sys
from pathlib import Path

for c in sorted(Path(sys.argv[1]).glob("census.*.json")):
    d = json.loads(c.read_text())
    ts = {k: [round(x * 1e3, 3) for x in v] for k, v in d.get("times_s", {}).items()}
    print(f" pid {d['pid']} module_imported={d['module_imported']} "
          f"env={d['env_TT_BIO_OF3_DEVICE_REFATOM']} calls={d['calls']} "
          f"gate={d['gate_branch']} ms={ts}")
