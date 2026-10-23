"""Run one command with AICLK sampled for its life; write rc, wall seconds and the clock summary as JSON.
  python clocked.py <out.json> <cmd...>
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample
import clocksample  # noqa: E402

out, cmd = sys.argv[1], sys.argv[2:]
t = time.time()
with clocksample.during(period=1.0) as clk:
    rc = subprocess.call(cmd)
json.dump({"cmd": cmd, "rc": rc, "wall_s": round(time.time() - t, 1), "aiclk": clk.summary()},
          open(out, "w"), indent=1)
sys.exit(rc)
