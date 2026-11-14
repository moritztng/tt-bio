"""Run a command on a card with the AICLK sampled during it, and record both.

    python perf/rfd4_land/clockrun.py --out <run.json> --tag <name> -- <cmd> ...

A number without a clock is not a measurement, so the wall time and the clock land in the
same JSON line. Below 1200 MHz the line is flagged `clock_artifact`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # perf/
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
import clocksample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

with clocksample.during() as clk:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

s = clk.summary().get(0, {})
rec = {"tag": a.tag, "cmd": cmd, "rc": p.returncode, "wall_s": round(wall, 3),
       "aiclk": clk.summary(), "aiclk_mhz": s.get("median"),
       "clock_line": clk.line(), "clock_artifact": bool(s.get("median", 0) < 1200),
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
       "sha": os.popen("git rev-parse HEAD").read().strip(),
       "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "stdout": p.stdout[-4000:], "stderr": p.stderr[-2000:]}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(rec["clock_line"])
print(f"{a.tag} rc={p.returncode} wall={wall:.2f}s")
sys.exit(p.returncode)
