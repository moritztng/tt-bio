#!/usr/bin/env python3
"""Time one tt-bio CLI run: wall, AICLK and host load sampled DURING it, one JSON line out.

    python3 perf/mgx_wh_matmul/fold_time.py --label boltz2-g1 --env TT_BIO_DEST_CARRY_GUARD=1 \
        --out times.jsonl -- predict fixture.yaml --model boltz2 --out_dir out/ ...

Everything after `--` goes to `python -m tt_bio.main`. The wall is the whole subprocess (load,
fold, write), as scripts/speed_bar.py's rungs were timed.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from perf.clocksample import during  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--env", action="append", default=[], metavar="K=V")
ap.add_argument("cli", nargs=argparse.REMAINDER)
a = ap.parse_args()
cli = a.cli[1:] if a.cli[:1] == ["--"] else a.cli
env = dict(os.environ, **dict(e.split("=", 1) for e in a.env))
loads, stop = [], threading.Event()


def sample_load():
    while not stop.is_set():
        loads.append(os.getloadavg()[0])
        stop.wait(10)


t = threading.Thread(target=sample_load, daemon=True)
t.start()
with during(period=5.0) as clk:
    t0 = time.time()
    rc = subprocess.call([sys.executable, "-m", "tt_bio.main"] + cli, cwd=REPO, env=env)
    wall = time.time() - t0
stop.set()
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
row = {"label": a.label, "env": a.env, "cli": cli, "rc": rc, "wall_s": round(wall, 1), "aiclk": clk.summary(),
       "load": {"min": min(loads), "median": sorted(loads)[len(loads) // 2], "max": max(loads),
                "nproc": os.cpu_count()} if loads else None,
       "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"), "commit": commit}
with open(a.out, "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
sys.exit(rc)
