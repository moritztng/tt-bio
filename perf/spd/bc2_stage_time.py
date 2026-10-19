#!/usr/bin/env python3
"""Seconds per BindCraft 2 design stage, by stamping each losses.csv row as it is written.

BindCraft 2 writes one row per design iteration with the stage in its `phase` column, but no timestamp,
so a trajectory's 2.6 s/it says nothing about which stage spent it. This polls the campaign's trajectory
folders and records the arrival time of every new row. Run it beside a `perf/spd/bench.py --model
bindcraft2` run, pointed at the same project folder; it exits when the run's pid does.

    python perf/spd/bc2_stage_time.py --project RUN/campaign_s101 --pid $BENCH_PID --out RUN/stage_time.jsonl

Each line: {t, traj, row, phase}. A row's cost is its own stamp minus the previous row's, so the first row
of a trajectory carries that trajectory's setup and is reported separately.
"""
import argparse, csv, json, os, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--project", required=True, type=Path)
p.add_argument("--pid", type=int, required=True)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--every", type=float, default=0.25)
a = p.parse_args()

seen = {}
with a.out.open("w", buffering=1) as fh:
    while True:
        alive = True
        try:
            os.kill(a.pid, 0)
        except OSError:
            alive = False
        for f in sorted(a.project.glob("1_Trajectories/*/*_losses.csv")):
            try:
                rows = list(csv.DictReader(f.open()))
            except OSError:
                continue
            n = seen.get(f.name, 0)
            if len(rows) > n:
                t = time.time()
                for i in range(n, len(rows)):
                    fh.write(json.dumps(dict(t=round(t, 3), traj=f.parent.name, row=i + 1,
                                             phase=rows[i].get("phase"))) + "\n")
                seen[f.name] = len(rows)
        if not alive:
            break
        time.sleep(a.every)
