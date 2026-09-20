#!/usr/bin/env python3
"""Pin and sample the AICLK around a command that does its own timing.

The two design models are not folds and `cell.py` cannot drive them: `bg_page.py` shells out to
the shipped `tt-bio design` CLI and reads its per-design stamps, and `rfd3_page.py` times
`RFD3Sampler.sample` inside the released entry point. Both are the harnesses the published cells
were taken with, so they are what this row reuses -- but neither pins a clock, and on Blackhole
the governor alone swings a fold 1.27-1.41x, so an unpinned design is a reading of the box mood.

This wrapper supplies exactly the missing part and nothing else. It forces FORCE_AICLK on the
node the child will use, keeps it there with the same watchdog `cell.py` uses, samples every chip
in the box at 4 Hz while the child runs, and writes the record beside the child's own output. The
Clock and Sampler classes are IMPORTED from cell.py rather than restated, so the design arms and
the fold arms are pinned and sampled by the same code.

    pinned_run.py --node 1 --clock 1350 --out clk.json -- <cmd> [args...]

The force is chip state and outlives both this process and its fd, so release is wired to the
normal path, to atexit and to the fatal signals -- a wrapper that died without releasing would
leave the card at burst indefinitely.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_cell():
    spec = importlib.util.spec_from_file_location("allm_cell", HERE / "cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True, help="/dev/tenstorrent/N the child uses")
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given after --")

    C = load_cell()
    # Clock.acquire() forces every node THIS process holds, off its own fd table. The wrapper
    # holds none -- the child does -- so the node is named explicitly instead.
    C._own_nodes = lambda: [a.node]

    rec = {"node": a.node, "clock_target": a.clock, "cmd": cmd,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "foreign_at_start": C.foreign_holders(),
           "loadavg_start": open("/proc/loadavg").read().split()[:3]}

    clock = C.Clock(a.clock) if a.clock else None
    if clock is not None:
        rec["clock_after_force"] = clock.acquire()
        print(f"pinned_run: forced AICLK -> {rec['clock_after_force']}", flush=True)
    samp = C.Sampler([a.node])
    samp.start()
    samp.take()                        # drop the pre-child window

    rc = 1
    try:
        p = subprocess.Popen(cmd)
        # Forward a stop to the child first, so its own release path runs before ours.
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, lambda sig, frm: p.send_signal(sig))
        rc = p.wait()
    finally:
        samp.stop.set()
        rec.update(samp.take())
        rec["rc"] = rc
        rec["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
        rec["foreign_at_end"] = C.foreign_holders()
        if clock is not None:
            rec["clock_reasserts"] = clock.reasserts
            clock.release()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in
                          ("rc", "aiclk_mean", "aiclk_min", "aiclk_max", "aiclk_n",
                           "clock_reasserts") if k in rec}, indent=1), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
