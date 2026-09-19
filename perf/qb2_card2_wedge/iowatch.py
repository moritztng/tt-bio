#!/usr/bin/env python3
"""Sample the wedge signature of a running fold: syscall progress, CPU, AICLK, output staleness.

The `blackhole-p300c-host-spin-wedge` presents as a process pinned near 100 % CPU with BOTH
`syscr` and `syscw` in /proc/<pid>/io frozen while its output file goes stale for minutes. CPU
alone cannot tell a live fold from a spin (a holder at 100 % CPU can be a corpse), so the
discriminator is syscall progress, and that is what this samples.

    iowatch.py --pid <pid> --node 2 --watch <out.json> --out samples.jsonl --period-s 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CLK_TCK = 100.0


def proc_io(pid: int) -> dict:
    d = {}
    for ln in Path(f"/proc/{pid}/io").read_text().splitlines():
        k, _, v = ln.partition(":")
        d[k] = int(v)
    return d


def proc_stat(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/stat").read_text()
    tail = raw[raw.rindex(")") + 2:].split()
    return {"state": tail[0], "utime": int(tail[11]), "stime": int(tail[12]),
            "threads": int(tail[17])}


def aiclk(node: int):
    try:
        return int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read_text())
    except OSError as e:
        return f"ERR:{e.errno}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--watch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period-s", type=float, default=15.0)
    ap.add_argument("--stall-s", type=float, default=90.0,
                    help="flag a wedge once syscr AND syscw have been frozen this long")
    a = ap.parse_args()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    f = a.out.open("w")
    prev = None
    frozen_since = None
    while Path(f"/proc/{a.pid}").exists():
        t = time.time()
        try:
            io, st = proc_io(a.pid), proc_stat(a.pid)
        except (OSError, ValueError):
            break
        watch_age = round(t - a.watch.stat().st_mtime, 1) if a.watch.exists() else None
        rec = {"t": round(t, 2), "utc": time.strftime("%H:%M:%SZ", time.gmtime(t)),
               "state": st["state"], "threads": st["threads"],
               "syscr": io["syscr"], "syscw": io["syscw"],
               "rchar": io["rchar"], "wchar": io["wchar"],
               "aiclk": aiclk(a.node), "watch_age_s": watch_age}
        if prev:
            dt = t - prev["t"]
            rec["d_syscr"] = io["syscr"] - prev["syscr"]
            rec["d_syscw"] = io["syscw"] - prev["syscw"]
            cpu_ticks = (st["utime"] + st["stime"]) - (prev["utime"] + prev["stime"])
            rec["pcpu"] = round(100.0 * cpu_ticks / CLK_TCK / dt, 1)
            if rec["d_syscr"] == 0 and rec["d_syscw"] == 0:
                frozen_since = frozen_since or prev["t"]
                rec["frozen_s"] = round(t - frozen_since, 1)
                rec["WEDGE"] = rec["frozen_s"] >= a.stall_s and rec["pcpu"] > 50.0
            else:
                frozen_since = None
        prev = dict(rec, t=t, utime=st["utime"], stime=st["stime"])
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(json.dumps(rec), flush=True)
        time.sleep(a.period_s)
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
