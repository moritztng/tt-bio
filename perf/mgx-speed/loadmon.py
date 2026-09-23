"""Box load, once a minute, split by the chip each runnable thread's process was pinned to.

The speed bar voids a rung whose host passed 1.5x nproc, and part of whglx's load is not ours to
remove (the live app on cards 24-27, a co-tenant on card 1). This says how much each card's
processes contribute, so that share can be subtracted and named instead of guessed.

Every 5 s it counts threads in state R, keyed by TT_VISIBLE_DEVICES of the owning process
(unreadable or unset -> "-"). Every 60 s it appends one JSON line: the 1-min loadavg, nproc and
the mean runnable count per card over that minute.

    python perf/mgx-speed/loadmon.py <out.jsonl> [minutes]
"""
import json
import os
import sys
import time
from collections import Counter


def card_of(pid, cache):
    if pid not in cache:
        try:
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            vis = [e[19:].decode() for e in env if e.startswith(b"TT_VISIBLE_DEVICES=")]
            cache[pid] = vis[0] if vis else "-"
        except OSError:
            cache[pid] = "-"
    return cache[pid]


def runnable():
    """R-state threads per card right now."""
    out, cache = Counter(), {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                stat = open(f"/proc/{pid}/task/{tid}/stat").read()
            except OSError:
                continue
            if stat[stat.rfind(")") + 2] == "R":
                out[card_of(pid, cache)] += 1
    return out


path, minutes = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
for _ in range(minutes):
    acc, n, t_end = Counter(), 0, time.time() + 60
    while time.time() < t_end:
        acc.update(runnable())
        n += 1
        time.sleep(5)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load1": os.getloadavg()[0],
           "nproc": os.cpu_count(), "runnable": {k: round(v / n, 1) for k, v in sorted(acc.items())}}
    with open(path, "a") as fp:
        fp.write(json.dumps(rec) + "\n")
