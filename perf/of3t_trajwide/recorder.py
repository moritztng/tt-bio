#!/usr/bin/env python3
"""Mirror the live run into the branch, because the branch is the only storage on this fleet
that survives a qb2 reboot.

qb2 rebooted on 2026-09-21 at 12:48Z and wiped /tmp, which held every steplog, every
done-marker and the archive of the pre-D149 arms. The w_k dumps are ~190 MB a rung and cannot
go in git; the RECORD can. This copies each arm's steplog into `perf/of3t_trajwide/runs/`,
writes a STATUS.md a reader can check liveness against (rung counter, log mtime, pid), and
commits and pushes when anything changed. One writer, so three chains cannot race the index.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

R = "/home/ttuser/of3t_runs/trajwide"
G = "perf/of3t_trajwide/runs"
ARMS = ["shipped", "shipped_aa2", "permute", "stale", "norebind", "zero",
        "theirs", "theirs_aa2"]


def sh(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()


def alive(pat):
    out = sh("pgrep", "-af", pat)
    return [l for l in out.splitlines() if "pgrep" not in l]


def row(arm):
    sl = os.path.join(R, f"steplog_{arm}.json")
    mk = os.path.join(G, f"{arm}.done")
    log = os.path.join(R, (f"ours_{arm}.log" if not arm.startswith("theirs") else f"{arm}.log"))
    d = {}
    if os.path.exists(sl):
        try:
            d = json.load(open(sl))
        except Exception:
            d = {}
    marker = open(mk).read().strip() if os.path.exists(mk) else ""
    age = int(time.time() - os.path.getmtime(log)) if os.path.exists(log) else None
    n = len(d.get("steps") or [])
    if d.get("complete") and n == 20 and "rc=0" in marker:
        st = "COMPLETE"
    elif marker:
        st = "FAILED" if "rc=0" not in marker else "COMPLETE"
    elif n:
        st = "RUNNING" if age is not None and age < 900 else "STALLED"
    else:
        st = "not started" if age is None else "starting"
    return dict(arm=arm, rungs=n, status=st, marker=marker,
                log_age_s=age, ref_tree=d.get("ref_tree"))


def status_md(rows):
    live = alive("trajwide.py")
    out = ["# of3t-trajwide live run", "",
           f"Written by `recorder.py` at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}.",
           "Check liveness by the rung counter moving and `log_age_s` staying small, not by",
           "the clock. An arm with a marker whose rc is not 0 is FAILED, not finished.", "",
           "    arm           rungs  status      log age  marker"]
    for r in rows:
        out.append(f"    {r['arm']:<13} {r['rungs']:>2}/20  {r['status']:<11} "
                   f"{str(r['log_age_s']) + ' s':>8}  {r['marker']}")
    out += ["", "Reference tree resolved in-process (D149): "
            + (next((r["ref_tree"] for r in rows if r["ref_tree"]), "not yet read")), "",
            "Live `trajwide.py` processes:", ""]
    out += [f"    {l}" for l in live] or ["    none"]
    out.append("")
    return "\n".join(out)


def once():
    os.makedirs(G, exist_ok=True)
    rows = [row(a) for a in ARMS]
    for a in ARMS:
        sl = os.path.join(R, f"steplog_{a}.json")
        if os.path.exists(sl):
            dst = os.path.join(G, f"steplog_{a}.json")
            data = open(sl).read()
            if not os.path.exists(dst) or open(dst).read() != data:
                open(dst, "w").write(data)
    open(os.path.join(G, "STATUS.md"), "w").write(status_md(rows))
    subprocess.run(["git", "add", G], check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
        n = sum(r["rungs"] for r in rows)
        subprocess.run(["git", "commit", "-q", "-m",
                        f"of3t-trajwide: live steplogs, {n} rungs across 8 arms"], check=False)
        subprocess.run(["git", "push", "-q", "origin", "wk/of3t-trajwide"], check=False)
        return True
    return False


if __name__ == "__main__":
    period = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    while True:
        try:
            once()
        except Exception as e:                                  # never take the loop down
            print(f"{time.strftime('%FT%TZ', time.gmtime())} recorder error: {e}", flush=True)
        time.sleep(period)
