#!/usr/bin/env python3
"""Is this /dev/tenstorrent holder doing work, or is it a host-spin corpse?

The BH p300c host-spin wedge burns 100 % of a core forever while holding its chip, and it ignores
SIGINT and SIGTERM (the handler is registered but never runs). It therefore looks exactly like a
busy fold to every tool we have: `ps` shows high CPU, `fuser` shows the fd held, and `benchlock`
counts it as a live co-tenant. On 2026-09-17 that cost the box both chips of board ...410D for
1 h 41 m and 53 min, and `c12-host-decomp` deferred 3.7 h rather than reclaim its OWN dead process,
having concluded the BOARD was wedged when the wedge was its own `decomp.py`.

The discriminator is FORWARD PROGRESS, not CPU. Sample /proc/<pid>/{io,stat} twice and compare: CPU
advancing while the syscall counters do not is the signature. This never kills anything -- it
reports, and a human or the owning row decides.

Calibrated against a genuinely live fold, not asserted. The v0.9.0 release gate's `esmfold2` leg on
qb2 dev1, sampled over 20 s at 100 % CPU, moved `syscr +0` but `syscw +37 744` and `wchar +5.45 MB`.
So a live fold can issue ZERO read syscalls -- counting only `syscr` would have called it a corpse --
while the write side runs three orders of magnitude above the threshold here. The two wedges of
2026-09-17 sat at 26 518 syscalls TOTAL across 53 min, most of that startup.

    exit 0  every holder checked is making progress (or no holders)
    exit 1  at least one holder is WEDGED on this evidence
    exit 2  usage/environment problem

Usage:
    python3 wedge_check.py                    # every /dev/tenstorrent holder
    python3 wedge_check.py --card 2           # holders of /dev/tenstorrent/2 only
    python3 wedge_check.py --pid 1571080      # one pid, device check skipped (for testing)
    python3 wedge_check.py --window 30        # sampling window in seconds (default 20)

KNOWN FALSE-POSITIVE DIRECTION, and it is the one that matters. A fold parked in a long polling wait
on the device can burn CPU without issuing syscalls for a while, and this tool would call that
wedged. So WEDGED here is EVIDENCE TO LOOK CLOSER, never permission to kill. The corroboration that
actually decided both calls on 2026-09-17 was the holder's OUTPUT DIRECTORY sitting at zero bytes
written for 53 min and 1 h 41 m -- a span no polling wait survives. Take that second reading, over
minutes not seconds, before anyone touches a chip holder. The safe direction is validated by
measurement: a live fold clears the threshold by ~3700x.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEV = Path("/dev/tenstorrent")
CLK = os.sysconf("SC_CLK_TCK")


def holders(cards):
    """pids holding /dev/tenstorrent/<card> for each requested card, via fuser."""
    out = {}
    for c in cards:
        node = DEV / str(c)
        if not node.exists():
            continue
        r = subprocess.run(["sudo", "-n", "fuser", str(node)],
                           capture_output=True, text=True)
        for tok in r.stderr.split() + r.stdout.split():
            tok = tok.rstrip("cefmrtw")
            if tok.isdigit():
                out.setdefault(int(tok), []).append(c)
    return out


def sample(pid):
    """(cpu_seconds, syscall_count, bytes_moved) or None if the pid is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        cpu = (int(stat[11]) + int(stat[12])) / CLK          # utime + stime
        # Own processes expose /proc/<pid>/io directly; another user's need sudo. Try the cheap
        # path first so the tool works unprivileged on a host where you own the holder.
        try:
            raw = Path(f"/proc/{pid}/io").read_text()
        except PermissionError:
            raw = subprocess.run(["sudo", "-n", "cat", f"/proc/{pid}/io"],
                                 capture_output=True, text=True).stdout
        io = {}
        for line in raw.splitlines():
            k, _, v = line.partition(":")
            io[k.strip()] = int(v)
        if not io:
            return None
        return cpu, io["syscr"] + io["syscw"], io["rchar"] + io["wchar"]
    except (FileNotFoundError, ProcessLookupError, IndexError, KeyError, ValueError):
        return None


def argv_of(pid):
    try:
        a = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        return " ".join(a.split("\0")).strip() or "?"
    except OSError:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, action="append",
                    help="restrict to holders of this node; repeatable (default: all)")
    ap.add_argument("--pid", type=int, action="append",
                    help="check these pids directly, skipping the device lookup")
    ap.add_argument("--window", type=float, default=20.0, help="sampling window, seconds")
    ap.add_argument("--quiet", action="store_true", help="exit status only")
    a = ap.parse_args()

    if a.pid:
        targets = {p: [] for p in a.pid}
    else:
        cards = a.card if a.card else [int(p.name) for p in DEV.iterdir() if p.name.isdigit()] \
            if DEV.exists() else []
        if not cards:
            print("wedge_check: no /dev/tenstorrent nodes", file=sys.stderr)
            return 2
        targets = holders(sorted(cards))

    if not targets:
        if not a.quiet:
            print("wedge_check: no holders, nothing to classify")
        return 0

    first = {p: sample(p) for p in targets}
    live = {p: s for p, s in first.items() if s}
    if not live:
        if not a.quiet:
            print("wedge_check: holders vanished before sampling")
        return 0
    time.sleep(a.window)

    wedged = False
    for pid, s0 in sorted(live.items()):
        s1 = sample(pid)
        if s1 is None:
            if not a.quiet:
                print(f"pid {pid}: EXITED during the window")
            continue
        dcpu, dsys, dbytes = s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2]
        busy = dcpu / a.window
        # Spinning hard on the CPU while issuing essentially no syscalls and moving no bytes.
        # A live fold does thousands of syscalls per second; a parked one burns no CPU either.
        bad = busy > 0.5 and dsys < 10 and dbytes < 4096
        wedged |= bad
        held = ",".join(str(c) for c in targets.get(pid, [])) or "-"
        if not a.quiet:
            print(f"pid {pid:>8}  dev[{held}]  cpu {busy*100:6.1f}%  "
                  f"syscalls +{dsys:<8} bytes +{dbytes:<12} "
                  f"{'WEDGED (spinning, no progress)' if bad else 'progressing'}")
            print(f"           {argv_of(pid)[:150]}")
            if bad:
                print("           ^ corroborate before acting: check this holder's output dir "
                      "mtime over MINUTES. A polling wait can look like this for seconds.")
    return 1 if wedged else 0


if __name__ == "__main__":
    sys.exit(main())
