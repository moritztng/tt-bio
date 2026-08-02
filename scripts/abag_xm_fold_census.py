#!/usr/bin/env python3
"""Census of this host's fold processes: card ownership, real CPU, and leaked workers.

Written after two incidents in this campaign that a naive `ps` could not distinguish:

  * An orphaned `spawn_main` device worker (ppid=1) held a card for 5.7 h at ~1 core while
    producing nothing. Worse, because it kept the card's lease flock, the driver's deadlock
    watchdog read the starved fold behind it as "legitimately parked" and reset its idle
    counter forever -- so the hung fold was IMMUNE to the watchdog and burned to its
    wall-clock timeout. Any `ppid=1` worker or predict process is a leak; nothing reaps it.

  * CPU read from the fold PARENT is meaningless. The device work runs in a `spawn_main`
    grandchild, so a healthy fold's parent sits near zero while its tree runs at ~1 core.
    Judging liveness from the parent understates the tree by ~150x and will call a working
    fold idle. Always sum the whole tree -- which is what the watchdog itself does.

  python3 scripts/abag_xm_fold_census.py [--seconds 60]

Reports, per fold: whether any process in its tree holds a /dev/tenstorrent fd (HOLDS vs
parked), the tree's CPU over the sample window, and any orphans found. Read-only.
"""
import argparse
import os
import re
import socket
import time

CLK = os.sysconf("SC_CLK_TCK") or 100


def _read(path, default=""):
    try:
        return open(path).read()
    except Exception:
        return default


def _stat(pid):
    s = _read(f"/proc/{pid}/stat")
    if not s:
        return None
    try:
        return s.rsplit(")", 1)[1].split()
    except Exception:
        return None


def ppid_of(pid):
    f = _stat(pid)
    return int(f[1]) if f else None


def jiffies(pid):
    f = _stat(pid)
    return (int(f[11]) + int(f[12])) if f else 0


def cmdline(pid):
    return _read(f"/proc/{pid}/cmdline").replace("\0", " ")


def pids():
    return [int(d) for d in os.listdir("/proc") if d.isdigit()]


def tree_of(root):
    """root plus all descendants, from one /proc pass."""
    kids = {}
    for p in pids():
        pp = ppid_of(p)
        if pp is not None:
            kids.setdefault(pp, []).append(p)
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, ()))
    return out


def holds_device(tree):
    for p in tree:
        try:
            for fd in os.listdir(f"/proc/{p}/fd"):
                try:
                    if "tenstorrent" in os.readlink(f"/proc/{p}/fd/{fd}"):
                        return True
                except OSError:
                    pass
        except Exception:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60, help="CPU sample window")
    a = ap.parse_args()

    folds, orphans = [], []
    for p in sorted(pids()):
        cl = cmdline(p)
        pp = ppid_of(p)
        if "tt_bio.main predict" in cl:
            m = re.search(r"abag_xm/([a-z0-9]+)\.yaml --model ([a-z0-9-]+)", cl)
            if not m:
                continue
            if pp and "tt_bio.main predict" in cmdline(pp):
                continue  # the fold's own forked twin, not a separate fold
            if pp == 1:
                orphans.append(("predict", p, f"{m.group(2)}/{m.group(1)}"))
            folds.append((p, pp, f"{m.group(2)}/{m.group(1)}"))
        elif "spawn_main" in cl and pp == 1:
            orphans.append(("spawn_main", p, "device worker"))

    trees = {p: tree_of(p) for p, _, _ in folds}
    t0 = {p: sum(jiffies(x) for x in trees[p]) for p in trees}
    time.sleep(a.seconds)

    print(f"host {socket.gethostname()}  ({a.seconds}s window)")
    print(f"  {'pid':>8} {'ppid':>7} {'card':>6} {'cores':>6}  target")
    for p, pp, tag in folds:
        cores = (sum(jiffies(x) for x in trees[p]) - t0[p]) / float(CLK * a.seconds)
        print(f"  {p:>8} {pp:>7} {'HOLDS' if holds_device(trees[p]) else 'parked':>6} "
              f"{cores:>6.2f}  {tag}")
    held = sum(1 for p, _, _ in folds if holds_device(trees[p]))
    print(f"  -> {held} holding a card, {len(folds) - held} parked")
    if orphans:
        print("  LEAKED (ppid=1, nothing reaps these; they hold cards and blind the "
              "watchdog):")
        for kind, p, what in orphans:
            print(f"    {kind} pid {p} — {what}")
    else:
        print("  leaked processes: none")


if __name__ == "__main__":
    main()
