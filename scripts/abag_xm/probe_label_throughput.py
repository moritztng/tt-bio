#!/usr/bin/env python3
"""Is the label pool starved, or is it the wrong width? Answer in 20 seconds.

Four passes in a row have read a labelling rate off `labels.json` mtimes and attributed it
to the pool's `--workers` setting. The rate is not set by the width. It is set by the CPU
share the pool gets, and on qb1 that share is whatever the co-tenant leaves:

    22:10Z  1.84 cores   3.80 dirs/min      pool at --workers 12
    23:10Z  1.33 cores   2.73 dirs/min      pool at --workers 12
    00:10Z  0.52 cores   1.07 dirs/min      pool at --workers 48
    00:40Z  0.19 cores   (no output)        pool at --workers 48

Every one of those is the same pool doing the same work at a different core share. Reading
the last two as "the 48-worker bump made it slower" is the error this probe exists to stop.
The invariant is per-dir CPU cost, not per-dir wall time:

    dirs/min = 60 * cores_available_to_pool / cost_core_seconds_per_dir

`--cost 29.1` is pass 38's measurement (12 workers, 218 s/dir, 1.60 cores), and it is an
opendde figure; the other three models have not been measured separately. Treat it as an
order-of-magnitude constant, not a calibrated one.

Width still matters, for one reason only: a pool of W workers cannot absorb more than W
cores. When the co-tenant exits and hands back ~30, a 48-wide pool takes them and a 12-wide
pool leaves 18 on the floor. That is the whole case for staying wide, and it is why a
low measured rate under co-tenancy is NOT a reason to narrow the pool.

Read-only. Touches /proc and the galaxy tree's directory names; writes nothing.

    python3 scripts/abag_xm/probe_label_throughput.py --base ~/abag_xm/deepn/galaxy
"""
import argparse
import os
import time
from pathlib import Path

HZ = os.sysconf("SC_CLK_TCK")


def _stat(pid):
    """(cpu_ticks, ppid) for one pid, or None if it vanished."""
    try:
        raw = Path("/proc/%s/stat" % pid).read_text()
    except OSError:
        return None
    f = raw.rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12]), int(f[1])


def _snapshot():
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        s = _stat(p)
        if s:
            out[int(p)] = s
    return out


def _descendants(root, snap):
    kids = {}
    for pid, (_, ppid) in snap.items():
        kids.setdefault(ppid, []).append(pid)
    seen, todo = {root}, [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in seen:
                seen.add(c)
                todo.append(c)
    return seen


def _cmdline(pid):
    try:
        return Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


def _labeler_pid():
    for p in os.listdir("/proc"):
        if p.isdigit() and "abag_xm_deepn_label.py" in _cmdline(int(p)) and "grep" not in _cmdline(int(p)):
            s = _stat(p)
            # the launcher shell also matches; take the python whose parent is not itself a match
            if s and "python" in _cmdline(int(p)).split(" ")[0]:
                return int(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="~/abag_xm/deepn/galaxy")
    ap.add_argument("--pid", type=int, default=0, help="labeler pid; auto-detected when 0")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cost", type=float, default=29.1, help="CPU cost per dir, core-seconds (pass 38)")
    ap.add_argument("--handover-cores", type=float, default=30.0)
    args = ap.parse_args()

    pid = args.pid or _labeler_pid()
    if not pid:
        print("labeler ABSENT -- no abag_xm_deepn_label.py process; the supervisor should relaunch it")
        return 1

    a, ta = _snapshot(), _stat("self")
    boot_a = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    time.sleep(args.seconds)
    b = _snapshot()
    boot_b = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]

    tree = _descendants(pid, b)
    pool = sum(b[p][0] - a.get(p, (0, 0))[0] for p in tree if p in b) / HZ / args.seconds
    box = (sum(boot_b) - sum(boot_a)) / HZ / args.seconds
    idle = ((boot_b[3] + boot_b[4]) - (boot_a[3] + boot_a[4])) / HZ / args.seconds
    ncpu = os.cpu_count()

    rest = []
    for p, (t, _) in b.items():
        if p in tree:
            continue
        c = (t - a.get(p, (0, 0))[0]) / HZ / args.seconds
        if c > 0.5:
            rest.append((c, p))
    rest.sort(reverse=True)

    base = Path(os.path.expanduser(args.base))
    pending = 0
    total = 0
    per_model = []
    for m in sorted(d for d in base.iterdir() if d.is_dir()):
        dirs = [x for x in m.iterdir() if x.is_dir()]
        lab = sum(1 for x in dirs if (x / "labels.json").exists())
        per_model.append((m.name, lab, len(dirs)))
        pending += len(dirs) - lab
        total += len(dirs)

    print("cores %d | pool %.2f (%.1f pct) | other tenants %.2f | idle %.2f"
          % (ncpu, pool, 100.0 * pool / ncpu, box - idle - pool, idle))
    print("pool width: %d direct children of pid %d" % (len(_descendants(pid, b) & set(k for k, v in b.items() if v[1] == pid)), pid))
    for c, p in rest[:3]:
        print("  co-tenant %5.2f cores  pid %-8d %s" % (c, p, _cmdline(p)[:96]))
    print()
    for name, lab, tot in per_model:
        print("  %-12s %5d / %-5d pending %4d" % (name, lab, tot, tot - lab))
    print("  %-12s %5d / %-5d pending %4d" % ("TOTAL", total - pending, total, pending))
    print()

    now_rate = 60.0 * pool / args.cost
    hand_rate = 60.0 * args.handover_cores / args.cost
    print("cost %.1f core-s/dir (pass 38, opendde)" % args.cost)
    print("  at the current %.2f cores : %6.2f dirs/min -> %8.1f h for %d pending"
          % (pool, now_rate, (pending / now_rate / 60.0) if now_rate else float("inf"), pending))
    print("  at a handed-back %.0f cores: %6.2f dirs/min -> %8.2f h for %d pending"
          % (args.handover_cores, hand_rate, pending / hand_rate / 60.0, pending))
    print()
    print("A low rate here is a co-tenant reading, not a width reading. Do not narrow the pool on it:")
    print("a W-wide pool cannot absorb more than W cores when the co-tenant exits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
