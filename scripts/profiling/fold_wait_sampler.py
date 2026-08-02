#!/usr/bin/env python3
"""Where does a fold's device thread go when it is not on a CPU?

Motivation (state doc §29). Across the 32-way concurrency curve the per-fold host CPU is
essentially constant (381 core-s at N=1, 394 core-s at N=32, +3.4%) while the per-fold wall grows
15.2% (339.7 s -> 391.5 s). CPU that does not grow with a wall that does means the extra time is
spent **off CPU**. Every host *resource* has been measured and excluded (cores, threads, PCIe root
complex, DRAM bandwidth, NUMA, NFS, daemons), which rules out contention for a resource but says
nothing about *what call the thread is blocked in*. This samples exactly that.

Design notes, because the naive version perturbs what it measures:

* A fold is a process tree (``timeout`` -> ``tt_bio.main predict`` -> forked child ->
  ``multiprocessing.spawn`` grandchild) and ~130 of its ~131 threads are parked torch/OMP pool
  threads sitting in ``futex_wait_queue`` forever. Sampling all of them across 32 folds is ~4k
  /proc reads per sample; at any useful rate that is a couple of cores of measurement overhead on
  the very box whose CPU behaviour is in question.
* So: a slow scan (default every 5 s) re-identifies each fold's single hottest thread by CPU delta,
  and a fast loop samples only those (one per fold) for state + wchan. 32 threads at 100 Hz is
  ~3.2k reads/s, which is noise.

Verified against a live 8-way box before use: the device thread reads 1.067 cores and **100% of its
samples in state R**, i.e. at N=8 it never blocks. That is the control the N=32 run is compared to.

Output: one JSON summary to stdout, plus ``--jsonl`` per-fold rows. The load-bearing numbers are
``on_cpu_frac`` (share of samples in R) and the ``wchan`` histogram of everything else.

Read-only: /proc reads and nothing else. Safe to attach to another worker's folds.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _stat_fields(pid: int, tid: int | None = None) -> list[str] | None:
    p = f"/proc/{pid}/stat" if tid is None else f"/proc/{pid}/task/{tid}/stat"
    raw = _read(p)
    if raw is None:
        return None
    # comm can contain spaces and parens; everything after the last ')' is positional
    try:
        rest = raw[raw.rindex(") ") + 2 :].split()
    except ValueError:
        return None
    return rest


def _cpu_ticks(pid: int, tid: int | None = None) -> int | None:
    f = _stat_fields(pid, tid)
    if f is None or len(f) < 13:
        return None
    return int(f[11]) + int(f[12])  # utime, stime (0-based after state)


def _state(pid: int, tid: int) -> str | None:
    f = _stat_fields(pid, tid)
    return f[0] if f else None


def device_procs() -> dict[int, str]:
    """Device-owning grandchildren, keyed by pid -> the target its parent is folding.

    The grandchild is a bare ``multiprocessing.spawn`` process, so its own cmdline says nothing
    about which fold it belongs to; the label comes from walking up to the ``tt_bio.main predict``
    ancestor.
    """
    out: dict[int, str] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        cmd = _read(f"/proc/{pid}/cmdline")
        if not cmd or "spawn_main" not in cmd:
            continue
        label = "?"
        walk = pid
        for _ in range(4):
            f = _stat_fields(walk)
            if f is None or len(f) < 2:
                break
            walk = int(f[1])  # ppid
            pcmd = _read(f"/proc/{walk}/cmdline")
            if pcmd and "tt_bio.main" in pcmd:
                parts = [x for x in pcmd.split("\0") if x]
                label = next((x for x in parts if x.endswith((".yaml", ".yml"))), "?")
                break
        out[pid] = os.path.basename(label)
    return out


def threads_of_interest(pid: int, window: float) -> list[tuple[int, str]]:
    """The two threads that carry the device round trip, tagged by role.

    ``main`` (tid == pid) is the thread that enqueues work and then blocks in
    ``FDMeshCommandQueue::finish_nolock``; the hottest *other* thread is tt-metal's
    ``read_completion_queue`` reader, which polls the device completion queue and signals the
    condvar main is sleeping on. Sampling only the hotter of the two hides half the round trip --
    whichever one is waiting is by construction the one not burning CPU.
    """
    try:
        tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        return []
    before = {t: _cpu_ticks(pid, t) for t in tids}
    time.sleep(window)
    best, best_d = None, 0
    for t in tids:
        if t == pid:
            continue
        a, b = before.get(t), _cpu_ticks(pid, t)
        if a is None or b is None:
            continue
        if b - a > best_d:
            best, best_d = t, b - a
    out = [(pid, "main")]
    if best is not None:
        out.append((best, "completion"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=120.0, help="seconds to sample")
    ap.add_argument("--hz", type=float, default=100.0, help="fast-loop sample rate per thread")
    ap.add_argument("--rescan", type=float, default=5.0, help="seconds between hot-thread rescans")
    ap.add_argument("--rescan-window", type=float, default=0.2, help="CPU-delta window for the rescan")
    ap.add_argument("--jsonl", default=None, help="append one JSON row per fold at the end")
    ap.add_argument("--label", default="", help="free-form tag copied into the output (e.g. 'N=32')")
    args = ap.parse_args()

    period = 1.0 / args.hz
    end = time.time() + args.duration

    # per-pid: Counter of (state, wchan), plus first/last cpu ticks for the cores figure
    hist: dict[tuple[int, str], collections.Counter] = collections.defaultdict(collections.Counter)
    labels: dict[int, str] = {}
    cpu0: dict[int, tuple[float, int]] = {}
    cpu1: dict[int, tuple[float, int]] = {}
    watch: dict[int, list[tuple[int, str]]] = {}
    next_rescan = 0.0

    while time.time() < end:
        now = time.time()
        if now >= next_rescan:
            procs = device_procs()
            labels.update(procs)
            for pid in procs:
                cur = watch.get(pid)
                if not cur or any(_state(pid, t) is None for t, _ in cur):
                    got = threads_of_interest(pid, args.rescan_window)
                    if got:
                        watch[pid] = got
                c = _cpu_ticks(pid)
                if c is not None:
                    cpu0.setdefault(pid, (time.time(), c))
                    cpu1[pid] = (time.time(), c)
            for pid in list(watch):
                if pid not in procs:
                    watch.pop(pid, None)
            next_rescan = time.time() + args.rescan

        for pid, tids in list(watch.items()):
            for tid, role in tids:
                st = _state(pid, tid)
                if st is None:
                    continue
                if st == "R":
                    hist[(pid, role)][("R", "-")] += 1
                else:
                    w = (_read(f"/proc/{pid}/task/{tid}/wchan") or "-").strip() or "-"
                    hist[(pid, role)][(st, w)] += 1
        time.sleep(period)

    rows = []
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (pid, role), h in sorted(hist.items()):
        total = sum(h.values())
        if not total:
            continue
        agg[role].update(h)
        t0, c0 = cpu0.get(pid, (0, 0))
        t1, c1 = cpu1.get(pid, (0, 0))
        cores = ((c1 - c0) / HZ / (t1 - t0)) if t1 > t0 else None
        rows.append(
            {
                "label": args.label,
                "pid": pid,
                "role": role,
                "target": labels.get(pid, "?"),
                "samples": total,
                "on_cpu_frac": h[("R", "-")] / total,
                "tree_cores": round(cores, 3) if cores is not None else None,
                "waits": {f"{s}:{w}": n for (s, w), n in h.most_common() if s != "R"},
            }
        )

    def _mean(role: str, key: str):
        vals = [r[key] for r in rows if r["role"] == role and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "label": args.label,
        "folds": len({r["pid"] for r in rows}),
        "samples": sum(sum(c.values()) for c in agg.values()),
        "on_cpu_frac_main": _mean("main", "on_cpu_frac"),
        "on_cpu_frac_completion": _mean("completion", "on_cpu_frac"),
        "cores_mean": _mean("main", "tree_cores"),
        "waits_main": {f"{s}:{w}": n for (s, w), n in agg["main"].most_common(10) if s != "R"},
        "waits_completion": {f"{s}:{w}": n for (s, w), n in agg["completion"].most_common(10) if s != "R"},
    }
    print(json.dumps(summary, indent=1))
    if args.jsonl:
        with open(args.jsonl, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.write(json.dumps({"_summary": summary}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
