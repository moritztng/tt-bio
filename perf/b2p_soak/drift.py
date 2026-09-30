#!/usr/bin/env python3
"""What a long campaign leaks, sampled beside it rather than from inside it.

A campaign that ruins a researcher's week does not crash: it creeps. Host memory climbs a few
GB per trajectory until the box OOM-kills it at trajectory 17, a file handle is left per
prediction until the process hits its limit, or the compile cache fills the disk the project
folder is on. None of that is visible in the campaign's own log, and a sampler that lives
inside the campaign process dies with it.

So this runs as its own process, is handed a pid, and appends one JSON object per sample to a
JSONL file until that pid exits. Every reading is a number a defect can be stated in:

  rss, hwm, threads        /proc/<pid>/status, the process itself
  fds                      /proc/<pid>/fd, the handle count
  maps                     /proc/<pid>/maps line count -- a device mmap leak shows here first
  children_rss             design workers and MPNN subprocesses, summed
  project_bytes            what the campaign has written, so a disk filling is attributable
  cache_bytes              the JAX and tt-metal caches, which grow per length bucket
  disk_free, mem_available the two limits the box actually enforces
  aiclk                    the card's own clock, because a slowdown at trajectory 15 is a
                           clock story until proven otherwise

Usage: drift.py <pid> <out.jsonl> [--every 20] [--project DIR] [--cache DIR ...]
"""
import argparse
import json
import os
import pathlib
import time


def _status(pid: int) -> dict:
    out = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "Threads", "FDSize"):
                out[key.lower()] = int(value.split()[0]) * (1024 if key.startswith("Vm") else 1)
    except OSError:
        pass
    return out


def _fds(pid: int) -> int | None:
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return None


def _maps(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/maps") as maps:
            return sum(1 for _ in maps)
    except OSError:
        return None


def _children(pid: int) -> list[int]:
    kids = []
    try:
        for task in os.listdir(f"/proc/{pid}/task"):
            kids += open(f"/proc/{pid}/task/{task}/children").read().split()
    except OSError:
        return []
    return [int(kid) for kid in kids]


def _tree_rss(pid: int) -> int:
    total, seen, stack = 0, set(), [pid]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _status(current).get("vmrss", 0)
        stack += _children(current)
    return total


def _bytes(path: str | None) -> int | None:
    if not path or not os.path.exists(path):
        return None
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _mem_available() -> int | None:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _aiclk_path() -> str | None:
    """`tt_aiclk` for the chip this campaign was pinned to.

    `TT_VISIBLE_DEVICES` counts cards in PCI bus order and `tenstorrent!N` does not, so the
    naive node reads whichever card sysfs enumerated first -- an idle one at 800 MHz on qb1
    (`bcx-oplin`). Same ordering as `perf/bcx_stack/stack.py::sysfs_node`, without the import.
    """
    root = "/sys/class/tenstorrent"
    try:
        nodes = sorted(os.listdir(root),
                       key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
    except OSError:
        return None
    visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] or "0")
    if not visible.isdigit() or not 0 <= int(visible) < len(nodes):
        return None
    return f"{root}/{nodes[int(visible)]}/tt_aiclk"


_AICLK = _aiclk_path()


def _aiclk() -> int | None:
    """The card's own clock. The node is a right-aligned string, not a plain integer."""
    if not _AICLK:
        return None
    try:
        return int(pathlib.Path(_AICLK).read_text().strip())
    except (OSError, ValueError):
        return None


def sample(pid: int, project: str | None, caches: list[str]) -> dict:
    status = _status(pid)
    return {"t": round(time.time(), 2), "utc": time.strftime("%FT%TZ", time.gmtime()),
            "alive": os.path.exists(f"/proc/{pid}"),
            "rss": status.get("vmrss"), "hwm": status.get("vmhwm"),
            "threads": status.get("threads"), "fds": _fds(pid), "maps": _maps(pid),
            "tree_rss": _tree_rss(pid),
            "project_bytes": _bytes(project),
            "cache_bytes": {c: _bytes(c) for c in caches},
            "disk_free": (os.statvfs(project or ".").f_bavail
                          * os.statvfs(project or ".").f_frsize),
            "mem_available": _mem_available(), "load1": round(os.getloadavg()[0], 2),
            "aiclk": _aiclk()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", type=int)
    ap.add_argument("out")
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--project")
    ap.add_argument("--cache", action="append", default=[])
    args = ap.parse_args()
    # A walk of a growing project folder is the expensive part of a sample, so it is done at a
    # coarser cadence than the cheap /proc readings.
    with open(args.out, "a", buffering=1) as out:
        while True:
            row = sample(args.pid, args.project, args.cache)
            out.write(json.dumps(row) + "\n")
            if not row["alive"]:
                return 0
            time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
