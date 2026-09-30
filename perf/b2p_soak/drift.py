#!/usr/bin/env python3
"""What a long campaign leaks, sampled beside it rather than from inside it.

A campaign that ruins a researcher's week does not crash: it creeps. Host memory climbs a few
GB per trajectory until the box OOM-kills it at trajectory 17, a file handle is left per
prediction until the process hits its limit, or the compile cache fills the disk the project
folder is on. None of that is visible in the campaign's own log, and a sampler that lives
inside the campaign process dies with it.

So this runs as its own process, is handed a pid, and appends one JSON object per sample to a
JSONL file until that pid exits. Every reading is a number a defect can be stated in:

  rss, hwm, threads        summed over the process TREE, since the pid handed over is usually a
                           wrapper (`timeout`, a shell) holding 1 MB and one thread
  fds                      open handles over the tree
  maps                     /proc/<pid>/maps lines over the tree -- a device mmap leak shows here
                           first
  procs, root_rss, root_hwm how many processes the tree holds, and the root's own figures
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


def _tree(pid: int) -> list[int]:
    """Every pid in the process tree rooted at this one, the root first.

    The pid a launcher hands over is usually a wrapper -- `timeout`, a shell -- holding 1 MB and
    one thread, so its own handle and thread counts say nothing about the campaign underneath it.
    Every reading that can leak is therefore taken over the whole tree.
    """
    found, stack = [], [pid]
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.append(current)
        stack += _children(current)
    return found


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


def _disk_free(path: str | None) -> int | None:
    """Free bytes on the filesystem the project is being written to.

    The project folder does not exist yet when the sampler starts: the campaign creates it after
    its own preflight, which is minutes of weight loading later. `statvfs` of a path that is not
    there raises, and the first version of this killed the sampler at its first sample and left a
    soak running with an empty drift series. So walk up to the first directory that exists.
    """
    candidate = os.path.abspath(path or ".")
    while candidate and not os.path.exists(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent
    try:
        stat = os.statvfs(candidate)
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


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
    tree = _tree(pid)
    status = [_status(member) for member in tree]
    fds = [_fds(member) for member in tree]
    maps = [_maps(member) for member in tree]
    root = _status(pid)
    return {"t": round(time.time(), 2), "utc": time.strftime("%FT%TZ", time.gmtime()),
            "alive": os.path.exists(f"/proc/{pid}"),
            "procs": len(tree),
            # Tree-wide, because the pid handed over is usually a wrapper.
            "rss": sum(one.get("vmrss", 0) for one in status),
            "hwm": sum(one.get("vmhwm", 0) for one in status),
            "threads": sum(one.get("threads", 0) for one in status),
            "fds": sum(count for count in fds if count is not None),
            "maps": sum(count for count in maps if count is not None),
            "root_rss": root.get("vmrss"), "root_hwm": root.get("vmhwm"),
            "project_bytes": _bytes(project),
            "cache_bytes": {c: _bytes(c) for c in caches},
            "disk_free": _disk_free(project),
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
