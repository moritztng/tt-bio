#!/usr/bin/env python3
"""Node-parameterised clock sampler and prerequisite validator.

`control.py` is a pinned import and its clock reader and node check are both hard-wired to
node 0. Card 0's PCIe endpoint was quarantined by the host's containment guard at 29130.7 s of
uptime, 2.649 s into this row's first session, and its ARC has not answered since, so this row
runs on node 1. Editing the pinned helper would break its hash, so the node-specific halves are
re-implemented here and `test_node_control.py` proves this file reproduces `control.py`'s own
verdicts when the node is 0.

Everything node-agnostic -- `snapshot`, `digest`, `holders`, `own_nodes`, `coverage` and its
1350 MHz / 10 ms / no-read-error gate -- is still the pinned helper's.
"""
from __future__ import annotations

import json
import os
import select
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import control                                                                # noqa: E402

CARD_ATTRS = ("tt_aiclk", "tt_card_type", "tt_asic_id")


def aiclk_path(node: int) -> Path:
    return Path("/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % node)


def identity(node: int) -> dict:
    """What the chip says about itself, read straight off sysfs by the run that uses it."""
    root = Path("/sys/class/tenstorrent/tenstorrent!%d" % node)
    out = {"node": node}
    for a in CARD_ATTRS:
        try:
            out[a] = (root / a).read_text().strip()
        except OSError as e:
            out[a] = repr(e)
    try:
        out["pci"] = os.path.basename(os.path.realpath(root / "device"))
    except OSError as e:                                                      # noqa: BLE001
        out["pci"] = repr(e)
    return out


def ancestry(pid: int, read=None) -> list:
    """`pid` and every ancestor of it, youngest first. Used to tell one of my own forks from a
    genuinely foreign holder.

    The holder census reads `/proc`, and `subprocess` reaches an exec through a fork that
    briefly carries the PARENT's argv and every one of the parent's open fds, so a process that
    has the device open and is sampling holders at 10 Hz will occasionally photograph its own
    pre-exec fork and read it as a foreign holder of its own card. That is what took this row's
    first node-1 session down. The fix is the correct semantic anyway: "busy" means held by
    something that is not me and not mine.
    """
    read = read or (lambda p: Path("/proc/%d/stat" % p).read_text())
    chain, seen = [], set()
    while pid and pid not in seen:
        chain.append(pid)
        seen.add(pid)
        try:
            pid = int(read(pid).rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return chain


def foreign(holders_rows, node: int, pid: int, read=None) -> list:
    """Holders of `node` that are neither `pid` nor descended from it."""
    dev = "/dev/tenstorrent/%d" % node
    return [h for h in holders_rows
            if dev in h["nodes"] and pid not in ancestry(h["pid"], read)]


def validate(snapshot: dict, node: int, opened: bool = False, resample=None,
             retries: int = 2, pause: float = 0.25) -> None:
    """control.validate_snapshot, with the assigned node as an argument instead of a constant.

    A holder inside my own process tree is mine. A holder outside it has to still be there on a
    re-sample before the session is refused, so a photographed fork cannot fail a quiet host and
    a real co-tenant cannot slip through.
    """
    if (snapshot["containment"] != "active"
            or snapshot["module_srcversion"] != "A10759A24565BC5BBE903C5"):
        raise RuntimeError("containment or driver prerequisite failed: %s"
                           % {k: snapshot[k] for k in ("containment", "module_srcversion")})
    busy = foreign(snapshot["holders"], node, os.getpid())
    for _ in range(retries if busy else 0):
        if resample is None:
            resample = control.snapshot
        time.sleep(pause)
        busy = foreign(resample()["holders"], node, os.getpid())
        if not busy:
            break
    if busy:
        raise RuntimeError("assigned node busy: %s" % busy)
    if opened and snapshot["own_nodes"] != [dev_path(node)]:
        raise RuntimeError("unexpected device opens: %s" % snapshot["own_nodes"])


def dev_path(node: int) -> str:
    return "/dev/tenstorrent/%d" % node


def foreign_holders(rows, node: int, pid: int) -> list:
    """Every observation in which a process outside my tree held ANY node. All four nodes, not
    just the assigned one: a co-tenant anywhere on this host is evidence about the measurement."""
    out = []
    for row in rows:
        others = [h for h in row.get("holders", []) if pid not in ancestry(h["pid"])]
        if others:
            out.append({"monotonic_ns": row["monotonic_ns"], "holders": others})
    return out


def sample(path, node, owner):
    """~1 kHz tt_aiclk reads with monotonic brackets, plus a 10 Hz device-holder census."""
    stop = threading.Event()

    def observe():
        with Path(path).with_name("holders.jsonl").open("w") as out:
            while not stop.is_set():
                row = {"monotonic_ns": time.monotonic_ns(), "utc_ns": time.time_ns()}
                try:
                    row.update(holders=control.holders(), owner_nodes=control.own_nodes(owner))
                except BaseException as e:                                    # noqa: BLE001
                    row["error"] = repr(e)
                out.write(json.dumps(row) + "\n")
                out.flush()
                stop.wait(0.1)

    monitor = threading.Thread(target=observe)
    monitor.start()
    node_path = str(aiclk_path(node))
    try:
        with Path(path).open("w") as out:
            while not select.select([sys.stdin], [], [], 0.001)[0]:
                row = {"read_start_ns": time.monotonic_ns(), "utc_ns": time.time_ns(),
                       "node": node_path}
                try:
                    row["MHz"] = int(Path(node_path).read_text())
                except (OSError, ValueError) as e:
                    row["error"] = repr(e)
                row["read_end_ns"] = time.monotonic_ns()
                out.write(json.dumps(row) + "\n")
                out.flush()
    finally:
        stop.set()
        monitor.join(timeout=10)


if __name__ == "__main__":
    sample(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
