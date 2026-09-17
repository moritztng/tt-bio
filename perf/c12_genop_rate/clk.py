#!/usr/bin/env python3
"""Force and sample Blackhole AICLK on one or more nodes, for the duration of a measurement.

Forcing goes through tt-kmd's SMC message queue (TENSTORRENT_IOCTL_SMC_MSG, ARC message
FORCE_AICLK = 0x33), which tt-kmd multiplexes over every open fd, so this needs no UMD and does
not disturb the fold's own messages. Sampling reads `tt_aiclk` off sysfs.

Lifted from perf/c10_fold_census/force_aiclk.py (the ioctl layout and the 0x33 message) and
perf/c10_fold_census/node_control.py (the sysfs sampler), generalised to hold the clock for as
long as the parent measurement runs and to sample every node it forces. Runs as its own process
so the 1 kHz sampler never takes the fold's GIL: a sampler thread inside the folding process
would show up as host time in the very number this row is trying to split.

    clk.py --nodes 2,3 --target 1350 --out clock.jsonl
    # ... run the measurement ... then SIGTERM this process; it releases the force on the way out.

Each line of the jsonl is {"t": <time.time()>, "<node>": <MHz>, ...}. An interval qualifies only
if every sample inside it reads the target, there is no read error, and no gap exceeds 10 ms --
the same gate perf/c10_fold_census/control.py applies, re-implemented here for N nodes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import struct
import sys
import time
from pathlib import Path

IOC = (0xFA << 8) | 17
POST, POLL = 1 << 0, 1 << 1
FORCE_AICLK = 0x33


def smc(fd, msg_type, *args):
    msg = [msg_type] + list(args) + [0] * (7 - len(args))
    fcntl.ioctl(fd, IOC, struct.pack("=IIII8I", 48, POST, 0, 0, *msg))
    deadline = time.time() + 2.0
    while time.time() < deadline:
        buf = bytearray(struct.pack("=IIII8I", 48, POLL, 0, 0, *([0] * 8)))
        try:
            fcntl.ioctl(fd, IOC, buf, True)
        except OSError as e:
            if e.errno == 11:
                time.sleep(0.005)
                continue
            raise
        resp = struct.unpack("=IIII8I", bytes(buf))[4:]
        return resp[0] & 0xFF, resp[0] >> 16
    raise TimeoutError("no ARC response")


def aiclk(node):
    try:
        return int(Path("/sys/class/tenstorrent/tenstorrent!%d/tt_aiclk" % node).read_text())
    except OSError as e:
        return "ERR:%s" % e.errno


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--target", type=int, default=1350)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period-ms", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    a = ap.parse_args()
    nodes = [int(x) for x in a.nodes.split(",")]

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    fds = {n: os.open("/dev/tenstorrent/%d" % n, os.O_RDWR | os.O_APPEND) for n in nodes}
    forced = {}
    try:
        for n, fd in fds.items():
            st, ret = smc(fd, FORCE_AICLK, a.target)
            forced[n] = {"status": "0x%02X" % st, "ret": ret, "before": aiclk(n)}
            print("node%d FORCE_AICLK(%d) status=0x%02X ret=%d" % (n, a.target, st, ret),
                  flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w") as f:
            f.write(json.dumps({"forced": forced, "target": a.target,
                                "period_ms": a.period_ms}) + "\n")
            t_end = time.time() + a.max_s
            while not stop["now"] and time.time() < t_end:
                f.write(json.dumps(dict({"t": time.time()},
                                        **{str(n): aiclk(n) for n in nodes})) + "\n")
                f.flush()
                time.sleep(a.period_ms / 1e3)
    finally:
        for n, fd in fds.items():
            try:
                smc(fd, FORCE_AICLK, 0)
                print("node%d released" % n, flush=True)
            except Exception as e:                                            # noqa: BLE001
                print("node%d release FAILED %r" % (n, e), flush=True)
            os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
