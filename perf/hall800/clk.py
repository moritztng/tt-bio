#!/usr/bin/env python3
"""Hold and sample Blackhole AICLK on one node for the duration of a measurement.

Same mechanism as perf/c10_fold_census/force_aiclk.py and c12_profiled_fold/clk.py: tt-kmd
SMC message queue, ARC message FORCE_AICLK = 0x33, sysfs tt_aiclk for the read-back. The one
addition is the re-assert. tt-kmd re-sends an AICLK-low aggregate on every legacy (non-O_APPEND)
device open, so the fold process clobbers a clock this script forced before it started. Forcing
once and trusting it silently measures the governor instead of the target, which is exactly the
artifact this row is not allowed to report, so the force is re-sent every --reassert-ms.

    clk.py --node 0 --target 1350 --out clock.jsonl &
    ... run the fold ... then SIGTERM; the force is released on the way out.

Each sample line is {"t": <epoch>, "mhz": <MHz>}. summarise() prints min/mean/max and the
fraction of samples at or above the target, which is what a perf claim has to quote.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import struct
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
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--target", type=int, default=1350)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period-ms", type=float, default=20.0)
    ap.add_argument("--reassert-ms", type=float, default=250.0)
    ap.add_argument("--max-s", type=float, default=3600.0)
    a = ap.parse_args()

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    try:
        st, ret = smc(fd, FORCE_AICLK, a.target)
        print("node%d FORCE_AICLK(%d) status=0x%02X ret=%d before=%s"
              % (a.node, a.target, st, ret, aiclk(a.node)), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w") as f:
            f.write(json.dumps({"node": a.node, "target": a.target,
                                "force_status": "0x%02X" % st}) + "\n")
            t_end = time.time() + a.max_s
            next_reassert = time.time() + a.reassert_ms / 1e3
            while not stop["now"] and time.time() < t_end:
                now = time.time()
                if now >= next_reassert:
                    try:
                        smc(fd, FORCE_AICLK, a.target)
                    except Exception as e:                                    # noqa: BLE001
                        f.write(json.dumps({"t": now, "reassert_error": repr(e)}) + "\n")
                    next_reassert = now + a.reassert_ms / 1e3
                f.write(json.dumps({"t": now, "mhz": aiclk(a.node)}) + "\n")
                f.flush()
                time.sleep(a.period_ms / 1e3)
    finally:
        try:
            smc(fd, FORCE_AICLK, 0)
            print("node%d released" % a.node, flush=True)
        except Exception as e:                                                # noqa: BLE001
            print("node%d release FAILED %r" % (a.node, e), flush=True)
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
