#!/usr/bin/env python3
"""Force and sample the Blackhole ARC clock for the length of one measurement.

The force mechanism (ARC message 0x33 through tt-kmd IOCTL 17) is the one `tt_bio/aiclk.py` uses on
branch wk/qb2-aiclk-force (commit ca8244881, "Hold the Blackhole card clock through a fold"); it is
vendored here because that module is not on main and a perf harness may not depend on an unmerged
branch. `sample()` reads the same telemetry sysfs the fleet's cell_now.py reads.
"""
from __future__ import annotations

import atexit
import fcntl
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_IOCTL_SMC_MSG = (0xFA << 8) | 17
_POST, _POLL = 1 << 0, 1 << 1
_FORCE_AICLK = 0x33
_LAYOUT = "=IIII8I"

_fds: dict[int, int] = {}


def nodes_open_by_this_process() -> list[int]:
    """The /dev/tenstorrent/N this process has open. TT_VISIBLE_DEVICES is a UMD logical id, not a
    device node, so the force has to be aimed off our own fd table."""
    out = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent/"):
            tail = target.rsplit("/", 1)[1]
            if tail.isdigit():
                out.add(int(tail))
    return sorted(out)


def aiclk(node: int) -> int:
    return int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read_text())


def _smc(fd: int, msg_type: int, *args: int) -> int:
    msg = [msg_type] + list(args) + [0] * (7 - len(args))
    fcntl.ioctl(fd, _IOCTL_SMC_MSG, struct.pack(_LAYOUT, 48, _POST, 0, 0, *msg))
    buf = bytearray(struct.pack(_LAYOUT, 48, _POLL, 0, 0, *([0] * 8)))
    for _ in range(400):
        try:
            fcntl.ioctl(fd, _IOCTL_SMC_MSG, buf, True)
        except OSError as e:
            if e.errno == 11:
                time.sleep(0.005)
                continue
            raise
        return struct.unpack(_LAYOUT, bytes(buf))[4] & 0xFF
    raise TimeoutError(f"no ARC response to 0x{msg_type:02X}")


def force(mhz: int, nodes: list[int] | None = None) -> list[int]:
    """Hold every open chip at `mhz`. Returns the nodes held. Released at exit."""
    held = []
    for node in nodes if nodes is not None else nodes_open_by_this_process():
        fd = _fds.get(node) or os.open(f"/dev/tenstorrent/{node}", os.O_RDWR | os.O_APPEND)
        _fds[node] = fd
        status = _smc(fd, _FORCE_AICLK, mhz)
        if status != 0:
            raise OSError(f"FORCE_AICLK({mhz}) refused on node {node}, status 0x{status:02X}")
        held.append(node)
    return held


def release() -> None:
    for node, fd in list(_fds.items()):
        try:
            _smc(fd, _FORCE_AICLK, 0)
        finally:
            os.close(fd)
            del _fds[node]


atexit.register(release)


class Sampler:
    """Samples AICLK at ~500 Hz in a SUBPROCESS for as long as it runs.

    A sampler thread does not work here: ttnn holds the GIL across a device call, so a thread gets
    one sample per timed interval (n=1, measured) and the clock would be recorded before and after
    the measurement rather than during it. A separate process is not blocked by our GIL.
    """

    def __init__(self, node: int, hz: float = 500.0):
        self.node, self.dt = node, 1.0 / hz
        self.tmp = tempfile.NamedTemporaryFile("r+", prefix=f"aiclk{node}_", delete=False)
        code = ("import sys,time\n"
                "p=sys.argv[1]; dt=float(sys.argv[2])\n"
                "while True:\n"
                "    print(open(p).read().strip(), flush=True)\n"
                "    time.sleep(dt)\n")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", code,
             f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk", str(self.dt)],
            stdout=self.tmp, stderr=subprocess.DEVNULL)

    def start(self) -> None:
        pass                                   # the process is already sampling

    def stop(self) -> dict:
        self.proc.terminate()                  # by handle, never by name
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.tmp.flush()
        self.tmp.seek(0)
        s = [int(x) for x in self.tmp.read().split() if x.isdigit()]
        self.tmp.close()
        os.unlink(self.tmp.name)
        return {"n": len(s), "min": min(s) if s else 0, "max": max(s) if s else 0,
                "mean": round(sum(s) / len(s), 1) if s else 0}
