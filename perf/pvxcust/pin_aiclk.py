#!/usr/bin/env python3
"""Hold FORCE_AICLK at a target for as long as this process lives, then release.

`perf/c10_bare_baseline/force_aiclk.py` forces for a fixed number of seconds, which is a
probe, not a pin: a measurement that runs longer than the argument finishes on the
governor's clock and is unlabelled. This one forces, writes READY, and blocks until it is
signalled, so the pin's lifetime is the measurement's lifetime by construction.

Also samples the clock to a log while it holds, so a run that drifts off the pin says so
in its own artifact rather than being discovered later.
"""
import fcntl, os, signal, struct, sys, time
from pathlib import Path

IOC = (0xFA << 8) | 17
POST, POLL = 1 << 0, 1 << 1
FORCE_AICLK = 0x33
RUN = {"go": True}


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


def main():
    node, target = int(sys.argv[1]), int(sys.argv[2])
    root = Path(f"/sys/class/tenstorrent/tenstorrent!{node}")
    fd = os.open(f"/dev/tenstorrent/{node}", os.O_RDWR | os.O_APPEND)
    signal.signal(signal.SIGTERM, lambda *a: RUN.__setitem__("go", False))
    signal.signal(signal.SIGINT, lambda *a: RUN.__setitem__("go", False))
    try:
        status, _ = smc(fd, FORCE_AICLK, target)
        print(f"PIN node={node} target={target} status=0x{status:02X} "
              f"aiclk={int((root/'tt_aiclk').read_text())}", flush=True)
        print("READY", flush=True)
        off = 0
        while RUN["go"]:
            time.sleep(5.0)
            clk = int((root / "tt_aiclk").read_text())
            if clk != target:
                off += 1
                print(f"OFF-PIN aiclk={clk} (target {target}) n={off}", flush=True)
        print(f"off_pin_samples={off}", flush=True)
    finally:
        smc(fd, FORCE_AICLK, 0)
        os.close(fd)
        print("RELEASED", flush=True)


if __name__ == "__main__":
    main()
