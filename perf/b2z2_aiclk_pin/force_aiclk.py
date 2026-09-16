#!/usr/bin/env python3
"""Send Blackhole ARC FORCE_AICLK through tt-kmd's SMC message queue and watch what the chip does.

tt-kmd owns the ARC message queue and multiplexes it over every open fd
(TENSTORRENT_IOCTL_SMC_MSG), so this needs no UMD and does not disturb a running job's own
messages. Message layout is UMD's: message[0] is the type, message[1..7] the arguments
(blackhole_arc_message_queue.cpp:102). A response status of 0xFF means the firmware does not
know the message.
"""
import fcntl, os, struct, sys, time
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
            if e.errno == 11:          # EAGAIN, still pending
                time.sleep(0.005)
                continue
            raise
        resp = struct.unpack("=IIII8I", bytes(buf))[4:]
        return resp[0] & 0xFF, resp[0] >> 16
    raise TimeoutError("no ARC response")


def main():
    node, target, secs = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    root = Path(f"/sys/class/tenstorrent/tenstorrent!{node}")
    hw = next(root.glob("device/hwmon/hwmon*"))

    def tele():
        return (int((root / "tt_aiclk").read_text()),
                int((hw / "power1_input").read_text()) / 1e6,
                int((hw / "temp1_input").read_text()) / 1e3)

    fd = os.open(f"/dev/tenstorrent/{node}", os.O_RDWR | os.O_APPEND)
    try:
        print("before          aiclk=%d power=%.1fW temp=%.1fC" % tele(), flush=True)
        status, ret = smc(fd, FORCE_AICLK, target)
        print("FORCE_AICLK(%d) -> status=0x%02X ret=%d" % (target, status, ret), flush=True)
        t0 = time.time()
        while time.time() - t0 < secs:
            time.sleep(1.0)
            print("  t=%2.0fs aiclk=%d power=%.1fW temp=%.1fC" % ((time.time() - t0,) + tele()),
                  flush=True)
        status, ret = smc(fd, FORCE_AICLK, 0)
        print("FORCE_AICLK(0) release -> status=0x%02X ret=%d" % (status, ret), flush=True)
        for _ in range(5):
            time.sleep(1.0)
            print("  released aiclk=%d power=%.1fW temp=%.1fC" % tele(), flush=True)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
