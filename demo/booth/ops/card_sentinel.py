#!/usr/bin/python3 -I
"""Reboot the box at once when a Blackhole card drops off the PCIe bus.

    sudo /usr/local/sbin/booth-card-sentinel [--observe]

A card that no longer answers on the bus returns all-ones for every telemetry register. The driver
scales power and current in 32-bit arithmetic (tenstorrent-2.11.0 telemetry.c), so those files read
the wrapped products, 4293967296 uW and 4294966296 mA; only in0_input reads 4294967295. On qb2 that reading was followed by a host lockup every time it was seen: 66 of 66 between
2026-09-15 and 2026-10-07, 32 to 197 s later, with nothing in the kernel log. The screen sits at
1 to 3 fps until the lockup, then frozen until the hardware watchdog resets the board 150 s after
that. Rebooting at the first sign trades those 3 to 5 minutes for the ~45 s the box takes to boot.

Runs as root (booth-card-sentinel.service): only root can reboot without a clean shutdown, and a
clean shutdown stops the engine, which touches the dead card. --observe only logs, for running
next to something that must not be rebooted.
"""
import argparse
import glob
import os
import time

ONES = 0xFFFFFFFF
# what each file reads for an all-ones register: the driver multiplies a u32 and keeps 32 bits
GONE = {"power1_input": str(ONES * 1000000 % 2**32), "curr1_input": str(ONES * 1000 % 2**32),
        "in0_input": str(ONES)}


def read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return ""


def cards(hwmon):
    return {os.path.basename(os.path.realpath(f"{h}/device")): h
            for h in sorted(glob.glob(f"{hwmon}/hwmon*")) if read(f"{h}/name") == "blackhole"}


def gone(h):
    return any(read(f"{h}/{f}") == v for f, v in GONE.items())


def say(msg, kmsg):
    print(msg, flush=True)
    try:   # through netconsole this reaches another machine before the box goes
        with open(kmsg, "w") as f:
            f.write(f"<2>booth-card-sentinel: {msg}\n")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--observe", action="store_true", help="log only, never reboot")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between reads")
    ap.add_argument("--confirm", type=int, default=3, help="all-ones reads in a row before rebooting")
    ap.add_argument("--hwmon", default="/sys/class/hwmon")
    ap.add_argument("--sysrq", default="/proc/sysrq-trigger")
    ap.add_argument("--kmsg", default="/dev/kmsg")
    a = ap.parse_args()

    mode = " (observe only)" if a.observe else ""
    say(f"watching {sorted(cards(a.hwmon))}{mode}", a.kmsg)
    streak = {}
    while True:
        for dev, h in cards(a.hwmon).items():   # listed every pass: a board reset renumbers hwmon
            streak[dev] = streak.get(dev, 0) + 1 if gone(h) else 0
            if streak[dev] == 1:
                say(f"{dev} reads all-ones: the card is off the bus", a.kmsg)
            if streak[dev] != a.confirm:
                continue
            if a.observe:
                say(f"{dev} off the bus {a.confirm} reads in a row; would reboot now", a.kmsg)
                continue
            say(f"{dev} off the bus {a.confirm} reads in a row; rebooting before the host locks up", a.kmsg)
            for key in "sub":   # sync, remount read-only, reboot; s and u are queued, so give each a second
                with open(a.sysrq, "a") as f:
                    f.write(key)
                time.sleep(1)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
