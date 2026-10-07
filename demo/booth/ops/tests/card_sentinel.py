"""The card sentinel reboots on a card that stays off the bus, and on nothing else. Uses a fake
hwmon tree and a fake sysrq file, so it runs anywhere and reboots nothing.

    python3 demo/booth/ops/tests/card_sentinel.py

Cases: healthy cards (no reboot), one all-ones read that recovers, as a board reset might give
(no reboot), a card that stays all-ones (sysrq s, u, b in that order), and --observe (logs, no
reboot).
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SENTINEL = Path(__file__).resolve().parents[1] / "card_sentinel.py"
GONE = str(0xFFFFFFFF)


def tree(root):
    for i, bdf in enumerate(["0000:01:00.0", "0000:02:00.0"]):
        dev = root / "devices" / bdf
        dev.mkdir(parents=True)
        h = root / "hwmon" / f"hwmon{i}"
        h.mkdir(parents=True)
        (h / "device").symlink_to(dev)
        (h / "name").write_text("blackhole\n")
        (h / "power1_input").write_text("89000000\n")
        (h / "curr1_input").write_text("111000\n")
    other = root / "hwmon" / "hwmon9"
    other.mkdir()
    (other / "name").write_text("k10temp\n")
    (other / "power1_input").write_text(GONE + "\n")   # not a card: must be ignored
    return root / "hwmon" / "hwmon1"


def run(script, *extra):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        card = tree(root)
        sysrq, kmsg = root / "sysrq", root / "kmsg"
        p = subprocess.Popen([sys.executable, "-I", SENTINEL, "--every", "0.1", "--hwmon", root / "hwmon",
                              "--sysrq", sysrq, "--kmsg", kmsg, *extra],
                             stdout=subprocess.PIPE, text=True)
        try:
            time.sleep(0.5)
            script(card)
            time.sleep(3.5)
        finally:
            p.terminate()
        out = p.communicate(timeout=5)[0]
        keys = sysrq.read_text() if sysrq.exists() else ""
        return keys, out


def healthy(card):
    pass


def blip(card):
    (card / "power1_input").write_text(GONE + "\n")
    time.sleep(0.15)
    (card / "power1_input").write_text("89000000\n")


def dead(card):
    (card / "curr1_input").write_text(GONE + "\n")


fails = 0
for name, script, extra, want_keys, want_text in [
        ("healthy", healthy, (), "", None),
        ("one all-ones read", blip, (), "", "0000:02:00.0 reads all-ones"),
        ("card stays off the bus", dead, (), "sub", "rebooting"),
        ("observe", dead, ("--observe",), "", "would reboot")]:
    keys, out = run(script, *extra)
    ok = keys == want_keys and (want_text is None or want_text in out)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: sysrq {keys!r}  {(out.strip().splitlines() or [""])[-1]}")
sys.exit(1 if fails else 0)
