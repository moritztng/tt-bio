#!/usr/bin/env python3
"""The gate channel of pair_channel_quiet must test LIVENESS, not argv.

Why this exists. On 2026-09-19 this row held benchlock for 67 minutes and measured nothing.
Its guard hard-failed on `release_gate pid 37787 live (owner pvx-land)` on 1406 consecutive
ticks. That process was blocked in flock waiting for the very lock this row was holding: zero
CPU, no device fd, and unable to perturb anything. The guard matched its argv and called it a
folding neighbour, so holding the lock manufactured the reason to keep holding it.

The three tests are a set on purpose. Admitting an idle gate is only safe if a BUSY gate is
still refused, so the other two burn real CPU and hold a real fd under the same argv the
first one sleeps under. Both mutations were run rather than argued, on pc:

    argv-only, i.e. the original defect   -> test 1 fails, 2 and 3 pass
    gate check deleted, always admit      -> all three fail

So no single edit both restores the bug and leaves the suite green.

Device-free: `C14_PCQ_DEV_PREFIX` points the guard at four ordinary files, which fuser
reports on exactly as it does on /dev/tenstorrent/N.
"""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

GUARD = Path(__file__).with_name("pair_channel_quiet.py")
# Unique per run, and handed to the guard as C14_PCQ_GATE_RE, so a real release gate running on
# the same host cannot decide these assertions either way. Without it the first test asserts on
# whatever else the box happens to be doing: on pc it failed against two live of3t-diffusion
# gates, which the guard had classified perfectly correctly.
MARKER = f"release_gate_selftest_{os.getpid()}_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def devdir(tmp_path):
    d = tmp_path / "dev"
    d.mkdir()
    for n in range(4):
        (d / str(n)).write_text("")
    return d


def run_guard(devdir, card=0):
    return subprocess.run(
        [sys.executable, str(GUARD), "--card", str(card),
         "--maxload", "999", "--drift", "999", "--settle", "0"],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "C14_PCQ_DEV_PREFIX": str(devdir), "C14_PCQ_GATE_RE": MARKER},
    )


def spawn(code, *args):
    p = subprocess.Popen([sys.executable, "-c", code, MARKER, *args])
    time.sleep(1.5)          # let it be visible to ps and start burning, if it burns
    return p


def kill(p):
    p.kill()
    p.wait(timeout=30)


def test_a_sleeping_gate_is_not_a_blocker(devdir):
    """A gate parked in benchlock's queue burns nothing and holds nothing. Admit."""
    p = spawn("import time; time.sleep(600)")
    try:
        r = run_guard(devdir)
    finally:
        kill(p)
    assert "present but IDLE" in r.stdout, r.stdout
    assert str(p.pid) in r.stdout, r.stdout
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_folding_gate_still_hard_fails(devdir):
    """NEGATIVE CONTROL: same argv, real CPU. Must refuse, or the test above proves nothing."""
    p = spawn("x = 0\nwhile True:\n    x += 1")
    try:
        r = run_guard(devdir)
    finally:
        kill(p)
    assert "NON-stationary neighbour. Hard fail." in r.stdout, r.stdout
    assert "ticks in 2 s" in r.stdout, r.stdout
    assert r.returncode == 1, r.stdout + r.stderr


def test_a_gate_holding_a_device_fd_hard_fails(devdir):
    """The other half of liveness: an fd on a chip counts even while the gate sleeps."""
    p = spawn("import sys, time; f = open(sys.argv[2]); time.sleep(600)", str(devdir / "0"))
    try:
        r = run_guard(devdir)
    finally:
        kill(p)
    assert "device fd on" in r.stdout, r.stdout
    assert r.returncode == 1, r.stdout + r.stderr
