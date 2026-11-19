"""A device open cannot park forever and take every other chip on the host with it.

`device_lease.device_init_lock` serializes every device open and close on a host on purpose
(UMD's bring-up deadlocks when raced). So one open that never returns used to hold every
other chip's open behind it: qb1 2026-09-17 (all four cards), .114 2026-10-11 (21 opens, 24
min), qb2 2026-10-17 (57 min in the bring-up probe). Three bounds close that, each pinned here:

* a chip whose ARC firmware is dead is refused from sysfs in seconds, before any lease or lock
  (`runtime.refuse_dead_chips`);
* a holder still inside the lock at `INIT_LOCK_DEADLINE_S` ends, the kernel drops its flock,
  and the next opener proceeds; the bring-up probe has the same bound;
* a waiter says who holds the lock and since when.

Card-free: sysfs is a temp tree, the holder is a child process, and no device node is opened.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio import device_lease as dl, runtime as rt


# --- ARC heartbeat precheck -----------------------------------------------------------------

@pytest.fixture
def sysfs(monkeypatch, tmp_path):
    """A fake /sys/class/tenstorrent with chips 0 and 1 on nodes 0 and 1."""
    monkeypatch.setattr(rt, "_SYSFS_CLASS", str(tmp_path))
    monkeypatch.setattr(rt, "umd_index_to_dev_node", lambda: {0: 0, 1: 1})
    monkeypatch.setattr(rt, "HEARTBEAT_WINDOW_S", 0.5)

    def chip(node, value):
        d = tmp_path / f"tenstorrent!{node}"
        d.mkdir(exist_ok=True)
        if value is not None:
            (d / "tt_heartbeat").write_text(f"{value}\n")
        return d / "tt_heartbeat"
    return chip


def _tick(path, stop):
    n = 1000
    while not stop.is_set():
        n += 1
        tmp = path.with_suffix(".tmp")
        tmp.write_text(f"{n}\n")
        os.replace(tmp, path)    # sysfs never shows a half-written value; a plain write would
        time.sleep(0.1)          # 10 per second, as every measured chip ticks


def test_a_stuck_heartbeat_is_refused_within_the_window(sysfs):
    sysfs(0, 1234)
    t = time.time()
    with pytest.raises(rt.ChipNeedsResetError, match=r"chip\(s\) \[0\] need a reset.*stayed at 1234"):
        rt.refuse_dead_chips([0])
    assert time.time() - t < 2.0


def test_a_dead_arc_sentinel_is_refused_at_once(sysfs):
    sysfs(0, 0xFFFFFFFF)
    t = time.time()
    with pytest.raises(rt.ChipNeedsResetError, match="0xffffffff"):
        rt.refuse_dead_chips([0])
    assert time.time() - t < 0.2


def test_unreadable_telemetry_is_refused(sysfs):
    sysfs(0, None)
    os.mkdir(rt._SYSFS_CLASS + "/tenstorrent!0/tt_heartbeat")   # read fails with EISDIR
    with pytest.raises(rt.ChipNeedsResetError, match="cannot be read"):
        rt.refuse_dead_chips([0])


def test_a_live_chip_passes_fast_and_only_the_dead_one_is_named(sysfs):
    live = sysfs(0, 1000)
    sysfs(1, 77)
    stop = threading.Event()
    threading.Thread(target=_tick, args=(live, stop), daemon=True).start()
    try:
        t = time.time()
        assert rt.dead_heartbeats([0]) == {}
        assert time.time() - t < 0.4
        assert set(rt.dead_heartbeats([0, 1])) == {1}
    finally:
        stop.set()


def test_unknown_is_not_dead(sysfs):
    sysfs(0, None)              # an older kmd: no tt_heartbeat attribute
    rt.refuse_dead_chips([0, 5])  # and chip 5 is not in sysfs at all


def test_get_device_refuses_a_dead_chip_before_taking_a_lease(monkeypatch):
    T = pytest.importorskip("tt_bio.tenstorrent")
    leased = []
    monkeypatch.setattr(T, "_device", None)
    monkeypatch.setattr(T, "_device_lease", None)
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(dl, "physical_cards", lambda: ["0"])
    monkeypatch.setattr(dl, "arm_orphan_guard", lambda: None)
    monkeypatch.setattr(dl.CardSetLease, "acquire", lambda self: leased.append(1))
    monkeypatch.setattr(rt, "dead_heartbeats", lambda idx: {0: "its ARC heartbeat stayed at 9 for 2 s"})
    monkeypatch.setattr(rt, "umd_index_to_dev_node", lambda: {0: 0})
    with pytest.raises(rt.ChipNeedsResetError, match="tt-smi -r 0"):
        T.get_device()
    assert leased == [], "a refused chip must leave no lease behind"


# --- deadline on the locked section, and the waiter's report ---------------------------------

# Takes the bring-up lock and then blocks in one C call that holds the GIL, the way
# ttnn.open_device spins on a dirty chip. No Python-level timer could end it.
STUCK = r"""
import os, re, sys
sys.path.insert(0, os.environ["REPO"])
from tt_bio import device_lease as dl
dl.DEVICE_INIT_LOCK_PATH = os.environ["LOCK"]
dl.INIT_LOCK_DEADLINE_S = float(os.environ["DEADLINE"])
with dl.device_init_lock("open"):
    open(os.environ["READY"], "w").close()
    re.match(r"(a+)+$", "a" * 40 + "b")
print("returned")
"""

NEXT = r"""
import json, os, sys, time
sys.path.insert(0, os.environ["REPO"])
from tt_bio import device_lease as dl
dl.DEVICE_INIT_LOCK_PATH = os.environ["LOCK"]
with dl.device_init_lock("open"):
    t = time.time()
print(json.dumps(t))
"""


def _env(d, **kw):
    env = dict(os.environ, REPO=REPO, LOCK=os.path.join(d, "open.lock"),
               READY=os.path.join(d, "ready"), TT_VISIBLE_DEVICES="3",
               TT_BIO_LEASE_HOLDER="worker:stuck-test")
    env.update(kw)
    return env


def _wait_for(path, timeout=30):
    end = time.time() + timeout
    while not os.path.exists(path):
        assert time.time() < end, f"{path} never appeared"
        time.sleep(0.01)


def test_a_holder_stuck_past_the_deadline_releases_and_the_next_opener_proceeds():
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time()
        a = subprocess.Popen([sys.executable, "-c", STUCK], env=_env(d, DEADLINE="3"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _wait_for(os.path.join(d, "ready"))
        b = subprocess.run([sys.executable, "-c", NEXT], env=_env(d, TT_VISIBLE_DEVICES="1"),
                           capture_output=True, text=True, timeout=60)
        a_out, a_err = a.communicate(timeout=30)
        assert a.returncode == 1 and "returned" not in a_out, (a.returncode, a_out, a_err)
        assert "Timeout" in a_err, a_err
        assert b.returncode == 0, b.stderr
        took = json.loads(b.stdout) - t0
        assert 2.5 < took < 20, f"next opener got the lock {took:.1f}s in (deadline 3s)"
        # the waiter said who held it and since when, then that it got it
        assert f"held by pid {a.pid} (worker:stuck-test, open, TT_VISIBLE_DEVICES=3" in b.stderr, b.stderr
        assert " since 20" in b.stderr and "s ago" in b.stderr, b.stderr
        assert "took the device bring-up lock after" in b.stderr, b.stderr


def test_an_uncontended_open_is_silent_and_leaves_no_holder_record():
    with tempfile.TemporaryDirectory() as d:
        b = subprocess.run([sys.executable, "-c", NEXT], env=_env(d), capture_output=True,
                           text=True, timeout=60)
        assert b.returncode == 0 and b.stderr == "", b.stderr
        assert open(os.path.join(d, "open.lock")).read() == ""


def test_a_waiter_does_not_erase_the_holders_record():
    """The lock file used to be opened with "w", so every waiter truncated it."""
    with tempfile.TemporaryDirectory() as d:
        a = subprocess.Popen([sys.executable, "-c", STUCK], env=_env(d, DEADLINE="30"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            _wait_for(os.path.join(d, "ready"))
            with open(os.path.join(d, "open.lock"), "a+") as f:   # what a waiter's open does
                assert f"held by pid {a.pid}" in dl._lock_holder(f)
            rec = json.load(open(os.path.join(d, "open.lock")))
            assert rec["pid"] == a.pid and rec["purpose"] == "open"
        finally:
            a.kill()
            a.wait()


# A bring-up probe whose synchronize_device blocks in native code: SIGALRM cannot reach it.
PROBE = r"""
import os, re, sys
sys.path.insert(0, os.environ["REPO"])
from tt_bio import tenstorrent as T
T.ttnn.from_torch = lambda *a, **k: "t"
T.ttnn.add = lambda *a, **k: "t"
T.ttnn.synchronize_device = lambda d: re.match(r"(a+)+$", "a" * 40 + "b")
T._close_device_locked = lambda d: None
T._assert_local_dispatch(object(), timeout_s=1)
print("returned")
"""


def test_a_probe_blocked_in_native_code_ends_the_process():
    pytest.importorskip("tt_bio.tenstorrent")
    t = time.time()
    p = subprocess.run([sys.executable, "-c", PROBE], env=dict(os.environ, REPO=REPO),
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 1 and "returned" not in p.stdout, (p.returncode, p.stdout, p.stderr)
    assert "Timeout" in p.stderr and time.time() - t < 60, p.stderr
