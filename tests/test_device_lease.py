"""Proof that the physical-card device lease serializes same-host opens and self-heals.

Device-free: exercises the tt_bio.device_lease enforcement primitive directly (the same
object get_device() acquires before ttnn.open_device), so it runs anywhere and in CI. A
companion on-hardware test (test_device_lease_hw.py) reproduces the exact flagship-vs-RFD3
two-process open on a real TT card.

Run: python3 tests/test_device_lease.py
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio.device_lease import DeviceLease, DeviceInUseError, lease_dir  # noqa: E402

# A child that acquires card 0's lease, writes a "ready" marker, holds for `hold`
# seconds, then releases (unless it is killed first).
HOLDER = r"""
import os, sys, time
sys.path.insert(0, os.environ["REPO"])
from tt_bio.device_lease import DeviceLease
lease = DeviceLease(card="0").acquire()
open(os.environ["READY"], "w").write(str(os.getpid()))
time.sleep(float(os.environ["HOLD"]))
lease.release()
"""


def _env(d, ready, hold):
    e = dict(os.environ)
    e.update(REPO=REPO, TT_BIO_LEASE_DIR=d, TT_BIO_LEASE_HOST="testhost",
             READY=ready, HOLD=str(hold))
    return e


def _spawn_holder(d, ready, hold):
    if os.path.exists(ready):
        os.remove(ready)
    p = subprocess.Popen([sys.executable, "-c", HOLDER], env=_env(d, ready, hold))
    for _ in range(200):
        if os.path.exists(ready):
            return p, int(open(ready).read())
        time.sleep(0.05)
    p.kill()
    raise AssertionError("holder never acquired the lease")


def test_serialization_blocks_then_succeeds(d):
    """A second opener WAITS while a live holder has the card, then acquires once freed."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready1"), hold=3.0)
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=30).acquire()  # must block ~3s, not collide
    waited = time.time() - t0
    lease.release()
    p.wait()
    assert waited >= 2.5, f"second opener did not wait for the holder (waited {waited:.2f}s)"
    print(f"  serialization: 2nd open blocked {waited:.2f}s until holder (pid {holder_pid}) freed  OK")


def test_timeout_errors_cleanly(d):
    """A contended card fails LOUDLY with the holder's identity, never a silent collision."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready2"), hold=5.0)
    try:
        DeviceLease(card="0", timeout=1).acquire()
        raise AssertionError("expected DeviceInUseError on a contended card")
    except DeviceInUseError as e:
        assert str(holder_pid) in str(e), f"error must name the holder pid: {e}"
        print(f"  timeout: contended open raised DeviceInUseError naming pid {holder_pid}  OK")
    finally:
        p.kill(); p.wait()


def test_sigkill_reclaim(d):
    """A SIGKILLed holder (no chance to release) leaves NO phantom claim: kernel drops flock."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready3"), hold=60.0)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=10).acquire()  # must succeed ~immediately
    waited = time.time() - t0
    lease.release()
    assert waited < 2.0, f"stale SIGKILLed lease was not reclaimed promptly ({waited:.2f}s)"
    print(f"  SIGKILL reclaim: card re-leased {waited:.2f}s after holder killed  OK")


def test_sigterm_release(d):
    """SIGTERM frees the card via the kernel flock-drop on process death."""
    p, holder_pid = _spawn_holder(d, os.path.join(d, "ready4"), hold=60.0)
    os.kill(p.pid, signal.SIGTERM)
    p.wait()
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=10).acquire()
    waited = time.time() - t0
    lease.release()
    assert waited < 2.0, f"SIGTERMed lease not freed promptly ({waited:.2f}s)"
    print(f"  SIGTERM release: card re-leased {waited:.2f}s after holder terminated  OK")


def test_no_self_deadlock_against_dispatch_lease(d):
    """A worker opening its OWN dispatch-leased card must NOT deadlock on it.

    The fleet dispatcher writes the lease JSON WITHOUT holding an flock, so device-open
    acquires immediately and just overwrites the metadata with its live pid."""
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "testhost-card0.json")
    with open(path, "w") as f:  # simulate fleet.sh's dispatch-time write (no flock held)
        json.dump({"host": "testhost", "card": "0", "holder": "worker:flagship",
                   "pid": 999999, "acquired": time.time()}, f)
    t0 = time.time()
    lease = DeviceLease(card="0", timeout=5).acquire()
    waited = time.time() - t0
    meta = json.load(open(path))
    lease.release()
    assert waited < 1.0, f"deadlocked against own dispatch lease ({waited:.2f}s)"
    assert meta["pid"] == os.getpid(), "device-open did not claim the lease as its own pid"
    print(f"  no self-deadlock: claimed own dispatch lease in {waited:.2f}s  OK")


def test_clean_release(d):
    """Context-manager exit frees the card immediately."""
    with DeviceLease(card="0", timeout=5):
        pass
    t0 = time.time()
    with DeviceLease(card="0", timeout=5):
        pass
    assert time.time() - t0 < 1.0
    print("  clean release: re-leased immediately after context exit  OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        # The main process must lease into the SAME dir/host as the spawned holders so
        # they actually contend on one file. Holders inherit these via _env().
        os.environ["TT_BIO_LEASE_DIR"] = d
        os.environ["TT_BIO_LEASE_HOST"] = "testhost"
        print(f"lease dir under test: {d}")
        test_no_self_deadlock_against_dispatch_lease(d)
        test_clean_release(d)
        test_serialization_blocks_then_succeeds(d)
        test_timeout_errors_cleanly(d)
        test_sigkill_reclaim(d)
        test_sigterm_release(d)
    print("ALL DEVICE-LEASE UNIT TESTS PASSED")
