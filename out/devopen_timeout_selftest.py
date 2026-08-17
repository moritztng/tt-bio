"""Prove the device-open bound kills a process wedged in a native call while it holds the host-wide
device-init lock, and that the worker queued behind it then acquires the lock and proceeds.

This is the `japanfold-wh-cutover-deploy.md` section 21 failure reproduced without hardware: a card
whose ttnn.open_device never returns holds `_device_init_lock` forever, every other worker queues
behind it, and killing the holder only hands the lock to the next bad card. The fix bounds how long
the lock may be *held* (never how long a caller may wait for it, which is a change that already
reintroduced a UMD deadlock once).

Two cases are run. With the GIL released, a Python-level timer could in principle have worked. With
the GIL held, it could not: a threading.Timer never gets scheduled and a Python SIGALRM handler never
runs, because both need the interpreter. That second case is why the bound is signal.alarm with the
default disposition, enforced by the kernel. It is also the case we cannot rule out for a native
bring-up call deadlocked in a cross-process mutex.

Progress is reported through a raw pipe rather than a multiprocessing.Queue: Queue hands the write
to a background feeder thread, which is itself frozen when the GIL is held, so it would report
nothing in exactly the case being tested.

Run: python3 out/devopen_timeout_selftest.py   (no hardware, no ttnn, no torch)
"""
import contextlib
import ctypes
import multiprocessing as mp
import os
import sys
import time
import types

# Load the lock machinery out of tt_bio/tenstorrent.py without importing ttnn or torch: take the
# source between the lock-path constant and the first function that touches the device API.
_src = open(os.path.join(os.path.dirname(__file__), "..", "tt_bio", "tenstorrent.py")).read()
mod = types.ModuleType("locktest")
mod.__dict__.update(os=os, contextlib=contextlib)
exec(compile(_src[_src.index("_DEVICE_INIT_LOCK_PATH ="):_src.index("def _open_device_locked")],
             "tenstorrent.py", "exec"), mod.__dict__)

mod._DEVICE_INIT_LOCK_PATH = LOCKPATH = "/tmp/tt-bio-device-open.lock.selftest"
mod._DEVICE_OPEN_TIMEOUT_S = BOUND_S = 5


def _note(fd, tag):
    os.write(fd, f"{tag} {time.time():.3f}\n".encode())


def hang_holding_the_lock(fd, hold_gil):
    """Stand in for a card whose ttnn.open_device never returns.

    ctypes.CDLL releases the GIL around the call; ctypes.PyDLL does not."""
    libc = (ctypes.PyDLL if hold_gil else ctypes.CDLL)("libc.so.6", use_errno=True)
    with mod._device_init_lock():
        _note(fd, "holder-has-lock")
        libc.usleep(600 * 1000 * 1000)      # 600 s: never returns on its own
    _note(fd, "holder-returned-normally")   # must never be reached


def queued_behind(fd):
    """A healthy worker that wants the lock while the bad one holds it."""
    _note(fd, "queued-start")
    with mod._device_init_lock():
        _note(fd, "queued-acquired")


def run_case(hold_gil):
    print(f"\n=== the hang holds the GIL: {hold_gil}"
          f"{'   <- the case a Python-level timer cannot handle' if hold_gil else ''} ===")
    ctx = mp.get_context("fork")
    r, w = os.pipe()
    holder = ctx.Process(target=hang_holding_the_lock, args=(w, hold_gil))
    holder.start()
    waiter = None
    events, t0 = {}, None
    deadline = time.time() + BOUND_S + 30

    with os.fdopen(r) as rf:
        os.close(w)
        while time.time() < deadline and "queued-acquired" not in events:
            line = rf.readline()
            if not line:
                break
            tag, ts = line.split()
            ts = float(ts)
            t0 = ts if t0 is None else t0
            events[tag] = ts
            print(f"  t+{ts - t0:5.1f}s  {tag}")
            if tag == "holder-has-lock":
                r2, w2 = os.pipe()
                waiter = ctx.Process(target=queued_behind, args=(w2,))
                waiter.start()
                os.close(w2)
                rf2 = os.fdopen(r2)
                for _ in range(2):
                    l2 = rf2.readline()
                    if not l2:
                        break
                    tag2, ts2 = l2.split()
                    events[tag2] = float(ts2)
                    print(f"  t+{float(ts2) - t0:5.1f}s  {tag2}")

    holder.join(30)
    if waiter:
        waiter.join(30)
    print(f"  holder exitcode={holder.exitcode} (-14 == SIGALRM, delivered by the kernel), "
          f"waiter exitcode={waiter.exitcode if waiter else None}")

    ok = (holder.exitcode == -14
          and waiter is not None and waiter.exitcode == 0
          and "queued-acquired" in events
          and "holder-returned-normally" not in events)
    held = events.get("queued-acquired", 0) - events.get("holder-has-lock", 0)
    print(f"  the queued worker got the lock {held:.1f}s after the bad one took it "
          f"(bound is {BOUND_S}s): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = all([run_case(False), run_case(True)])
    print("\nRESULT:", "PASS" if ok else "FAIL",
          "- in both cases the kernel killed the wedged holder at the bound and the worker queued "
          "behind it acquired the lock and proceeded, so the pool drains instead of bleeding down."
          if ok else "- unexpected exit codes, see above")
    with contextlib.suppress(OSError):
        os.unlink(LOCKPATH)
    sys.exit(0 if ok else 1)
