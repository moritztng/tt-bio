"""CPU-only test for the local-dispatch probe watchdog.

No Tenstorrent device: the three ttnn calls the probe makes are stubbed, so this covers
the watchdog's control flow, not device behaviour.

Background: ``_assert_local_dispatch`` dispatches one trivial op at device open so a chip
that came up remote-only fails at startup instead of silently accepting jobs. It handled a
chip that THROWS on dispatch but not one that never returns -- three Tier-A folds hung
inside it for 11-36 minutes at 100% CPU with weights never loaded, deaf to SIGINT and
SIGTERM because the spin is in native ttnn. A watchdog thread now bounds it and takes the
process down, since signals do not land while the main thread is in that call.

Verifies:
  1. A probe that does not return within the timeout exits the process with the dedicated
     code, after printing why.
  2. A probe that returns promptly is untouched -- the watchdog must not fire on a healthy
     open, which is the failure mode that would take out good folds.

Run: python3 tests/test_dispatch_probe_watchdog.py
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_TIMEOUT_EXIT = 87

CHILD = textwrap.dedent('''
    import os, sys, time
    sys.path.insert(0, {root!r})
    os.environ["TT_BIO_DISPATCH_PROBE_TIMEOUT_S"] = {timeout!r}
    import tt_bio.tenstorrent as T
    T.ttnn.from_torch = lambda *a, **k: object()
    T.ttnn.add = lambda *a, **k: None
    T.ttnn.synchronize_device = lambda *a, **k: time.sleep({sleep})
    T._assert_local_dispatch(object())
    print("RETURNED_NORMALLY")
''')


def _run(timeout, sleep):
    src = CHILD.format(root=str(ROOT), timeout=str(timeout), sleep=sleep)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       timeout=180)
    return r.returncode, r.stdout + r.stderr


def test_watchdog_fires_on_a_hang():
    rc, out = _run(timeout=2, sleep=30)
    assert rc == PROBE_TIMEOUT_EXIT, f"expected exit {PROBE_TIMEOUT_EXIT}, got {rc}\n{out}"
    assert "did not return in 2s" in out, f"no explanation printed:\n{out}"
    assert "RETURNED_NORMALLY" not in out
    print(f"[PASS] hanging probe exits {PROBE_TIMEOUT_EXIT} and says why")


def test_watchdog_silent_on_a_healthy_open():
    rc, out = _run(timeout=30, sleep=0)
    assert rc == 0, f"watchdog fired on a healthy probe (exit {rc}):\n{out}"
    assert "RETURNED_NORMALLY" in out, f"probe did not complete:\n{out}"
    assert "did not return" not in out
    print("[PASS] healthy probe returns normally, watchdog silent")


if __name__ == "__main__":
    test_watchdog_fires_on_a_hang()
    test_watchdog_silent_on_a_healthy_open()
    print("\nALL DISPATCH-PROBE WATCHDOG TESTS PASS")
