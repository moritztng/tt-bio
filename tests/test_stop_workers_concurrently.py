"""Stopping N local workers costs what stopping one does.

`_stop_worker_processes` used to SIGINT, then join, one worker at a time. A worker
whose model takes ~20 s to tear down (OpenFold3, OpenBind) made a 4-card run on a
p150a box sit ~80 s after its last result was on disk, which erased most of the
4-card speed-up. Host-only: the workers are plain processes that take 2 s to exit.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio import main as M  # noqa: E402


def _slow_to_stop(ready):
    def _bye(*_):
        time.sleep(2.0)
        sys.exit(0)

    signal.signal(signal.SIGINT, _bye)
    ready.release()
    while True:
        time.sleep(0.1)


def _deaf(ready):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.release()
    while True:
        time.sleep(0.1)


def _start(target, n):
    ctx = mp.get_context("spawn")
    ready = ctx.Semaphore(0)
    procs = [ctx.Process(target=target, args=(ready,)) for _ in range(n)]
    for p in procs:
        p.start()
    for _ in procs:  # every child has its handler in place
        assert ready.acquire(timeout=120)
    return procs


def test_four_workers_stop_in_one_workers_time():
    procs = _start(_slow_to_stop, 4)
    t0 = time.monotonic()
    M._stop_worker_processes(procs)
    took = time.monotonic() - t0
    assert not any(p.is_alive() for p in procs)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert took < 5.0, f"stopping 4 workers took {took:.1f} s; serial would be ~8 s"


def test_a_worker_deaf_to_signals_is_still_killed():
    procs = _start(_deaf, 2)
    t0 = time.monotonic()
    M._stop_worker_processes(procs)
    assert not any(p.is_alive() for p in procs)
    # SIGINT grace + SIGTERM grace, shared across both workers, then SIGKILL.
    assert time.monotonic() - t0 < 12 + 8 + 3 + 2
