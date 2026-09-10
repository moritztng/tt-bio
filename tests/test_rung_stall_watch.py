"""A fold that stops writing has stopped folding, and the harness must not wait out its budget.

opendde at 1536 tokens froze at `trunk 9/10` on qb1 card 0 and burned the remaining 23 minutes
of its 2700 s budget without writing another byte. It was NOT idle while it did that: 111 % CPU,
two threads busy-polling, main thread in futex_wait, RSS frozen to the byte and no kernel
compiled for 14 minutes. So a CPU- or RSS-based liveness check answers "yes, it is working" on
a run that is going nowhere; log growth is what actually distinguishes them.

Exercises `run_rung.watch` itself, not a copy of its loop, against real subprocesses.
Host-only: no device, no model.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

HARNESS = Path(__file__).resolve().parents[1] / "perf" / "bh1536" / "run_rung.py"
_spec = importlib.util.spec_from_file_location("bh1536_run_rung", HARNESS)
rr = importlib.util.module_from_spec(_spec)
sys.modules["bh1536_run_rung"] = rr
_spec.loader.exec_module(rr)


def _run(script: str, log: Path, **kw):
    """Start `script` with its output in `log` and hand it to the real watch().

    `start_new_session=True` matches production and is NOT optional here: `kill_tree` signals
    the process GROUP, and a subprocess left in the caller's group means that group is pytest's.
    Learned the hard way -- the first run of this without it killed the test session and the ssh
    around it. `kill_tree` now refuses to signal its own group as well, so the mistake is caught
    from both ends.
    """
    with log.open("wb") as fh:
        proc = subprocess.Popen(["/bin/sh", "-c", script], stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        return rr.watch(proc, log, time.time(), poll=0.2, **kw)


def test_a_process_that_stops_writing_is_called_stalled():
    """The opendde shape: it says something, then goes quiet while still running."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "fold.log"
        t0 = time.time()
        killed, stalled, _, _ = _run("echo 'trunk 9/10'; sleep 60", log,
                                     budget=300, stall=3)
        assert stalled is True, "a silent process was not detected as stalled"
        assert killed is True, "a stalled process must be killed, not left holding the card"
        # And killed on the STALL, not by waiting out the budget: that is the whole saving.
        assert time.time() - t0 < 20, "it waited far longer than the stall window"
        assert "trunk 9/10" in log.read_text(), "the last thing it said must survive"


def test_a_process_that_keeps_writing_is_left_alone():
    """The negative control. A fold that is talking must never be cut, however long it runs --
    a stall detector that fires on a healthy run destroys the ladder instead of speeding it."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "fold.log"
        killed, stalled, _, _ = _run(
            "for i in 1 2 3 4 5 6 7 8; do echo \"trunk $i/8\"; sleep 0.5; done", log,
            budget=300, stall=2)
        assert stalled is False, "a live, talking process was called stalled"
        assert killed is False, "a live process was killed"
        assert log.read_text().count("trunk") == 8, "it did not run to completion"


def test_a_talking_process_still_hits_the_wall_clock_budget():
    """The two stops are independent: --budget still applies to a run that never goes quiet."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "fold.log"
        killed, stalled, _, _ = _run("while true; do echo tick; sleep 0.2; done", log,
                                     budget=3, stall=60)
        assert killed is True, "the budget did not stop a talking run"
        assert stalled is False, "a talking run must be TIMEOUT, not STALLED"


def test_stall_zero_disables_the_check():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "fold.log"
        killed, stalled, _, _ = _run("echo start; sleep 60", log, budget=3, stall=0)
        assert stalled is False, "--stall 0 must disable the stall check"
        assert killed is True, "the budget should still have stopped it"


def test_a_clean_exit_is_neither_killed_nor_stalled():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "fold.log"
        killed, stalled, _, _ = _run("echo done", log, budget=300, stall=60)
        assert (killed, stalled) == (False, False)


def test_stalled_counts_as_measured_but_contended_does_not():
    """`measured.py` decides whether chain.sh re-walks a rung. A stall IS a result -- the model
    did not complete at that size -- while CONTENDED means nothing ran at all."""
    spec = importlib.util.spec_from_file_location(
        "bh1536_measured", HARNESS.parent / "measured.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "STALLED" in m.MEASURED
    assert "CONTENDED" not in m.MEASURED


def test_kill_tree_reaches_a_child_that_ignores_sigterm():
    """The whole point: a fold's spawned WORKER is what holds the chip, and a frozen one does
    not take SIGTERM. Measured twice on this ladder -- opendde's worker lived 44 more minutes
    holding card 0's lease after its parent was gone, opendde-abag's 11 more. The grandchild
    here traps SIGTERM and sleeps, standing in for one wedged inside ttnn."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "fold.log"
        marker = Path(td) / "grandchild.pid"
        script = (f"sh -c 'trap \"\" TERM; echo $$ > {marker}; sleep 300' & "
                  f"echo parent; wait")
        with log.open("wb") as fh:
            proc = subprocess.Popen(["/bin/sh", "-c", script], stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(100):
                if marker.is_file() and marker.read_text().strip():
                    break
                time.sleep(0.1)
            gpid = int(marker.read_text().strip())
            assert _alive(gpid), "the SIGTERM-ignoring grandchild never started"
            rr.kill_tree(proc, grace=2.0)

        assert proc.poll() is not None, "the fold itself survived kill_tree"
        assert not _alive(gpid), (
            "the grandchild that ignores SIGTERM survived; this is the orphan that holds the card")


def test_kill_tree_refuses_to_signal_its_own_process_group():
    """The guard on the guard. A proc started WITHOUT start_new_session shares this process's
    group, and signalling that group would take down the harness and its shell. It must fall
    back to killing just that process."""
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 60"])   # deliberately same group
    assert os.getpgid(proc.pid) == os.getpgrp(), "test premise: same group"
    rr.kill_tree(proc, grace=1.0)
    assert proc.poll() is not None, "the target process was not killed"
    # And we are still here, which is the assertion that matters.
