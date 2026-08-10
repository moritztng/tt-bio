"""A predict run whose workers have all died must end, not poll forever.

The incident this pins: 26 campaign folds on a 32-chip galaxy each sat 45-100 minutes
at ~1 pct CPU with a zero-byte log and an idle chip. Their single spawned worker had
exited at device open, so no job could ever complete, the controller's run status
stayed "running", and `_stream_run` polled it at 2 Hz until an external watchdog
killed the process group. Two defects, one per fix:

* `_stream_run` never looked at its own worker processes.
* the worker sends fd 1 and 2 to /dev/null, so its exit reason went nowhere.

CPU-only: no device, no model, no fold. The dead worker is a real process that has
exited, so `is_alive()`/`exitcode` behave exactly as they do in production.
"""

import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio.distributed import ControllerClient, ControllerServer  # noqa: E402
from tt_bio.main import _stream_run  # noqa: E402


def _exit_immediately():
    """Stand-in for a worker that dies at device open."""
    os._exit(3)


def test_stream_run_ends_when_every_worker_is_dead():
    tmpdir = Path(tempfile.mkdtemp(prefix="tt-bio-test-deadworker-"))
    server = ControllerServer("127.0.0.1", 0, tmpdir / "controller.sqlite3")
    server.serve_in_background()
    client = ControllerClient(f"http://127.0.0.1:{server.port}")
    try:
        run = client.create_run({
            "data": str(tmpdir),
            "out_dir": str(tmpdir),
            "result_dir": str(tmpdir),
            "config": {"model": "boltz2"},
            "jobs": [{"id": "a", "name": "a.yaml", "input_b64": ""},
                     {"id": "b", "name": "b.yaml", "input_b64": ""}],
        })

        proc = mp.get_context("spawn").Process(target=_exit_immediately)
        proc.start()
        proc.join(timeout=30)
        assert not proc.is_alive(), "test setup: the stand-in worker should have exited"

        t0 = time.time()
        failed = _stream_run(client, run["run_id"], total=2, n_workers=1,
                             debug=True, log=False, procs=[proc])
        elapsed = time.time() - t0

        # Both jobs are unfinished and unfinishable, so both count as failed, and the
        # caller gets a non-zero failure count to exit on.
        assert failed == 2, f"expected 2 failed, got {failed}"
        assert elapsed < 30, f"_stream_run took {elapsed:.1f}s; it should give up at once"
        assert client.run_status(run["run_id"]) == "canceled"
    finally:
        server.shutdown()


def test_stream_run_without_procs_is_unchanged():
    """A controller run (remote workers, procs=None) must not self-cancel."""
    tmpdir = Path(tempfile.mkdtemp(prefix="tt-bio-test-remoteworker-"))
    server = ControllerServer("127.0.0.1", 0, tmpdir / "controller.sqlite3")
    server.serve_in_background()
    client = ControllerClient(f"http://127.0.0.1:{server.port}")
    try:
        run = client.create_run({
            "data": str(tmpdir), "out_dir": str(tmpdir), "result_dir": str(tmpdir),
            "config": {"model": "boltz2"},
            "jobs": [{"id": "a", "name": "a.yaml", "input_b64": ""}],
        })
        client.cancel_run(run["run_id"])  # the only way this run can end
        failed = _stream_run(client, run["run_id"], total=1, n_workers=1,
                             debug=True, log=False, procs=None)
        assert failed == 0
        assert client.run_status(run["run_id"]) == "canceled"
    finally:
        server.shutdown()


def test_worker_fatal_error_survives_output_silencing():
    """The silenced worker still reports a fatal error on the original fd 2.

    Runs in a child process: `_silence_subprocess_output` replaces fd 1 and 2 for
    good, so it cannot be called in the test process.
    """
    import subprocess

    snippet = (
        "import sys, traceback\n"
        "from tt_bio.worker import _silence_subprocess_output, _fatal\n"
        "fd = _silence_subprocess_output()\n"
        "print('this stdout must be swallowed')\n"
        "sys.stderr.write('this stderr must be swallowed\\n')\n"
        "try:\n"
        "    raise RuntimeError('device open failed')\n"
        "except Exception:\n"
        "    _fatal(fd, '[worker tt0] device open failed, exiting')\n"
    )
    with tempfile.NamedTemporaryFile("r+", suffix=".log") as log:
        subprocess.run([sys.executable, "-c", snippet], check=True,
                       stdout=log, stderr=log,
                       cwd=str(Path(__file__).resolve().parents[1]))
        log.seek(0)
        text = log.read()
    assert "[worker tt0] device open failed, exiting" in text, text
    assert "RuntimeError: device open failed" in text, text
    assert "must be swallowed" not in text, text


def _display_chosen_for(stderr_is_tty: bool) -> str:
    """Return the display class `_stream_run` picks for this stderr, debug off."""
    import tt_bio.main as m

    chosen = []

    class _Rec:
        name = "?"

        def __init__(self, queue, **_kw): chosen.append(self.name)
        def start(self): pass
        def stop(self): pass

    class _Prog(_Rec): name = "ProgressDisplay"
    class _Dbg(_Rec): name = "DebugDisplay"

    class _Stderr:
        def isatty(self): return stderr_is_tty
        def write(self, _s): return 0
        def flush(self): pass

    tmpdir = Path(tempfile.mkdtemp(prefix="tt-bio-test-display-"))
    server = ControllerServer("127.0.0.1", 0, tmpdir / "controller.sqlite3")
    server.serve_in_background()
    client = ControllerClient(f"http://127.0.0.1:{server.port}")
    saved = (m.ProgressDisplay, m.DebugDisplay, m._sys.stderr)
    try:
        m.ProgressDisplay, m.DebugDisplay = _Prog, _Dbg
        m._sys.stderr = _Stderr()
        run = client.create_run({
            "data": str(tmpdir), "out_dir": str(tmpdir), "result_dir": str(tmpdir),
            "config": {"model": "boltz2"},
            "jobs": [{"id": "a", "name": "a.yaml", "input_b64": ""}],
        })
        client.cancel_run(run["run_id"])  # end the poll loop at once
        _stream_run(client, run["run_id"], total=1, n_workers=1,
                    debug=False, log=False, procs=None)
    finally:
        m.ProgressDisplay, m.DebugDisplay, m._sys.stderr = saved
        server.shutdown()
    assert len(chosen) == 1, f"expected one display, got {chosen}"
    return chosen[0]


def test_redirected_run_is_observable():
    """A fold whose output is a file must write progress lines, not a 0-byte log.

    Rich's Live renders only its final frame to a non-tty, so a redirected run
    produced a zero-byte log for its whole duration. Every fleet watchdog reads
    log growth to tell a working fold from a hung one, so it could distinguish
    neither and reaped healthy folds on a wall-clock cap instead.
    """
    assert _display_chosen_for(stderr_is_tty=False) == "DebugDisplay"
    # An interactive run keeps the Rich view it always had.
    assert _display_chosen_for(stderr_is_tty=True) == "ProgressDisplay"


if __name__ == "__main__":
    test_stream_run_ends_when_every_worker_is_dead()
    test_stream_run_without_procs_is_unchanged()
    test_worker_fatal_error_survives_output_silencing()
    test_redirected_run_is_observable()
    print("PASS")
