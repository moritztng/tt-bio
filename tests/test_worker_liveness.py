"""Host-only tests for the predict CLI's worker-liveness handling.

These cover the three defects behind the "predict hangs forever" failure: the
CLI polled a run whose local workers had all exited, the workers' fatal went to
/dev/null, and an orphaned worker kept its card lease. No device needed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio.distributed import ControllerClient, ControllerServer  # noqa: E402
from tt_bio.main import _stream_run  # noqa: E402
from tt_bio.worker import (  # noqa: E402
    _install_orphan_guard,
    _report_fatal,
    _silence_subprocess_output,
)


def _controller(tmpdir: Path):
    server = ControllerServer("127.0.0.1", 0, tmpdir / "controller.sqlite3")
    server.serve_in_background()
    return server, ControllerClient(f"http://127.0.0.1:{server.port}")


def _make_run(client: ControllerClient, tmpdir: Path) -> str:
    return client.create_run({
        "data": str(tmpdir / "in.yaml"),
        "out_dir": str(tmpdir / "out"),
        "result_dir": str(tmpdir / "res"),
        "config": {"model": "esmfold2"},
        "jobs": [{"id": "prot", "name": "prot", "input_b64": ""}],
    })["run_id"]


def _dead_proc() -> mp.process.BaseProcess:
    """A spawned worker process that has already exited, like one that failed to
    open its chip."""
    proc = mp.get_context("spawn").Process(target=time.sleep, args=(0,))
    proc.start()
    proc.join(30)
    assert not proc.is_alive()
    return proc


def _stream_in_thread(client, run_id, procs, timeout=20.0):
    """Run _stream_run off-thread so a regression fails the test instead of
    hanging the suite. Returns (returned_value, raised_exception, finished)."""
    box: dict = {}

    def target():
        try:
            box["value"] = _stream_run(client, run_id, total=1, n_workers=1,
                                       debug=True, log=False, local_procs=procs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    return box.get("value"), box.get("error"), not thread.is_alive()


def test_stream_run_aborts_when_local_workers_die():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        server, client = _controller(tmpdir)
        try:
            run_id = _make_run(client, tmpdir)
            _, error, finished = _stream_in_thread(client, run_id, [_dead_proc()])
            assert finished, "_stream_run polled on with every local worker dead"
            assert isinstance(error, RuntimeError), f"expected RuntimeError, got {error!r}"
            assert "every local worker exited" in str(error)
        finally:
            server.shutdown()


def test_stream_run_ignores_dead_workers_once_run_is_terminal():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        server, client = _controller(tmpdir)
        try:
            run_id = _make_run(client, tmpdir)
            client.cancel_run(run_id)
            value, error, finished = _stream_in_thread(client, run_id, [_dead_proc()])
            assert finished, "_stream_run hung on a terminal run"
            assert error is None, f"terminal run must not raise: {error!r}"
            assert value == 0
        finally:
            server.shutdown()


def test_silence_subprocess_output_preserves_real_stderr():
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(read_fd)
            os.dup2(write_fd, 2)
            _silence_subprocess_output()
            _report_fatal("boom\n")
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        got = pipe.read()
    os.waitpid(pid, 0)
    assert b"boom" in got, f"fatal was swallowed, read {got!r}"


def _orphan_child():
    # A pid that is not our parent, standing in for a dispatcher that already died.
    _install_orphan_guard(dispatcher_pid=os.getpid())
    time.sleep(60)  # only reached if the guard failed to fire


def test_orphan_guard_exits_when_dispatcher_is_gone():
    proc = mp.get_context("fork").Process(target=_orphan_child)
    proc.start()
    proc.join(15)
    assert not proc.is_alive(), "orphaned worker kept running; it would hold its card lease"
    assert proc.exitcode == 70, f"expected exit 70, got {proc.exitcode}"
