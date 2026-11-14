"""Host-only tests for the predict CLI's worker-liveness handling.

These cover the three defects behind the "predict hangs forever" failure: the
CLI polled a run whose local workers had all exited, the workers' fatal went to
/dev/null, and an orphaned worker kept its card lease. No device needed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_bio.host_controller import ControllerClient, ControllerServer  # noqa: E402
from tt_bio.main import _stream_run  # noqa: E402
from tt_bio.worker import (  # noqa: E402
    _cleanup_worker_capture,
    _install_orphan_guard,
    _report_fatal,
    _silence_subprocess_output,
)


def _controller(tmpdir: Path):
    server = ControllerServer(0, tmpdir / "controller.sqlite3")
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
            _cleanup_worker_capture()      # what run_worker_loop's finally does
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        got = pipe.read()
    os.waitpid(pid, 0)
    assert b"boom" in got, f"fatal was swallowed, read {got!r}"


# The two native fatals this capture exists for. Both abort out of a library thread,
# write to fd 2 and exit without unwinding, so no Python exception is ever raised and
# _report_fatal never runs. On /dev/null the first left issue #12 as an opaque exit 14
# and the second left #14 as a 0-byte log.
NATIVE_FATALS = {
    "mpi_init": b"*** An error occurred in MPI_Init_thread\n*** MPI_ERRORS_ARE_FATAL\n",
    "l1_overflow": b"TT_THROW: Statically allocated circular buffers on core range "
                   b"[(x=0,y=0) - (x=10,y=9)] grow to 1765888 B which is beyond max L1 "
                   b"size of 1572864 B\n",
}


def _native_abort_worker(kind):
    _silence_subprocess_output()
    os.write(2, NATIVE_FATALS[kind])
    os._exit(14)


@pytest.mark.parametrize("kind", sorted(NATIVE_FATALS))
def test_silenced_worker_native_abort_is_captured(kind):
    from tt_bio.worker import read_worker_capture, worker_capture_path

    proc = mp.get_context("spawn").Process(target=_native_abort_worker, args=(kind,))
    proc.start()
    proc.join(60)
    assert proc.exitcode == 14
    cap = read_worker_capture(proc.pid)
    needle = NATIVE_FATALS[kind].split(b"\n")[0].decode()
    assert needle in cap, f"native abort swallowed, read {cap!r}"
    assert not worker_capture_path(proc.pid).exists(), "the tail must be consumed, not left in /tmp"


def _clean_exit_worker():
    _silence_subprocess_output()
    from tt_bio.worker import _cleanup_worker_capture

    _cleanup_worker_capture()


def test_clean_worker_leaves_no_capture_file():
    from tt_bio.worker import worker_capture_path

    proc = mp.get_context("spawn").Process(target=_clean_exit_worker)
    proc.start()
    proc.join(60)
    assert proc.exitcode == 0
    assert not worker_capture_path(proc.pid).exists(), "clean exit littered /tmp"


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


def _alive(pid: int) -> bool:
    """Zombie counts as dead: under a subreaper the corpse lingers as a child."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


_PDEATHSIG_PARENT = """
import os, sys, time
sys.path.insert(0, {repo!r})
from tt_bio.device_lease import install_parent_death_guard
armed = {armed!r}
if os.fork() == 0:
    install_parent_death_guard(os.getppid())
    with open(armed, "w") as f:
        f.write(str(os.getpid()))
    time.sleep(60)          # mid-job: never polls getppid() again
    os._exit(1)
for _ in range(200):        # vanish only once the child has armed
    if os.path.exists(armed):
        break
    time.sleep(0.05)
os._exit(0)
"""


def test_parent_death_guard_kills_child_when_var_names_parent(tmp_path):
    """The mid-job case the between-lease check cannot reach: the child is asleep
    inside a call and still dies with its parent, releasing card lease and chip."""
    armed = tmp_path / "armed"
    src = _PDEATHSIG_PARENT.format(repo=str(Path(__file__).resolve().parents[1]),
                                   armed=str(armed))
    subprocess.run([sys.executable, "-c", src], timeout=60, check=True)
    assert armed.exists(), "child never armed the guard"
    child = int(armed.read_text())
    deadline = time.time() + 5
    while time.time() < deadline and _alive(child):
        time.sleep(0.1)
    assert not _alive(child), f"pid {child} outlived its parent; it would hold its card lease"


def test_arm_orphan_guard_ignores_a_stale_var(monkeypatch):
    """A detached fleet job inherits TT_BIO_PARENT_PID from a dead grandparent.
    Arming there would kill it, so the test is pid-exact, not "the variable is set".
    If this regresses, this process exits 70 rather than failing an assert."""
    import tt_bio.device_lease as dl

    monkeypatch.setattr(dl, "_ARMED", False)
    monkeypatch.setenv("TT_BIO_PARENT_PID", str(os.getpid()))  # our own pid is never our parent
    before = signal.getsignal(signal.SIGTERM)
    dl.arm_orphan_guard()
    assert dl._ARMED is False
    assert signal.getsignal(signal.SIGTERM) is before, "guard must not touch SIGTERM disposition"


_WEDGED_WORKER_PARENT = """
import multiprocessing as mp, os, signal, sys, time
sys.path.insert(0, {repo!r})


def worker(armed):
    from tt_bio.device_lease import install_parent_death_guard
    install_parent_death_guard(int(os.environ["TT_BIO_PARENT_PID"]), kill_after_s=1.0)
    # A worker's handler, as run_worker_loop installs it, and then the wedge: a call stuck
    # in C never returns to the interpreter, so a pending SIGTERM never takes effect.
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    signal.pthread_sigmask(signal.SIG_BLOCK, {{signal.SIGTERM, signal.SIGINT}})
    with open(armed, "w") as f:
        f.write(str(os.getpid()))
    time.sleep(120)


if __name__ == "__main__":
    from tt_bio.device_lease import orphan_guard_env
    os.environ.update(orphan_guard_env())
    mp.get_context("spawn").Process(target=worker, args=({armed!r},)).start()
    time.sleep(120)     # `timeout` kills us here
"""


def test_wedged_worker_dies_after_timeout_kills_its_cli(tmp_path):
    """2026-11-13: `timeout ... tt-bio predict` killed the CLI; its spawn worker, wedged in
    the bring-up dispatch probe, ignored the SIGTERM and held card 3 for 1 h 48 min."""
    armed = tmp_path / "armed"
    script = tmp_path / "parent.py"
    script.write_text(_WEDGED_WORKER_PARENT.format(
        repo=str(Path(__file__).resolve().parents[1]), armed=str(armed)))
    proc = subprocess.Popen(["timeout", "-s", "TERM", "4", sys.executable, str(script)])
    try:
        assert proc.wait(30) == 124, "timeout should have killed the parent"
        assert armed.exists(), "worker never armed the guard"
        child = int(armed.read_text())
        deadline = time.time() + 10
        while time.time() < deadline and _alive(child):
            time.sleep(0.1)
        alive = _alive(child)
        if alive:
            os.kill(child, signal.SIGKILL)
        assert not alive, f"wedged worker {child} outlived its killed CLI; it would hold its card"
    finally:
        proc.kill()


def test_reaper_leaves_with_the_process_it_guards(tmp_path):
    """The reaper must not outlive a guarded process that finishes normally."""
    src = (f"import os, sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
           "from tt_bio import device_lease as dl\n"
           "dl.install_parent_death_guard(os.getppid())\n"
           "print(dl._REAPER.pid, flush=True)\n")
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                         timeout=30, check=True).stdout
    reaper = int(out.split()[0])
    deadline = time.time() + 5
    while time.time() < deadline and _alive(reaper):
        time.sleep(0.05)
    assert not _alive(reaper), f"reaper {reaper} outlived the process it guarded"


def test_device_spawning_children_get_the_parent_pid():
    """Every spawner whose child opens a device must hand it TT_BIO_PARENT_PID, or the
    child's arm_orphan_guard sees no parent and never arms."""
    from tt_bio.device_lease import orphan_guard_env

    assert orphan_guard_env() == {"TT_BIO_PARENT_PID": str(os.getpid())}
    root = Path(__file__).resolve().parents[1] / "tt_bio"
    for rel in ("esmc.py", "saprot.py", "boltzgen/cli/boltzgen.py", "main.py"):
        assert "orphan_guard_env()" in (root / rel).read_text(), rel
