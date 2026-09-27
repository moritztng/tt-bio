"""The stderr filter's forked child must hold only the two fds it reads and writes.

`_install_nanobind_leak_stderr_filter` forks a child that lives for the whole run. It used to
inherit every fd the parent had open at the fork, so an `flock` held at that moment could never be
released: closing the parent's fd dropped one reference and the child kept the other. BindCraft 2
runs its trajectory under a compile flock and first imports the tt-bio device stack inside it, so
a second trajectory in the same process hung forever (bcx-p10-duotraj, 2026-09-27).

Host-only: no device, no weights. Each case runs in a fresh interpreter because the filter
rewires fd 2 of the process that installs it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# TT_BIO_DEBUG_STDERR stops the import-time install, so each script installs the filter itself at
# the point it chooses, the way a lazy import inside a caller's lock does.
_PRELUDE = """
    import fcntl, os, sys, time
    import tt_bio.main as m
    def filter_child():
        me = os.getpid()
        with open(f"/proc/{me}/task/{me}/children") as f:
            (pid,) = f.read().split()
        return int(pid)
"""


def _run(body: str, **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "TT_VISIBLE_DEVICES": "", "TT_BIO_DEBUG_STDERR": "1"}
    code = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300, **kw)


def test_a_lock_held_across_the_fork_is_released_when_the_parent_closes_it(tmp_path):
    """The regression. Before the fix the re-lock never succeeds while the child lives."""
    out = _run(f"""
        path = {str(tmp_path / "compile.lock")!r}
        held = open(path, "w")
        fcntl.flock(held, fcntl.LOCK_EX)
        m._install_nanobind_leak_stderr_filter()
        held.close()
        again = open(path, "w")
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fcntl.flock(again, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("RELOCK=ok")
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    print("RELOCK=blocked")
                    break
                time.sleep(0.05)
        print("CHILD_FDS=" + ",".join(sorted(os.listdir(f"/proc/{{filter_child()}}/fd"))))
    """)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "RELOCK=ok" in out.stdout, out.stdout
    fds = next(ln for ln in out.stdout.splitlines() if ln.startswith("CHILD_FDS="))
    assert len(fds.split("=", 1)[1].split(",")) == 2, fds


def test_the_filter_still_drops_a_nanobind_leak_report_and_forwards_the_rest():
    out = _run("""
        m._install_nanobind_leak_stderr_filter()
        os.write(2, b"before\\n")
        os.write(2, b"nanobind: leaked 3 instances!\\n - leaked instance 0x1 of type X\\n"
                    b"nanobind: this is likely caused by a reference counting issue\\n"
                    b"See https://nanobind.readthedocs.io/en/latest/refleaks.html\\n")
        os.write(2, b"after\\n")
        print("python-level", file=sys.stderr, flush=True)
        time.sleep(0.5)
    """)
    assert out.returncode == 0, out.stderr[-3000:]
    # Python-level stderr goes straight to the original fd, so only the fd-level lines are ordered.
    assert "before\nafter\n" in out.stderr and "python-level\n" in out.stderr, out.stderr
    assert "nanobind" not in out.stderr, out.stderr


def test_the_child_dies_with_its_parent():
    """A surviving filter child keeps a dup of fd 2 and a dispatcher never sees EOF."""
    proc = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_PRELUDE) + textwrap.dedent("""
            m._install_nanobind_leak_stderr_filter()
            print(filter_child(), flush=True)
            time.sleep(300)
        """)],
        cwd=REPO, env={**os.environ, "TT_VISIBLE_DEVICES": "", "TT_BIO_DEBUG_STDERR": "1"},
        stdout=subprocess.PIPE, text=True)
    child = int(proc.stdout.readline())
    assert Path(f"/proc/{child}").exists()
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    deadline = time.monotonic() + 10
    while Path(f"/proc/{child}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{child}").exists(), f"filter child {child} outlived its parent"


def test_a_stderr_pipe_reaches_eof_when_the_run_ends():
    """fd 2 is a pipe here, as it is for a dispatcher shard. communicate() returns only at EOF."""
    t0 = time.monotonic()
    out = _run("""
        m._install_nanobind_leak_stderr_filter()
        os.write(2, b"shard done\\n")
    """)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stderr.endswith("shard done\n"), out.stderr
    assert time.monotonic() - t0 < 120
