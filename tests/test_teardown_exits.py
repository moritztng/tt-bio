"""A process that used tt-bio ends on its own, with the status it chose, and honours SIGTERM.

Issue #20: a BindCraft 2 campaign that finished printed its last line and then died with exit 139,
one that hit an allocator refusal died 139 four seconds later or hung with every thread in
`futex_do_wait`, and the hung one ignored SIGTERM. All of it is teardown, after Python is done:
the C++ static destructors of tt-metal and XLA, run inside `exit()` with their worker threads
still alive. `runtime.end_after_teardown` is the last atexit hook and hands `exit()` straight to
`_exit` with the status Python chose.

A destructor that kills the process stands in for the ones that crashed and hung: it is
registered with `__cxa_atexit`, the way a C++ library registers its statics, so it runs inside
`exit()` exactly where theirs do. Host-only: no device, no weights.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# A C++-style static destructor that aborts, and a C-level write to stderr from an object
# finalised at interpreter shutdown, which is where nanobind prints its leak report.
_PRELUDE = """
    import atexit, builtins, ctypes, sys
    libc = ctypes.CDLL(None)
    libc.__cxa_atexit(libc.abort, None, None)
    libc.fprintf.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    class AliveAtShutdown:
        def __del__(self):
            libc.fprintf(ctypes.c_void_p.in_dll(libc, "stderr").value,
                         b"nanobind: leaked 1 instances!\\n")
            libc.fflush(None)
            print("python stderr still prints", file=sys.stderr, flush=True)

    builtins._alive = AliveAtShutdown()
"""


def _run(body: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "TT_VISIBLE_DEVICES": ""}
    env.pop("TT_BIO_DEBUG_STDERR", None)
    env.update(env_extra or {})
    code = textwrap.dedent(_PRELUDE) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)


_HOOK = textwrap.dedent("""
    from tt_bio import runtime
    atexit.register(runtime.end_after_teardown)
""")


@pytest.mark.parametrize("ending, status", [("pass", 0), ("sys.exit(3)", 3),
                                            ("raise MemoryError('refused')", 1)])
def test_the_status_python_chose_survives_a_destructor_that_dies(ending, status):
    out = _run(_HOOK + ending)
    assert out.returncode == status, out.stderr[-2000:]


def test_without_the_hook_that_destructor_owns_the_status():
    """The control: the stand-in really does kill an ordinary exit, so the test above means it."""
    out = _run("sys.exit(3)")
    assert out.returncode == -signal.SIGABRT, out.stderr[-2000:]


def test_the_leak_report_is_dropped_and_python_stderr_is_not():
    out = _run(_HOOK + "raise MemoryError('refused')")
    assert "nanobind: leaked" not in out.stderr
    assert "python stderr still prints" in out.stderr
    assert "MemoryError: refused" in out.stderr


def test_debug_keeps_the_leak_report():
    out = _run(_HOOK, {"TT_BIO_DEBUG_STDERR": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert "nanobind: leaked 1 instances!" in out.stderr


def test_importing_the_device_module_installs_the_hook():
    """The wiring: every process that can open a card gets the hook, with no call of its own."""
    out = _run("import tt_bio.tenstorrent\nsys.exit(3)")
    assert out.returncode == 3, out.stderr[-3000:]


def test_importing_tt_bio_main_forks_nothing():
    """`tt_bio.main` is imported lazily at device open, with JAX's threads already running. It
    used to fork a stderr filter on import there, which Python warns will likely deadlock."""
    code = textwrap.dedent("""
        import os, threading
        stop = threading.Event()
        threading.Thread(target=stop.wait, daemon=True).start()
        import tt_bio.main  # noqa: F401
        me = os.getpid()
        with open(f"/proc/{me}/task/{me}/children") as f:
            print("CHILDREN=" + f.read().strip())
    """)
    env = {**os.environ, "TT_VISIBLE_DEVICES": ""}
    env.pop("TT_BIO_DEBUG_STDERR", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    children = out.stdout.split("CHILDREN=")[-1].strip()
    assert children == "", f"importing tt_bio.main left child process(es) {children}"


def test_sigterm_ends_a_process_that_imported_the_device_stack():
    """Nothing tt-bio installs at import takes SIGTERM over: it ends the process, promptly."""
    code = "import tt_bio.tenstorrent, time\nprint('READY', flush=True)\ntime.sleep(600)\n"
    env = {**os.environ, "TT_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert proc.stdout.readline().strip() == "READY"
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=30)
        assert rc == -signal.SIGTERM
        assert time.monotonic() - t0 < 5
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


@pytest.mark.parametrize("debug", [False, True])
def test_ttnns_own_leak_report_is_gone_unless_debug(debug):
    """The real thing: importing ttnn leaves ~580 nanobind objects alive at shutdown, and the
    report of them is hundreds of lines at the end of every run."""
    env = {**os.environ, "TT_VISIBLE_DEVICES": ""}
    env.pop("TT_BIO_DEBUG_STDERR", None)
    if debug:
        env["TT_BIO_DEBUG_STDERR"] = "1"
    out = subprocess.run([sys.executable, "-c", "import tt_bio.tenstorrent"], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert ("nanobind: leaked" in out.stderr) == debug
