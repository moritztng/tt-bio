"""Proof that the host-wide device bring-up lock serializes, and says so when it cannot.

Device-free: it exercises tt_bio.device_lease.device_init_lock directly, the same context
manager _open_device_locked() wraps around ttnn.open_device, so it runs anywhere and in CI.

The interesting case is a lock path the process cannot write, which is what happens on a
shared box when another unix account created /tmp/tt-bio-device-open.lock with a default
umask (whglx, 2026-09-13: owner tt-admin, mode 664, the agent account not in that group).
Until this test existed, that case fell through to an unguarded `yield` and every open on the
box raced with no warning printed. test_two_processes_serialize_when_shared_path_unwritable
measures 6s of overlap against the old code and none against the new.

Run: python3 tests/test_device_init_lock.py, or as part of the release suite via pytest.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_bio import device_lease as dl

# A child that takes the bring-up lock, holds it, and records the interval it held it for.
HOLDER = r"""
import json, os, sys, time
sys.path.insert(0, os.environ["REPO"])
from tt_bio import device_lease as dl
dl.DEVICE_INIT_LOCK_PATH = os.environ["LOCK"]
with dl.device_init_lock():
    t0 = time.time()
    time.sleep(float(os.environ["HOLD"]))
    t1 = time.time()
open(os.environ["OUT"], "w").write(json.dumps([t0, t1]))
"""


@contextlib.contextmanager
def _lock_at(path, **env):
    """Point the bring-up lock at `path`, with a clean per-process resolution memo."""
    old_path, old_memo, old_env = dl.DEVICE_INIT_LOCK_PATH, dl._init_lock_path, {}
    for k, v in env.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v
    dl.DEVICE_INIT_LOCK_PATH, dl._init_lock_path = path, None
    try:
        yield
    finally:
        dl.DEVICE_INIT_LOCK_PATH, dl._init_lock_path = old_path, old_memo
        for k, v in old_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _take_lock(path, **env):
    """Enter and leave the lock once. Returns (resolved path, stderr text)."""
    err = io.StringIO()
    with _lock_at(path, **env), contextlib.redirect_stderr(err):
        with dl.device_init_lock():
            pass
        resolved = dl._init_lock_path
    return resolved, err.getvalue()


def _unwritable_file(path):
    open(path, "w").close()
    os.chmod(path, 0o444)       # the owner is denied too, same errno as the cross-account case
    return path


def test_shared_path_is_used_unchanged_when_writable():
    """The box where the lock already works must behave exactly as it did before."""
    with tempfile.TemporaryDirectory() as d:
        shared = os.path.join(d, "tt-bio-device-open.lock")
        resolved, err = _take_lock(shared)
        assert resolved == shared, resolved
        assert err == "", err                      # no warning, nothing to warn about
        assert os.listdir(d) == ["tt-bio-device-open.lock"], os.listdir(d)


def test_falls_back_to_a_per_user_path_and_says_so():
    with tempfile.TemporaryDirectory() as d:
        shared = _unwritable_file(os.path.join(d, "tt-bio-device-open.lock"))
        resolved, err = _take_lock(shared)
        assert resolved == f"{shared}.uid{os.getuid()}", resolved
        assert os.path.exists(resolved)
        assert shared in err and "Permission denied" in err, err
        assert resolved in err and "different unix account" in err, err


def test_unserializable_bring_up_is_loud_and_still_proceeds():
    """No writable candidate anywhere: bring-up must keep working, but never quietly."""
    with tempfile.TemporaryDirectory() as d:
        ro = os.path.join(d, "ro")
        os.mkdir(ro)
        shared = os.path.join(ro, "tt-bio-device-open.lock")
        os.chmod(ro, 0o555)
        try:
            ran = []
            err = io.StringIO()
            with _lock_at(shared, HOME=ro, TMPDIR=ro, XDG_RUNTIME_DIR=ro), \
                    contextlib.redirect_stderr(err):
                with dl.device_init_lock():
                    ran.append(True)
                assert dl._init_lock_path == ""    # resolved to "nothing writable", once
                with dl.device_init_lock():        # second entry must not re-warn
                    ran.append(True)
            text = err.getvalue()
        finally:
            os.chmod(ro, 0o755)
        assert ran == [True, True]                 # bring-up is never blocked by this
        assert text.count("WARNING") == 1, text
        assert "NOT serialized" in text and "Permission denied" in text, text
        assert shared in text, text


def test_two_processes_serialize_when_shared_path_unwritable():
    """The negative control: this is the case that used to race silently.

    Measured, not inferred: two children hold the lock for HOLD seconds each and write the
    interval they held it for. Serialized means the two intervals do not overlap.
    """
    hold = 1.5
    with tempfile.TemporaryDirectory() as d:
        shared = _unwritable_file(os.path.join(d, "tt-bio-device-open.lock"))
        outs = [os.path.join(d, f"out{i}.json") for i in range(2)]
        procs = [subprocess.Popen(
            [sys.executable, "-c", HOLDER],
            env=dict(os.environ, REPO=REPO, LOCK=shared, HOLD=str(hold), OUT=o),
            stderr=subprocess.DEVNULL) for o in outs]
        for p in procs:
            assert p.wait(timeout=60) == 0
        (a0, a1), (b0, b1) = [json.load(open(o)) for o in outs]
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        assert overlap < 0.1, (f"bring-up was NOT serialized: the two holds overlapped by "
                               f"{overlap:.3f}s of {hold}s")
        assert a1 - a0 >= hold and b1 - b0 >= hold


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t:.1f}s)")
    print("all device-init-lock tests passed")
