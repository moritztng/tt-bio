"""A forked stderr-filter child must not keep the parent's file locks alive.

An `fcntl.flock` belongs to the open file description, not to the process, so a child that
inherits the descriptor holds the lock after the parent closes its own copy. BindCraft 2
takes exactly that kind of lock around every compile (`bindcraft/af2.py:19-27`), releasing
it by letting a `with open(...)` close, and `tt_bio.main`'s stderr filter forks a child
that lives for the whole run.

A 10-round BindCraft 2 gradient arm deadlocked on it: round 10 blocked for 469.6 s in
`one_worker_compiles`, `/proc/locks` named the one process as both the holder and the
blocked waiter, `wchan` read `locks_lock_inode_wait` at 0 % CPU (which looks exactly like
contention with a co-tenant and is not that), and killing the forked child released the
lock and let the round finish. See `state/perf10/bcx-HOSTFLOOR.md`.

Host-only: no device, no weights. The subprocess is what makes the fork real.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Take a lock, fork, optionally clear the child's inherited descriptors, wait for the child
# to say it is done, drop the parent's own copy, then ask for the lock again without
# blocking. Without the fix the second request fails, which is the deadlock in miniature.
PROBE = """
import fcntl, os, sys
sys.path.insert(0, {repo!r})
from tt_bio.main import _close_inherited_fds

lock_path = sys.argv[1]
handle = open(lock_path, "w")
fcntl.flock(handle, fcntl.LOCK_EX)

hold_r, hold_w = os.pipe()          # the parent releases this to end the child
done_r, done_w = os.pipe()          # the child reports that it has finished closing
if os.fork() == 0:
    os.close(hold_w)
    os.close(done_r)
    if {close_them}:
        _close_inherited_fds({{0, 1, 2, hold_r, done_w}})
    os.write(done_w, b".")
    os.read(hold_r, 1)              # live as long as the parent does
    os._exit(0)
os.close(hold_r)
os.close(done_w)
assert os.read(done_r, 1) == b"."

handle.close()                      # the parent's own release
again = open(lock_path, "w")
try:
    fcntl.flock(again, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("ACQUIRED")
except BlockingIOError:
    print("STILL HELD BY THE CHILD")
os.close(hold_w)
"""


def _probe(tmp_path: Path, close_them: bool) -> str:
    script = PROBE.format(repo=str(REPO), close_them=close_them)
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(script),
                          str(tmp_path / "compile.lock")],
                         capture_output=True, text=True, timeout=60,
                         env={**os.environ, "TT_BIO_DEBUG_STDERR": "1"})
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_an_inherited_descriptor_holds_the_parents_flock(tmp_path):
    """The control. Without the fix the lock outlives the parent's own release."""
    assert _probe(tmp_path, close_them=False) == "STILL HELD BY THE CHILD"


def test_closing_inherited_descriptors_releases_the_parents_flock(tmp_path):
    assert _probe(tmp_path, close_them=True) == "ACQUIRED"


KEEPS = """
import os, sys, tempfile
sys.path.insert(0, {repo!r})
from tt_bio.main import _close_inherited_fds

inherited = tempfile.TemporaryFile()
read_fd, write_fd = os.pipe()
report_r, report_w = os.pipe()
if os.fork() == 0:
    _close_inherited_fds({{0, 1, 2, read_fd, report_w}})
    os.fstat(read_fd)                      # kept: this is what the filter drains
    try:
        os.fstat(inherited.fileno())
        os.write(report_w, b"LEAKED")
    except OSError:
        os.write(report_w, b"CLOSED")
    os._exit(0)
os.close(report_w)
got = b""
while True:
    chunk = os.read(report_r, 64)
    if not chunk:
        break
    got += chunk
os.waitpid(-1, 0)
print(got.decode())
"""


def test_the_filter_child_keeps_only_what_it_forwards():
    """The kept descriptors survive and everything else the parent had open does not."""
    out = subprocess.run([sys.executable, "-c",
                          textwrap.dedent(KEEPS.format(repo=str(REPO)))],
                         capture_output=True, text=True, timeout=60,
                         env={**os.environ, "TT_BIO_DEBUG_STDERR": "1"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "CLOSED"
