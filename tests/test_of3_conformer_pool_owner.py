"""OpenFold3's conformer worker processes do not outlive the process that started them.

They are forkserver children, so nothing told them when their owner died: every
`tt-bio predict --model openfold3` on a host where a worker gets >= 24 threads left a
forkserver, 16 workers and a resource tracker behind, holding each other's pipes (and the
caller's stdout) open forever. Host-only: starts the pool in a child, SIGKILLs the child,
and checks the workers are gone.
"""
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

CHILD = textwrap.dedent("""
    import os, sys, time
    from tt_bio._vendor.openfold3.core.data.primitives.structure import query as q
    assert q.CONFORMER_PROCS > 1, q.CONFORMER_PROCS
    assert q.start_conformer_processes(wait=True)
    print(" ".join(str(p) for p in q._CONFORMER_PROCS_POOL._processes), flush=True)
    time.sleep(600)
""")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split(")")[1].split()[0] != "Z"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc")
def test_conformer_workers_exit_when_their_owner_is_killed():
    env = dict(os.environ, OMP_NUM_THREADS="32")
    child = subprocess.Popen([sys.executable, "-c", CHILD], env=env, stdout=subprocess.PIPE, text=True)
    workers = [int(p) for p in child.stdout.readline().split()]
    assert len(workers) > 1
    child.send_signal(signal.SIGKILL)
    child.wait()
    deadline = time.time() + 20
    while time.time() < deadline and any(_alive(p) for p in workers):
        time.sleep(0.5)
    assert not [p for p in workers if _alive(p)]
