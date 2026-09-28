"""``--msa_searches K`` bounds the offline searches on a host across processes.

A Galaxy runs one ``tt-bio predict`` per job, so the bound has to hold between separate
processes, not threads. Each case starts real OS processes that call the real
``compute_msa_offline`` with a stand-in ``colabfold_search`` on PATH. The stand-in records
when it ran and at what nice value, so overlap and priority are read from what happened.
"""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

STUB = """#!/bin/sh
# colabfold_search FASTA DB OUT ...: note start, nice, end; write an a3m per query.
echo "start $(date +%s.%N) $(cut -d' ' -f19 /proc/$$/stat)" >> "$LOG"
sleep "${HOLD:-0.6}"
out="$3"; mkdir -p "$out"
grep '^>' "$1" | cut -c2- | while read -r n; do printf '>q\\nMK\\n' > "$out/$n.a3m"; done
echo "end $(date +%s.%N)" >> "$LOG"
"""

SEARCH = textwrap.dedent("""
    import sys
    from pathlib import Path
    from tt_bio import main as m
    m.MSA_SEARCHES = int(sys.argv[3]) or None
    m.compute_msa_offline({sys.argv[1]: "MKTAYIAKQR"}, sys.argv[1], Path(sys.argv[2]),
                          sys.argv[4], pair=False)
""")


@pytest.fixture
def host(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    stub = bin_ / "colabfold_search"
    stub.write_text(STUB)
    stub.chmod(0o755)
    db = tmp_path / "db"          # the slots are per database, so each test has its own
    db.mkdir()
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "LOG": str(tmp_path / "log"),
           "TMPDIR": str(tmp_path)}
    return tmp_path, db, env


def _launch(host, n, k, **extra):
    tmp, db, env = host
    env = {**env, **extra}
    return [subprocess.Popen([sys.executable, "-c", SEARCH, f"s{i}", str(tmp / f"msa{i}"),
                              str(k), str(db)], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
            for i in range(n)]


def _intervals(tmp):
    starts, ends, nices = [], [], []
    for line in (tmp / "log").read_text().split("\n"):
        f = line.split()
        if f and f[0] == "start":
            starts.append(float(f[1]))
            nices.append(int(f[2]))
        elif f and f[0] == "end":
            ends.append(float(f[1]))
    return sorted(starts), sorted(ends), nices


def _peak(starts, ends):
    ev = sorted([(t, 1) for t in starts] + [(t, -1) for t in ends])
    live = peak = 0
    for _t, d in ev:
        live += d
        peak = max(peak, live)
    return peak


def _finish(procs, timeout=60):
    outs = []
    for p in procs:
        out, _ = p.communicate(timeout=timeout)
        assert p.returncode == 0, out
        outs.append(out)
    return outs


def test_k_bounds_searches_across_processes(host):
    tmp = host[0]
    outs = _finish(_launch(host, 6, 2))
    starts, ends, nices = _intervals(tmp)
    assert len(starts) == len(ends) == 6
    assert _peak(starts, ends) == 2
    assert any("waiting for one of 2 MSA search slots" in o for o in outs)
    for i in range(6):
        assert (tmp / f"msa{i}" / f"s{i}.a3m").exists()


def test_without_k_they_all_search_at_once(host):
    """The control: the same six processes, no limit, overlap six ways."""
    tmp = host[0]
    _finish(_launch(host, 6, 0, HOLD="2"))
    starts, ends, _ = _intervals(tmp)
    assert _peak(starts, ends) == 6


def test_a_search_runs_at_the_lowest_cpu_priority(host):
    tmp = host[0]
    _finish(_launch(host, 1, 1))
    assert _intervals(tmp)[2] == [19]


def test_a_killed_searcher_frees_its_slot(host):
    """K=1: the first searcher is SIGKILLed mid-search; the second must still search."""
    tmp = host[0]
    first = _launch(host, 1, 1, HOLD="30")[0]
    for _ in range(100):
        if (tmp / "log").exists():
            break
        time.sleep(0.1)
    second = _launch(host, 1, 1)[0]
    time.sleep(1.5)
    assert second.poll() is None          # waiting behind the slot the first holds
    os.kill(first.pid, signal.SIGKILL)
    first.wait()
    out, _ = second.communicate(timeout=30)
    assert second.returncode == 0, out
    starts, _ends, _ = _intervals(tmp)
    assert len(starts) == 2
