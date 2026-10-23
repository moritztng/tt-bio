"""A gate leg on a lent JapanFold chip stops when the chip is recalled, and starts nothing after.

A recall SIGINTs the borrower: the fold's process group (glx_elastic's give-back), every process
pinned to the chip (the agent's lend mode, aiand-bio 6c6ab7287) or the pool job's group. On .110 on
2026-10-23 the gate drivers read the interrupted fold as a failed attempt and started the next one
on the chip JapanFold was resetting; chips 1 and 4 were marked dead. Host-only: every "fold" here is
a small Python process, no device is opened.
"""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import gate_fanout as gf      # noqa: E402

#: A fold whose chip is taken back mid-run: it SIGINTs its own group, as a recall does.
RECALLED_FOLD = "import os, signal, time; time.sleep(0.3); os.killpg(0, signal.SIGINT); time.sleep(60)"


def _driver(tmp_path, body: str) -> subprocess.CompletedProcess:
    """Run `body` as a gate driver with gate_guard's teardown armed and a second leg (a fold on
    another chip) live beside it. The driver retries up to three times, the way capacity_gate's
    rungs and full_parity_gate's seeds do, and writes one line per attempt it starts."""
    script = textwrap.dedent(f"""
        import os, signal, subprocess, sys, time, threading
        sys.path.insert(0, {str(REPO / 'scripts')!r})
        import gate_guard
        gate_guard.install_teardown()
        other = subprocess.Popen([sys.executable, "-c",
            "import signal, sys, time\\n"
            "signal.signal(signal.SIGINT, lambda *a: (open({str(tmp_path / 'other.int')!r}, 'w').close(), sys.exit(0)))\\n"
            "time.sleep(600)"], start_new_session=True)
        gate_guard.track_leg(other.pid)
        attempts = open({str(tmp_path / 'attempts')!r}, "a")
        def attempt():
    """) + textwrap.indent(textwrap.dedent(body), "    ") + textwrap.dedent("""
        def loop():
            for i in range(3):
                attempts.write(f"{i}\\n"); attempts.flush()
                attempt()
        t = threading.Thread(target=loop); t.start()
        while t.is_alive():
            t.join(timeout=0.5)
        sys.exit(0)
    """)
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=120)
    r.wall = time.monotonic() - t0
    return r


def _assert_stopped(tmp_path, r):
    assert r.returncode == 128 + signal.SIGINT, f"the gate exited {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert (tmp_path / "attempts").read_text().split() == ["0"], "the gate started another attempt"
    assert (tmp_path / "other.int").exists(), "the leg on the other chip never got the SIGINT"
    assert r.wall < 30, f"the gate took {r.wall:.0f} s to stop"


def test_capacity_gate_stops_on_a_recalled_fold(tmp_path):
    r = _driver(tmp_path, f"""
        import capacity_gate as cg, pathlib
        class W:
            def cmd(self, argv, env): return argv
        log = pathlib.Path({str(tmp_path / 'leg.log')!r})
        cg.execute(W(), [sys.executable, "-c", {RECALLED_FOLD!r}], log, mode="off",
                   hook_out=log.with_suffix(".hook"), hookdir=log.parent, out_dir=log.with_suffix(".out"))
    """)
    _assert_stopped(tmp_path, r)


def test_full_parity_gate_stops_on_a_recalled_fold(tmp_path):
    r = _driver(tmp_path, f"""
        import full_parity_gate as fp, pathlib
        with open({str(tmp_path / 'leg.log')!r}, "w") as logf:
            fp._run_local_fold([sys.executable, "-c", {RECALLED_FOLD!r}], pathlib.Path({str(tmp_path)!r}), logf, 120)
    """)
    _assert_stopped(tmp_path, r)


def test_release_gate_stops_on_a_recalled_fold(tmp_path):
    pytest.importorskip("torch")                # release_gate imports it at module level
    rg = __import__("importlib.util").util
    spec = rg.spec_from_file_location("release_gate_under_test", REPO / "scripts" / "release_gate.py")
    mod = rg.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._release_driver_device = lambda: None
    started = []
    with pytest.raises(KeyboardInterrupt):
        for rung in (256, 512, 768):            # the size ladder: one fold per rung
            started.append(rung)
            mod._run_fold([sys.executable, "-c", RECALLED_FOLD], 120)
    assert started == [256], f"the ladder went on to rungs {started[1:]} after a recalled fold"


def test_its_own_timeout_is_not_read_as_a_recall():
    """capacity_gate's stall kill and every gate's fold timeout send SIGTERM too; only an exit the
    gate did not cause stops it."""
    import gate_guard
    gate_guard.stop_if_stopped_outside(0)
    gate_guard.stop_if_stopped_outside(1)
    gate_guard.stop_if_stopped_outside(-99)
    with pytest.raises(KeyboardInterrupt):
        gate_guard.stop_if_stopped_outside(-signal.SIGINT)


def test_a_recall_of_the_pool_job_reaches_the_leg_driver(tmp_path):
    """The leg runs in its own session under flock and timeout, so the job's group never held it.
    The job shell passes the SIGINT on, waits for the driver, and returns its exit code."""
    tree = tmp_path / "trees" / ("a" * 12)
    tree.mkdir(parents=True)
    out = tmp_path / "out"
    got = tmp_path / "driver.int"
    driver = (f"import signal, sys, time\n"
              f"signal.signal(signal.SIGINT, lambda *a: (open({str(got)!r}, 'w').close(), time.sleep(1), sys.exit(130)))\n"
              f"open({str(tmp_path / 'driver.up')!r}, 'w').close()\n"
              f"time.sleep(600)")
    h = gf.Host("h", {"arch": "bh", "card_type": "p", "root": str(tmp_path)}, "a" * 40)
    cmd = h.command(gf.Leg("x", [sys.executable, "-c", driver], "parity"), None, str(out))
    job = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)      # as pool.sh starts a job
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "driver.up").exists():
            assert time.monotonic() < deadline and job.poll() is None, "the leg never started"
            time.sleep(0.1)
        os.killpg(job.pid, signal.SIGINT)
        rc = job.wait(timeout=30)
    finally:
        if job.poll() is None:
            os.killpg(job.pid, signal.SIGKILL)
    assert got.exists(), "the SIGINT to the pool job never reached the leg's driver"
    assert rc == 130, f"the job returned {rc}, not the driver's exit code"
    mark = f"GATE_ATTEMPT={out}"
    def environ(p):
        try:
            return p.read_bytes().split(b"\0")
        except OSError:
            return []
    left = [p.parent.name for p in Path("/proc").glob("[0-9]*/environ") if mark.encode() in environ(p)]
    assert not left, f"processes of the attempt outlived it: {left}"
