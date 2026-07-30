#!/usr/bin/env python3
"""AbAg-XM frontier endgame supervisor (post-reboot recovery + finish).

qb1 rebooted 2026-07-30 09:04 UTC (kernel 6.8.0-124 -> 6.8.0-136), killing the
three in-flight A-label retries (9j4c, 9q6y, 9q6z), the pool supervisor, and
the 9q6y/9wpm pool builds. All 252 folds and all 240 B labels survived on
disk; 9/12 A labels and 3/12 B-pool labels are done. This script finishes the
tail with per-item isolation (p13 lesson: one bad item must never kill the
loop):

  1. A labels for 9j4c/9q6y/9q6z — direct abag_xm_labels.py invocations,
     identical cmd/env to abag_xm_frontier_label.py, pair_workers=8.
  2. B-pool builds for the 9 targets without pool labels (9j4c first —
     heaviest matrix), concurrency sized to a worker budget.
  3. When 12/12 A labels + 12/12 pool labels exist: run
     abag_xm_frontier_analysis.py, save the section-7 markdown body to
     logs/analysis_section7.md, touch ENDGAME_DONE.

Launch detached on qb1:  cd $WT && setsid nohup python3 -u \
  scripts/abag_xm_frontier_endgame.py </dev/null >/dev/null 2>&1 &
Log: ~/abag_xm/frontier/logs/endgame.log
"""
import json, os, subprocess, sys, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/abag-xm-seeds-vs-samples-oracle-frontier-p2")
BASE = Path.home() / "abag_xm" / "frontier"
GT = Path.home() / "abag_xm" / "ground_truth"
POOL = BASE / "B_pool"
LOG = BASE / "logs" / "endgame.log"
DONE_MARKER = BASE / "ENDGAME_DONE"
TARGETS = ["9q6y", "9tmp", "9gei", "9fte", "9wpm", "9qrv",
           "9ma0", "9q6z", "9j4c", "9uoi", "9m8l", "9ldx"]
A_PW, POOL_PW = 14, 14
MAX_WORKERS = 30          # qb1 has 32 threads; only the two 9j4c jobs remain
POLL_S = 30
LABEL_VENV_PY = Path.home() / ".abag_xm_label_venv" / "bin" / "python3"
SHARED_VENV = Path("/home/ttuser/tt-bio-dev/env")
THREAD_CAPS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
POOL_ORDER = ["9j4c", "9ldx", "9m8l", "9ma0", "9q6y", "9q6z", "9qrv", "9uoi", "9wpm"]


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def label_env(caps=True):
    sp = next(iter(SHARED_VENV.glob("lib/python*/site-packages")), None)
    pp = ":".join(str(x) for x in [sp, WT] if x)
    env = {**os.environ, "PYTHONPATH": pp,
           "PATH": str(Path.home() / ".local" / "bin") + os.pathsep + os.environ.get("PATH", "")}
    if caps:
        env.update({v: "2" for v in THREAD_CAPS})
    return env


class Job:
    def __init__(self, name, cmd, nworkers, env, out_path, log_path):
        self.name = name
        self.cmd = cmd
        self.nworkers = nworkers
        self.env = env
        self.out_path = Path(out_path)
        self.log_path = Path(log_path)
        self.proc = None
        self.t0 = None
        self.retries_left = 1

    def launch(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        lf = self.log_path.open("a")
        self.proc = subprocess.Popen(self.cmd, stdout=lf, stderr=subprocess.STDOUT,
                                     env=self.env, cwd=str(WT), start_new_session=True)
        self.t0 = time.time()
        self.state = "running"
        log(f"launch {self.name} pid={self.proc.pid} w={self.nworkers}")

    def poll(self):
        """True while the job should stay tracked (running or just relaunched)."""
        if self.proc.poll() is None:
            return True
        wall = round(time.time() - self.t0)
        ok = self.proc.returncode == 0 and self.out_path.exists()
        if ok:
            self.state = "ok"
            log(f"done {self.name} wall={wall}s")
        elif self.retries_left > 0:
            self.retries_left -= 1
            log(f"RETRY {self.name} rc={self.proc.returncode} out_exists={self.out_path.exists()} wall={wall}s")
            time.sleep(10)
            self.launch()
            return True
        else:
            self.state = "failed"
            log(f"FAILED {self.name} rc={self.proc.returncode} out_exists={self.out_path.exists()} wall={wall}s")
        return False


def a_label_job(target):
    out_dir = BASE / "A" / target
    rd = next(out_dir.glob("opendde_results_*"))
    (out_dir / ".label_lock").unlink(missing_ok=True)   # stale lock from the killed retry
    cmd = [str(LABEL_VENV_PY), str(WT / "scripts" / "abag_xm_labels.py"), str(rd),
           str(GT / f"{target}.cif"), str(WT / "examples" / "abag_xm" / f"{target}.yaml"),
           "--out", str(out_dir / "labels.json"), "--pair_workers", str(A_PW)]
    return Job(f"A/{target}", cmd, A_PW, label_env(caps=True),
               out_dir / "labels.json", BASE / "logs" / f"endgame_A_{target}.log")


def pool_job(target):
    cmd = [sys.executable, str(WT / "scripts" / "abag_xm_frontier_pool.py"), target]
    env = label_env(caps=False)
    env["POOL_PAIR_WORKERS"] = str(POOL_PW)
    return Job(f"pool/{target}", cmd, POOL_PW, env,
               POOL / target / "labels.json", POOL / target / "pool_build.log")


def a_done():
    return [t for t in TARGETS if (BASE / "A" / t / "labels.json").exists()]


def pools_done():
    return [t for t in TARGETS if (POOL / t / "labels.json").exists()]


def run_analysis():
    log("all labels complete: running analysis")
    r = subprocess.run([sys.executable, str(WT / "scripts" / "abag_xm_frontier_analysis.py")],
                       capture_output=True, text=True, cwd=str(WT), timeout=3600)
    (BASE / "logs" / "analysis_section7.md").write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""))
    if r.returncode != 0:
        log(f"analysis FAILED rc={r.returncode}: {r.stderr.strip()[-300:]}")
        return False
    try:
        d = json.loads((BASE / "analysis.json").read_text())
        c = (d.get("n_A_labeled"), d.get("n_B_targets_complete"), d.get("n_B_pools"))
        log(f"analysis ok: counters n_A={c[0]} n_B_complete={c[1]} n_pools={c[2]}")
        if c != (12, 12, 12):
            log("analysis counters INCOMPLETE — marker withheld")
            return False
    except Exception as e:
        log(f"analysis.json unreadable: {e}")
        return False
    DONE_MARKER.touch()
    log("ENDGAME_DONE")
    return True


def main():
    log(f"endgame start: A_done={len(a_done())}/12 pools_done={len(pools_done())}/12")
    running = []
    a_pending = [t for t in TARGETS if not (BASE / "A" / t / "labels.json").exists()]
    pool_pending = [t for t in POOL_ORDER if not (POOL / t / "labels.json").exists()]
    while True:
        running = [j for j in running if j.poll()]
        inflight = sum(j.nworkers for j in running)
        while a_pending and inflight + A_PW <= MAX_WORKERS:
            j = a_label_job(a_pending.pop(0))
            j.launch()
            running.append(j)
            inflight += j.nworkers
        while pool_pending and inflight + POOL_PW <= MAX_WORKERS:
            j = pool_job(pool_pending.pop(0))
            j.launch()
            running.append(j)
            inflight += j.nworkers
        if not running and not a_pending and not pool_pending:
            break
        time.sleep(POLL_S)
    a_ok, p_ok = a_done(), pools_done()
    log(f"work phase over: A={len(a_ok)}/12 pools={len(p_ok)}/12")
    if len(a_ok) == 12 and len(p_ok) == 12:
        run_analysis()
    else:
        log("INCOMPLETE — not running analysis; see FAILED lines above")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"supervisor exception (isolated, will not retry): {e!r}")
        raise
