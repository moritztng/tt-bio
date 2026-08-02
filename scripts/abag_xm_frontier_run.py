#!/usr/bin/env python3
"""AbAg-XM seeds-vs-samples frontier driver (state doc §6 Step 2).

Runs one card's static share of the campaign: its Arm-A jobs (200-sample
predicts) first, then its Arm-B units (10-sample predicts, one per
(target, seed-block)). One JSON record per finished job is appended to
~/abag_xm/frontier/progress.jsonl (host-local).

  --plan DIR                      write jobs_card{0..7}.json into DIR, exit
  --jobs FILE --card N            run the jobs in FILE sequentially
    [--timeout 7200] [--host_threads T]

Card assignment (frozen runbook): cards 0..7 = qb1:0-3, qb2:0-3. Arm A:
target index i -> card i % 8 (cards 0-3 get two). Arm B: unit (target,
seed-block j) -> card j % 8. Seeds: Arm A 42; Arm B 1000+10*j, disjoint.

Resume: a job whose out_dir holds a complete fold (results.json status ok
with exactly n_samples CIFs) is skipped.

Spawn-deadlock watchdog: the predict spawns its device worker with
mp "spawn" from a multi-threaded parent; the forked child can deadlock
before exec (observed 2026-07-29: child forked, zero syscalls, parent
parked in do_select forever, zero log output). Signature: whole process
tree at ~0 CPU, no tenstorrent fds, no results.json. Killed and retried
in-place (up to 3 attempts per job). Exemption: while a sibling worker
holds the card's lease flock our predict is parked in the acquire loop
(observed 2026-07-30 on card7: misread as deadlock) — waiting, never killed.
"""
import argparse, fcntl, json, os, signal, socket, subprocess, sys, time
from pathlib import Path

TARGETS = ["9q6y", "9tmp", "9gei", "9fte", "9wpm", "9qrv",
           "9ma0", "9q6z", "9j4c", "9uoi", "9m8l", "9ldx"]
ARM_A_SAMPLES, ARM_A_SEED = 200, 42
ARM_B_SAMPLES = 10
ARM_B_J = list(range(20))
MPS = 5
BASE = Path.home() / "abag_xm" / "frontier"
PROGRESS = BASE / "progress.jsonl"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
MSA_DB = Path.home() / ".boltz" / "msa_db"
CKPT_SHA256 = "5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d"
RECYCLING_STEPS, SAMPLING_STEPS = 10, 200  # resolved tt-bio defaults (D4a), unset on cmdline
WT = Path(__file__).resolve().parent.parent
MAX_DEADLOCK_RETRIES = 2  # 3 attempts total per job


def fold_python():
    """Interpreter that can run `-m tt_bio.main` (same candidates as abag_xm_generate.py)."""
    cands = [str(WT / "env" / "bin" / "python3"),
             str(Path.home() / "tt-bio" / "env" / "bin" / "python3"),
             sys.executable]
    for c in cands:
        if not Path(c).exists():
            continue
        r = subprocess.run([c, "-m", "tt_bio.main", "--help"],
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(WT)})
        if r.returncode == 0:
            return c
    raise SystemExit("no interpreter can run `-m tt_bio.main --help`: " + ", ".join(cands))


def plan(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {i: [] for i in range(8)}
    for i, t in enumerate(TARGETS):
        jobs[i % 8].append({"arm": "A", "target": t, "seed": ARM_A_SEED,
                            "n_samples": ARM_A_SAMPLES,
                            "out_dir": str(BASE / "A" / t)})
    for j in ARM_B_J:
        for t in TARGETS:
            jobs[j % 8].append({"arm": "B", "target": t, "seed": 1000 + 10 * j,
                                "seed_j": j, "n_samples": ARM_B_SAMPLES,
                                "out_dir": str(BASE / "B" / f"{t}_seed{j}")})
    for i in range(8):
        (out_dir / f"jobs_card{i}.json").write_text(json.dumps(jobs[i], indent=1))
        print(f"card {i}: {len(jobs[i])} jobs "
              f"{sum(1 for x in jobs[i] if x['arm'] == 'A')} A + "
              f"{sum(1 for x in jobs[i] if x['arm'] == 'B')} B")


def result_dir(job):
    return Path(job["out_dir"]) / f"opendde_results_{job['target']}"


def count_cifs(job):
    st = result_dir(job) / "structures"
    return len(list(st.glob("*.cif"))) if st.is_dir() else 0


def fold_status(job):
    rj = result_dir(job) / "results.json"
    try:
        data = json.loads(rj.read_text())
        rec = data[0] if isinstance(data, list) else data
        return rec.get("status")
    except Exception:
        return None


def complete(job):
    return fold_status(job) == "ok" and count_cifs(job) == job["n_samples"]


def record(job, card, wall_s, status):
    rec = {"ts": time.time(), "arm": job["arm"], "target": job["target"],
           "seed": job["seed"], "seed_j": job.get("seed_j"),
           "host": socket.gethostname(), "card": card,
           "wall_s": round(wall_s, 1), "tt_bio_commit": GIT_HEAD,
           "checkpoint_sha256": CKPT_SHA256,
           "recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
           "mps": MPS, "n_cifs": count_cifs(job), "status": status,
           "out_dir": job["out_dir"]}
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _proc_tree(root):
    """Descendant pids of root (inclusive) plus summed utime+stime jiffies."""
    kids, jiffies = {}, {}
    for pid_s in os.listdir("/proc"):
        if not pid_s.isdigit():
            continue
        try:
            fields = Path(f"/proc/{pid_s}/stat").read_text().rsplit(")", 1)[1].split()
            ppid, ut, stt = int(fields[1]), int(fields[11]), int(fields[12])
        except Exception:
            continue
        kids.setdefault(ppid, []).append(int(pid_s))
        jiffies[int(pid_s)] = ut + stt
    tree, stack = [], [root]
    while stack:
        pid = stack.pop()
        tree.append(pid)
        stack.extend(kids.get(pid, ()))
    return tree, sum(jiffies.get(p, 0) for p in tree)


def _tree_has_device(tree):
    for pid in tree:
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if "tenstorrent" in os.readlink(f"/proc/{pid}/fd/{fd}"):
                        return True
                except OSError:
                    pass
        except OSError:
            pass
    return False


def _card_held_elsewhere():
    """True when this host's device-lease lock for our TT_VISIBLE_DEVICES card is
    flock-held at all. Reachable from the watchdog only when OUR tree has no
    tenstorrent fds (checked first), so a held lock means a sibling worker owns
    the card and our predict is blocked in the lease acquire loop — waiting,
    not deadlocked. flock is kernel-released on death, so held == live holder.
    """
    visible = os.environ.get("TT_VISIBLE_DEVICES", "")
    phys = visible.split(",")[0] if visible else "0"
    lock = (Path.home() / ".coworker" / "state" / "leases"
            / f"{socket.gethostname()}-card{phys}.json")
    try:
        fd = os.open(lock, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def run_job(job, card, py, timeout, host_threads, log):
    yaml = WT / "examples" / "abag_xm" / f"{job['target']}.yaml"
    cmd = [py, "-m", "tt_bio.main", "predict", str(yaml),
           "--model", "opendde-abag", "--out_dir", job["out_dir"],
           "--diffusion_samples", str(job["n_samples"]),
           "--max_parallel_samples", str(MPS),
           "--msa_dir", str(MSA_DIR), "--msa_db_path", str(MSA_DB),
           "--seed", str(job["seed"]), "--override", "--write_pae",
           "--host_threads", str(host_threads)]
    env = {**os.environ, "PYTHONPATH": str(WT), "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    with open(log, "a") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        idle_snaps, last_j = 0, None
        status = None
        while True:
            rc = p.poll()
            if rc is not None:
                break
            time.sleep(30)
            if time.time() - t0 > timeout:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                return time.time() - t0, "timeout"
            if complete(job):
                continue
            tree, j = _proc_tree(p.pid)
            if _tree_has_device(tree):
                idle_snaps, last_j = 0, None
                continue
            # a deadlocked tree still burns ~1% in the parent's select loop;
            # a healthy no-device phase (featurization) saturates cores.
            if last_j is not None and j - last_j < 300:
                # a sibling worker holding the card lease means we are parked
                # in the flock acquire loop, not deadlocked — don't kill.
                idle_snaps = 0 if _card_held_elsewhere() else idle_snaps + 1
            else:
                idle_snaps = 0
            last_j = j
            if idle_snaps >= 6:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                return time.time() - t0, "spawn-deadlock"
        wall = time.time() - t0
    if rc != 0:
        return wall, f"failed:rc{rc}"
    st = fold_status(job)
    return wall, st if st in ("ok",) else f"failed:{st}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, metavar="DIR")
    ap.add_argument("--jobs", default=None, metavar="FILE")
    ap.add_argument("--card", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--host_threads", type=int, default=8)
    a = ap.parse_args()
    if a.plan:
        plan(a.plan)
        return
    if not a.jobs or a.card is None:
        ap.error("--jobs and --card are required without --plan")
    jobs = json.loads(Path(a.jobs).read_text())
    py = fold_python()
    BASE.mkdir(parents=True, exist_ok=True)
    print(f"driver card {a.card}: {len(jobs)} jobs, python={py}, "
          f"timeout={a.timeout}, host_threads={a.host_threads}", flush=True)
    for k, job in enumerate(jobs):
        tag = f"{job['arm']}/{job['target']}" + (f"/j{job['seed_j']}" if job["arm"] == "B" else "")
        if complete(job):
            print(f"[{k+1}/{len(jobs)}] {tag} already complete, skip", flush=True)
            continue
        log = Path(job["out_dir"]).with_suffix(".log")
        Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            wall, status = run_job(job, a.card, py, a.timeout, a.host_threads, log)
            rec = record(job, a.card, wall, status)
            print(f"[{k+1}/{len(jobs)}] {tag} {status} wall={rec['wall_s']}s "
                  f"n_cifs={rec['n_cifs']}", flush=True)
            if status != "spawn-deadlock" or attempt >= MAX_DEADLOCK_RETRIES:
                break
            attempt += 1
            print(f"[{k+1}/{len(jobs)}] {tag} retry {attempt} after spawn-deadlock", flush=True)


if __name__ == "__main__":
    GIT_HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WT,
                              capture_output=True, text=True).stdout.strip()
    main()
