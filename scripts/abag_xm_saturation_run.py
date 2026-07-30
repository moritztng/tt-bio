#!/usr/bin/env python3
"""AbAg-XM oracle-saturation-depth driver (state doc abag-xm-oracle-saturation-depth §6).

Runs one card's static share of the campaign. Every job is fully specified in
the jobs JSON: target, model, seed, n_samples, out_dir, timeout_s, mps.
One JSON record per finished job is appended to ~/abag_xm/saturation/progress.jsonl
(host-local; qb2's file is mirrored to qb1 as progress_qb2.jsonl by the labeler
rsync, same convention as the frontier).

  --plan DIR --model opendde-abag   write jobs_card{0..7}.json into DIR, exit
  --jobs FILE --card N              run the jobs in FILE sequentially
    [--timeout 7200] [--host_threads T]

Card assignment: LPT (longest-projected-first to the currently least-loaded of
8 cards = qb1:0-3, qb2:0-3). The frozen runbook said "sort desc, deal"; LPT is
the same greedy family and balances strictly better (max load 7.8h vs 9.6h
strict round-robin on the §3 projections). Chunks of one target never share a
card under LPT here.

Resume: a job whose out_dir holds a complete fold (results.json status ok with
exactly n_samples CIFs) is skipped. Spawn-deadlock watchdog + lease-exemption
identical to abag_xm_frontier_run.py (proven in the frontier campaign).
"""
import argparse, fcntl, json, os, signal, socket, subprocess, sys, time
from pathlib import Path

MPS = 5
BASE = Path.home() / "abag_xm" / "saturation"
PROGRESS = BASE / "progress.jsonl"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
MSA_DB = Path.home() / ".boltz" / "msa_db"
RECYCLING_STEPS, SAMPLING_STEPS = 10, 200  # resolved tt-bio defaults (D4a), unset on cmdline
WT = Path(__file__).resolve().parent.parent
MAX_DEADLOCK_RETRIES = 2  # 3 attempts total per job

# tt_bio model id -> (results-dir prefix per tt_bio/main.py predict_results_dir_name,
# output parent dir per state doc §6, base seed per §3 — disjoint per model and from
# the frontier's 42/1000-1199).
MODELS = {"opendde-abag": ("opendde", "opendde", 5000),
          "protenix-v2": ("protenix", "protenix", 7000),
          "boltz2": ("boltz2", "boltz2", 9000)}

# Projected 1000-sample wall_s per target, state doc §3 (known 11: 342 + 5*(wall_200
# - 342) from measured frontier walls; new 5: 2.3 s/sample/100 residues).
PROJ = {"9q6y": 21252, "9q6z": 18922, "9m8l": 15202, "9ma0": 14282, "9tmp": 13657,
        "9qrv": 11497, "9ldx": 10502, "9wpm": 8582, "9gei": 8427, "9fte": 6802,
        "9uoi": 5712, "9nl0": 10853, "9l9y": 13498, "9gfr": 12233, "9mnu": 13774,
        "9zen": 5632}
CHUNK = {"9q6y", "9q6z"}  # projection > ~5h -> two 500-sample chunks (§3), seeds base/base+1000
CHEAP8 = ["9zen", "9uoi", "9fte", "9gei", "9wpm", "9ldx", "9nl0", "9qrv"]  # §3 fallback set
TIMEOUT_FACTOR = 1.6
FIXED_S = 342.0  # frontier cost fit (contended, mps 5); used only for chunk split


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


def jobs_for_model(model, targets, scale=1.0):
    prefix, out_parent, base_seed = MODELS[model]
    jobs = []
    for t in targets:
        proj = round(PROJ[t] * scale)
        # opendde keeps the frozen §3 chunk set; Q2 models chunk on the §3 rule
        # (>4h scaled projection) since their costs differ.
        chunk = t in CHUNK if model == "opendde-abag" else proj > 4 * 3600
        if chunk:
            chunk_proj = FIXED_S + (proj - FIXED_S) / 2.0
            for j in (0, 1):
                jobs.append({"target": t, "chunk": j, "model": model,
                             "seed": base_seed + 1000 * j, "n_samples": 500,
                             "proj_s": round(chunk_proj),
                             "timeout_s": round(TIMEOUT_FACTOR * chunk_proj),
                             "mps": MPS,
                             "out_dir": str(BASE / out_parent / f"{t}_c{j}")})
        else:
            jobs.append({"target": t, "chunk": None, "model": model,
                         "seed": base_seed, "n_samples": 1000,
                         "proj_s": proj, "timeout_s": round(TIMEOUT_FACTOR * proj),
                         "mps": MPS,
                         "out_dir": str(BASE / out_parent / t)})
    return jobs


def plan(out_dir, model, targets, scale=1.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = sorted(jobs_for_model(model, targets, scale), key=lambda j: -j["proj_s"])
    loads, cards = [0.0] * 8, [[] for _ in range(8)]
    for job in jobs:  # LPT: longest first onto the currently least-loaded card
        c = min(range(8), key=lambda i: loads[i])
        cards[c].append(job)
        loads[c] += job["proj_s"]
    for i in range(8):
        (out_dir / f"jobs_card{i}.json").write_text(json.dumps(cards[i], indent=1))
        print(f"card {i}: {len(cards[i])} jobs, projected {loads[i]/3600:.2f} h: "
              + ",".join(j["target"] + (f"_c{j['chunk']}" if j["chunk"] is not None else "")
                         for j in cards[i]))


def result_dir(job):
    prefix = MODELS[job["model"]][0]
    return Path(job["out_dir"]) / f"{prefix}_results_{job['target']}"


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
    rec = {"ts": time.time(), "model": job["model"], "target": job["target"],
           "chunk": job.get("chunk"), "seed": job["seed"],
           "host": socket.gethostname(), "card": card,
           "wall_s": round(wall_s, 1), "tt_bio_commit": GIT_HEAD,
           "checkpoint_sha256": job.get("ckpt_sha256"),
           "recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
           "mps": job.get("mps", MPS), "n_samples": job["n_samples"],
           "n_cifs": count_cifs(job), "status": status,
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
    the card and our predict is parked in the acquire loop — waiting, never killed.
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
           "--model", job["model"], "--out_dir", job["out_dir"],
           "--diffusion_samples", str(job["n_samples"]),
           "--max_parallel_samples", str(job.get("mps", MPS)),
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
    ap.add_argument("--model", default="opendde-abag", choices=sorted(MODELS))
    ap.add_argument("--targets", default="all",
                    help="'all' (16, §3) or 'cheap8' (Q2 fallback set, §3)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply §3 opendde projections (Q2 models, from their canary)")
    ap.add_argument("--jobs", default=None, metavar="FILE")
    ap.add_argument("--card", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=7200,
                    help="fallback; a job's own timeout_s wins (per-job bound, §3)")
    ap.add_argument("--host_threads", type=int, default=8)
    a = ap.parse_args()
    if a.plan:
        targets = sorted(PROJ) if a.targets == "all" else CHEAP8
        plan(a.plan, a.model, targets, a.scale)
        return
    if not a.jobs or a.card is None:
        ap.error("--jobs and --card are required without --plan")
    jobs = json.loads(Path(a.jobs).read_text())
    py = fold_python()
    BASE.mkdir(parents=True, exist_ok=True)
    print(f"driver card {a.card}: {len(jobs)} jobs, python={py}, "
          f"host_threads={a.host_threads}", flush=True)
    for k, job in enumerate(jobs):
        tag = f"{job['model']}/{job['target']}" + (
            f"/c{job['chunk']}" if job.get("chunk") is not None else "")
        if complete(job):
            print(f"[{k+1}/{len(jobs)}] {tag} already complete, skip", flush=True)
            continue
        log = Path(job["out_dir"]).with_suffix(".log")
        Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
        timeout = job.get("timeout_s") or a.timeout
        attempt = 0
        while True:
            wall, status = run_job(job, a.card, py, timeout, a.host_threads, log)
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
