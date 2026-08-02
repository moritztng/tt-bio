#!/usr/bin/env python3
"""AbAg-XM deep-N saturation full-panel driver (state doc abag-xm-deepn-saturation-fullpanel).

Lineage: adapted from abag_xm_saturation_run.py (oracle-saturation-depth, proven on qb1/qb2 at
N=1000 x 16 targets). Same job-JSON + claim-queue + resume-on-complete + deadlock-watchdog
machinery; what changes is the campaign shape:

  - 164 targets x 4 models (esmfold2 added) x N ladder 64/256/512 (1024 only per stop rule).
  - Seeds disjoint per model AND from every prior campaign (frontier 42 + 1000-1199, saturation
    5000/7000/9000): opendde 20000, protenix 30000, boltz2 40000, esmfold2 50000. Chunked jobs
    take base+1000*j.
  - Projections come from tier_a's own measured 50-sample walls (qb1-contended, mps 5):
    proj(N) = wall50 * N/50. The pilot re-fits s/sample; planning stays conservative.
  - Chunking: proj_chunk > 4h splits (time), and boltz2 never folds >256 samples per job
    (host RAM ~221 MB/sample, sat-depth p42). Chunk sizes divide the rung (powers of two).
  - --msa_cache_only always: the 164-target cache is complete from tier_a; a miss is a bug,
    not a reason to re-search (the opendde paired-MSA history makes this non-negotiable).

  --select_pilot            write BASE/pilot_targets.json (deterministic rule), exit
  --plan_queue DIR --rung N [--targets pilot|all]   write one qb1 claim queue, exit
  --queue DIR --card N [--wait_card]                pull longest-first until empty

DONE_CHECK safety convention (binding): this driver never prints a literal "NN.N%" string.
"""
import argparse, fcntl, json, os, re, signal, socket, subprocess, sys, time
from pathlib import Path

MPS = 5
MPS_BOLTZ2 = 8  # width-capped after the ceil-chunking fix (b62301f5) + L1 budget rule
BOLTZ2_CHUNK_CAP = 256  # samples per boltz2 job (host RAM ~221 MB/sample)
TIME_CHUNK_S = 4 * 3600
BASE = Path.home() / "abag_xm" / "deepn"
PROGRESS = BASE / "progress.jsonl"
PLAN_INPUTS = BASE / "plan_inputs.json"
PILOT_JSON = BASE / "pilot_targets.json"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"  # cache-only: never paired with --msa_db_path
WT = Path(__file__).resolve().parent.parent
MAX_DEADLOCK_RETRIES = 2  # 3 attempts total per job
# tt-metal OOM families: DRAM/L1 allocator exhaustion and static L1 circular-buffer
# clashes (the latter never contains the allocator's wording).
OOM_SIGS = ("Out of Memory: Not enough space to allocate", "clash with L1 buffers")

# tt_bio model id -> (results-dir prefix, out_parent, base seed). Resolved defaults at this
# HEAD (recorded in every progress record): recycling 10/10/3/10, sampling 200/200/200/100.
MODELS = {"opendde-abag": ("opendde", "opendde", 20000),
          "protenix-v2": ("protenix", "protenix", 30000),
          "boltz2": ("boltz2", "boltz2", 40000),
          "esmfold2": ("esmfold2", "esmfold2", 50000)}
RESOLVED = {"opendde-abag": (10, 200), "protenix-v2": (10, 200),
            "boltz2": (3, 200), "esmfold2": (10, 100)}
TIMEOUT_FACTOR = 3.0


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


def load_plan_inputs():
    return json.loads(PLAN_INPUTS.read_text())


def select_pilot():
    """16 targets: 8 GT + 8 non-GT, each evenly spaced over n_residues (deterministic)."""
    pi = load_plan_inputs()
    items = [(t, d["n_res"] or 0) for t, d in pi.items() if d.get("yaml")]
    gt = sorted([i for i in items if pi[i[0]]["gt"]], key=lambda x: x[1])
    nogt = sorted([i for i in items if not pi[i[0]]["gt"]], key=lambda x: x[1])

    def spread(rows, k):
        return [rows[round(i * (len(rows) - 1) / (k - 1))][0] for i in range(k)]

    picks = sorted(spread(gt, 8) + spread(nogt, 8))
    PILOT_JSON.write_text(json.dumps({"rule": "8 GT + 8 non-GT, evenly spaced over "
                                      "n_residues (rank i*(n-1)/7), sorted", "targets": picks},
                                     indent=1))
    print(f"pilot ({len(picks)}): {','.join(picks)} -> {PILOT_JSON}")


def jobs_for(model, targets, rung, pi):
    prefix, out_parent, base_seed = MODELS[model]
    mps = MPS_BOLTZ2 if model == "boltz2" else MPS
    jobs = []
    for t in targets:
        wall50 = pi[t]["wall50"].get(model)
        if wall50 is None:
            print(f"WARN: no tier_a wall for {model}/{t}, skipping")
            continue
        proj = wall50 * rung / 50.0
        n_chunks = 1
        while proj / n_chunks > TIME_CHUNK_S:
            n_chunks *= 2
        if model == "boltz2":
            while rung // n_chunks > BOLTZ2_CHUNK_CAP:
                n_chunks *= 2
        for j in range(n_chunks):
            n = rung // n_chunks
            cproj = proj / n_chunks
            name = f"{t}_n{rung}" + (f"_c{j}" if n_chunks > 1 else "")
            jobs.append({"target": t, "rung": rung,
                         "chunk": j if n_chunks > 1 else None,
                         "model": model, "seed": base_seed + 1000 * j,
                         "n_samples": n, "proj_s": round(cproj),
                         "timeout_s": round(TIMEOUT_FACTOR * cproj), "mps": mps,
                         "out_dir": str(BASE / out_parent / name)})
    return jobs


def plan_queue(out_root, rung, targets_sel):
    """One claim queue for qb1 (the only host this campaign runs on unless Moritz powers qb2)."""
    pi = load_plan_inputs()
    if targets_sel == "pilot":
        targets = json.loads(PILOT_JSON.read_text())["targets"]
    else:
        targets = sorted(t for t, d in pi.items() if d.get("yaml"))
    jobs = []
    for model in sorted(MODELS):
        jobs += jobs_for(model, targets, rung, pi)
    jobs.sort(key=lambda j: -j["proj_s"])
    qdir = Path(out_root)
    qdir.mkdir(parents=True, exist_ok=True)
    for rank, job in enumerate(jobs):
        tag = f"{job['model']}_{job['target']}_n{job['rung']}" + (
            f"_c{job['chunk']}" if job["chunk"] is not None else "")
        (qdir / f"job_{rank:04d}_{tag}.json").write_text(json.dumps(job, indent=1))
    total = sum(j["proj_s"] for j in jobs)
    print(f"{len(jobs)} jobs, {total/3600:.1f} projected card-h "
          f"({total/3600/2:.1f} h wall on 2 cards, {total/3600/4:.1f} h on 4) -> {qdir}")


def oom(log):
    try:
        tail = Path(log).read_text()[-200000:]
        return any(sig in tail for sig in OOM_SIGS)
    except Exception:
        return False


def claim(jf):
    """Atomically take a queue job for this driver. Returns False if someone else has it."""
    try:
        fd = os.open(str(jf) + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
    os.close(fd)
    return True


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
           "rung": job["rung"], "chunk": job.get("chunk"), "seed": job["seed"],
           "host": socket.gethostname(), "card": card,
           "wall_s": round(wall_s, 1), "tt_bio_commit": GIT_HEAD,
           "recycling_steps": RESOLVED[job["model"]][0],
           "sampling_steps": RESOLVED[job["model"]][1],
           "msa_cache_only": True,
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


def _host_oom_since(t0, pids):
    """The kernel's oom-kill line for one of `pids`, if the host OOM-killer fired since t0.

    See abag_xm_saturation_run.py for the full history: a global-OOM kill looks exactly like a
    spawn deadlock from outside, and retries deterministically reproduce it (job sizing, not
    flake). Report the real cause.
    """
    out = ""
    for cmd in (["journalctl", "-k", "--since", "-6h", "--no-pager"],
                ["sudo", "-n", "dmesg", "-T"],
                ["dmesg", "-T"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception:
            continue
        if r.stdout.strip():
            out = r.stdout
            break
    if not out:
        return None
    want = {str(p) for p in pids}
    for line in reversed(out.splitlines()):
        if "Killed process" not in line and "oom-kill:" not in line:
            continue
        if any(re.search(rf"\b{p}\b", line) for p in want):
            return line.strip()[:300]
    return None


def _tree_rss_gb(tree):
    total = 0
    for pid in tree:
        try:
            total += int(Path(f"/proc/{pid}/statm").read_text().split()[1])
        except Exception:
            continue
    return total * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3


def _avail_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return float("nan")


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
    # -u: without it the fold stdout is block-buffered into the log file and a SIGKILL from
    # the deadlock watchdog discards the buffer, leaving a 0-byte log for precisely the runs
    # that need explaining -- and oom() then reads that empty log and never fires.
    cmd = [py, "-u", "-m", "tt_bio.main", "predict", str(yaml),
           "--model", job["model"], "--out_dir", job["out_dir"],
           "--diffusion_samples", str(job["n_samples"]),
           "--seed", str(job["seed"]), "--override", "--write_pae",
           "--host_threads", str(host_threads)]
    if job["model"] == "esmfold2":
        # Single-sequence by design (D12/A.5, tier_a protocol): this leg measures the
        # no-MSA regime, so MSA flags must NOT be passed (a warm shared-cache hit would
        # silently MSA-condition the fold -- and the MSA path is also what blew DRAM to
        # 12 GB on 9q6y). No --max_parallel_samples: the runtime auto-chunks the sample
        # batch by TT_ESMFOLD2_DIFFUSION_BUDGET; forcing mps caused the pilot OOMs.
        cmd += ["--recycling_steps", "10", "--sampling_steps", "100",
                "--single_sequence"]
    else:
        cmd += ["--max_parallel_samples", str(job.get("mps", MPS)),
                "--msa_dir", str(MSA_DIR), "--msa_cache_only"]
    env = {**os.environ, "PYTHONPATH": str(WT), "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    with open(log, "a") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        idle_snaps, last_j = 0, None
        seen_pids, peak_rss = {p.pid}, 0.0
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
                idle_snaps, last_j, seen_pids = 0, None, set(tree)
                peak_rss = max(peak_rss, _tree_rss_gb(tree))
                continue
            seen_pids |= set(tree)
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
                oom = _host_oom_since(t0, seen_pids)
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                if oom:
                    print(f"  host OOM-killer took a worker (peak tree RSS "
                          f"{peak_rss:.0f} GB, {_avail_gb():.0f} GB avail now): {oom}",
                          flush=True)
                    return time.time() - t0, "host-oom"
                return time.time() - t0, "spawn-deadlock"
        wall = time.time() - t0
    if rc != 0:
        return wall, f"failed:rc{rc}"
    st = fold_status(job)
    return wall, st if st in ("ok",) else f"failed:{st}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select_pilot", action="store_true")
    ap.add_argument("--plan_queue", default=None, metavar="DIR")
    ap.add_argument("--rung", type=int, default=None, choices=(64, 256, 512, 1024))
    ap.add_argument("--targets", default="all", choices=("pilot", "all"))
    ap.add_argument("--queue", default=None, metavar="DIR")
    ap.add_argument("--wait_card", action="store_true",
                    help="with --queue: before each job, wait for this card's device lease "
                         "to be free, so a driver can be launched behind a running arm")
    ap.add_argument("--card", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=7200,
                    help="fallback; a job's own timeout_s wins (per-job bound)")
    ap.add_argument("--host_threads", type=int, default=8)
    a = ap.parse_args()
    if a.select_pilot:
        select_pilot()
        return
    if a.plan_queue:
        if a.rung is None:
            ap.error("--plan_queue needs --rung")
        plan_queue(a.plan_queue, a.rung, a.targets)
        return
    if a.card is None or not a.queue:
        ap.error("--card plus --queue is required without --plan_queue")
    py = fold_python()
    BASE.mkdir(parents=True, exist_ok=True)
    print(f"driver card {a.card}: queue {a.queue}, python={py}, "
          f"host_threads={a.host_threads}", flush=True)

    def do(job, pos):
        tag = f"{job['model']}/{job['target']}/n{job['rung']}" + (
            f"/c{job['chunk']}" if job.get("chunk") is not None else "")
        if complete(job):
            print(f"[{pos}] {tag} already complete, skip", flush=True)
            return
        log = Path(job["out_dir"]).with_suffix(".log")
        Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
        timeout = job.get("timeout_s") or a.timeout
        attempt = 0
        while True:
            wall, status = run_job(job, a.card, py, timeout, a.host_threads, log)
            rec = record(job, a.card, wall, status)
            print(f"[{pos}] {tag} {status} wall={rec['wall_s']}s "
                  f"n_cifs={rec['n_cifs']}", flush=True)
            if status.startswith("failed") and job.get("mps", MPS) > 1 and oom(log):
                # The task mandates: on an OOM, narrow the chunk, record it, keep going.
                # Floor is mps=1: the largest targets OOM even at mps=2, and tier_a
                # proves they fold when the device holds a single pipeline.
                job["mps"] = max(1, job.get("mps", MPS) // 2)
                print(f"[{pos}] {tag} L1 OOM -> retry at mps {job['mps']}", flush=True)
                continue
            if status == "spawn-deadlock" and attempt >= MAX_DEADLOCK_RETRIES \
                    and job.get("mps", MPS) > 1:
                # Retries at an unchanged width are exhausted and the log cannot tell a real
                # L1 OOM from a true deadlock, so treat it as the width problem it may well
                # be. Same escape as the OOM path, one step, then give up for real.
                job["mps"] = max(1, job.get("mps", MPS) // 2)
                attempt = 0
                print(f"[{pos}] {tag} deadlock retries exhausted -> "
                      f"narrowing to mps {job['mps']}", flush=True)
                continue
            if status != "spawn-deadlock" or attempt >= MAX_DEADLOCK_RETRIES:
                break
            attempt += 1
            print(f"[{pos}] {tag} retry {attempt} after spawn-deadlock", flush=True)

    qdir = Path(a.queue)
    while True:
        pending = [f for f in sorted(qdir.glob("job_*.json"))
                   if not Path(str(f) + ".claim").exists()]
        if not pending:
            print("queue empty", flush=True)
            return
        # Claim only once the card is actually ours: a job's timeout runs from launch, so a
        # fold must never sit parked in tt_bio's flock acquire loop burning its own budget.
        if a.wait_card and _card_held_elsewhere():
            time.sleep(60)
            continue
        for jf in pending:
            if claim(jf):
                do(json.loads(jf.read_text()), jf.stem)
                break
        else:
            time.sleep(30)


if __name__ == "__main__":
    GIT_HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WT,
                              capture_output=True, text=True).stdout.strip()
    main()
