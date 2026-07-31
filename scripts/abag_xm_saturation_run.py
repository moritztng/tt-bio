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
# boltz2: width-capped after the ceil-chunking fix (b62301f5) + p3 L1 budget rule
# (917504 B/sample at 256 tok, tokens^2 scaling, 80 MB chunk budget -> all 16 targets <= 8).
MPS_BOLTZ2 = 8
BASE = Path.home() / "abag_xm" / "saturation"
PROGRESS = BASE / "progress.jsonl"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
MSA_DB = Path.home() / ".boltz" / "msa_db"
RECYCLING_STEPS, SAMPLING_STEPS = 10, 200  # resolved tt-bio defaults (D4a), unset on cmdline
WT = Path(__file__).resolve().parent.parent
MAX_DEADLOCK_RETRIES = 2  # 3 attempts total per job
OOM_SIG = "Out of Memory: Not enough space to allocate"  # tt-metal L1 allocator

# tt_bio model id -> (results-dir prefix per tt_bio/main.py predict_results_dir_name,
# output parent dir per state doc §6, base seed per §3 — disjoint per model and from
# the frontier's 42/1000-1199 — extra predict args, recycling/sampling actually used).
#
# The extra-args field exists because the models genuinely disagree here and a single
# global constant cannot express it: opendde-abag and protenix-v2 deliberately leave
# recycling/sampling UNSET on the command line so tt-bio's resolved defaults (10/200)
# apply, whereas esmfold2's own default is 3/200 and is wrong in both directions, so it
# must pass 10/100 explicitly. esmfold2 also runs single-sequence (no MSA), which is a
# real asymmetry versus the other three and must be stated in the datasheet, not hidden.
MODELS = {"opendde-abag": ("opendde", "opendde", 5000, (), (10, 200)),
          "protenix-v2": ("protenix", "protenix", 7000, (), (10, 200)),
          "boltz2": ("boltz2", "boltz2", 9000, (), (10, 200)),
          "esmfold2": ("esmfold2", "esmfold2", 11000,
                       ("--recycling_steps", "10", "--sampling_steps", "100",
                        "--single_sequence"), (10, 100))}

# Projected 1000-sample wall_s per target, state doc §3 (known 11: 342 + 5*(wall_200
# - 342) from measured frontier walls; new 5: 2.3 s/sample/100 residues).
PROJ = {"9q6y": 21252, "9q6z": 18922, "9m8l": 15202, "9ma0": 14282, "9tmp": 13657,
        "9qrv": 11497, "9ldx": 10502, "9wpm": 8582, "9gei": 8427, "9fte": 6802,
        "9uoi": 5712, "9nl0": 10853, "9l9y": 13498, "9gfr": 12233, "9mnu": 13774,
        "9zen": 5632}
CHUNK = {"9q6y", "9q6z", "9ma0"}  # >~5h projected -> two 500-sample chunks, seeds base/base+1000
CHEAP8 = ["9zen", "9uoi", "9fte", "9gei", "9wpm", "9ldx", "9nl0", "9qrv"]  # §3 fallback set
# The batched-multiplicity path writes every CIF only at fold END, so a wall-clock
# timeout at 99% destroys the whole fold (9ma0 lost 6.35 card-h at 1.6x). Real hangs are
# caught by the no-device/no-jiffies deadlock watchdog in run_job, which fires in ~3 min;
# the wall-clock bound only has to be loose enough to survive host contention (qb1 folds
# ran 1.39-1.56x their projection under 4 concurrent folds + a sibling 14-core labeler).
TIMEOUT_FACTOR = 3.0
QB1_CONTENTION = 1.45  # measured qb1 wall / projection, used only where no wall exists yet
FIXED_S = 342.0  # frontier cost fit (contended, mps 5); used only for chunk split

PANEL_DIR = Path(__file__).resolve().parent.parent / "examples" / "abag_xm"
_RES_CACHE = {}


def residues(target):
    """Total residue count across a target's chains, from its panel YAML.

    Only needed for the 148 panel targets with no measured wall. Parsed with a regex
    rather than a YAML load so this stays dependency-free; verified to parse all 164
    panel files with zero misses (every one is 2 or 3 single-line protein chains)."""
    if target not in _RES_CACHE:
        import re
        txt = (PANEL_DIR / f"{target}.yaml").read_text()
        seqs = re.findall(r"^\s*sequence:\s*([A-Za-z]+)\s*$", txt, re.M)
        if not seqs:
            raise SystemExit(f"{target}: no sequences parsed from its panel YAML")
        _RES_CACHE[target] = sum(len(s) for s in seqs)
    return _RES_CACHE[target]


def projection(target):
    """Projected 1000-sample wall_s. Measured entries in PROJ win; everything else uses
    the panel's own residue count via the rule PROJ = FIXED_S + 23*residues (i.e. the
    2.3 s/sample/100-residue marginal at 1000 samples).

    The rule reproduces PROJ's five rule-derived entries (9nl0 9l9y 9gfr 9mnu 9zen)
    EXACTLY, so it is the same rule, not a re-fit. Against the eleven entries that have
    measured walls it spans 0.74-1.26x (median 0.93) and the miss is structured -- it
    over-predicts small targets and under-predicts the largest -- so it is used only to
    set a loose per-job timeout (TIMEOUT_FACTOR on top) and must NOT be quoted as a cost
    estimate for the panel."""
    if target in PROJ:
        return PROJ[target]
    return round(FIXED_S + 23 * residues(target))


def panel_targets():
    """All 164 AbAg-XM panel targets, from the YAMLs themselves rather than a copied list
    (161 are scorable; the 3 without a scorable Ab-Ag interface still fold and simply
    yield no DockQ -- they must render as blanks, never as zeros)."""
    return sorted(p.stem for p in PANEL_DIR.glob("*.yaml"))


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


def measured_walls():
    """Measured opendde 1000-sample walls from the campaign's own progress records.

    More accurate than the projections (which under-predicted by up to 1.56x), so Q2
    planning and the chunk decision run off real data. A chunked target's single-run
    equivalent is the sum of its chunk walls minus the duplicated fixed trunk passes.
    """
    per = {}
    for f in (PROGRESS, PROGRESS.with_name("progress_qb2.jsonl"),
              PROGRESS.with_name("progress_qb1.jsonl")):
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("model") != "opendde-abag" or r.get("status") != "ok"
                    or r.get("n_samples", 0) < 500):
                continue
            d = per.setdefault(r["target"], {})
            k = r.get("chunk")
            d[k] = max(d.get(k, 0.0), r["wall_s"])
    walls = {}
    for t, d in per.items():
        if None in d:
            walls[t] = d[None]
        elif len(d) >= 2:
            walls[t] = sum(d.values()) - FIXED_S * (len(d) - 1)
    return walls


def jobs_for_model(model, targets, scale=1.0, n_samples=1000):
    prefix, out_parent, base_seed, extra, (recyc, samp) = MODELS[model]
    walls = measured_walls()
    mps = MPS_BOLTZ2 if model == "boltz2" else MPS
    jobs = []
    common = {"model": model, "mps": mps, "extra_args": list(extra),
              "recycling_steps": recyc, "sampling_steps": samp}
    for t in targets:
        # PROJ/measured walls are 1000-sample figures; rescale to the requested depth
        # through the marginal only, since the fixed trunk pass does not scale with N.
        full = walls.get(t, projection(t) * QB1_CONTENTION)
        proj = round((FIXED_S + (full - FIXED_S) * n_samples / 1000.0) * scale)
        # The frozen-16 opendde arm keeps its hand-checked §3 chunk set verbatim so the
        # in-flight campaign's job identities do not shift; every other (model, target)
        # -- including all 148 panel targets -- derives chunking from the >4h rule.
        chunk = (t in CHUNK) if (model == "opendde-abag" and t in PROJ) else proj > 4 * 3600
        if chunk:
            chunk_proj = FIXED_S + (proj - FIXED_S) / 2.0
            for j in (0, 1):
                jobs.append({**common, "target": t, "chunk": j,
                             "seed": base_seed + 1000 * j, "n_samples": n_samples // 2,
                             "proj_s": round(chunk_proj),
                             "timeout_s": round(TIMEOUT_FACTOR * chunk_proj),
                             "out_dir": str(BASE / out_parent / f"{t}_c{j}")})
        else:
            jobs.append({**common, "target": t, "chunk": None,
                         "seed": base_seed, "n_samples": n_samples,
                         "proj_s": proj, "timeout_s": round(TIMEOUT_FACTOR * proj),
                         "out_dir": str(BASE / out_parent / t)})
    return jobs


def plan(out_dir, model, targets, scale=1.0, n_samples=1000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = sorted(jobs_for_model(model, targets, scale, n_samples), key=lambda j: -j["proj_s"])
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


# Measured per-fold capacity ratio qb2/qb1 (9l9y 572 res on qb2 13383 s vs 9m8l 571 res
# on qb1 21120 s): qb1 carries a sibling 14-core labeler, so a fold there costs ~1.58x.
HOST_CAPACITY = {"tt-quietbox": 1.0, "tt-quietbox2": 1.58}


def plan_queue(out_root, models, scales):
    """Split all Q2 jobs into one CLAIM QUEUE per host (the hosts share no filesystem).

    Within a host the four card drivers pull longest-first from the queue, so a card that
    frees early takes the next-biggest job instead of idling on a static assignment.
    Across hosts the split is capacity-weighted LPT, which is the only balancing decision
    that cannot be made at claim time.
    """
    out_root = Path(out_root)
    jobs = []
    for model in models:
        jobs += jobs_for_model(model, sorted(PROJ), scales.get(model, 1.0))
    jobs.sort(key=lambda j: -j["proj_s"])
    hosts = sorted(HOST_CAPACITY)
    load = {h: 0.0 for h in hosts}
    bins = {h: [] for h in hosts}
    for job in jobs:  # LPT on load/capacity
        h = min(hosts, key=lambda x: load[x] / HOST_CAPACITY[x])
        bins[h].append(job)
        load[h] += job["proj_s"]
    for h in hosts:
        qdir = out_root / f"queue_{h}"
        qdir.mkdir(parents=True, exist_ok=True)
        for rank, job in enumerate(bins[h]):
            (qdir / f"job_{rank:03d}_{job['model']}_{job['target']}"
                    f"{'' if job['chunk'] is None else '_c%d' % job['chunk']}.json"
             ).write_text(json.dumps(job, indent=1))
        print(f"{h}: {len(bins[h])} jobs, {load[h]/3600:.1f} projected card-h "
              f"({load[h]/3600/4:.1f} h wall on 4 cards)")


def oom(log):
    try:
        return OOM_SIG in Path(log).read_text()[-200000:]
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
           "chunk": job.get("chunk"), "seed": job["seed"],
           "host": socket.gethostname(), "card": card,
           "wall_s": round(wall_s, 1), "tt_bio_commit": GIT_HEAD,
           "checkpoint_sha256": job.get("ckpt_sha256"),
           # per-model, not the old global constants: esmfold2 runs 10/100 while the
           # other three run the resolved 10/200, so a single pair would have written a
           # provenance block that was simply false for one of the four arms.
           "recycling_steps": job.get("recycling_steps", RECYCLING_STEPS),
           "sampling_steps": job.get("sampling_steps", SAMPLING_STEPS),
           "single_sequence": "--single_sequence" in job.get("extra_args", []),
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
    # -u: without it the fold stdout is block-buffered into the log file and a SIGKILL from
    # the deadlock watchdog discards the buffer, leaving a 0-byte log for precisely the runs
    # that need explaining -- and oom() then reads that empty log and never fires.
    cmd = [py, "-u", "-m", "tt_bio.main", "predict", str(yaml),
           "--model", job["model"], "--out_dir", job["out_dir"],
           "--diffusion_samples", str(job["n_samples"]),
           "--max_parallel_samples", str(job.get("mps", MPS)),
           "--msa_dir", str(MSA_DIR), "--msa_db_path", str(MSA_DB),
           "--seed", str(job["seed"]), "--override", "--write_pae",
           "--host_threads", str(host_threads)] + list(job.get("extra_args", []))
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
                    help="'all' (frozen 16, §3), 'cheap8' (Q2 fallback set, §3), or "
                         "'panel164' (the full AbAg-XM panel read from examples/abag_xm)")
    ap.add_argument("--n_samples", type=int, default=1000,
                    help="samples per (target, model); chunked halves split it. Oracle at "
                         "any m <= n_samples comes free by subsampling, so this sets depth")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply §3 opendde projections (Q2 models, from their canary)")
    ap.add_argument("--plan_queue", default=None, metavar="DIR",
                    help="write the Q2 per-host claim queues into DIR, exit")
    ap.add_argument("--jobs", default=None, metavar="FILE")
    ap.add_argument("--queue", default=None, metavar="DIR",
                    help="pull jobs longest-first from a claim queue until it is empty")
    ap.add_argument("--wait_card", action="store_true",
                    help="with --queue: before each job, wait for this card's device lease "
                         "to be free, so a Q2 driver can be launched behind a running arm")
    ap.add_argument("--card", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=7200,
                    help="fallback; a job's own timeout_s wins (per-job bound, §3)")
    ap.add_argument("--host_threads", type=int, default=8)
    a = ap.parse_args()
    if a.plan:
        targets = ({"all": sorted(PROJ), "cheap8": CHEAP8}.get(a.targets)
                   or (panel_targets() if a.targets == "panel164" else None))
        if targets is None:
            ap.error(f"--targets {a.targets!r}: expected all | cheap8 | panel164")
        plan(a.plan, a.model, targets, a.scale, a.n_samples)
        return
    if a.plan_queue:
        # Q2 canary scales at 9zen: protenix-v2 5701/4261, boltz2 2881/4261.
        plan_queue(a.plan_queue, ["protenix-v2", "boltz2"],
                   {"protenix-v2": 1.34, "boltz2": 0.68})
        return
    if a.card is None or not (a.jobs or a.queue):
        ap.error("--card plus one of --jobs/--queue is required without --plan")
    jobs = json.loads(Path(a.jobs).read_text()) if a.jobs else None
    py = fold_python()
    BASE.mkdir(parents=True, exist_ok=True)
    print(f"driver card {a.card}: {len(jobs) if jobs is not None else 'queue ' + a.queue}"
          f", python={py}, host_threads={a.host_threads}", flush=True)
    def do(job, pos):
        tag = f"{job['model']}/{job['target']}" + (
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
            if status.startswith("failed") and job.get("mps", MPS) > 2 and oom(log):
                # The task mandates: on an OOM, narrow the chunk, record it, keep going.
                job["mps"] = max(2, job.get("mps", MPS) // 2)
                print(f"[{pos}] {tag} L1 OOM -> retry at mps {job['mps']}", flush=True)
                continue
            if status == "spawn-deadlock" and attempt >= MAX_DEADLOCK_RETRIES \
                    and job.get("mps", MPS) > 2:
                # Retries at an unchanged width are exhausted and the log cannot tell a real
                # L1 OOM from a true deadlock, so treat it as the width problem it may well
                # be -- the task mandates narrowing and continuing rather than dropping a
                # target. Same escape as the OOM path, one step, then give up for real.
                job["mps"] = max(2, job.get("mps", MPS) // 2)
                attempt = 0
                print(f"[{pos}] {tag} deadlock retries exhausted -> "
                      f"narrowing to mps {job['mps']}", flush=True)
                continue
            if status != "spawn-deadlock" or attempt >= MAX_DEADLOCK_RETRIES:
                break
            attempt += 1
            print(f"[{pos}] {tag} retry {attempt} after spawn-deadlock", flush=True)

    if jobs is not None:
        for k, job in enumerate(jobs):
            do(job, f"{k+1}/{len(jobs)}")
        return
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
