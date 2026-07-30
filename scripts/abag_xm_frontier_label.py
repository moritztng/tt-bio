#!/usr/bin/env python3
"""AbAg-XM frontier trailing labeler (state doc §6 Step 3). Runs on qb1.

Loop: (1) rsync qb2's frontier fold tree into qb1's (disjoint dirs; qb2's
progress.jsonl comes over as progress_qb2.jsonl), (2) label every complete,
unlabeled fold with the unmodified p4 pipeline (abag_xm_labels.py), at most
NWORK concurrent processes, (3) build any Arm-B 200-sample pool whose 20
per-seed labels all exist (abag_xm_frontier_pool.py, after the folds), (4)
sleep. Exits when the CAMPAIGN_DONE marker exists and a full scan finds
nothing left to label.

labels.json lands in the fold's out_dir (~/abag_xm/frontier/<arm>/<T>[_seed<j>]/).
Label env mirrors abag_xm_labels_campaign.py: label venv python + shared
site-packages on PYTHONPATH, thread caps, ~/.local/bin on PATH for hmmscan.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/abag-xm-seeds-vs-samples-oracle-frontier-p2")
BASE = Path.home() / "abag_xm" / "frontier"
GT = Path.home() / "abag_xm" / "ground_truth"
DONE_MARKER = BASE / "CAMPAIGN_DONE"
NWORK = 4
PAIR_WORKERS = 4
SLEEP_S = 120
LABEL_VENV_PY = Path.home() / ".abag_xm_label_venv" / "bin" / "python3"
SHARED_VENV = Path("/home/ttuser/tt-bio-dev/env")
THREAD_CAPS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def label_env():
    sp = next(iter(SHARED_VENV.glob("lib/python*/site-packages")), None)
    pp = ":".join(str(x) for x in [sp, WT] if x)
    env = {**os.environ, "PYTHONPATH": pp,
           "PATH": str(Path.home() / ".local" / "bin") + os.pathsep + os.environ.get("PATH", "")}
    env.update({v: "2" for v in THREAD_CAPS})
    return env


def sync_qb2():
    """Best-effort pull of qb2's fold tree + progress. Tolerates qb2 being down."""
    try:
        for arm in ("A", "B"):
            subprocess.run(["rsync", "-a", "--timeout=60",
                            f"tt-quietbox2:abag_xm/frontier/{arm}/", str(BASE / arm) + "/"],
                           capture_output=True, timeout=300)
        r = subprocess.run(["scp", "-q", "tt-quietbox2:abag_xm/frontier/progress.jsonl",
                            str(BASE / "progress_qb2.jsonl")],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception as e:
        print(f"qb2 sync failed (tolerated): {e}", flush=True)
        return False


def expected_cifs(out_dir):
    return 200 if out_dir.parent.name == "A" else 10


def fold_complete(rd, n_exp):
    rj = rd / "results.json"
    try:
        data = json.loads(rj.read_text())
        rec = data[0] if isinstance(data, list) else data
        if rec.get("status") != "ok":
            return False
        st = rd / "structures"
        return st.is_dir() and len(list(st.glob("*.cif"))) == n_exp
    except Exception:
        return False


def pending_folds():
    todo = []
    for arm in ("A", "B"):
        arm_dir = BASE / arm
        if not arm_dir.is_dir():
            continue
        for out_dir in sorted(arm_dir.iterdir()):
            if not out_dir.is_dir():
                continue
            rds = list(out_dir.glob("opendde_results_*"))
            if not rds:
                continue
            rd = rds[0]
            if (out_dir / "labels.json").exists() or (out_dir / ".label_lock").exists():
                continue
            if fold_complete(rd, expected_cifs(out_dir)):
                todo.append((out_dir, rd))
    return todo


def label_one(out_dir, rd):
    target = rd.name.split("results_")[1]
    lock = out_dir / ".label_lock"
    lock.write_text(str(os.getpid()))
    t0 = time.time()
    try:
        cmd = [str(LABEL_VENV_PY), str(WT / "scripts" / "abag_xm_labels.py"), str(rd),
               str(GT / f"{target}.cif"), str(WT / "examples" / "abag_xm" / f"{target}.yaml"),
               "--out", str(out_dir / "labels.json"), "--pair_workers", str(PAIR_WORKERS)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=label_env(),
                               timeout=21600)
            ok = r.returncode == 0 and (out_dir / "labels.json").exists()
            if not ok:
                print(f"  stderr tail: {r.stderr.strip()[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            ok = False
        print(f"label {out_dir.parent.name}/{out_dir.name} {'ok' if ok else 'FAILED'} "
              f"wall={round(time.time()-t0)}s", flush=True)
    finally:
        lock.unlink(missing_ok=True)


POOL = BASE / "B_pool"
POOL_SCRIPT = WT / "scripts" / "abag_xm_frontier_pool.py"
N_SEEDS = 20


def pending_pools():
    """Targets whose 20 per-seed B labels all exist but the 200-sample pool is
    not built/locked yet. Runs LAST in each scan (folds first)."""
    bdir = BASE / "B"
    if not bdir.is_dir():
        return []
    targets = sorted({d.name.split("_seed")[0] for d in bdir.iterdir()
                      if d.is_dir() and "_seed" in d.name})
    out = []
    for t in targets:
        if (POOL / t / "labels.json").exists() or (POOL / t / ".pool_lock").exists():
            continue
        if all((bdir / f"{t}_seed{j}" / "labels.json").exists() for j in range(N_SEEDS)):
            out.append(t)
    return out


def pool_one(target):
    lock = POOL / target / ".pool_lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    t0 = time.time()
    try:
        env = label_env()
        env["POOL_PAIR_WORKERS"] = str(PAIR_WORKERS)
        try:
            r = subprocess.run([sys.executable, str(POOL_SCRIPT), target],
                               capture_output=True, text=True, env=env, timeout=25200)
            ok = r.returncode == 0 and (POOL / target / "labels.json").exists()
            if not ok:
                print(f"  pool stderr tail: {r.stderr.strip()[-300:]}", flush=True)
        except subprocess.TimeoutExpired:
            ok = False
        print(f"pool {target} {'ok' if ok else 'FAILED'} wall={round(time.time()-t0)}s",
              flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    print(f"labeler start: base={BASE} workers={NWORK} pair_workers={PAIR_WORKERS}", flush=True)
    while True:
        qb2_up = sync_qb2()
        todo = pending_folds()
        pools = pending_pools()
        if todo or pools:
            print(f"scan: {len(todo)} folds + {len(pools)} pools to label (qb2_up={qb2_up})",
                  flush=True)
            with ThreadPoolExecutor(max_workers=NWORK) as ex:
                list(ex.map(lambda t: label_one(*t), todo))
                list(ex.map(pool_one, pools))
        elif DONE_MARKER.exists():
            print("campaign done + nothing to label: labeler exits", flush=True)
            return
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()
