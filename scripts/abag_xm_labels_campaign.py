#!/usr/bin/env python3
"""Phase 4 campaign orchestrator: run the label driver over every completed
(target, gen) Tier-A fold and write one labels JSON per fold plus a labels.jsonl
index. CPU-only (no device) -- safe to run detached alongside generation.

Idempotent: a fold whose labels JSON already exists with a matching sample count
is skipped, so this can be re-run as Tier-A completes more pairs without redoing
finished work.

    PYTHONPATH=<wt> python3 scripts/abag_xm_labels_campaign.py [--workers N] [--force]

Output layout (persistent, not /tmp):
    ~/abag_xm/tier_a/labels/<model>_<target>.json   (full per-fold label block)
    ~/abag_xm/tier_a/labels/labels.jsonl            (one index line per fold)
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
OUT_BASE = Path.home() / "abag_xm" / "tier_a"
PROGRESS = OUT_BASE / "progress.jsonl"
LABELS_DIR = OUT_BASE / "labels"
LABELS_INDEX = LABELS_DIR / "labels.jsonl"
# Ground-truth reference structures. 143.8 MiB of append-only mmCIF ships as a Release
# asset, not as git blobs, so prefer the host data directory and fall back to the checkout.
_GT_HOST = Path.home() / "abag_xm" / "ground_truth"
GT = _GT_HOST if _GT_HOST.is_dir() else ROOT / "examples" / "ground_truth_structures"
YAML_DIR = ROOT / "examples" / "abag_xm"

RESULT_PREFIX = {"protenix-v2": "protenix", "opendde-abag": "opendde",
                 "opendde": "opendde", "boltz2": "boltz2"}
MODEL_DIR = {"protenix-v2": "protenix_v2", "opendde-abag": "opendde_abag",
             "opendde": "opendde_abag", "boltz2": "boltz2"}

# Label-venv: tmtools/anarci/pyarrow live here (not in the shared venv). When present,
# label sub-scripts run with this python + PYTHONPATH including the shared venv site-
# packages (for DockQ/gemmi/numpy). Falls back to sys.executable otherwise.
LABEL_VENV_PY = Path.home() / ".abag_xm_label_venv" / "bin" / "python3"
SHARED_VENV = Path("/home/ttuser/tt-bio-dev/env")


# Threads per label worker. Every numpy/BLAS pool in a label subprocess otherwise sizes
# itself to ALL cores: measured 61 threads per subprocess, so two workers put ~122 threads
# on a 32-core host that generation had already been carefully capped to fill exactly
# (4 folds x 8). That is the same oversubscription the Phase-1 --host_threads work removed
# from generation, and labelling alongside folds puts it straight back. Capped explicitly;
# raise it with --host_threads once the box is no longer generating.
#
# Full CPU footprint of this campaign, so it can be reasoned about rather than discovered:
# each label worker runs abag_xm_labels.py, which shells out to abag_xm_pairwise_matrix.py,
# which itself opens a pool of --n_workers (default 4) processes for the C(50,2) pairs. So
# the busy-process count is workers * 4 -- 8 at the defaults -- each now holding
# host_threads BLAS threads rather than one per core.
THREAD_CAP_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _user_bin_path():
    """PATH with ~/.local/bin ahead of it.

    CDR RMSD goes through ANARCI, which shells out to `hmmscan`. A non-interactive ssh PATH does
    not include ~/.local/bin, so on a host where HMMER is user-installed rather than in /usr/bin
    the binary is present and still not found -- and the failure surfaces only as a truncated
    multiprocessing traceback buried in a label record. That silently nulled CDR RMSD for every
    fold on qb2 while the other three metrics looked perfectly healthy.
    """
    local_bin = str(Path.home() / ".local" / "bin")
    cur = os.environ.get("PATH", "")
    return local_bin + (os.pathsep + cur if cur else "")


def _label_python_env(host_threads: int = 2):
    """Return (python, env) for invoking abag_xm_labels.py with all deps available."""
    cap = {v: str(max(1, host_threads)) for v in THREAD_CAP_VARS}
    cap["PATH"] = _user_bin_path()
    if LABEL_VENV_PY.exists():
        shared_sp = next(iter(SHARED_VENV.glob("lib/python*/site-packages")), None)
        pp = ":".join(str(x) for x in [shared_sp, ROOT] if x)
        return str(LABEL_VENV_PY), {**os.environ, **cap, "PYTHONPATH": pp}
    return sys.executable, {**os.environ, **cap, "PYTHONPATH": str(ROOT)}


def _preflight_hmmscan():
    """Warn loudly if hmmscan is unreachable: CDR RMSD would be null for the entire run.

    Deliberately a warning and not an abort -- DockQ, epitope Jaccard and interface lDDT do not
    need it, and losing three working metrics to protect one is the wrong trade. The point is
    that it must not be SILENT, which is how it went unnoticed for a whole host.
    """
    if shutil.which("hmmscan", path=_user_bin_path()) is None:
        print("!! hmmscan NOT FOUND on PATH (incl. ~/.local/bin) -- ANARCI cannot run, so "
              "cdr_rmsd will be null for EVERY fold in this run. Install HMMER "
              "(qb1 and qb2 both use 3.3.2) to fix.", flush=True)
        return False
    return True


def done_ok_pairs():
    seen = {}
    if PROGRESS.exists():
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                seen[(r["target"], r["model"])] = r
    return seen


def fold_fingerprint(result_dir: Path) -> str:
    """sha256 over the fold's structure bytes.

    Labels are only reusable for the exact structures they were computed from. The sample
    count is not enough of a key: a regenerated fold has the same 50 CIFs by count, so
    labels computed from the superseded ones would be silently kept -- and this campaign
    regenerates folds by design (the resume pass, and any config correction). Hashing 50
    CIFs costs a fraction of a second against the minutes the labelling itself takes.
    """
    h = hashlib.sha256()
    for f in sorted((result_dir / "structures").glob("*.cif")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def label_one(task):
    target, model, rec = task
    rd = Path(rec["result_dir"])
    native = GT / f"{target}.cif"
    yaml = YAML_DIR / f"{target}.yaml"
    out = LABELS_DIR / f"{MODEL_DIR[model]}_{target}.json"
    fp = fold_fingerprint(rd) if rd.exists() else None
    if out.exists() and not task_force:
        try:
            d = json.loads(out.read_text())
            if d.get("n_samples") == rec.get("n_cifs") and d.get("source_sha256") == fp:
                return {"target": target, "model": model, "status": "skipped",
                        "n_samples": d.get("n_samples")}
        except Exception:
            pass
    if not native.exists() or not yaml.exists() or not rd.exists():
        return {"target": target, "model": model, "status": "missing_inputs",
                "native": native.exists(), "yaml": yaml.exists(),
                "result_dir": rd.exists()}
    py, lenv = _label_python_env(task_host_threads)
    cmd = [py, str(SCRIPTS / "abag_xm_labels.py"),
           str(rd), str(native), str(yaml), "--out", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=lenv)
    wall = time.time() - t0
    if r.returncode != 0:
        return {"target": target, "model": model, "status": "failed",
                "rc": r.returncode, "wall_s": round(wall, 1),
                "stderr": r.stderr.strip()[-800:]}
    try:
        d = json.loads(out.read_text())
    except Exception as e:
        return {"target": target, "model": model, "status": "bad_json",
                "wall_s": round(wall, 1), "error": str(e)}
    # Stamp the structures these labels describe, so a later regeneration of this fold
    # invalidates them instead of being skipped.
    d["source_sha256"] = fp
    out.write_text(json.dumps(d))
    n = d.get("n_samples", 0)
    # sanity: every per-sample record must have a non-None dockq
    bad = [s.get("rank") for s in d.get("samples", [])
           if not (s.get("dockq", {}) or {}).get("dockq")]
    return {"target": target, "model": model, "status": "ok" if not bad else "null_dockq",
            "n_samples": n, "wall_s": round(wall, 1),
            "null_dockq_ranks": bad[:5]}


task_force = False
task_host_threads = 2


def main():
    global task_force, task_host_threads
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel label workers (CPU-bound; 2 is safe alongside folds)")
    ap.add_argument("--force", action="store_true",
                    help="re-label even if a labels JSON already exists")
    ap.add_argument("--host_threads", type=int, default=2,
                    help="thread cap per label worker; 2 is safe alongside generation, "
                         "raise it once the host is no longer folding")
    a = ap.parse_args()
    task_force = a.force
    task_host_threads = a.host_threads
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = done_ok_pairs()
    tasks = [(t, m, rec) for (t, m), rec in pairs.items()]
    tasks.sort()
    print(f"[campaign] {len(tasks)} ok pairs to label (workers={a.workers}, "
          f"host_threads={a.host_threads}, force={a.force})", flush=True)
    _preflight_hmmscan()
    if not tasks:
        print("[campaign] nothing to do; no ok pairs yet", flush=True)
        return
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(label_one, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(f"[label] {res['target']} {res['model']} status={res['status']} "
                  f"n={res.get('n_samples')} wall={res.get('wall_s')}s", flush=True)
    # append-new index: only write ok/failed lines, dedup by (target,model) keeping last
    results.sort(key=lambda r: (r["target"], r["model"]))
    with open(LABELS_INDEX, "a") as fp:
        for r in results:
            if r["status"] in ("ok", "skipped", "null_dockq", "failed"):
                fp.write(json.dumps(r) + "\n")
    n_ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    print(f"[campaign] done: {n_ok}/{len(tasks)} ok, "
          f"{sum(1 for r in results if r['status']=='null_dockq')} null_dockq, "
          f"{sum(1 for r in results if r['status']=='failed')} failed", flush=True)


if __name__ == "__main__":
    main()
