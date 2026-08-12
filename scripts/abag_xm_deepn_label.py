#!/usr/bin/env python3
"""AbAg-XM deep-N trailing labeler (state doc abag-xm-deepn-saturation-fullpanel).

Lineage: abag_xm_saturation_label.py with the deepn campaign shape. Per-sample DockQ ONLY
(`--per_sample_only`): every deliverable of this campaign is a best-of-N / ranked-top-1
statistic over per-sample DockQ, and the pairwise matrix is quadratic. labels.json lands next
to each fold, the analysis pass reads it.

Loop: scan {opendde,protenix,boltz2,esmfold2}/<target>_n<N>[_c<j>] for complete, unlabeled
folds whose target has a ground-truth CIF, label at most --workers concurrently, sleep,
repeat. A fold is complete when results.json says ok and the structures dir holds exactly the
fold's own `samples` count of CIFs (chunk-transparent). Exits when the CAMPAIGN_DONE marker
exists and a full scan finds nothing left.

  cd $WT && setsid nohup python3 -u scripts/abag_xm_deepn_label.py --workers 2 \
      </dev/null >>~/abag_xm/deepn/logs/label.log 2>&1 &
"""
import argparse, os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WT = Path(__file__).resolve().parent.parent
BASE = Path.home() / "abag_xm" / "deepn"
GT = Path.home() / "abag_xm" / "ground_truth"
DONE_MARKER = BASE / "CAMPAIGN_DONE"
MODEL_DIRS = ("opendde", "protenix", "boltz2", "esmfold2")
LABEL_VENV_PY = Path.home() / ".abag_xm_label_venv" / "bin" / "python3"
SHARED_VENV = Path("/home/ttuser/tt-bio-dev/env")  # read-only: DockQ deps live here
THREAD_CAPS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SLEEP_S = 180
LABEL_TIMEOUT_S = 21600


def label_env(shared_venv=SHARED_VENV):
    sp = next(iter(shared_venv.glob("lib/python*/site-packages")), None)
    pp = ":".join(str(x) for x in [sp, WT] if x)
    env = {**os.environ, "PYTHONPATH": pp,
           "PATH": str(Path.home() / ".local" / "bin") + os.pathsep + os.environ.get("PATH", "")}
    env.update({v: "2" for v in THREAD_CAPS})
    return env


def fold_complete(rd):
    """results.json ok AND exactly the fold's own declared sample count of CIFs."""
    try:
        data = json.loads((rd / "results.json").read_text())
        rec = data[0] if isinstance(data, list) else data
        if rec.get("status") != "ok":
            return False
        n_exp = int(rec.get("samples") or len(rec.get("all_runs") or []))
        if n_exp <= 0:
            return False
        st = rd / "structures"
        return st.is_dir() and len(list(st.glob("*.cif"))) == n_exp
    except Exception:
        return False


def pending_folds(base, models=MODEL_DIRS):
    todo = []
    for model in models:
        mdir = base / model
        if not mdir.is_dir():
            continue
        for out_dir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            if (out_dir / "labels.json").exists() or (out_dir / ".label_lock").exists():
                continue
            rds = list(out_dir.glob("*_results_*"))
            if not rds:
                continue
            target = rds[0].name.split("results_")[1]
            if not (GT / f"{target}.cif").exists():
                continue
            if fold_complete(rds[0]):
                todo.append((out_dir, rds[0]))
    return todo


def label_one(out_dir, rd, venv_py=LABEL_VENV_PY, shared_venv=SHARED_VENV):
    target = rd.name.split("results_")[1]
    lock = out_dir / ".label_lock"
    # Re-check at dequeue, not just at scan. pending_folds() builds the whole todo list in
    # one pass and the pool then drains it over hours, so with a second leg running on
    # another host every dir that leg finishes after our scan is still in our list and would
    # be labelled a second time. The write is atomic and deterministic, so a duplicate costs
    # nothing but CPU -- and CPU is the campaign's critical path.
    if (out_dir / "labels.json").exists() or lock.exists():
        return
    lock.write_text(str(os.getpid()))
    t0 = time.time()
    try:
        cmd = [str(venv_py), str(WT / "scripts" / "abag_xm_labels.py"), str(rd),
               str(GT / f"{target}.cif"), str(WT / "examples" / "abag_xm" / f"{target}.yaml"),
               "--out", str(out_dir / "labels.json"), "--per_sample_only"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=label_env(shared_venv),
                               timeout=LABEL_TIMEOUT_S)
            ok = r.returncode == 0 and (out_dir / "labels.json").exists()
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            ok, tail = False, [f"timeout after {LABEL_TIMEOUT_S}s"]
        print(f"{out_dir.parent.name}/{out_dir.name} "
              f"{'ok' if ok else 'FAILED'} {time.time() - t0:.0f}s {tail[0][:160]}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent single-threaded label processes")
    ap.add_argument("--base", type=Path, default=BASE,
                    help="campaign root to scan (default ~/abag_xm/deepn)")
    ap.add_argument("--models", type=str, default=",".join(MODEL_DIRS),
                    help="comma-separated model dirs to scan (default all four)")
    ap.add_argument("--label-venv-py", type=Path, default=LABEL_VENV_PY,
                    help="python of the label venv (default ~/.abag_xm_label_venv)")
    ap.add_argument("--shared-venv", type=Path, default=SHARED_VENV,
                    help="venv whose site-packages holds DockQ/gemmi (default qb1's)")
    ap.add_argument("--once", action="store_true", help="one scan, then exit")
    a = ap.parse_args()
    models = tuple(m.strip() for m in a.models.split(",") if m.strip())
    print(f"deepn labeler: workers={a.workers} base={a.base} models={models}", flush=True)
    while True:
        todo = pending_folds(a.base, models)
        if todo:
            print(f"scan: {len(todo)} pending -> "
                  + ",".join(f"{o.parent.name}/{o.name}" for o, _ in todo), flush=True)
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(lambda t: label_one(*t, venv_py=a.label_venv_py,
                                                shared_venv=a.shared_venv), todo))
        if a.once or (not todo and (a.base / DONE_MARKER.name).exists()):
            print("labeler done", flush=True)
            return
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()
