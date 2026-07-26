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
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
OUT_BASE = Path.home() / "abag_xm" / "tier_a"
PROGRESS = OUT_BASE / "progress.jsonl"
LABELS_DIR = OUT_BASE / "labels"
LABELS_INDEX = LABELS_DIR / "labels.jsonl"
GT = ROOT / "examples" / "ground_truth_structures"
YAML_DIR = ROOT / "examples" / "abag_xm"

RESULT_PREFIX = {"protenix-v2": "protenix", "opendde-abag": "opendde",
                 "opendde": "opendde", "boltz2": "boltz2"}
MODEL_DIR = {"protenix-v2": "protenix_v2", "opendde-abag": "opendde_abag",
             "opendde": "opendde_abag", "boltz2": "boltz2"}


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


def label_one(task):
    target, model, rec = task
    rd = Path(rec["result_dir"])
    native = GT / f"{target}.cif"
    yaml = YAML_DIR / f"{target}.yaml"
    out = LABELS_DIR / f"{MODEL_DIR[model]}_{target}.json"
    if out.exists() and not task_force:
        try:
            d = json.loads(out.read_text())
            if d.get("n_samples") == rec.get("n_cifs"):
                return {"target": target, "model": model, "status": "skipped",
                        "n_samples": d.get("n_samples")}
        except Exception:
            pass
    if not native.exists() or not yaml.exists() or not rd.exists():
        return {"target": target, "model": model, "status": "missing_inputs",
                "native": native.exists(), "yaml": yaml.exists(),
                "result_dir": rd.exists()}
    cmd = [sys.executable, str(SCRIPTS / "abag_xm_labels.py"),
           str(rd), str(native), str(yaml), "--out", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
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
    n = d.get("n_samples", 0)
    # sanity: every per-sample record must have a non-None dockq
    bad = [s.get("rank") for s in d.get("samples", [])
           if not (s.get("dockq", {}) or {}).get("dockq")]
    return {"target": target, "model": model, "status": "ok" if not bad else "null_dockq",
            "n_samples": n, "wall_s": round(wall, 1),
            "null_dockq_ranks": bad[:5]}


task_force = False


def main():
    global task_force
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel label workers (CPU-bound; 2 is safe alongside folds)")
    ap.add_argument("--force", action="store_true",
                    help="re-label even if a labels JSON already exists")
    a = ap.parse_args()
    task_force = a.force
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = done_ok_pairs()
    tasks = [(t, m, rec) for (t, m), rec in pairs.items()]
    tasks.sort()
    print(f"[campaign] {len(tasks)} ok pairs to label (workers={a.workers}, "
          f"force={a.force})", flush=True)
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
