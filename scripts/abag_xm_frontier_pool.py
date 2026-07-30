#!/usr/bin/env python3
"""AbAg-XM frontier Arm-B pool builder (state doc §6 Step 3).

For each target with all 20 per-seed labels.json present, merge into one
200-sample pool: concatenate the 20 samples arrays (annotated with seed_j and
pool_rank), then recompute ONLY the fold-level pieces on the pool
(pairwise_matrix + basin_clust — per-sample labels are context-free and come
from the per-fold runs unmodified). Writes
~/abag_xm/frontier/B_pool/<target>/labels.json.

The pool results dir is symlinks: pool rank k = j*10 + r (j = seed-block,
r = rank within its 10-sample fold), named per the layout convention
(<T>.cif for pool rank 0, <T>_model_<k>.cif for k >= 1).
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/abag-xm-seeds-vs-samples-oracle-frontier-p2")
BASE = Path.home() / "abag_xm" / "frontier"
POOL = BASE / "B_pool"
TARGETS = ["9q6y", "9tmp", "9gei", "9fte", "9wpm", "9qrv",
           "9ma0", "9q6z", "9j4c", "9uoi", "9m8l", "9ldx"]
N_SEEDS, N_PER = 20, 10
PAIR_WORKERS = int(os.environ.get("POOL_PAIR_WORKERS", "8"))
LABEL_VENV_PY = Path.home() / ".abag_xm_label_venv" / "bin" / "python3"
SHARED_VENV = Path("/home/ttuser/tt-bio-dev/env")


def label_env():
    sp = next(iter(SHARED_VENV.glob("lib/python*/site-packages")), None)
    pp = ":".join(str(x) for x in [sp, WT] if x)
    return {**os.environ, "PYTHONPATH": pp}


def fold_labels(target):
    """[(j, labels_dict)] for every seed-block with a labels.json, sorted by j."""
    out = []
    for j in range(N_SEEDS):
        p = BASE / "B" / f"{target}_seed{j}" / "labels.json"
        if p.exists():
            try:
                out.append((j, json.loads(p.read_text())))
            except Exception:
                pass
    return out


def cif_for_rank(rd, target, r):
    st = rd / "structures"
    return st / f"{target}.cif" if r == 0 else st / f"{target}_model_{r}.cif"


def build_pool(target, labeled):
    """Symlink 200 CIFs in pool-rank order. Returns the pool results_dir."""
    rd = POOL / target / f"opendde_results_{target}"
    st = rd / "structures"
    st.mkdir(parents=True, exist_ok=True)
    for j, d in labeled:
        src_rd = Path(d["results_dir"])
        for s in d["samples"]:
            r = s["rank"]
            k = j * N_PER + r
            src = cif_for_rank(src_rd, target, r)
            dst = st / (f"{target}.cif" if k == 0 else f"{target}_model_{k}.cif")
            if not src.exists():
                raise SystemExit(f"pool {target}: missing source CIF {src}")
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
    return rd


def main():
    only = set(sys.argv[1:])
    for target in TARGETS:
        if only and target not in only:
            continue
        out_json = POOL / target / "labels.json"
        if out_json.exists():
            print(f"{target}: pool labels exist, skip", flush=True)
            continue
        labeled = fold_labels(target)
        if len(labeled) != N_SEEDS:
            print(f"{target}: {len(labeled)}/{N_SEEDS} seed-blocks labeled, not ready",
                  flush=True)
            continue
        rd = build_pool(target, labeled)
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            matrix = Path(td) / "matrix.json"
            try:
                r = subprocess.run([str(LABEL_VENV_PY), str(WT / "scripts" / "abag_xm_pairwise_matrix.py"),
                                    str(rd), target, "--n_samples", "200",
                                    "--n_workers", str(PAIR_WORKERS), "--out", str(matrix)],
                                   capture_output=True, text=True, env=label_env(), timeout=21600)
            except subprocess.TimeoutExpired:
                # p13 lesson: one slow target must not kill the loop; 9j4c-class
                # matrices exceed 3 h at low worker counts (hit 10800 s first).
                print(f"{target}: pairwise TIMEOUT, skipped", flush=True)
                continue
            if r.returncode != 0:
                print(f"{target}: pairwise FAILED: {r.stderr.strip()[-300:]}", flush=True)
                continue
            basin = Path(td) / "basin.json"
            r2 = subprocess.run([str(LABEL_VENV_PY), str(WT / "scripts" / "abag_xm_basin_clust.py"),
                                 str(matrix), "--out", str(basin)],
                                capture_output=True, text=True, env=label_env(), timeout=3600)
            if r2.returncode != 0:
                print(f"{target}: basin FAILED: {r2.stderr.strip()[-300:]}", flush=True)
                continue
            pm = json.loads(matrix.read_text())
            bc = json.loads(basin.read_text())
        samples = []
        for j, d in labeled:
            for s in d["samples"]:
                rec = dict(s)
                rec["seed_j"] = j
                rec["pool_rank"] = j * N_PER + s["rank"]
                samples.append(rec)
        samples.sort(key=lambda x: x["pool_rank"])
        out = {"target": target, "results_dir": str(rd), "n_samples": len(samples),
               "pooled_from_seed_folds": [j for j, _ in labeled],
               "samples": samples, "pairwise_matrix": pm, "basin_clust": bc}
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, indent=2))
        print(f"{target}: pool labels written ({len(samples)} samples, "
              f"matrix wall={round(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    main()
