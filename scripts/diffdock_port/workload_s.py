"""WORKLOAD-S on the shipped command: the campaign's gated number (D-dfd-orch-80).

PROTOCOL.md section 2: one receptor, 64 distinct ligands by SMILES, one process, a discarded
warm-up batch of >= 8 ligands first, n >= 3 timed batches, each into a fresh empty out_dir,
`wall_s` from the first ligand's processing to the last ranked SDF on disk. Section 3's one-time
costs -- process start and import, checkpoint load to device, the receptor's ESM-2 embedding --
are measured separately as cold lines and excluded from `wall_s`, on this side exactly as the
B200's 0.24042 complexes/s was taken.

This calls `tt_bio.diffdock.run.dock`, which is the function `tt-bio dock` itself calls, with the
model loaded once as `dock` loads it for a screen. It is the shipped path, not a harness copy of it.
"""
import argparse, json, shutil, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--edge-tp", default="gemm", choices=["gemm", "tile"])
ap.add_argument("--name", default="wls")
args = ap.parse_args()

T0 = time.time()
from tt_bio.diffdock import run as dock_run
from tt_bio.diffdock.job import DockOptions, read_csv_jobs
t_import = time.time() - T0

WL = Path("/home/ttuser/scratch/dfd/wl/workload")
OUT = Path("/home/ttuser/scratch/dfd/perf") / args.name
WARMUP, TIMED, BATCHES = 8, 64, 3

jobs = read_csv_jobs(WL / "screen72.csv")
assert len(jobs) == WARMUP + TIMED, len(jobs)
warm, timed = jobs[:WARMUP], jobs[WARMUP:]
options = DockOptions(device="tt", **({"edge_tp": "tile"} if args.edge_tp == "tile" else {}))

t = time.time()
model = dock_run.load_model(options)
t_load = time.time() - t
print(f"cold: import {t_import:.2f}s, load_model (card open + weights) {t_load:.2f}s", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
t = time.time()
d = OUT / "warm"
shutil.rmtree(d, ignore_errors=True)
dock_run.dock(warm, options, d, isolate=True, model=model)
print(f"warm-up batch ({WARMUP} ligands, discarded): {time.time() - t:.1f}s", flush=True)

RESULT = Path(f"/home/ttuser/scratch/dfd/perf/{args.name}.json")


def summary(rows, partial):
    """The protocol's reading of the batches measured so far."""
    w = sorted(r["complexes_per_s"] for r in rows)
    med = w[len(w) // 2]
    spread = (w[-1] - w[0]) / med
    return {"workload": "WORKLOAD-S", "edge_tp": args.edge_tp, "receptor": "5SAK_ZRY",
            "ligands": TIMED, "batches": rows, "batches_done": len(rows),
            "complete": not partial and len(rows) == BATCHES,
            "every_batch_docked_all_ligands": all(r["docked"] == TIMED for r in rows),
            "median_complexes_per_s": med, "min_complexes_per_s": w[0],
            "max_complexes_per_s": w[-1], "spread_frac_of_median": round(spread, 4),
            "spread_ok": spread <= 0.10,
            "cold_lines": {"import_s": round(t_import, 3), "load_model_s": round(t_load, 3)},
            "b200_complexes_per_s": 0.24042, "ratio_vs_b200": round(med / 0.24042, 4)}


def flush(rows, partial=True):
    """Write the json after every batch. A 3.5 h window that dies in its last batch must not
    also lose the batches that already ran (the lesson `device_loop_grade.py` learned at
    ed3e94d8f: a leg that raises discarded every measurement before it)."""
    RESULT.write_text(json.dumps(summary(rows, partial), indent=1))


rows = []
for b in range(BATCHES):
    d = OUT / f"batch{b}"
    shutil.rmtree(d, ignore_errors=True)
    t = time.time()
    res = dock_run.dock(timed, options, d, isolate=True, model=model)
    wall = time.time() - t
    n_ok = len(res["docked"])
    sdfs = len(list(d.glob("*/rank1.sdf")))
    cps = n_ok / wall
    rows.append({"batch": b, "wall_s": round(wall, 3), "docked": n_ok,
                 "rank1_sdfs_on_disk": sdfs, "failed": len(res["failed"]),
                 "complexes_per_s": round(cps, 5),
                 "s_per_complex": round(wall / max(n_ok, 1), 3)})
    print(json.dumps(rows[-1]), flush=True)
    flush(rows)

out = summary(rows, partial=False)
flush(rows, partial=False)
print(json.dumps(out, indent=1), flush=True)
