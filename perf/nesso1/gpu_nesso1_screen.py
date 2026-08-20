"""The throughput leg: predictions/hour for one target against many compounds.

    python perf/nesso1/gpu_nesso1_screen.py --concurrency 1,2,4,8

Why this leg exists and why it is the headline rather than s/prediction. Nesso-1's job is virtual
screening: one target, millions of compounds. So the number that decides whether a Tenstorrent port
is worth building is predictions/hour at the concurrency the tool actually admits, not the latency
of one prediction.

And the tool admits exactly one batch size. `NessoInferenceDataModule.predict_dataloader` hardcodes
`batch_size=1`, its own docstring says "always batch_size=1", and the model cannot be handed a
larger batch without changing it: `_select_pocket_indices` reads `feats["mol_type"][0]`,
`pocket_crop` writes `pdistogram_full[0, ...]`, and the affinity pocket selection takes
`get_pocket_mask(...)[0]` -- one crop index set, applied to the whole batch. A B>1 forward would
silently crop every member of the batch to the FIRST member's pocket. There is therefore no batch
curve to sweep inside a process; the only throughput lever the shipped tool offers is process
concurrency (or `--devices N`, which is data-parallel across GPUs, not batching).

So the curve measured here is over concurrent processes on ONE card, plus the amortisation curve
inside a single process (N records in one `trainer.predict` call, which is what `nesso predict
<dir>` does). Both are reported: amortisation is free, concurrency costs memory.

Aggregate throughput is computed over the UNION of the workers' warm windows, not from the mean of
their durations -- the workers do not start their warm rep at the same instant, and averaging
durations would report a throughput the card never actually delivered.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import threading
import time

HERE = pathlib.Path(__file__).parent
RUN = HERE / "gpu_nesso1_run.py"


def sample_gpu(stop: threading.Event, out: list) -> None:
    q = "power.draw,utilization.gpu,memory.used,clocks.sm"
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout
            vals = [x.strip() for x in r.strip().splitlines()[0].split(",")]
            out.append({k: v for k, v in zip(q.split(","), vals)})
        except Exception:                                     # noqa: BLE001
            pass
        stop.wait(2.0)


def shard(src: pathlib.Path, root: pathlib.Path, c: int) -> list[pathlib.Path]:
    yamls = sorted(p for p in src.iterdir() if p.suffix == ".yaml")
    if root.exists():
        shutil.rmtree(root)
    dirs = []
    for i in range(c):
        d = root / ("w%d" % i)
        d.mkdir(parents=True)
        dirs.append(d)
    for i, y in enumerate(yamls):
        shutil.copy(y, dirs[i % c] / y.name)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(HERE / "inputs" / "screen"))
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=2, help="rep 0 cold and discarded, rep 1+ warm")
    ap.add_argument("--python", default="/work/v_nesso/bin/python")
    ap.add_argument("--results", default="/work/results")
    ap.add_argument("--out-root", default="/work/out/screen")
    ap.add_argument("--shard-root", default="/work/shards")
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    src = pathlib.Path(args.inputs)
    n_total = len(sorted(p for p in src.iterdir() if p.suffix == ".yaml"))
    results = pathlib.Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    summary = {"inputs": str(src), "n_records": n_total, "reps": args.reps, "cells": []}

    for c in [int(x) for x in args.concurrency.split(",")]:
        if n_total % c:
            print("skip concurrency %d: %d records does not divide evenly" % (c, n_total))
            continue
        dirs = shard(src, pathlib.Path(args.shard_root) / ("c%d" % c), c)
        reports = [results / ("screen_c%d_w%d.json" % (c, i)) for i in range(c)]
        stop = threading.Event()
        samples: list = []
        th = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
        th.start()
        t0 = time.time()
        procs = []
        for i, d in enumerate(dirs):
            cmd = [args.python, str(RUN), "--inputs", str(d),
                   "--out-dir", "%s/c%d_w%d" % (args.out_root, c, i),
                   "--report", str(reports[i]), "--reps", str(args.reps),
                   "--label", "screen_c%d_w%d" % (c, i)]
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, text=True))
        errs = [p.communicate(timeout=args.timeout)[1] for p in procs]
        wall = time.time() - t0
        stop.set()
        th.join(timeout=5)

        cell = {"concurrency": c, "records_per_worker": n_total // c, "n_records": n_total,
                "e2e_wall_s": round(wall, 3), "workers": [], "ok": True, "why": ""}
        starts, ends, warm_records = [], [], 0
        for i, rp in enumerate(reports):
            if not rp.exists():
                cell["ok"] = False
                cell["why"] += "worker %d wrote no report; stderr: %s | " % (i, errs[i][-400:])
                continue
            d = json.loads(rp.read_text())
            w = {"ok": d.get("ok"), "why": (d.get("why") or "")[:160],
                 "rep_s": d.get("rep_s"), "n": d.get("n_records"),
                 "preprocess_s": d.get("preprocess_s"), "esm_s": d.get("esm_s"),
                 "model_load_s": d.get("model_load_s"),
                 "peak_vram_alloc_B": d.get("peak_vram_alloc_B"),
                 "forward_s": (d.get("phases", {}).get("1", {}) or {}).get("forward"),
                 "predict_step_s": (d.get("phases", {}).get("1", {}) or {}).get("predict_step")}
            cell["workers"].append(w)
            if not d.get("ok"):
                # A worker's own exclusivity check fires whenever another compute app is on the
                # card. With C>1 that is US, by construction, so a failed exclusivity check is
                # expected here and is not evidence of a co-tenant; every other output guard
                # (all records written, finite affinity, ligand placed) still has to pass.
                if c == 1 or "NOT EXCLUSIVE" not in (d.get("why") or ""):
                    cell["ok"] = False
                    cell["why"] += "worker %d: %s | " % (i, (d.get("why") or "")[:200])
            win = (d.get("rep_windows") or [])
            if len(win) >= 2:
                starts.append(win[1][0])
                ends.append(win[-1][1])
                warm_records += (d.get("n_records") or 0) * (len(win) - 1)
        if starts and ends:
            union = max(ends) - min(starts)
            cell["warm_union_s"] = round(union, 3)
            cell["warm_records"] = warm_records
            cell["steady_pred_per_hour"] = round(warm_records / union * 3600.0, 1)
            cell["steady_s_per_pred"] = round(union / warm_records, 5)
        cell["e2e_pred_per_hour"] = round(n_total / wall * 3600.0, 1)
        cell["e2e_s_per_pred"] = round(wall / n_total, 5)
        if samples:
            pw = [float(s["power.draw"]) for s in samples if s.get("power.draw")]
            ut = [float(s["utilization.gpu"]) for s in samples if s.get("utilization.gpu")]
            mem = [float(s["memory.used"]) for s in samples if s.get("memory.used")]
            cell["gpu"] = {"n_samples": len(samples),
                           "power_W_mean": round(sum(pw) / len(pw), 1) if pw else None,
                           "power_W_max": max(pw) if pw else None,
                           "util_pct_mean": round(sum(ut) / len(ut), 1) if ut else None,
                           "mem_MiB_max": max(mem) if mem else None}
        summary["cells"].append(cell)
        print("c=%-2d  e2e %8.1f pred/h (%.4f s/pred)  steady %8s pred/h  power %s W  ok=%s %s"
              % (c, cell["e2e_pred_per_hour"], cell["e2e_s_per_pred"],
                 cell.get("steady_pred_per_hour"), (cell.get("gpu") or {}).get("power_W_mean"),
                 cell["ok"], cell["why"][:200]), flush=True)
        (results / "screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
