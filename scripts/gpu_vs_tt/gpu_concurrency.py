#!/usr/bin/env python3
"""GPU leg of the throughput-at-concurrency comparison: N folds in flight on one H200.

The committed head-to-head measured the H200 one fold at a time and Tenstorrent three
cards at once, then published a throughput-per-dollar claim from the pair. This script
measures what the GPU actually delivers when a customer simply runs more jobs on it:
N independent processes, each with its own CUDA context, folding the same target
back-to-back after a common barrier.

Three sharing modes, all measured the same way (the mode is not this script's business
beyond recording it -- the session driver starts or stops the MPS daemon around it):

* plain     -- N processes, the driver time-slices between contexts. The honest baseline.
* mps       -- same, with the CUDA MPS daemon running, so kernels from different processes
               execute concurrently instead of context-switching.
* mig       -- fixed hardware partitions; feasibility is probed by the session driver.

Each worker folds through ``gpu_bench.build_fold``, the same path the committed latency
numbers came from, so N=1 here must reproduce the committed warm median. That is the
harness's own correctness check and the first thing the session runs.

Usage (launcher):

    python3 gpu_concurrency.py --n 4 --folds 4 --model protenix-v2 \
        --checkpoint /root/ckpt/protenix-v2.pt --mode plain \
        --msa-a3m fixtures/prot117.a3m --seq-file fixtures/prot117.seq \
        --out /root/bench-results/conc_protenix_prot117_plain_n4.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import conc  # noqa: E402
import gpu_bench  # noqa: E402

SMI_FIELDS = "utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------

def worker(args) -> int:
    try:
        return _worker(args)
    except BaseException as exc:
        conc.mark_failed(Path(args.run_dir), args.worker_id, exc)
        raise


def _worker(args) -> int:
    import torch

    run_dir = Path(args.run_dir)
    wid = args.worker_id
    rung = next(r for r in gpu_bench.LADDERS[args.model] if r["name"] == args.rung)
    model_name = {"protenix-v2": "protenix-v2", "opendde": "opendde_v1"}[args.model]

    one_fold, meta, _runner = gpu_bench.build_fold(
        args.model, model_name, rung, str(run_dir / "input.json"),
        run_dir / f"w{wid}" / "dump", args.checkpoint, args.n_msa, samples=args.samples)

    # Cold fold: per-process CUDA context init and first-kernel autotune. Never timed --
    # a throughput number for a served workload is a warm number.
    cold_s, _pred = one_fold()

    released = conc.barrier(run_dir, wid, args.n)
    folds = []
    for _ in range(args.folds):
        t0 = time.monotonic()
        one_fold()
        folds.append([t0, time.monotonic()])

    conc.write_worker_result(run_dir, wid, dict(
        worker=wid, released=released, folds=folds,
        cold_s=round(cold_s, 3), load_s=meta["load_s"],
        n_msa=meta["n_msa"], n_token=meta["n_token"],
        peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3),
        pid=os.getpid(),
    ))
    return 0


# --------------------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------------------

class SmiSampler(threading.Thread):
    """Device telemetry on the same monotonic clock as the fold timestamps.

    Sampled rather than read once at the end because the number that matters is the mean
    over the window in which every worker was folding, not a peak that might have landed
    during a model load.
    """

    def __init__(self, period_s: float = 1.0):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.samples: list[tuple[float, list[float]]] = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={SMI_FIELDS}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                vals = [float(x) for x in out.split(",")]
                self.samples.append((t, vals))
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self):
        self._stop.set()

    def window_stats(self, t0: float, t1: float) -> dict:
        rows = [v for t, v in self.samples if t0 <= t <= t1]
        if not rows:
            return dict(smi_samples_in_window=0)
        cols = list(zip(*rows))
        names = SMI_FIELDS.split(",")
        out = dict(smi_samples_in_window=len(rows))
        for name, col in zip(names, cols):
            key = name.replace(".", "_")
            out[f"{key}_mean"] = round(sum(col) / len(col), 2)
            out[f"{key}_max"] = round(max(col), 2)
        return out


def _compute_apps() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return []


def _mps_state() -> dict:
    """What the driver actually thinks about MPS, recorded rather than assumed."""
    state = dict(pipe_dir=os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
    try:
        p = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n",
                           capture_output=True, text=True, timeout=10)
        state["get_server_list"] = (p.stdout + p.stderr).strip()[:400]
    except FileNotFoundError:
        state["get_server_list"] = "nvidia-cuda-mps-control not on PATH"
    except Exception as exc:
        state["get_server_list"] = f"probe failed: {exc!r}"
    return state


def launcher(args) -> int:
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        for pat in ("ready.*", "done.*.json", "failed.*"):
            for p in run_dir.glob(pat):
                p.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    seq = Path(args.seq_file).read_text().strip()
    a3m = Path(args.msa_a3m).resolve()
    rows = a3m.read_text().split("\n")
    assert rows[1] == seq, f"{a3m} query row does not match {args.seq_file}"
    n_msa = a3m.read_text().count(">")
    gpu_bench.write_input_json(run_dir / "input.json", a3m, seq, args.name)

    sampler = SmiSampler()
    sampler.start()

    env = dict(os.environ)
    t_launch = time.monotonic()
    procs = []
    for i in range(args.n):
        cmd = [sys.executable, str(HERE / "gpu_concurrency.py"),
               "--worker-id", str(i), "--n", str(args.n), "--folds", str(args.folds),
               "--model", args.model, "--rung", args.rung, "--samples", str(args.samples),
               "--run-dir", str(run_dir), "--n-msa", str(n_msa)]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        log = open(run_dir / f"w{i}.log", "wb")
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env=env), log))

    # Snapshot the process list once the barrier has released, as evidence that N
    # contexts really were resident rather than N launched and one surviving.
    apps_snapshot: list[str] = []

    def _snapshot():
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if len(list(run_dir.glob("ready.*"))) >= args.n:
                time.sleep(2.0)
                apps_snapshot.extend(_compute_apps())
                return
            time.sleep(0.5)

    snap = threading.Thread(target=_snapshot, daemon=True)
    snap.start()

    rcs = {}
    for i, p, log in procs:
        try:
            rcs[i] = p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rcs[i] = "timeout"
        log.close()
    sampler.stop()
    wall_s = time.monotonic() - t_launch

    results = conc.load_worker_results(run_dir)
    agg = conc.aggregate(results) if results else dict(clean=False, reason="no results")
    if agg.get("window_start") is not None:
        agg.update(sampler.window_stats(agg["window_start"], agg["window_end"]))

    out = dict(
        side="gpu", mode=args.mode, model=args.model, rung=args.rung,
        target=args.name, label=args.label, n_concurrent=args.n,
        folds_per_worker=args.folds, diffusion_samples=args.samples,
        n_msa=n_msa, seed=gpu_bench.SEED, recycling_steps=gpu_bench.CYCLES,
        sampling_steps=gpu_bench.STEPS,
        worker_rcs=rcs, wall_s=round(wall_s, 2),
        cold_s=[r.get("cold_s") for r in results],
        load_s=[r.get("load_s") for r in results],
        peak_alloc_gib=[r.get("peak_alloc_gib") for r in results],
        compute_apps_at_barrier=apps_snapshot,
        mps=_mps_state() if args.mode == "mps" else None,
        date=time.strftime("%Y-%m-%d"), **agg,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    ok = "CLEAN" if agg.get("clean") else f"DIRTY ({agg.get('reason')})"
    print(f"[{args.mode} N={args.n}] aggregate {agg.get('agg_folds_per_s')} folds/s "
          f"(window est {agg.get('agg_folds_per_s_window')}), "
          f"latency median {agg.get('latency_median_s')}s, "
          f"util {out.get('utilization_gpu_mean')}%, "
          f"power {out.get('power_draw_mean')}W, {ok}", file=sys.stderr, flush=True)
    return 0 if agg.get("clean") else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="protenix-v2", choices=["protenix-v2", "opendde"])
    ap.add_argument("--rung", default="L2-bf16-fusion-cache",
                    help="ladder rung; L2 is the vendor's shipping default and the rung "
                         "the committed head-to-head reported")
    ap.add_argument("--n", type=int, default=1, help="concurrent worker processes")
    ap.add_argument("--folds", type=int, default=4, help="timed folds per worker")
    ap.add_argument("--samples", type=int, default=1,
                    help="diffusion samples per fold; >1 is the in-process batching lever")
    ap.add_argument("--mode", default="plain", choices=["plain", "mps", "mig"])
    ap.add_argument("--msa-a3m", default=str(HERE / "fixtures" / "prot117.a3m"))
    ap.add_argument("--seq-file", default=str(HERE / "fixtures" / "prot117.seq"))
    ap.add_argument("--name", default="prot117")
    ap.add_argument("--label", default="prot.yaml sequence (117 aa)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=None)
    # worker-only
    ap.add_argument("--worker-id", type=int, default=None)
    ap.add_argument("--n-msa", dest="n_msa", type=int, default=None)
    args = ap.parse_args()

    if args.run_dir is None:
        args.run_dir = f"/tmp/gpuconc-{args.model}-{args.name}-{args.mode}-n{args.n}"
    if args.worker_id is not None:
        assert args.n_msa is not None, "worker needs --n-msa from the launcher"
        return worker(args)
    if args.out is None:
        ap.error("--out is required in launcher mode")
    return launcher(args)


if __name__ == "__main__":
    sys.exit(main())
