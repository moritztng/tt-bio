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
        args.model, model_name, rung,
        args.input_json or str(run_dir / "input.json"),
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

def _cpu_jiffies() -> tuple[float, float] | None:
    """(idle_including_iowait, total) from /proc/stat, cumulative since boot."""
    try:
        with open("/proc/stat") as fh:
            parts = [float(x) for x in fh.readline().split()[1:]]
    except Exception:
        return None
    if len(parts) < 5:
        return None
    return parts[3] + parts[4], sum(parts)


class SmiSampler(threading.Thread):
    """Device *and host* telemetry on the same monotonic clock as the fold timestamps.

    Sampled rather than read once at the end because the number that matters is the mean
    over the window in which every worker was folding, not a peak that might have landed
    during a model load.

    Host CPU is sampled alongside the GPU because at 117 aa this workload is host-bound
    (N=1 measured 33% GPU utilisation), so a concurrency curve that flattens is ambiguous
    on its own: the H200 may have saturated, or the rental's vCPUs may have. Only the pair
    of numbers distinguishes those, and the whole point of this benchmark is not to quote
    a device at less than its best.
    """

    def __init__(self, period_s: float = 1.0):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.samples: list[tuple[float, list[float]]] = []
        self.cpu_samples: list[tuple[float, tuple[float, float]]] = []
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
            jif = _cpu_jiffies()
            if jif is not None:
                self.cpu_samples.append((t, jif))
            self._stop.wait(self.period_s)

    def stop(self):
        self._stop.set()

    def window_stats(self, t0: float, t1: float) -> dict:
        out = {}
        # Host CPU: a difference of cumulative counters across the window, so it needs the
        # first and last sample inside it rather than a mean of instantaneous readings.
        cpu = [(t, v) for t, v in self.cpu_samples if t0 <= t <= t1]
        if len(cpu) >= 2:
            (_, (idle0, tot0)), (_, (idle1, tot1)) = cpu[0], cpu[-1]
            d_tot, d_idle = tot1 - tot0, idle1 - idle0
            if d_tot > 0:
                out["host_cpu_busy_pct"] = round(100.0 * (1.0 - d_idle / d_tot), 2)
                out["host_cpu_busy_cores"] = round(
                    (os.cpu_count() or 0) * (1.0 - d_idle / d_tot), 2)
        rows = [v for t, v in self.samples if t0 <= t <= t1]
        if not rows:
            out["smi_samples_in_window"] = 0
            return out
        cols = list(zip(*rows))
        names = SMI_FIELDS.split(",")
        out["smi_samples_in_window"] = len(rows)
        for name, col in zip(names, cols):
            key = name.replace(".", "_")
            out[f"{key}_mean"] = round(sum(col) / len(col), 2)
            out[f"{key}_max"] = round(max(col), 2)
        return out


def _effective_cpus() -> int:
    """Cores this container may actually use: the cgroup CPU quota if one is set,
    else the visible count. cgroup v2 cpu.max holds '<quota> <period>' ('max' if
    uncapped); v1 splits them across cpu.cfs_quota_us / cpu.cfs_period_us."""
    try:
        p = Path("/sys/fs/cgroup/cpu.max")
        if p.exists():
            q, per = p.read_text().split()[:2]
            if q != "max":
                return max(1, int(int(q) / int(per)))
    except Exception:
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, q // per)
    except Exception:
        pass
    return os.cpu_count() or 1


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

    # Distinct-targets control: worker i folds target i (its own seq + a3m + input
    # JSON), so an N-way point measures N different proteins, not N copies of one.
    # Direct answer to "did the MPS speedup only appear because it was always the
    # same prediction".
    # protenix prepends the query to the alignment and does not dedupe it against the
    # a3m's own rows, so the depth it actually consumes is unique(rows) + 1, not the
    # header count. prot117.a3m happens to carry a duplicate row (35 rows, 34 unique)
    # so the two agree there at 35; a hand-built a3m with 35 distinct rows lands on 36
    # and trips gpu_bench's fairness assertion. Compute the real expectation.
    def _msa_depth(a3m: Path) -> int:
        rows = a3m.read_text().split("\n")
        return len({rows[i + 1] for i, l in enumerate(rows) if l.startswith(">")}) + 1

    target_of_worker = None
    if args.targets:
        names = [t.strip() for t in args.targets.split(",") if t.strip()]
        assert len(names) == args.n, \
            f"distinct mode wants exactly one target per worker: {len(names)} vs N={args.n}"
        distinct_dir = HERE / "fixtures" / "distinct"
        per_worker = []
        for i, nm in enumerate(names):
            seq_i = (distinct_dir / f"{nm}.seq").read_text().strip()
            a3m_i = (distinct_dir / f"{nm}.a3m").resolve()
            rows = a3m_i.read_text().split("\n")
            assert rows[1] == seq_i, f"{a3m_i} query row does not match its .seq"
            per_worker.append((nm, seq_i, a3m_i, _msa_depth(a3m_i)))
        n_msa = per_worker[0][3]
        assert all(p[3] == n_msa for p in per_worker), "all targets need the same n_msa"
        # The control is only a control if it folds at the same alignment depth as the
        # same-target point it is compared against.
        assert n_msa == _msa_depth(Path(args.msa_a3m).resolve()), \
            f"distinct targets consume {n_msa} MSA rows, prot117 consumes " \
            f"{_msa_depth(Path(args.msa_a3m).resolve())}"
        target_of_worker = {}
        for i, (nm, seq_i, a3m_i, _n) in enumerate(per_worker):
            gpu_bench.write_input_json(run_dir / f"input_w{i}.json", a3m_i, seq_i, nm)
            target_of_worker[i] = nm
    else:
        seq = Path(args.seq_file).read_text().strip()
        a3m = Path(args.msa_a3m).resolve()
        rows = a3m.read_text().split("\n")
        assert rows[1] == seq, f"{a3m} query row does not match {args.seq_file}"
        n_msa = _msa_depth(a3m)
        gpu_bench.write_input_json(run_dir / "input.json", a3m, seq, args.name)

    sampler = SmiSampler()
    sampler.start()

    # Give each concurrent worker an equal share of the host's cores. Left to itself torch
    # sizes its thread pools from the full core count, so N workers ask for N x cores
    # threads and spend the difference in the scheduler -- at N=8 on a 24-vCPU rental that
    # is a host artefact masquerading as a GPU throughput ceiling. At N=1 the share is the
    # whole machine, so the control point stays comparable to the unpinned runs already
    # committed. Floor of 2 so a large N cannot starve a worker to a single thread.
    # The share must come from the EFFECTIVE core count: a vast.ai container sees every
    # host core (192 on machine 51172) while the cgroup quota caps it at ~23, and sizing
    # pools from the visible count oversubscribes the quota 8x (measured 2026-08-08:
    # omp=192 at N=1 depressed the control fold 7.22s vs 6.07-6.15s on uncapped hosts).
    cores = _effective_cpus()
    omp = args.omp_threads if args.omp_threads > 0 else max(2, cores // args.n)
    env = dict(os.environ, OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp))
    t_launch = time.monotonic()
    procs = []
    for i in range(args.n):
        cmd = [sys.executable, str(HERE / "gpu_concurrency.py"),
               "--worker-id", str(i), "--n", str(args.n), "--folds", str(args.folds),
               "--model", args.model, "--rung", args.rung, "--samples", str(args.samples),
               "--run-dir", str(run_dir), "--n-msa", str(n_msa)]
        if target_of_worker is not None:
            cmd += ["--input-json", str(run_dir / f"input_w{i}.json")]
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
        target_of_worker=target_of_worker,
        folds_per_worker=args.folds, diffusion_samples=args.samples,
        n_msa=n_msa, seed=gpu_bench.SEED, recycling_steps=gpu_bench.CYCLES,
        sampling_steps=gpu_bench.STEPS,
        worker_rcs=rcs, wall_s=round(wall_s, 2),
        omp_num_threads=omp, host_cpu_cores=os.cpu_count() or 0,
        container_cpu_cores=cores,
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
          f"power {out.get('power_draw_mean')}W, "
          f"host cpu {out.get('host_cpu_busy_pct')}% of {cores} "
          f"(omp {omp}), {ok}", file=sys.stderr, flush=True)
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
    ap.add_argument("--omp-threads", type=int, default=0,
                    help="threads per worker; 0 = an equal share of the host's cores")
    ap.add_argument("--msa-a3m", default=str(HERE / "fixtures" / "prot117.a3m"))
    ap.add_argument("--seq-file", default=str(HERE / "fixtures" / "prot117.seq"))
    ap.add_argument("--targets", default=None,
                    help="comma-separated names under fixtures/distinct/ for the "
                         "distinct-targets control: worker i folds target i, so N must "
                         "equal the number of names")
    ap.add_argument("--name", default="prot117")
    ap.add_argument("--label", default="prot.yaml sequence (117 aa)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default=None)
    # worker-only
    ap.add_argument("--worker-id", type=int, default=None)
    ap.add_argument("--n-msa", dest="n_msa", type=int, default=None)
    ap.add_argument("--input-json", default=None)
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
