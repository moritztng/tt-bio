#!/usr/bin/env python3
"""Tenstorrent leg of the throughput-at-concurrency comparison.

Same barrier, same estimator, same aggregate as the GPU leg (``conc.py``) -- that is the
whole point. The committed 0.211 folds/s was three qb1 cards with one process each, card 0
held back for another job, compared against a serial H200. This re-measures the TT side at
its own optimum: 1, 2, 3 and 4 concurrent cards, so the box number is a measurement rather
than 4x the single-card number.

Per-card concurrency above one process is not a thing on this stack and is not silently
skipped: ``tt_bio.tenstorrent.get_device`` takes an exclusive flock on the physical card
before opening it (``tt_bio/device_lease.py``), so a second process on the same chip blocks
and then fails with ``DeviceInUseError``. Point ``--cards 3,3`` at one card with a short
``--lease-timeout`` and this script records that refusal verbatim as the evidence.

Usage:

    TT_BIO_LEASE_TIMEOUT=60 python3 scripts/gpu_vs_tt/tt_concurrency.py \
        --cards 0,1,2,3 --folds 5 --model protenix-v2 \
        --out results/conc/tt_protenix_prot117_c4.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import conc  # noqa: E402
import tt_baseline  # noqa: E402


# --------------------------------------------------------------------------------------
# worker: one process, one card (TT_VISIBLE_DEVICES already set by the launcher)
# --------------------------------------------------------------------------------------

def worker(args) -> int:
    try:
        return _worker(args)
    except BaseException as exc:
        conc.mark_failed(Path(args.run_dir), args.worker_id, exc)
        raise


def _worker(args) -> int:
    run_dir = Path(args.run_dir)
    wid = args.worker_id
    msa_dir = run_dir / f"w{wid}" / "msa"

    one_fold, meta, state = tt_baseline.build_fold(
        args.model, msa_dir, Path(args.target), Path(args.msa_a3m),
        samples=args.samples, hoist=args.hoist, instrument=args.instrument)

    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"

    released = conc.barrier(run_dir, wid, args.n)
    folds = []
    for _ in range(args.folds):
        t0 = time.monotonic()
        one_fold()
        folds.append([t0, time.monotonic()])

    conc.write_worker_result(run_dir, wid, dict(
        worker=wid, card=os.environ.get("TT_VISIBLE_DEVICES"), released=released,
        folds=folds, cold_s=round(cold_s, 3), load_s=meta["load_s"],
        n_msa=meta["n_msa"], plddt=cold_metrics.get("plddt"),
        n_tokens=cold_metrics.get("n_tokens"),
        n_residues=cold_metrics.get("n_residues"),
        card_info={k: meta[k] for k in ("card_type", "aiclk_mhz") if k in meta},
        phase_times=meta.get("phase_times"),
        pid=os.getpid(),
    ))
    state.reset()
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


# --------------------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------------------

class TtSmiSampler(threading.Thread):
    """Per-card telemetry on the fold timestamps' monotonic clock.

    Measured draw, not the 300 W TBP ceiling: the committed W/fold table used TDP on both
    sides, and the GPU leg gets a real reading from nvidia-smi, so the TT side has to get
    a real reading too or the energy comparison tilts by construction.
    """

    def __init__(self, tt_smi: str | None, cards: list[int], period_s: float = 5.0):
        super().__init__(daemon=True)
        self.tt_smi, self.cards, self.period_s = tt_smi, cards, period_s
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()

    def run(self):
        if not self.tt_smi:
            return
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                out = subprocess.run([self.tt_smi, "-s"], capture_output=True,
                                     text=True, timeout=30).stdout
                devs = json.loads(out).get("device_info", [])
                row = {}
                for c in self.cards:
                    if c < len(devs):
                        row[c] = devs[c].get("telemetry", {}) or {}
                self.samples.append((t, row))
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self):
        self._stop.set()

    def window_stats(self, t0: float, t1: float) -> dict:
        rows = [r for t, r in self.samples if t0 <= t <= t1]
        if not rows:
            return dict(tt_smi_samples_in_window=0)
        out: dict = dict(tt_smi_samples_in_window=len(rows))
        # Telemetry key names have moved between tt-smi releases; take whichever of the
        # usual spellings is present rather than hard-coding one and silently reporting
        # nothing.
        for key in ("power", "voltage", "current", "aiclk", "asic_temperature"):
            vals = []
            for r in rows:
                for tel in r.values():
                    v = tel.get(key)
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        continue
            if vals:
                out[f"{key}_mean_per_card"] = round(sum(vals) / len(vals), 2)
                out[f"{key}_max_per_card"] = round(max(vals), 2)
                out[f"{key}_sum_mean_all_cards"] = round(sum(vals) / len(rows), 2)
        return out


def _find_tt_smi() -> str | None:
    for c in (Path(sys.executable).parent / "tt-smi",
              Path.home() / ".local" / "bin" / "tt-smi",
              Path("/usr/local/bin/tt-smi")):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _ttnn_version() -> str:
    try:
        from importlib.metadata import version
        return version("ttnn")
    except Exception:
        return "unknown"


def launcher(args) -> int:
    cards = [int(c) for c in args.cards.split(",") if c.strip() != ""]
    n = len(cards)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for pat in ("ready.*", "done.*.json", "failed.*"):
        for p in run_dir.glob(pat):
            p.unlink()

    sampler = TtSmiSampler(_find_tt_smi() if args.telemetry else None, cards)
    sampler.start()

    procs = []
    t_launch = time.monotonic()
    for i, card in enumerate(cards):
        env = dict(os.environ, TT_VISIBLE_DEVICES=str(card))
        if args.lease_timeout:
            env["TT_BIO_LEASE_TIMEOUT"] = str(args.lease_timeout)
        env.setdefault("PYTHONPATH", str(REPO_ROOT))
        cmd = [sys.executable, str(HERE / "tt_concurrency.py"),
               "--worker-id", str(i), "--n", str(n), "--folds", str(args.folds),
               "--model", args.model, "--samples", str(args.samples),
               "--target", args.target, "--msa-a3m", args.msa_a3m,
               "--cards", args.cards, "--run-dir", str(run_dir)]
        if args.hoist:
            cmd.append("--hoist")
        if args.instrument:
            cmd.append("--instrument")
        log = open(run_dir / f"w{i}.log", "wb")
        procs.append((i, card, subprocess.Popen(cmd, stdout=log,
                                                stderr=subprocess.STDOUT, env=env), log))

    rcs, tails = {}, {}
    for i, card, p, log in procs:
        try:
            rcs[i] = p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rcs[i] = "timeout"
        log.close()
        tails[i] = (run_dir / f"w{i}.log").read_text(errors="replace").splitlines()[-15:]
    sampler.stop()
    wall_s = time.monotonic() - t_launch

    results = conc.load_worker_results(run_dir)
    agg = conc.aggregate(results) if results else dict(
        clean=False, reason="no worker produced a result")
    if agg.get("window_start") is not None and args.telemetry:
        agg.update(sampler.window_stats(agg["window_start"], agg["window_end"]))

    out = dict(
        side="tenstorrent", model=args.model, machine=socket.gethostname(),
        cards=cards, n_concurrent=n, folds_per_worker=args.folds,
        diffusion_samples=args.samples, target=args.target,
        timed_region=("model.fold only (featurization hoisted, CIF write suppressed)"
                      if args.hoist else
                      "predict_one (featurize + fold + CIF write)"),
        phase_times=[r.get("phase_times") for r in results],
        ttnn_version=_ttnn_version(), tt_bio_git=tt_baseline._git_sha(),
        recycling_steps=tt_baseline.RECYCLING_STEPS,
        sampling_steps=tt_baseline.SAMPLING_STEPS, seed=tt_baseline.SEED,
        worker_rcs=rcs, worker_log_tails=tails, wall_s=round(wall_s, 2),
        cold_s=[r.get("cold_s") for r in results],
        load_s=[r.get("load_s") for r in results],
        plddt=[r.get("plddt") for r in results],
        card_of_worker={r["worker"]: r.get("card") for r in results},
        date=time.strftime("%Y-%m-%d"), **agg,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    ok = "CLEAN" if agg.get("clean") else f"DIRTY ({agg.get('reason')})"
    print(f"[tt cards={cards}] aggregate {agg.get('agg_folds_per_s')} folds/s "
          f"(window est {agg.get('agg_folds_per_s_window')}), "
          f"latency median {agg.get('latency_median_s')}s, {ok}",
          file=sys.stderr, flush=True)
    return 0 if agg.get("clean") else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", default="0,1,2,3",
                    help="physical cards, one worker each. Repeat a card (e.g. 3,3) to "
                         "record what happens when two processes want the same chip.")
    ap.add_argument("--model", default="protenix-v2", choices=["protenix-v2", "opendde"])
    ap.add_argument("--folds", type=int, default=5, help="timed folds per worker")
    ap.add_argument("--samples", type=int, default=1,
                    help="diffusion samples per fold; >1 is the in-process batching lever")
    ap.add_argument("--hoist", action="store_true",
                    help="time model.fold only (featurization hoisted out, CIF write "
                         "suppressed), matching the GPU harness's timed region")
    ap.add_argument("--instrument", action="store_true",
                    help="raw timed region, but record the per-fold featurize/fold/write "
                         "split in each worker's result")
    ap.add_argument("--target", default=str(REPO_ROOT / "examples" / "prot.yaml"))
    ap.add_argument("--msa-a3m", default=str(HERE / "fixtures" / "prot117.a3m"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--lease-timeout", type=float, default=None,
                    help="TT_BIO_LEASE_TIMEOUT for the workers; set it low (e.g. 20) for "
                         "the same-card probe so the refusal is quick")
    ap.add_argument("--telemetry", dest="telemetry", action="store_true", default=True)
    ap.add_argument("--no-telemetry", dest="telemetry", action="store_false")
    ap.add_argument("--out", default=None)
    # worker-only
    ap.add_argument("--worker-id", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    if args.run_dir is None:
        tag = args.cards.replace(",", "-")
        args.run_dir = f"/tmp/ttconc-{args.model}-{Path(args.target).stem}-c{tag}"
    if args.worker_id is not None:
        assert args.n is not None, "worker needs --n from the launcher"
        return worker(args)
    if args.out is None:
        ap.error("--out is required in launcher mode")
    return launcher(args)


if __name__ == "__main__":
    sys.exit(main())
