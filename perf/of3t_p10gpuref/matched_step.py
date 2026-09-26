#!/usr/bin/env python3
"""One OpenFold3 training step on a CUDA box, with the recycle count pinned.

Why this exists: the PERF10 bar is "<= 10x an H200 step", and the H200 step in the record
(`state/of3t-gpugap.md:46`) is n=2 steady samples whose recycle count was drawn U{0..3} rather
than pinned. Our TT step pins `--cycles 4`. So the two sides do a different amount of trunk work
per step, in the direction that makes the published ratio HARDER than a matched one.

It drives upstream's own training entry point in-process -- the same four calls
`openfold3/run_openfold.py:train` makes, against the same `datasets/train_pdb_subset.yaml`
`test_training_full.py` uses -- and adds three runtime patches. Nothing in the upstream tree is
edited.

  1. RECYCLE PIN (`--pin-recycles K`). `projects/of3_all_atom/model.py:650` draws
     `synced_generator.integers(low=0, high=shared.num_recycles + 1)` when `self.training`, so
     num_cycles = draw + 1 is U{1..4}, mean 2.5, against our pinned 4. The pin replaces that one
     generator with a constant. `--pin-recycles -1` leaves upstream's draw alone.
  2. RECYCLE RECORD. `model.forward` returns `{"recycles": n}` and nothing consumes it, so the
     forward is wrapped to stash it. Every step below reports the count it actually ran.
  3. PER-STEP CLOCK. A Lightning callback times each step around `torch.cuda.synchronize()`, and
     a 0.5 s sampler reads `clocks.sm` so each step carries the SM clock sampled DURING it, not a
     nameplate. Same discipline as the AICLK rule on the TT side.

`t_batch` is batch_start -> batch_end (compute). `t_iter` is batch_start -> next batch_start, so
it also carries the dataloader, and it is the one comparable to Lightning's progress bar, which
is where the record's 11/8/7 s reading came from. Both are reported; neither is derived.
"""

import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------- clock sampler


class ClockSampler:
    """Poll SM clock / util / power until stopped. Samples carry a wall timestamp so each
    step can be given the clock that was true while it ran."""

    QUERY = "clocks.sm,utilization.gpu,power.draw,temperature.gpu,clocks.max.sm"

    def __init__(self, period=0.5):
        self.period = period
        self.samples = []  # (t, sm, util, power, temp, maxsm)
        self._stop = threading.Event()
        self._thread = None

    def _read(self):
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return [float(x) for x in out.split(",")]

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), *self._read()))
            except Exception:
                pass
            self._stop.wait(self.period)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def window(self, t0, t1):
        """Clock stats over [t0, t1]. Returns None when the window caught no sample."""
        sm = [s[1] for s in self.samples if t0 <= s[0] <= t1]
        util = [s[2] for s in self.samples if t0 <= s[0] <= t1]
        if not sm:
            return None
        return {
            "sm_mhz_min": min(sm), "sm_mhz_median": statistics.median(sm),
            "sm_mhz_max": max(sm), "n": len(sm),
            "util_median": statistics.median(util),
        }


# ---------------------------------------------------------------- patches

_STATE = {"recycles": None}


def install_patches(pin_recycles, sampler, records):
    import pytorch_lightning as pl
    from openfold3.projects.of3_all_atom.model import OpenFold3

    # (2) record what the model actually ran
    _orig_forward = OpenFold3.forward

    def _forward(self, *a, **k):
        out = _orig_forward(self, *a, **k)
        o = out[1] if isinstance(out, tuple) and len(out) == 2 else out
        if isinstance(o, dict) and "recycles" in o:
            _STATE["recycles"] = int(o["recycles"])
        return out

    OpenFold3.forward = _forward

    # (1) pin the draw
    if pin_recycles is not None:
        class _Pinned:
            def __init__(self, v):
                self.v = v

            def integers(self, *a, **k):
                return self.v

        _orig_init = OpenFold3.__init__

        def _init(self, *a, **k):
            _orig_init(self, *a, **k)
            self.synced_generator = _Pinned(pin_recycles)

        OpenFold3.__init__ = _init

    # (3) per-step timing, appended to whatever callbacks the runner built
    import torch

    class StepTimer(pl.Callback):
        def __init__(self):
            self.t_start = None
            self.prev_start = None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            torch.cuda.synchronize()
            self.t_start = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                               dataloader_idx=0):
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            rec = {
                "step": len(records),
                "batch_idx": batch_idx,
                "recycles": _STATE["recycles"],
                "num_cycles": None if _STATE["recycles"] is None else _STATE["recycles"] + 1,
                "t_batch_s": t_end - self.t_start,
                "t_iter_s": None if self.prev_start is None else self.t_start - self.prev_start,
                "clock": sampler.window(self.t_start, t_end),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            torch.cuda.reset_peak_memory_stats()
            self.prev_start = self.t_start
            records.append(rec)
            ti = rec["t_iter_s"]
            print(f"STEP {rec['step']:>3} cycles={rec['num_cycles']}"
                  f" t_batch={rec['t_batch_s']:.3f}s"
                  f" t_iter={'-' if ti is None else format(ti, '.3f')}s"
                  f" clock={rec['clock']}", flush=True)

    _orig_trainer_init = pl.Trainer.__init__

    def _trainer_init(self, *a, **k):
        cbs = list(k.get("callbacks") or [])
        cbs.append(StepTimer())
        k["callbacks"] = cbs
        _orig_trainer_init(self, *a, **k)

    pl.Trainer.__init__ = _trainer_init


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner-yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=10, help="epoch_len; max_epochs is forced to 1")
    ap.add_argument("--pin-recycles", type=int, default=-1,
                    help="num_recycles to pin (num_cycles = this + 1). -1 leaves upstream's draw")
    ap.add_argument("--samples", type=int, default=48,
                    help="architecture.shared.diffusion.no_samples")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from openfold3.core.config import config_utils
    from openfold3.entry_points.import_utils import _configure_torch_backend

    _configure_torch_backend()

    cfg = config_utils.load_yaml(Path(args.runner_yaml))
    cfg = config_utils.deep_update(cfg, {
        "experiment_settings": {"output_dir": args.output_dir},
        "data_module_args": {"epoch_len": args.steps},
        "checkpoint_config": {"save_top_k": 0, "save_last": False},
        "pl_trainer_args": {
            "max_epochs": 1,
            "log_every_n_steps": 1,
            # no validation: this row measures a training step, and the sanity check
            # plus the 4-structure val pass would be most of the wall clock.
            "num_sanity_val_steps": 0,
            "limit_val_batches": 0,
        },
        "model_update": {"custom": {"architecture": {"shared": {
            "diffusion": {"no_samples": args.samples},
        }}}},
    })

    pin = None if args.pin_recycles < 0 else args.pin_recycles
    records = []
    sampler = ClockSampler()
    install_patches(pin, sampler, records)

    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
    from openfold3.entry_points.validator import TrainingExperimentConfig

    expt_config = TrainingExperimentConfig.model_validate(cfg)
    runner = TrainingExperimentRunner(expt_config)

    import torch
    meta = {
        "label": args.label,
        "pin_recycles": pin,
        "no_samples": args.samples,
        "steps_requested": args.steps,
        "seed": cfg["experiment_settings"]["seed"],
        "token_budget": 384,
        "precision": cfg["pl_trainer_args"]["precision"],
        "batch_size": cfg["data_module_args"]["batch_size"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    sampler.start()
    t0 = time.perf_counter()
    try:
        runner.setup()
        runner.run()
    finally:
        wall = time.perf_counter() - t0
        sampler.stop()

    # steady = every step but the first; the first carries CUDA/cudnn warmup and autotune
    steady = [r for r in records[1:] if r["t_batch_s"] is not None]
    seen = [r["recycles"] for r in records]
    summary = {"meta": meta, "wall_s": wall, "n_steps": len(records), "steps": records}
    summary["recycles_seen"] = seen
    if not records or all(r is None for r in seen):
        summary["INSTRUMENT_FAILED"] = (
            "no step reported a recycle count: the forward patch matched nothing, so neither "
            "the pin nor the draw can be verified from this run")
    elif pin is not None and any(r != pin for r in seen):
        summary["INSTRUMENT_FAILED"] = f"pin={pin} requested but steps ran {seen}"
    if steady:
        tb = [r["t_batch_s"] for r in steady]
        allsm = [s[1] for s in sampler.samples]
        summary["steady"] = {
            "n": len(tb),
            "t_batch_min": min(tb), "t_batch_median": statistics.median(tb),
            "t_batch_mean": statistics.fmean(tb), "t_batch_max": max(tb),
            "t_batch_stdev": statistics.stdev(tb) if len(tb) > 1 else 0.0,
            "spread_s": max(tb) - min(tb),
            "by_cycles": {
                str(c): {
                    "n": len(g),
                    "median": statistics.median(g),
                    "min": min(g), "max": max(g),
                }
                for c in sorted({r["num_cycles"] for r in steady})
                if (g := [r["t_batch_s"] for r in steady if r["num_cycles"] == c])
            },
            "sm_mhz_min_during": min(allsm) if allsm else None,
            "sm_mhz_median_during": statistics.median(allsm) if allsm else None,
            "sm_mhz_max_during": max(allsm) if allsm else None,
            "clock_samples": len(allsm),
        }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary.get("steady", {}), indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
