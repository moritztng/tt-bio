#!/usr/bin/env python
"""Measure ABodyBuilder3's own training step, at its own global batch of 64, on one GPU.

Runs upstream's model, loss, dataloader and Trainer configuration. It is a runner beside
`src/abodybuilder3/stages/train.py`, not a reimplementation: everything that costs time comes from
the upstream package. The differences from `train.py`, all of them deliberate:

  * params come from `params.yaml` read with ruamel.yaml instead of `dvc.api.params_show()`, which
    reads the same file with the same parser. It has to be ruamel and not pyyaml: pyyaml is YAML 1.1
    and leaves `lr: 5e-4` as the string "5e-4", which RAdam then compares against a float and
    crashes. `base.debug` is forced False; upstream ships it True, which caps a run at one batch per
    epoch and cannot be timed.
  * `Fabric`/`SLURM_JOB_GPUS` rank plumbing is dropped. devices=1, nodes=1, so
    accumulate_grad_batches = 64/8 = 8, exactly what train.py computes for a single GPU.
  * strategy="auto" instead of "ddp": at one device DDP wraps the model in a DistributedDataParallel
    with a single rank, which adds a process group and a no-op allreduce to every step. `--ddp`
    turns it back on, and the delta is reported rather than assumed.
  * logger=False instead of DVCLiveLogger; `--dvclive` turns it back on and the delta is reported.
  * validation is off (limit_val_batches=0, num_sanity_val_steps=0). A validation pass is not a
    training step, and s/step is what is being measured.
  * a StepTimer callback, and `trainer.should_stop` once enough steps are recorded.

Modes:
  real      per-step wall clock, full pipeline. The headline number.
  dataonly  iterate the same dataloader with no model, to price the host data pipeline. Valid to
            subtract because params.yaml sets num_workers: 0, so loading never overlaps compute.
  profile   torch.profiler over a few steps, for CUDA kernel time per step.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import lightning.pytorch as pl
import ml_collections
import torch
from box import Box
from lightning.pytorch.callbacks import Callback

from abodybuilder3.lightning_module import ABB3DataModule, LitABB3
from abodybuilder3.stages.finetune import FineTuneCallback


def nvidia_smi() -> dict:
    def q(fields, extra=()):
        cmd = ["nvidia-smi", f"--query-{fields[0]}={','.join(fields[1])}",
               "--format=csv,noheader,nounits", *extra]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"unavailable: {e}"
        return [l.strip() for l in out.splitlines() if l.strip()]

    return {
        "gpu": q(("gpu", ["name", "memory.total", "clocks.sm", "clocks.max.sm", "power.draw",
                          "power.limit", "temperature.gpu", "utilization.gpu", "persistence_mode"])),
        # Co-tenancy detector: any pid here that is not ours means the physical card is shared.
        "compute_apps": q(("compute-apps", ["pid", "process_name", "used_memory"])),
    }


def host_info() -> dict:
    def read(path, n=1):
        try:
            return Path(path).read_text().strip().splitlines()[:n]
        except Exception:  # noqa: BLE001
            return []

    model = [l.split(":", 1)[1].strip() for l in read("/proc/cpuinfo", 200)
             if l.startswith("model name")]
    mem = [l for l in read("/proc/meminfo", 3)]
    return {
        "cpu_model": model[0] if model else "?",
        "cpu_threads": os.cpu_count(),
        "meminfo": mem,
        "loadavg": read("/proc/loadavg"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "uname": platform.platform(),
    }


class StepTimer(Callback):
    """One recorded interval = one optimizer step at global batch 64.

    The clock starts at the end of the previous optimizer step, so the interval contains all
    `accumulate` micro-batch loads, their forward and backward, and the optimizer update. That is
    the whole step as upstream runs it, host work included.
    """

    def __init__(self, accumulate: int, warmup: int, steps: int):
        self.accumulate = accumulate
        self.warmup = warmup
        self.target = warmup + steps
        self.micro = 0
        self.t_prev: float | None = None
        self.step_s: list[float] = []
        self.tokens: list[int] = []
        self.batch_tokens: list[int] = []

    def _mark(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.t_prev is None:
            self.t_prev = self._mark()
        self.batch_tokens.append(int(batch["seq_mask"].shape[-1]))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.micro += 1
        if self.micro % self.accumulate:
            return
        now = self._mark()
        self.step_s.append(now - self.t_prev)
        self.tokens.append(max(self.batch_tokens[-self.accumulate:]))
        self.t_prev = now
        if len(self.step_s) >= self.target:
            trainer.should_stop = True

    def timed(self) -> list[float]:
        return self.step_s[self.warmup:]


def spread(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "stdev": statistics.stdev(xs) if n > 1 else 0.0,
        "min": xs[0],
        "max": xs[-1],
        "p05": xs[max(0, int(0.05 * n) - 1)],
        "p95": xs[min(n - 1, int(0.95 * n))],
    }


def build(params: Box, stage: int):
    loss_config = ml_collections.config_dict.ConfigDict(params.loss)
    model_config = ml_collections.config_dict.ConfigDict(params.model)
    optimiser_config = ml_collections.config_dict.ConfigDict(params.optimiser)
    rel_pos_dim = 64
    c_z = 2 * rel_pos_dim + 1
    if params.model.edge_chain_feature:
        c_z += 3
    model_config.c_z = c_z
    if params.language.model is not None:
        model_config.c_s = 1024
    model = LitABB3(model_config, loss_config, optimiser_config)
    data = ABB3DataModule(
        data_dir=params.base.data_dir,
        batch_size=8,
        legacy=not params.base.all_data,
        edge_chain_feature=params.model.edge_chain_feature,
        num_workers=params.base.num_workers,
        pin_memory=params.base.pin_memory,
        use_plm_embeddings=params.language.model is not None,
    )
    batch_size = params.train.batch_size if stage == 1 else params.finetune.batch_size
    return model, data, int(batch_size / 8)


def run_real(args, params: Box) -> dict:
    pl.seed_everything(params.base.seed)
    model, data, accumulate = build(params, args.stage)
    timer = StepTimer(accumulate, args.warmup, args.steps)
    callbacks: list[Callback] = [timer]
    if args.stage == 2:
        # Upstream's own stage-2 switch: enables the three violation losses and drops dropout.
        callbacks.append(FineTuneCallback(
            use_annealing=params.finetune.use_annealing,
            dropout=params.finetune.dropout,
            turn_off_scheduler=params.finetune.turn_off_scheduler,
            learning_rate=params.finetune.learning_rate,
        ))
    logger = False
    if args.dvclive:
        from dvclive.lightning import DVCLiveLogger
        logger = DVCLiveLogger(dir=f"dvclive/timing_stage{args.stage}")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        accumulate_grad_batches=accumulate,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        strategy="ddp" if args.ddp else "auto",
        limit_train_batches=1.0,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        precision="bf16-mixed",
    )
    smi_before = nvidia_smi()
    t0 = time.perf_counter()
    trainer.fit(model, data)
    wall = time.perf_counter() - t0
    return {
        "accumulate_grad_batches": accumulate,
        "global_batch": accumulate * 8,
        "steps_recorded": len(timer.step_s),
        "warmup_discarded": args.warmup,
        "step_s": spread(timer.timed()),
        "step_s_all": timer.step_s,
        "padded_tokens_per_step": spread([float(t) for t in timer.tokens[args.warmup:]]),
        "micro_batch_padded_tokens": spread([float(t) for t in timer.batch_tokens]),
        "fit_wall_s": wall,
        "peak_cuda_alloc_GB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_cuda_reserved_GB": torch.cuda.max_memory_reserved() / 2**30,
        "smi_before": smi_before,
        "smi_after": nvidia_smi(),
    }


def run_dataonly(args, params: Box) -> dict:
    """Price the host data pipeline: torch.load, feature build, collate. No model, no GPU."""
    pl.seed_everything(params.base.seed)
    _, data, accumulate = build(params, args.stage)
    data.setup("fit")
    loader = data.train_dataloader()
    n_micro = accumulate * (args.warmup + args.steps)
    per_step, group = [], []
    t_prev = time.perf_counter()
    it = iter(loader)
    for i in range(n_micro):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        group.append(int(batch["seq_mask"].shape[-1]))
        del batch
        if (i + 1) % accumulate == 0:
            now = time.perf_counter()
            per_step.append(now - t_prev)
            t_prev = now
            group = []
    return {
        "accumulate_grad_batches": accumulate,
        "micro_batches": n_micro,
        "train_samples": len(data.train_dataset),
        "data_s_per_step": spread(per_step[args.warmup:]),
    }


def run_profile(args, params: Box) -> dict:
    """CUDA kernel time per step, from torch.profiler over a few steps after a warmup."""
    pl.seed_everything(params.base.seed)
    model, data, accumulate = build(params, args.stage)
    callbacks: list[Callback] = []
    if args.stage == 2:
        callbacks.append(FineTuneCallback(
            use_annealing=params.finetune.use_annealing,
            dropout=params.finetune.dropout,
            turn_off_scheduler=params.finetune.turn_off_scheduler,
            learning_rate=params.finetune.learning_rate,
        ))
    warm, prof = args.warmup, args.steps

    class Sched(Callback):
        def __init__(self):
            self.micro = 0
            self.steps = 0
            self.p = None
            self.t0 = None
            self.wall = None

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.micro += 1
            if self.micro % accumulate:
                return
            self.steps += 1
            if self.steps == warm:
                torch.cuda.synchronize()
                self.p = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, with_stack=False,
                )
                self.p.__enter__()
                self.t0 = time.perf_counter()
            elif self.steps == warm + prof:
                torch.cuda.synchronize()
                self.wall = time.perf_counter() - self.t0
                self.p.__exit__(None, None, None)
                trainer.should_stop = True

    sched = Sched()
    callbacks.append(sched)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, num_nodes=1,
        accumulate_grad_batches=accumulate, logger=False, callbacks=callbacks,
        max_epochs=args.epochs, strategy="auto",
        limit_train_batches=1.0, limit_val_batches=0, num_sanity_val_steps=0,
        enable_checkpointing=False, enable_progress_bar=False, precision="bf16-mixed",
    )
    trainer.fit(model, data)
    ka = sched.p.key_averages()
    cuda_self = sum(getattr(e, "self_device_time_total", 0) or
                    getattr(e, "self_cuda_time_total", 0) for e in ka)
    cpu_self = sum(e.self_cpu_time_total for e in ka)
    top = sorted(
        ((e.key,
          (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) / 1e3
          / prof,
          e.count / prof) for e in ka),
        key=lambda r: -r[1],
    )[:25]
    return {
        "profiled_steps": prof,
        "profiled_wall_s_per_step": sched.wall / prof,
        "cuda_self_ms_per_step": cuda_self / 1e3 / prof,
        "cpu_self_ms_per_step": cpu_self / 1e3 / prof,
        "note": "profiler adds overhead, so profiled_wall_s_per_step is NOT the headline; "
                "cuda_self is device-busy time and is what the split uses",
        "top_kernels_ms_per_step": [{"op": k, "cuda_ms": round(ms, 3), "calls": round(c, 1)}
                                    for k, ms, c in top],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "dataonly", "profile"), default="real")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--dvclive", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")  # upstream train.py line 1 of __main__
    from ruamel.yaml import YAML  # dvc's own parser, so the values match params_show()
    params = Box(YAML(typ="safe").load(Path(args.params).read_text()))
    for k in ("lr", "weight_decay"):
        assert isinstance(params.optimiser[k], float), (k, params.optimiser[k])
    params.base.debug = False

    rec = {
        "tag": args.tag,
        "mode": args.mode,
        "stage": args.stage,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "args": vars(args),
        "host": host_info(),
        "nvidia_smi_idle": nvidia_smi(),
        "params_used": {k: params[k] for k in ("base", "train", "finetune", "model", "optimiser",
                                               "language")},
    }
    rec["result"] = {"real": run_real, "dataonly": run_dataonly, "profile": run_profile}[
        args.mode](args, params)
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out = args.out or f"timing_{args.tag or 'run'}_{args.mode}_stage{args.stage}.json"
    Path(out).write_text(json.dumps(rec, indent=1, default=str))
    print(f"\n=== wrote {out} ===")
    r = rec["result"]
    if args.mode == "real":
        s = r["step_s"]
        print(f"stage {args.stage} global batch {r['global_batch']} "
              f"({r['accumulate_grad_batches']} x 8): "
              f"median {s['median']:.4f} s/step over n={s['n']} "
              f"(p05 {s['p05']:.4f}, p95 {s['p95']:.4f}, stdev {s['stdev']:.4f})")
        print(f"peak cuda alloc {r['peak_cuda_alloc_GB']:.2f} GB")
    elif args.mode == "dataonly":
        s = r["data_s_per_step"]
        print(f"host data pipeline: median {s['median']:.4f} s/step (n={s['n']})")
    else:
        print(f"cuda self {r['cuda_self_ms_per_step']:.1f} ms/step, "
              f"cpu self {r['cpu_self_ms_per_step']:.1f} ms/step, "
              f"profiled wall {r['profiled_wall_s_per_step']:.4f} s/step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
