#!/usr/bin/env python3
"""Produce the OF3 training reference, and time their step, on one GPU.

Two modes, one script, because both must run against the same weights and the same frozen batches.

  --mode bundle  drives upstream's own OpenFold3AllAtom LightningModule through a real pl.Trainer
                 over the frozen batches, capturing w_0, per-step d_k = w_k - w_0, the step-1
                 gradients, every loss term, the LR, the unclipped per-sample grad norm, the clip
                 coefficient their grad_manager actually applied, and the gradient PRESENCE pattern
                 per parameter (PROTOCOL 3b: None must stay None, never zeros_like).

  --mode timing  the same module and batches at their shipped precision, timing the optimizer step
                 over --steps, warmup discarded, median with p05/p95. Per-rank batch is 1, which
                 upstream asserts at projects/of3_all_atom/runner.py:390.

Nothing here reimplements their training step. The capture is done with wrappers that observe and
then call through, so the arithmetic that produces every number is theirs.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_batches(d: Path):
    manifest = json.loads((d / "batches_manifest.json").read_text())
    out = []
    for entry in manifest["steps"]:
        p = d / entry["file"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != entry["sha256"]:
            raise SystemExit(f"batch {p.name} sha256 {got} != manifest {entry['sha256']}")
        out.append(torch.load(p, weights_only=False))
    return manifest, out


class FrozenBatches(torch.utils.data.Dataset):
    """Yields already-collated batches in the frozen order, so the loader adds no randomness."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


def frozen_datamodule(loader):
    """Minimal LightningDataModule carrying the frozen loader.

    Their on_train_epoch_start / on_train_epoch_end log
    `self.trainer.datamodule.next_dataset_indices`, which only their DataModule has. Passing a
    dataloader alone leaves `trainer.datamodule` as None and both hooks raise. The attribute is
    used for logging only (grep says these are its only two readers in runner.py), so a stub that
    reports the frozen order keeps their hooks on their normal path without standing in for their
    DataModule anywhere that matters.
    """
    import pytorch_lightning as pl

    class FrozenDataModule(pl.LightningDataModule):
        next_dataset_indices = "frozen"

        def train_dataloader(self):
            return loader

    return FrozenDataModule()


class FrozenSampler(torch.utils.data.SequentialSampler):
    """SequentialSampler that answers to `.epoch`.

    Their `on_train_epoch_start` logs `sampler.epoch`, which only their OF3DistributedSampler
    carries. Adding the attribute keeps their hook on its normal path; it does not change what is
    sampled, because the order is already the frozen one.
    """

    epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# Lightning swaps the sampler out unless told not to (use_distributed_sampler=False below), and if
# it ever does the swap anyway their logging hook hits a bare SequentialSampler. Giving the base
# class the attribute costs nothing and keeps a log line from ending a 20-step run.
torch.utils.data.SequentialSampler.epoch = 0


def build_module(upstream_cfg_overrides: dict):
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
    for dotted, value in upstream_cfg_overrides.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return cfg, OpenFold3AllAtom(model_config=cfg)


def mode_bundle(args):
    import pytorch_lightning as pl

    manifest, batches = load_batches(args.batches)
    batches = batches[: args.steps]
    overrides = {
        "settings.lr_scheduler.warmup_no_steps": args.warmup_no_steps,
        "architecture.shared.use_confidence_emb_prob": 0.8,
        "architecture.shared.diffusion.use_conditioning_prob": 0.8,
    }
    pl.seed_everything(args.seed, workers=True)
    cfg, module = build_module(overrides)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)

    record = {
        "seed": args.seed,
        "n_steps": len(batches),
        "warmup_no_steps": args.warmup_no_steps,
        "precision": args.precision,
        "batches_manifest": manifest,
        "steps": [],
    }

    # w_0. PROTOCOL A1: the compared quantity is d_k = w_k - w_0, so w_0 is published as tensors,
    # not only as hashes -- a downstream row that cannot reconstruct w_0 cannot form d_k.
    w0 = {k: v.detach().float().cpu().clone() for k, v in module.model.state_dict().items()}
    torch.save(w0, out / "weights" / "w0.pt")
    record["w0"] = {
        "file": "weights/w0.pt",
        "n_tensors": len(w0),
        "n_params": int(sum(v.numel() for v in w0.values())),
        "sha256": hashlib.sha256((out / "weights" / "w0.pt").read_bytes()).hexdigest(),
        "per_tensor_sha256": {k: tensor_sha256(v) for k, v in w0.items()},
    }

    captured = {"loss_terms": None, "unclipped_norm": None, "clip_coef": None,
                "disabled": None, "grad_presence": None, "lr_used": None}

    # module.loss is an nn.Module held in _modules, so it cannot be replaced by a plain function
    # (nn.Module.__setattr__ refuses). Patch its bound forward instead: __call__ looks up
    # self.forward, so this observes their loss without standing in for it.
    orig_loss_forward = module.loss.forward

    def loss_forward_hook(batch, outputs, _return_breakdown=False):
        result = orig_loss_forward(batch, outputs, _return_breakdown=_return_breakdown)
        if _return_breakdown:
            captured["loss_terms"] = {k: float(v) for k, v in result[1].items()}
        return result

    module.loss.forward = loss_forward_hook

    gm = module.grad_manager
    orig_clip = gm._clip_grads
    orig_accum = gm.clip_and_accumulate
    orig_sync = gm.sync_and_average_grads

    def clip_hook(logging_info=None, disabled_params=None):
        from openfold3.core.utils.grad_manager import compute_global_norm

        enabled = [p for n, p in gm._params_to_update.items()
                   if n not in (disabled_params or set())]
        norm, with_grad = compute_global_norm(parameters=enabled)
        captured["unclipped_norm"] = float(norm)
        # Their own expression, recomputed for the record rather than guessed:
        # clip_coef = max_norm / max(global_norm, max_norm).
        if gm.max_grad_norm is not None and with_grad:
            captured["clip_coef"] = float(
                gm.max_grad_norm / max(float(norm), gm.max_grad_norm)
            )
        else:
            captured["clip_coef"] = 1.0
        return orig_clip(logging_info=logging_info, disabled_params=disabled_params)

    def accum_hook(logging_info=None, disabled_params=None):
        captured["disabled"] = sorted(disabled_params or set())
        # Presence is read BEFORE accumulation, on the per-sample grads, which is where a
        # parameter that never received one is still None rather than a zero.
        captured["grad_presence"] = {
            n: (p.grad is not None) for n, p in gm._params_to_update.items()
        }
        return orig_accum(logging_info=logging_info, disabled_params=disabled_params)

    def sync_hook():
        result = orig_sync()
        if args.save_step1_grads and not (out / "grads_step1.pt").exists():
            grads = {n: (p.grad.detach().double().cpu().clone() if p.grad is not None else None)
                     for n, p in gm._params_to_update.items()}
            torch.save(grads, out / "grads_step1.pt")
        return result

    gm._clip_grads = clip_hook
    gm.clip_and_accumulate = accum_hook
    gm.sync_and_average_grads = sync_hook

    class Capture(pl.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            # The LR the optimizer USES at this step. Their step() calls opt.step() and then
            # lr_schedulers().step(), so by on_train_batch_end the group already holds the NEXT
            # step's value. Reading it at the end is off by one, and at step 1 that turns the
            # measured lr = 0 into 1.8e-06 -- exactly the fact this bundle exists to record.
            captured["lr_used"] = float(pl_module.optimizers().param_groups[0]["lr"])

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            w = pl_module.model.state_dict()
            deltas, per_tensor = {}, {}
            for k, v in w.items():
                d = v.detach().float().cpu() - w0[k]
                deltas[k] = d
                per_tensor[k] = {
                    "delta_l2": float(torch.linalg.vector_norm(d.double())),
                    "delta_max_abs": float(d.abs().max()) if d.numel() else 0.0,
                    "w_sha256": tensor_sha256(v.detach().float().cpu()),
                }
            step = batch_idx + 1
            if step in args.save_delta_steps:
                torch.save(deltas, out / "weights" / f"d{step:03d}.pt")
            flat = torch.cat([d.reshape(-1).double() for d in deltas.values()])
            record["steps"].append({
                "step": step,
                "pdb_id": batch["pdb_id"],
                "n_tokens": int(batch["token_mask"].sum()),
                "lr_used": captured["lr_used"],
                "lr_after_sched_step": float(pl_module.optimizers().param_groups[0]["lr"]),
                "loss_terms": captured["loss_terms"],
                "unclipped_grad_norm": captured["unclipped_norm"],
                "clip_coef": captured["clip_coef"],
                "disabled_params": captured["disabled"],
                "n_disabled": len(captured["disabled"] or []),
                "n_grad_absent": sum(1 for v in (captured["grad_presence"] or {}).values() if not v),
                "grad_presence_sha256": hashlib.sha256(
                    json.dumps(captured["grad_presence"], sort_keys=True).encode()
                ).hexdigest(),
                "delta_global_l2": float(torch.linalg.vector_norm(flat)),
                "delta_saved": step in args.save_delta_steps,
                "per_tensor": per_tensor,
            })
            print(f"step {step}: lr_used={record['steps'][-1]['lr_used']:.6g} "
                  f"loss={captured['loss_terms'].get('loss'):.6g} "
                  f"|d_k|={record['steps'][-1]['delta_global_l2']:.6g} "
                  f"clip={captured['clip_coef']:.6g} "
                  f"absent={record['steps'][-1]['n_grad_absent']}", flush=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        callbacks=[Capture()],
    )
    ds = FrozenBatches(batches)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, sampler=FrozenSampler(ds), num_workers=0
    )
    trainer.fit(module, datamodule=frozen_datamodule(loader))

    body = json.dumps(record, indent=2, sort_keys=True) + "\n"
    (out / "trajectory.json").write_text(body)
    print("trajectory sha256", hashlib.sha256(body.encode()).hexdigest())
    return 0


def mode_timing(args):
    import pytorch_lightning as pl

    manifest, batches = load_batches(args.batches)
    overrides = {
        "architecture.shared.use_confidence_emb_prob": 0.8,
        "architecture.shared.diffusion.use_conditioning_prob": 0.8,
    }
    pl.seed_everything(args.seed, workers=True)
    cfg, module = build_module(overrides)

    times = []

    class Timer(pl.Callback):
        def on_train_batch_start(self, *a, **k):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_train_batch_end(self, *a, **k):
            torch.cuda.synchronize()
            times.append(time.perf_counter() - self.t0)

    reps = (args.steps // len(batches)) + 1
    seq = (batches * reps)[: args.steps]
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=args.precision, max_epochs=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, use_distributed_sampler=False, callbacks=[Timer()],
    )
    ds = FrozenBatches(seq)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, sampler=FrozenSampler(ds), num_workers=0
    )
    trainer.fit(module, datamodule=frozen_datamodule(loader))

    kept = times[args.warmup:]
    tok = [int(b["token_mask"].sum()) for b in seq][args.warmup:]
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "precision": args.precision,
        "per_rank_batch": 1,
        "token_budget": 384,
        "steps_total": len(times),
        "warmup_discarded": args.warmup,
        "n": len(kept),
        "median_s": statistics.median(kept),
        "p05_s": sorted(kept)[int(0.05 * len(kept))],
        "p95_s": sorted(kept)[int(0.95 * len(kept))],
        "stdev_s": statistics.stdev(kept) if len(kept) > 1 else 0.0,
        "tokens_median": statistics.median(tok),
        "all_s": kept,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "all_s"}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bundle", "timing"], required=True)
    ap.add_argument("--batches", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--precision", default="32-true")
    ap.add_argument("--warmup-no-steps", type=int, default=1000)
    ap.add_argument("--save-step1-grads", action="store_true")
    ap.add_argument("--save-delta-steps", default="3,5,10,20")
    args = ap.parse_args()
    args.save_delta_steps = {int(x) for x in args.save_delta_steps.split(",") if x}
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    return mode_bundle(args) if args.mode == "bundle" else mode_timing(args)


if __name__ == "__main__":
    sys.exit(main())
