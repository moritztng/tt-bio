"""Checkpoint and resume for the ABodyBuilder3 reproduction's complete step.

``tt_bio/train/checkpoint.py`` is the adapter round-trip for the LoRA recipe's ``AdamW``. The
reproduction is not a LoRA fine-tune and its optimizer is ``torch.optim.RAdam``, because that
is what ``params.yaml`` pins -- so it needs its own round-trip, held to the same rule and with
one deliberate difference in the signature.

**The optimizer argument is required here, and that is the point.** ``load_adapter``'s ``opt``
is optional, and omitting it is silent: the weights load, both moment estimates stay at a fresh
optimizer's zeros, the step count driving bias correction resets, and the run continues showing
a plausible loss curve for a different optimisation problem. Over a nine-day run with ~9-47
watchdog resets that is 9-47 silent restarts of the optimiser state. So there is no way to call
this without the optimizer: :func:`load_run_state` takes the ``TrainStep`` itself and restores
every piece of state a step reads.

**What "every piece" means, enumerated rather than trusted.** A resume is only worth having if
the next step is the step the uninterrupted run would have taken, so the saved set is:

* the float32 masters -- ``TrainStep.mirror``, not the device tensors. The device copy is
  written from the mirror each step, so the mirror is the authoritative value;
* both RAdam moments and RAdam's own per-parameter step counter, which drives bias correction
  and RAdam's variance rectification;
* the global step, which drives the learning-rate schedule and the data order;
* the dropout generator's state and its call count. Dropout is part of this recipe
  (``dropout_rate: 0.1``, applied twice per block), so a resume that re-seeds it replays masks
  the run already used and the trajectory is no longer the one that was interrupted.

With all four restored the resumed run is **bit-identical** to the uninterrupted one, which is
what ``tests/test_abb3_resume.py`` asserts. That is a stronger claim than a continuous-looking
loss curve and it costs nothing to check, so it is the check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

__all__ = ["save_run_state", "load_run_state", "latest_checkpoint", "RunState"]

_SPLIT = "|"
#: Bumped when the saved set changes. A resume that silently ignored a field it did not know
#: about is the failure this file exists to prevent, so an unknown version is refused.
FORMAT = 2


class RunState(dict):
    """The metadata a checkpoint carries beside the tensors, as a plain dict."""

    @property
    def step(self) -> int:
        return int(self["global_step"])


def _flat_params(step) -> list:
    return list(step.mirror)


def save_run_state(path, step, *, global_step: int, metrics: dict | None = None,
                   provenance: dict | None = None, history_tail: list | None = None) -> Path:
    """Write one checkpoint atomically. Returns the path.

    Atomically because the process is killed by a watchdog reset rather than asked to stop, and
    a reset landing inside a 102 MB write leaves a truncated file that a resume would either
    refuse or, worse, load partially. Written to a temporary name in the same directory and
    renamed, so the newest complete checkpoint is always the newest file.
    """
    from safetensors.numpy import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mirrors = _flat_params(step)
    tensors, opt_steps = {}, []
    radam = step.optimizer
    for i, mirror in enumerate(mirrors):
        tensors[f"master{_SPLIT}{i}"] = mirror.detach().cpu().numpy().astype(np.float32)
        st = radam.state.get(mirror, {})
        for key in ("exp_avg", "exp_avg_sq"):
            t = st.get(key)
            tensors[f"{key}{_SPLIT}{i}"] = (
                t.detach().cpu().numpy().astype(np.float32) if t is not None
                else np.zeros_like(tensors[f"master{_SPLIT}{i}"]))
        s = st.get("step")
        opt_steps.append(float(s.item() if torch.is_tensor(s) else (s or 0)))
    tensors[f"rng{_SPLIT}dropout"] = step.dropout.generator.get_state().numpy()
    meta = {
        "format": str(FORMAT),
        "global_step": str(int(global_step)),
        "n_params": str(len(mirrors)),
        "opt_steps": json.dumps(opt_steps),
        "dropout": json.dumps({"rate": step.dropout.rate, "calls": step.dropout.calls}),
        "recipe": json.dumps(step.recipe),
        "accumulate": str(step.accumulate),
        "metrics": json.dumps(metrics or {}),
        "provenance": json.dumps(provenance or {}),
        "history_tail": json.dumps(history_tail or []),
    }
    tmp = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(path)
    return path


def load_run_state(path, step, *, upload=None) -> RunState:
    """Restore masters, both RAdam moments, RAdam's step counter and the dropout RNG.

    ``step`` is the ``TrainStep`` itself, so there is no call shape that loads the weights and
    leaves the optimizer at a fresh one's values. Returns the metadata; the caller reads
    ``global_step`` from it to place the learning-rate schedule and the data order.

    ``upload`` is the host-to-device uploader, defaulting to the one the step's own weights came
    up through. It exists so the restore can be tested without a card: everything this function
    has to get right -- the moments, RAdam's counter, the dropout generator -- is host state,
    and requiring a device to check it would put the gate somewhere CI cannot reach.
    """
    from safetensors.numpy import load_file
    path = Path(path)
    blob = load_file(str(path))
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        meta = json.loads(fh.read(n)).get("__metadata__", {})
    if int(meta.get("format", -1)) != FORMAT:
        raise ValueError(f"{path} is checkpoint format {meta.get('format')!r}, this build reads "
                         f"{FORMAT}. Refusing rather than loading the fields it recognises: a "
                         f"partial restore is the silent-optimizer failure with extra steps")
    mirrors = _flat_params(step)
    have = len([k for k in blob if k.startswith(f"master{_SPLIT}")])
    if have != len(mirrors):
        raise ValueError(f"{path} holds {have} masters and this model has {len(mirrors)} "
                         f"parameters. A partial load reproduces the initial model's loss and "
                         f"reads as 'training did nothing'")
    if int(meta["accumulate"]) != step.accumulate:
        raise ValueError(f"{path} was written at accumulate={meta['accumulate']} and this "
                         f"process runs accumulate={step.accumulate}. The global batch is the "
                         f"axis the published recipe pins, so resuming across a change in it "
                         f"would continue a different recipe")
    radam = step.optimizer
    opt_steps = json.loads(meta["opt_steps"])
    if upload is None:
        from ..abodybuilder3 import to_device_fp32 as upload
    for i, mirror in enumerate(mirrors):
        arr = blob[f"master{_SPLIT}{i}"]
        with torch.no_grad():
            mirror.copy_(torch.from_numpy(arr.reshape(tuple(mirror.shape))))
        st = radam.state.setdefault(mirror, {})
        for key in ("exp_avg", "exp_avg_sq"):
            st[key] = torch.from_numpy(
                blob[f"{key}{_SPLIT}{i}"].reshape(tuple(mirror.shape)).copy())
        # RAdam keeps its step as a scalar tensor and reads `.item()` on it. A python float
        # here would work until the first `torch.compile` or `foreach` path, which take the
        # tensor form; matching what RAdam itself writes is free.
        st["step"] = torch.tensor(opt_steps[i], dtype=torch.float32)
        # The device copy is written from the master, exactly as the optimizer does each step.
        # Uploading fp32 from an fp32 master is exact, so after this the device and the master
        # agree bit for bit rather than to within a rounding.
        step.params[i].value = upload(mirror.detach())
    drop = json.loads(meta["dropout"])
    step.dropout.generator.set_state(
        torch.from_numpy(blob[f"rng{_SPLIT}dropout"]).to(torch.uint8))
    step.dropout.calls = int(drop["calls"])
    out = RunState(meta)
    for key in ("opt_steps", "dropout", "recipe", "metrics", "provenance", "history_tail"):
        out[key] = json.loads(meta[key])
    out["global_step"] = int(meta["global_step"])
    return out


def latest_checkpoint(dir, prefix: str = "step") -> Path | None:
    """The newest complete checkpoint in ``dir``, or None.

    By filename rather than mtime: the step number is what a resume needs and a copied or
    rsynced directory loses mtime ordering. Partial writes cannot appear here because
    :func:`save_run_state` renames into place, and the temporary name is dot-prefixed.
    """
    d = Path(dir)
    if not d.is_dir():
        return None
    found = sorted(p for p in d.glob(f"{prefix}-*.safetensors") if not p.name.startswith("."))
    return found[-1] if found else None
