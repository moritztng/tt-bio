"""Lightning callback that times upstream RFD4-Proteina's own training step (R4T G4 GPU baseline).

Loaded into upstream's `rfd4-train` with
    +callbacks.r4t_timer._target_=r4t_step_timer.StepTimer +callbacks.r4t_timer.out=<path.jsonl>
and PYTHONPATH pointing at this directory. It changes nothing in the step; it only reads clocks.

Per step it writes one JSON line: global step, recycle depth (the value upstream's own
handle_recycling returned), the device step (batch start to batch end, synchronized), the full step (previous batch end to
this one, so dataloader waits are inside it), loss and its terms, and peak allocated memory.
"""

import json
import time

import lightning as L
import torch


class StepTimer(L.Callback):
    def __init__(self, out: str):
        self.out, self.t_prev, self.t0, self.rec = out, None, None, None
        self.f = open(out, "a", buffering=1)

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def on_fit_start(self, trainer, pl_module):
        # Record the recycle depth upstream actually drew for the step: wrap its own method, return value untouched.
        inner = pl_module.handle_recycling

        def handle_recycling(*a, **kw):
            n = inner(*a, **kw)
            self.rec = n
            return n

        pl_module.handle_recycling = handle_recycling

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step = int(pl_module.global_step)  # read before training_step bumps it
        self.rec = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = self._sync()
        if self.t_prev is None:
            self.t_prev = self.t0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        t1 = self._sync()
        k = int(outputs.get("exploration_k", 1)) if isinstance(outputs, dict) else 1
        phys = int(batch["padding_mask"].shape[0])
        terms = {n: float(v.float().mean()) for n, v in (outputs or {}).get("losses_per_sample", {}).items()}
        rec = {
            "step": self.step, "batch_idx": batch_idx, "n_recycle": self.rec,
            "physical_bs": phys, "examples": phys // max(k, 1), "tokens": int(batch["padding_mask"].shape[1]),
            "device_s": t1 - self.t0, "full_s": t1 - self.t_prev,
            "loss": float(outputs["loss"]) if isinstance(outputs, dict) and "loss" in outputs else None,
            "terms": terms,
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None,
            "t_end_unix": time.time(),
        }
        self.f.write(json.dumps(rec) + "\n")
        self.t_prev = t1
