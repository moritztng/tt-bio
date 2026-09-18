"""LoRA fine-tuning over ``tt_bio.autograd``: adapters, AdamW, checkpoints.

The patterns are tt-train's, copied rather than linked because ``ttml`` cannot be built
against the ttnn 0.68.0 wheel (it ships 4 headers in ``tt_metal/api/tt-metalium/`` where a
build needs the tree, and a source-built tt-metal would put a second ``libtt_metal.so`` in
the same process as the wheel's). The three that matter:

* ``ttml/modules/lora.py`` -- ``linear(x, W, b) + linear(linear(x, A), B) * alpha/rank``,
  with ``W`` frozen, ``A`` kaiming-uniform at bound ``1/sqrt(in_features)`` and **``B``
  exactly zero**. B=0 is load-bearing rather than conventional: it makes the adapted
  forward bit-identical to the base model's at step 0, so a before/after held-out
  comparison is against the shipped model and not a perturbed copy of it.
* ``optimizers/adamw_full_precision.cpp`` -- fp32 master weights built once from the bf16
  weights (:28-30) with fp32 ``exp_avg``/``exp_avg_sq`` (:33-42), the bf16 weight the
  forward reads written back by typecasting the master down each step (:97-99), and beta
  powers carried multiplicatively (:64-65) rather than recomputed as ``pow(beta, step)``.
  ``hallgrad-build`` measured only 0.209 of an eps=0.01 step surviving a bf16 round-trip,
  so an update accumulated in bf16 silently stops moving. The master is where that is
  fixed.
* ``serialization/safetensors.hpp`` -- named tensors round-tripped through safetensors,
  which is already a tt-bio dependency.

The masters and the Adam moments live on the HOST in fp32 rather than in DRAM. That is a
measured choice for an adapter and not a general one: see ``perf/ptxft/opt_cost.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import ttnn

from . import autograd as ag

__all__ = ["LoraConfig", "lora_factors", "lora_linear", "AdamW", "af3_lr",
           "save_adapter", "load_adapter", "to_host", "to_device"]


# --------------------------------------------------------------------- host <-> device

def to_host(t, *, dtype=None):
    """Read a ttnn tensor to a float32 numpy array. One PCIe read."""
    import numpy as np
    import torch
    out = ttnn.to_torch(t).to(torch.float32).numpy()
    return out.astype(dtype) if dtype is not None else np.ascontiguousarray(out)


def to_device(arr, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Push a numpy array to the device, rounding to ``dtype``. One PCIe write."""
    import torch
    return ttnn.from_torch(torch.from_numpy(arr).to(torch.float32),
                           dtype=dtype, layout=layout, device=device)


# --------------------------------------------------------------------- LoRA

@dataclass
class LoraConfig:
    """``ttml.modules.LoraConfig``'s fields, minus the ones we have no mechanism for.

    ``lora_dropout`` is deliberately absent: the tape has no dropout op, and adding one
    would need a reproducible on-device RNG whose gradient is a mask. tt-train's SFT
    example runs 0.05 on 500 steps of Shakespeare; a handful of structures is a different
    regime and rank is the regulariser here.
    """

    rank: int = 8
    alpha: float = 16.0
    use_rslora: bool = False
    targets: tuple = ()

    @property
    def scaling(self) -> float:
        """``alpha/rank``, or ``alpha/sqrt(rank)`` under rank-stabilised LoRA."""
        return (self.alpha / math.sqrt(self.rank)) if self.use_rslora else (self.alpha / self.rank)


def lora_factors(in_features: int, out_features: int, cfg: LoraConfig, device, *,
                 dtype=ttnn.bfloat16, rng=None):
    """``(A, B)`` in ttnn's ``(in, out)`` layout, both taped and requiring gradients.

    tt-train stores torch-style ``A (rank, in)`` and ``B (out, rank)``; ttnn's linear takes
    ``(in, out)``, so these are ``A (in, rank)`` and ``B (rank, out)``. A is uniform on
    ``+/- 1/sqrt(in_features)`` (``_create_lora_A``, lora.py:56-60) and B is exactly zero
    (``_create_lora_B``, :64-67).
    """
    import numpy as np
    rng = np.random.default_rng() if rng is None else rng
    bound = 1.0 / math.sqrt(in_features)
    a = rng.uniform(-bound, bound, size=(in_features, cfg.rank)).astype(np.float32)
    b = np.zeros((cfg.rank, out_features), dtype=np.float32)
    return (ag.Tensor(to_device(a, device, dtype=dtype), requires_grad=True),
            ag.Tensor(to_device(b, device, dtype=dtype), requires_grad=True))


def lora_linear(x: ag.Tensor, w: ag.Tensor, a: ag.Tensor, b: ag.Tensor,
                bias: Optional[ag.Tensor] = None, *, scaling: float, config=None) -> ag.Tensor:
    """``linear(x, w, bias) + linear(linear(x, a), b) * scaling``.

    Composed entirely from the tape's own ``linear``, ``add`` and ``scale``, so dA and dB
    are produced by ``linear``'s already-gradchecked weight rule rather than by a new
    backward. ``w`` keeps ``requires_grad = False``, which prunes its branch in ``_tape``
    and so costs nothing to carry -- tt-train freezes it the same way
    (``lora.py:92``).
    """
    base = ag.linear(x, w, bias, config=config)
    down = ag.linear(x, a, config=config)
    up = ag.linear(down, b, config=config)
    return ag.add(base, ag.scale(up, scaling))


# --------------------------------------------------------------------- schedule

def af3_lr(step: int, lr: float, *, warmup_steps: int = 1000,
           decay_every_n_steps: int = 50000, decay_factor: float = 0.95) -> float:
    """Protenix's `AlphaFold3LRScheduler`, `protenix/utils/lr_scheduler.py:85-91`.

    Linear warmup to `lr` over `warmup_steps`, then a step decay of `decay_factor` every
    `decay_every_n_steps`. Upstream's defaults are lr 1.8e-3 with warmup 1000
    (`configs/configs_base.py:74-76`, and `train_demo.sh` runs lr 1e-3 warmup 2000).
    The warmup is not decoration on a randomly initialised network: Adam's first update
    has magnitude ~lr per element whatever the gradient is, so without it the first pass
    over the data moves every weight the full step size before the second moment has any
    history.
    """
    if step <= warmup_steps:
        return step / warmup_steps * lr
    return lr * (decay_factor ** (step // decay_every_n_steps))


# --------------------------------------------------------------------- optimizer

class AdamW:
    """AdamW with fp32 master weights, after ``adamw_full_precision.cpp``.

    ``params`` is a name -> ``autograd.Tensor`` dict, which is tt-train's
    ``serialization::NamedParameters``. Masters and moments are fp32 numpy on host; each
    step reads the gradient back, updates in fp32, and writes the weight down to the
    device dtype. For an adapter that is a few hundred KB of PCIe against a multi-second
    step, and it keeps the accumulation 24 mantissa bits above the bf16 weight the forward
    reads.

    Decoupled weight decay, i.e. ``theta -= lr * (mhat / (sqrt(vhat) + eps) + wd * theta)``,
    which is AdamW's whole point and what tt-train's kernel implements.
    """

    def __init__(self, params: Dict[str, ag.Tensor], *, lr: float = 3e-4,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01,
                 clip_norm: float = 10.0, schedule=None):
        import numpy as np
        self.params = params
        self.lr, self.eps, self.weight_decay = float(lr), float(eps), float(weight_decay)
        # Upstream clips at 10 (`configs/configs_base.py:80`, applied by
        # `torch.nn.utils.clip_grad_norm_` at `runner/train.py:572`). 0 disables it, which
        # is upstream's own switch. `schedule` is a callable step -> lr; None holds lr.
        self.clip_norm = float(clip_norm)
        self.schedule = schedule
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.steps = 0
        # Multiplicative beta powers rather than pow(beta, step): cheap, and exactly
        # reproducible across a reload because the powers themselves are checkpointed.
        self.beta1_pow, self.beta2_pow = 1.0, 1.0
        self.master = {n: to_host(t.value).astype(np.float32) for n, t in params.items()}
        # Kept for the CUMULATIVE update control. The per-step kept ratio answers
        # "did this one step survive the cast", which is the right question only when the
        # weight the forward reads IS the accumulator. With an fp32 master behind a bf16
        # device copy -- tt-train's arrangement -- a single step below bf16 spacing is
        # SUPPOSED to round to 0 or to a whole spacing, so the per-step ratio scatters far
        # from 1 while the run is working perfectly. What has to hold is that the device
        # weight has travelled as far as the master has, over the run.
        self.init_master = {n: v.copy() for n, v in self.master.items()}
        self.init_device = {n: to_host(t.value).astype(np.float32).reshape(
            self.master[n].shape) for n, t in params.items()}
        self.exp_avg = {n: np.zeros_like(v) for n, v in self.master.items()}
        self.exp_avg_sq = {n: np.zeros_like(v) for n, v in self.master.items()}

    def zero_grad(self) -> None:
        for t in self.params.values():
            t.grad = None

    def step(self) -> dict:
        """One update. Returns the per-parameter update magnitudes, for the control."""
        import numpy as np
        self.steps += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1 = 1.0 - self.beta1_pow
        bc2 = 1.0 - self.beta2_pow
        lr = self.lr if self.schedule is None else float(self.schedule(self.steps))
        # Global-norm clipping, computed once over every gradient before any of them is
        # applied. Per-parameter clipping would be a different algorithm: it changes the
        # DIRECTION of the update, not just its length.
        gnorm = self.grad_norm()
        clip = (min(1.0, self.clip_norm / gnorm)
                if (self.clip_norm > 0 and gnorm > 0) else 1.0)
        report = {}
        for name, t in self.params.items():
            if t.grad is None:
                continue
            g = to_host(t.grad).astype(np.float32).reshape(self.master[name].shape)
            if clip != 1.0:
                g = g * clip
            m = self.exp_avg[name]
            v = self.exp_avg_sq[name]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            theta = self.master[name]
            upd = lr * ((m / bc1) / (np.sqrt(v / bc2) + self.eps)
                        + self.weight_decay * theta)
            before = theta.copy()
            theta -= upd
            # The control the brief demands, measured rather than asserted: the step the
            # weight the FORWARD reads actually took, after rounding to the device dtype.
            # A master update that vanishes in the cast is an expensive no-op.
            dev_before = to_host(t.value).astype(np.float32).reshape(theta.shape)
            t.value = to_device(theta, t.value.device(), dtype=t.value.dtype)
            dev_after = to_host(t.value).astype(np.float32).reshape(theta.shape)
            want = float(np.linalg.norm(theta - before))
            kept = float(np.linalg.norm(dev_after - dev_before))
            report[name] = {"master_step": want, "device_step": kept,
                            "kept": (kept / want) if want > 0 else float("nan"),
                            "grad_norm": float(np.linalg.norm(g))}
        self.last_lr, self.last_clip, self.last_grad_norm = lr, clip, gnorm
        return report

    def displacement(self) -> dict:
        """How far the master and the device weight have moved since construction.

        ratio near 1 says every increment the master accumulated reached the weight
        the forward actually reads. This is the control that survives a bf16 device copy;
        the per-step one does not.
        """
        import numpy as np
        m = d = 0.0
        for n, t in self.params.items():
            m += float(np.sum((self.master[n] - self.init_master[n]) ** 2))
            cur = to_host(t.value).astype(np.float32).reshape(self.master[n].shape)
            d += float(np.sum((cur - self.init_device[n]) ** 2))
        m, d = math.sqrt(m), math.sqrt(d)
        return {"master": m, "device": d, "ratio": (d / m) if m > 0 else float("nan")}

    def grad_norm(self) -> float:
        """Global L2 norm of the gradients, for clipping and for the trajectory log."""
        import numpy as np
        tot = 0.0
        for t in self.params.values():
            if t.grad is not None:
                gg = to_host(t.grad).astype(np.float32)
                tot += float(gg.ravel() @ gg.ravel())
        return math.sqrt(tot)

    def state_dict(self) -> dict:
        return {"steps": self.steps, "lr": self.lr, "beta1": self.beta1,
                "beta2": self.beta2, "eps": self.eps,
                "weight_decay": self.weight_decay, "clip_norm": self.clip_norm,
                "beta1_pow": self.beta1_pow, "beta2_pow": self.beta2_pow}

    def load_state_dict(self, d: dict) -> None:
        for k in ("steps", "lr", "beta1", "beta2", "eps", "weight_decay", "clip_norm",
                  "beta1_pow", "beta2_pow"):
            if k in d:
                setattr(self, k, d[k])


# --------------------------------------------------------------------- checkpoints

_SPLIT = "|"


def save_adapter(path, opt: AdamW, *, meta: Optional[dict] = None) -> None:
    """Write the fp32 masters plus the Adam moments to one safetensors file.

    The MASTER is what gets saved, not the bf16 weight the device holds: the master is the
    authoritative value and the device copy is a rounded view of it. Saving the rounded
    view would lose exactly the low bits the master exists to keep, so a save/reload cycle
    would quietly decay the adapter.
    """
    from safetensors.numpy import save_file
    tensors = {}
    for name, arr in opt.master.items():
        tensors[f"master{_SPLIT}{name}"] = arr
        tensors[f"exp_avg{_SPLIT}{name}"] = opt.exp_avg[name]
        tensors[f"exp_avg_sq{_SPLIT}{name}"] = opt.exp_avg_sq[name]
    header = {"optimizer": json.dumps(opt.state_dict()),
              "tt_bio_adapter": "1"}
    if meta:
        header["meta"] = json.dumps(meta)
    save_file(tensors, str(path), metadata=header)


def load_adapter(path, params: Dict[str, ag.Tensor], device, *, opt: Optional[AdamW] = None):
    """Load an adapter into ``params`` (and optionally an optimizer's moments).

    Returns the saved metadata dict. Every name in ``params`` must be present in the file
    and no extras are tolerated: a silently partial load is how a reload reproduces the
    BASE model's loss and gets read as "the fine-tune did nothing".
    """
    from safetensors.numpy import load_file
    import numpy as np
    blob = load_file(str(path))
    have = {k.split(_SPLIT, 1)[1] for k in blob if k.startswith(f"master{_SPLIT}")}
    want = set(params)
    if have != want:
        raise ValueError(f"adapter/parameter mismatch: missing {sorted(want - have)}, "
                         f"unexpected {sorted(have - want)}")
    for name, t in params.items():
        arr = blob[f"master{_SPLIT}{name}"].astype(np.float32)
        t.value = to_device(arr, device, dtype=t.value.dtype)
        if opt is not None:
            opt.master[name] = arr.copy()
            opt.exp_avg[name] = blob[f"exp_avg{_SPLIT}{name}"].astype(np.float32).copy()
            opt.exp_avg_sq[name] = blob[f"exp_avg_sq{_SPLIT}{name}"].astype(np.float32).copy()
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    md = header.get("__metadata__", {})
    if opt is not None and "optimizer" in md:
        opt.load_state_dict(json.loads(md["optimizer"]))
    return json.loads(md["meta"]) if "meta" in md else {}
