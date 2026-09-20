"""``AdamW`` with fp32 master weights, the AF3 schedule, and the two controls.

Tier 2. Three things live in this module that a docstring somewhere else would not enforce.

**The optimizer refuses a bf16 master.** Not a warning. ``hallgrad-build`` measured only
**0.209** of an eps=0.01 step surviving a bf16 round-trip while the gradient still looked
healthy, so an update accumulated at bf16 stops moving the weight long before anything in the
loss curve says so. The master is the fix and it is only a fix if it is fp32, so a request for
anything else is refused at construction where it is cheap to act on.

**The step control is on the CUMULATIVE displacement ratio, never per-step.** With an fp32
master behind a bf16 device copy, a single step below bf16 spacing is SUPPOSED to round to
zero -- the per-step kept ratio scatters far from 1 on a run that is working perfectly, so a
per-step assertion misfires on health. What has to hold is that the device weight has
travelled as far as the master has, over the run. ``displacement()`` reports it and
``check_displacement()`` asserts ``0.9 < ratio < 1.1``.

**The all-reduce is inside ``step()``, not in a callback.** ``step()`` raises
:class:`~tt_bio.train.mesh.UnreducedGradients` when the data-parallel axis is wider than one
chip and no axis was given, because forgetting it trains N diverging replicas behind N loss
curves that all look right. See ``tt_bio/train/mesh.py`` for why a callback cannot warn.

The patterns are tt-train's, copied rather than linked because ``ttml`` cannot be built
against the ttnn 0.68.0 wheel. ``optimizers/adamw_full_precision.cpp``: fp32 masters built
once from the bf16 weights (:28-30) with fp32 ``exp_avg``/``exp_avg_sq`` (:33-42), the bf16
weight the forward reads written back by typecasting the master down each step (:97-99), and
beta powers carried multiplicatively (:64-65) rather than recomputed as ``pow(beta, step)``.

Masters and moments live on the HOST in fp32 rather than in DRAM. ``ttnn.moreh_adamw`` takes
``param_in`` as bfloat16 or bfloat8_b only, so the device cannot express an fp32 master at all
and the placement is forced rather than picked. What was measured is the price of it at adapter
scale: ``perf/ptxft/optcheck.py --moreh`` times the host step at a median 7.83 ms for one
block's adapter (72 tensors, 204,032 elements, 0.82 MB) at 1350 MHz, against a step measured in
seconds, so the PCIe round trip is not worth optimising away. That is an adapter reading and not
a general one. The script reproduces the number, it does not store it: the run it came from is
``ptxft-build``'s.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional

from .mesh import Axis, UnreducedGradients
from .tensors import to_device, to_host

if TYPE_CHECKING:
    from .. import autograd as ag

__all__ = ["AdamW", "af3_lr", "DISPLACEMENT_BAND"]


# The band the cumulative ratio has to stay inside. Wide enough that bf16 rounding of a
# healthy run does not trip it, tight enough that a master whose updates are not reaching the
# device weight does. Both failures it exists to catch are order-of-magnitude, not percent.
DISPLACEMENT_BAND = (0.9, 1.1)


def af3_lr(step: int, lr: float, *, warmup_steps: int = 1000,
           decay_every_n_steps: int = 50000, decay_factor: float = 0.95,
           base_lr: float = 0.0, plateau_until: int | None = None) -> float:
    """The AlphaFold-family learning rate, in the two closed forms upstreams actually use.

    Both are a linear warmup to `lr` over `warmup_steps`, offset by `base_lr`, followed by a
    decay of `decay_factor` every `decay_every_n_steps`. They differ in what sits between,
    and `plateau_until` is the whole of the difference.

    `plateau_until=None` is Protenix's `AlphaFold3LRScheduler`,
    `protenix/utils/lr_scheduler.py:85-91`. No plateau, and the decay exponent counts from
    step zero, so the first decay lands at `decay_every_n_steps`. Upstream's defaults are lr
    1.8e-3 with warmup 1000 (`configs/configs_base.py:74-76`, and `train_demo.sh` runs lr
    1e-3 warmup 2000).

    `plateau_until=S` is the AlphaFold 2 supplement schedule, which is what OpenFold3 ships
    as `AlphaFoldLRScheduler` (`openfold3/core/utils/lr_schedulers.py`; defaults base_lr 0.0,
    max_lr 1e-3, warmup_no_steps 1000, start_decay_after_n_steps 50000, decay_every_n_steps
    50000, decay_factor 0.95). The rate holds flat at `lr` until step `S`, and the decay
    exponent counts from `S` and starts at 1, so the first decay lands immediately after the
    plateau instead of a whole period later. Passing `plateau_until` is not a per-model
    branch: it is the one parameter the two schedules disagree on, and naming it is what lets
    a single function carry both.

    The warmup is not decoration on a randomly initialised network: Adam's first update has
    magnitude ~lr per element whatever the gradient is, so without it the first pass over the
    data moves every weight the full step size before the second moment has any history.
    """
    if step <= warmup_steps:
        return base_lr + step / warmup_steps * lr
    if plateau_until is None:
        return lr * (decay_factor ** (step // decay_every_n_steps))
    if step <= plateau_until:
        return lr
    return lr * (decay_factor ** ((step - plateau_until) // decay_every_n_steps + 1))

# --------------------------------------------------------------------- optimizer

def _require_fp32_master(dtype) -> None:
    """Refuse anything but an fp32 master, at construction.

    Measured, not defensive: only **0.209** of an eps=0.01 step survived a bf16 round-trip in
    ``hallgrad-build`` while the gradient's own norm still looked healthy. So the failure has
    no signature in the loss, the gradient norm or the step count -- the weight simply stops
    moving -- and the only place it is cheap to catch is before the run starts.
    """
    name = getattr(dtype, "name", None) or str(dtype)
    name = name.rsplit(".", 1)[-1].lower()
    if name in ("float32", "fp32", "f32"):
        return
    raise ValueError(
        f"master_dtype={dtype!r} refused; AdamW keeps an fp32 master and only an fp32 master. "
        f"Measured: 0.209 of an eps=0.01 step survives a bfloat16 round-trip, so an update "
        f"accumulated at {name} stops moving the weight while the gradient still reads "
        f"healthy. The device copy the forward reads stays bfloat16 -- that is the point of "
        f"having a master -- and it is written by casting the master down each step")


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
                 clip_norm: float = 10.0, schedule=None,
                 data_parallel: Optional[Axis] = None, master_dtype="float32"):
        import numpy as np
        _require_fp32_master(master_dtype)
        self.master_dtype = "float32"
        # The DP axis, not a device count and not a flag. `step()` reduces across it, and a
        # width of 1 makes that a no-op rather than a special case.
        self.data_parallel = data_parallel
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

    def step(self, *, replicas=None) -> dict:
        """One update. Returns the per-parameter update magnitudes, for the control.

        ``replicas`` is the per-chip gradient for each parameter,
        ``{name: [g_chip0, g_chip1, ...]}``, when the DP axis is wider than one chip. It is
        reduced here. Omitting it on a wide axis raises
        :class:`~tt_bio.train.mesh.UnreducedGradients` rather than stepping on one replica's
        gradient, because that failure is invisible in every metric a user watches.
        """
        import numpy as np
        self._reduce(replicas)
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
        self.last_report = report
        return report

    def _reduce(self, replicas) -> None:
        """Sum each parameter's gradient across the DP axis, in place on the tape.

        Deliberately not a mean: the divisor is the global batch the caller pinned, and
        dividing by chip count here is exactly the substitution ``accelerate`` makes at
        ``data_loader.py:347-348`` that turns one recipe into a different one per box.
        """
        width = 1 if self.data_parallel is None else self.data_parallel.width
        if width == 1:
            if replicas:
                raise ValueError(
                    "per-chip gradients were passed but the optimizer has no data-parallel "
                    "axis wider than one chip. Hand it data_parallel=mesh.axis('dp')")
            return
        if not replicas:
            raise UnreducedGradients(
                f"the data-parallel axis {self.data_parallel} is {width} chips wide and "
                f"step() was given no per-chip gradients to reduce. Stepping on one "
                f"replica's gradient trains {width} diverging models, and every one of them "
                f"has a loss curve that falls. Pass replicas={{name: [g_per_chip, ...]}}")
        missing = [n for n, t in self.params.items() if t.grad is not None and n not in replicas]
        if missing:
            raise UnreducedGradients(
                f"these parameters have a gradient but no per-chip entry: {sorted(missing)}. "
                f"A partially reduced step is the same failure as an unreduced one")
        # One call for the whole parameter set, not one per parameter. On a mesh device the
        # two are the same work; across the launcher's processes the per-parameter form pays
        # a barrier per adapter tensor, which is dozens per step.
        for name, summed in self.data_parallel.reduce_all(replicas).items():
            self.params[name].grad = summed

    def check_displacement(self, band=DISPLACEMENT_BAND) -> dict:
        """Assert the device weight moved as far as the master did. The step control.

        On the CUMULATIVE ratio, which is the only form that survives a bf16 device copy: a
        single step below bf16 spacing is supposed to round to zero, so the per-step kept
        ratio scatters far from 1 on a healthy run and asserting on it fails the good case.
        Over a run the two displacements have to agree, and a master accumulating updates
        that never reach the weight the forward reads is what this catches.

        Call it after enough steps to have moved: on step 0 there is no displacement and the
        ratio is nan, which is reported rather than passed.
        """
        d = self.displacement()
        lo, hi = band
        r = d["ratio"]
        if self.steps == 0:
            raise RuntimeError("nothing has stepped yet; there is no displacement to check")
        if not (r == r):  # nan: master has not moved at all
            raise AssertionError(
                f"the master has not moved after {self.steps} steps (displacement "
                f"{d['master']:.3e}), so nothing was learned. Check that the loss reached "
                f"the parameters: a frozen tensor accumulates no gradient and a pruned "
                f"branch produces none")
        if not (lo < r < hi):
            raise AssertionError(
                f"cumulative displacement ratio {r:.4f} is outside {band}: the master moved "
                f"{d['master']:.4e} and the device weight the forward reads moved "
                f"{d['device']:.4e}. Below the band the updates are dying in the cast to "
                f"{self.params and next(iter(self.params.values())).value.dtype}; above it "
                f"the device weight is being written by something other than this optimizer")
        return d

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

