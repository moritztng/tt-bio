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
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

from .mesh import Axis, UnreducedGradients
from .tensors import to_device, to_host

if TYPE_CHECKING:
    from .. import autograd as ag

__all__ = ["AdamW", "Muon", "EMA", "linear_warmup", "af3_lr", "DISPLACEMENT_BAND",
           "round_to_device_dtype", "unit_roundoff"]


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
    # `warmup_steps=0` is "no warmup", not a division by zero. A short run has to be able to
    # say it: at 30 steps the 1000-step default leaves the rate at 9e-6 of 3e-4, the weights
    # barely move, and two arms compared over those steps agree because neither trained.
    if warmup_steps and step <= warmup_steps:
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


def _write(t, theta) -> None:
    """Hand a stepped master to its leaf: a device weight is re-uploaded in its own dtype, a
    `bridge.HostParam` (host torch glue) is updated in place."""
    if hasattr(t, "write"):
        t.write(theta)
    else:
        t.value = to_device(theta, t.value.device(), dtype=t.value.dtype)


def round_to_device_dtype(arr, dtype):
    """``arr`` as the device would hold it, computed on the host. No PCIe.

    ``to_device`` writes ``torch.from_numpy(arr).to(float32)`` through
    ``ttnn.from_torch(dtype=...)``, so for a dtype torch can express the value that lands on
    the card is exactly this cast. Returns ``None`` for a dtype it cannot reproduce
    (``bfloat8_b`` is a block format with a shared exponent and has no torch scalar type),
    and the caller falls back to reading the card.

    This exists because ``step()`` used to read every weight back TWICE per step -- once
    before the write and once after -- purely to report how much of the update survived the
    cast. At OF3T's census (3,152 parameters, 381,302,188 elements) those two reads are
    0.66 s of PCIe against 0.36 s for the same two casts on host
    (``perf/of3t_p10host/out/optsplit.json``).
    """
    import torch
    name = str(getattr(dtype, "name", dtype)).rsplit(".", 1)[-1].lower()
    t = {"bfloat16": torch.bfloat16, "float32": torch.float32,
         "uint32": None, "bfloat8_b": None}.get(name, None)
    if t is None:
        return None
    if t is torch.float32:
        return arr
    return torch.from_numpy(arr).to(t).to(torch.float32).numpy()


class _Moments(dict):
    """``exp_avg`` / ``exp_avg_sq``: a name -> array map whose entries appear at first use.

    A missing moment is a zero moment, and a zero moment is ``np.zeros_like(master[name])``,
    so building it on the first read gives the same array the eager construction gave --
    later, and only for a parameter something actually asked about. ``step()`` mutates what
    it gets in place (``m *= beta1``), so the entry is cached on materialisation and the same
    array comes back every step.

    A dict subclass rather than a lazy property because the readers are spread across the
    package and all of them want a plain mapping: ``save_adapter`` reads by name
    (``tt_bio/train/checkpoint.py::save_adapter``) and materialises what it is about to write,
    ``load_adapter`` assigns by name (``tt_bio/train/checkpoint.py::load_adapter``), and a checkpoint resume replaces the
    attribute outright. None of them needs to know this class exists.

    Why it is worth having at all: ``np.zeros_like`` is ``empty_like`` followed by
    ``copyto(0)``, so it WRITES every page and an eagerly built moment is fully resident, not
    a cheap untouched mapping. Measured at the crop-384 training census -- 3152 parameters,
    381,302,188 elements -- the pair cost **2.82 GiB** standing through a forward and a
    backward that never read them (``perf/of3t_optorder/``).
    """

    __slots__ = ("_master",)

    def __init__(self, master):
        super().__init__()
        self._master = master

    def __missing__(self, name):
        import numpy as np
        v = np.zeros_like(self._master[name])
        self[name] = v
        return v



def unit_roundoff(dtype) -> float:
    """Half an ulp, relative, of a device weight's dtype under round-to-nearest.

    bfloat16 keeps 7 explicit mantissa bits, so 2**-8. Matched on the lowercased name because
    ttnn prints ``DataType.BFLOAT16`` and torch ``torch.bfloat16``; a case-sensitive match
    read every ttnn weight as fp32 and failed every short run at the default warmup.
    """
    return 2.0 ** -8 if "bfloat16" in str(dtype).lower() else 2.0 ** -24

def linear_warmup(lr: float, steps: int):
    """``LambdaLR(min((step + 1) / steps, 1))`` as RFD4-Proteina builds it (`proteina.py:475`).

    Read at the count of COMPLETED steps, as ``AdamW.step`` reads every schedule, so the first
    update runs at ``lr / steps``: LambdaLR evaluates its lambda at 0 on construction.
    """
    return lambda step: lr * min((step + 1) / max(steps, 1), 1.0)


@dataclass(frozen=True)
class Muon:
    """Route 2-D weights to Muon inside ``AdamW``; everything else stays on Adam.

    RFD4-Proteina trains this way (`utils/optim_utils.py::build_optimizers`): every trainable
    2-D parameter whose name matches none of ``exclude`` takes Muon's update, the rest (norms,
    biases, embeddings, output heads) Adam's. Defaults are ``torch.optim.Muon``'s, which
    upstream does not override, so **Muon's weight decay is 0.1** while its Adam has none.

    Upstream clips each optimizer's gradients separately (`proteina.py`, one
    ``clip_gradients`` per optimizer), so the two groups get their own global norm here too.
    A parameter with no gradient is skipped outright, as ``torch.optim.Muon`` skips it.

    The Newton-Schulz iteration runs in bfloat16 on host torch with upstream's own op order
    (`utils/grouped_muon.py::_orthogonalize`); it is part of the update rule, not a precision
    choice of ours. ``adjust_lr_fn="original"`` scales by ``sqrt(max(1, out / in))`` and so
    needs the torch (out, in) orientation: set ``rows_are_outputs=False`` for weights stored
    (in, out), as ttnn linears are. ``match_rms_adamw`` (the RFD4 release) is symmetric.
    """

    exclude: tuple = ()
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: tuple = (3.4445, -4.775, 2.0315)
    ns_steps: int = 5
    eps: float = 1e-7
    weight_decay: float = 0.1
    adjust_lr_fn: Optional[str] = "match_rms_adamw"
    rows_are_outputs: bool = True
    schedule: Optional[Callable] = None

    def takes(self, name: str, shape) -> bool:
        return (len(shape) == 2 and min(shape) > 0
                and not any(re.search(p, name) for p in self.exclude))

    def ratio(self, shape) -> float:
        if self.adjust_lr_fn in (None, "original"):
            out, inp = shape if self.rows_are_outputs else shape[::-1]
            return math.sqrt(max(1.0, out / inp))
        if self.adjust_lr_fn == "match_rms_adamw":
            return 0.2 * math.sqrt(max(shape))
        raise ValueError(f"unknown Muon adjust_lr_fn {self.adjust_lr_fn!r}")

    def orthogonalize(self, update):
        """Upstream's bf16 Newton-Schulz on one matrix; float32 numpy in and out."""
        import torch
        if self.ns_steps >= 100:
            raise ValueError("Muon ns_steps must be below 100")
        a, b, c = self.ns_coefficients
        x = torch.from_numpy(update)[None].bfloat16()
        flip = x.shape[-2] > x.shape[-1]
        if flip:
            x = x.transpose(-1, -2)
        x.div_(x.norm(dim=(-2, -1), keepdim=True).clamp(min=self.eps))
        for _ in range(self.ns_steps):
            gram = torch.bmm(x, x.transpose(-1, -2))
            x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
        x = x.transpose(-1, -2) if flip else x
        return x[0].float().numpy()


class EMA:
    """The exponential moving average RFD4-Proteina keeps (`utils/simple_ema.py`, decay 0.999).

    torch's ``AveragedModel`` with ``get_ema_multi_avg_fn``: the first ``update`` COPIES the
    weights (it runs after the first optimizer step, so the average starts from step 1, not
    from initialisation), every later one is ``ema += (1 - decay) * (w - ema)``. Kept over the
    fp32 master, which is the weight upstream averages.
    """

    def __init__(self, decay: float = 0.999):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.n_averaged = 0
        self.shadow: Dict[str, "np.ndarray"] = {}

    def update(self, master: Dict[str, "np.ndarray"]) -> None:
        import numpy as np
        if self.n_averaged == 0:
            self.shadow = {n: np.array(v, np.float32, copy=True) for n, v in master.items()}
        else:
            w = np.float32(1.0 - self.decay)
            for n, v in master.items():
                e = self.shadow[n]
                e += w * (v - e)
        self.n_averaged += 1


class AdamW:
    """AdamW with fp32 master weights, after ``adamw_full_precision.cpp``.

    ``params`` is a name -> ``autograd.Tensor`` dict, which is tt-train's
    ``serialization::NamedParameters``. Masters and moments are fp32 numpy on host; each
    step reads the gradient back, updates in fp32, and writes the weight down to the
    device dtype. For an adapter that is a few hundred KB of PCIe against a multi-second
    step, and it keeps the accumulation 24 mantissa bits above the bf16 weight the forward
    reads.

    **Construction allocates the master and one copy of the initial weights, and nothing
    else.** The moments are built by the first ``step()``, which runs after the backward, so
    they are absent from the forward and the backward rather than resident and unread. At
    full-model scale that is the difference between 7.04 GiB and 2.82 GiB of host memory
    standing under the tape; see ``_Moments`` and ``perf/of3t_optorder/``.

    Decoupled weight decay, i.e. ``theta -= lr * (mhat / (sqrt(vhat) + eps) + wd * theta)``,
    which is AdamW's whole point and what tt-train's kernel implements.
    """

    #: ``{name: bool mask}`` of elements no step moves. `hold` replaces it, never mutates it.
    held: Dict[str, "np.ndarray"] = {}

    def __init__(self, params: Dict[str, ag.Tensor], *, lr: float = 3e-4,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01,
                 clip_norm: float = 10.0, schedule=None,
                 data_parallel: Optional[Axis] = None, master_dtype="float32",
                 muon: Optional[Muon] = None, shard: bool = False):
        import numpy as np
        _require_fp32_master(master_dtype)
        # ZeRO-1 across the launcher's ranks: each rank steps only the parameters it owns
        # (whole tensors, `launcher.shard_edges`) and builds moments only for those, then the
        # stepped masters are gathered so every rank still holds the same full master and
        # writes the same device weights. Host RAM for moments and host-Adam time divide by
        # the width; see `_step_sharded`.
        self.shard = bool(shard)
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
        #: The largest learning rate a step has run at. The AlphaFold 2 schedule starts at
        #: lr(0) = 0, so a one-step run moves nothing and is correct to.
        self.peak_lr = 0.0
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
        # ONE copy, where there were two. `init_master` and `init_device` were built from the
        # same `t.value`, three lines apart, with nothing between them that writes a
        # parameter -- `to_host(v).astype(float32)` twice over unchanged bfloat16 is the same
        # bytes, and the `.reshape(master[n].shape)` the second one carried is reshaping an
        # array to its own shape. So they were a duplicate, and at the crop-384 census (3152
        # tensors, 381,302,188 elements) the duplicate was **1.40 GiB** measured, standing
        # for the whole run (`perf/of3t_optorder/out/`).
        # `init` is the master AT CONSTRUCTION, and the master at construction is the device
        # weight promoted to fp32 -- `to_host` widens, it does not add information. So a
        # bfloat16 parameter's `init` carries 8 mantissa bits in a 24-bit box, and holding it
        # at the parameter's own dtype is LOSSLESS, not an approximation: `displacement()`
        # widens it back and gets the same bytes. At the crop-384 census that is 1.42 GiB
        # down to 0.71 GiB standing for the whole run, beside a master that cannot be
        # narrowed because it is the accumulator.
        self.init = {n: self._narrow(v, params[n].value.dtype)
                     for n, v in self.master.items()}
        # `exp_avg` and `exp_avg_sq` are built at the FIRST STEP, not here. Only `step()`
        # reads or writes them, and `step()` runs after the backward, so eager construction
        # put 2.82 GiB of zeros under the forward and the backward for nothing. Eager is not
        # free the way an untouched allocation would be: `np.zeros_like` is `empty_like`
        # followed by `copyto(0)`, which writes every page, so the moments were fully
        # resident from construction rather than faulted in on demand.
        self.exp_avg = _Moments(self.master)
        self.exp_avg_sq = _Moments(self.master)
        # Muon's routing, decided once from names and shapes. Empty when `muon` is None, and
        # then every branch below is the plain AdamW it always was.
        self.muon = muon
        self.muon_names = frozenset(n for n, v in self.master.items()
                                    if muon is not None and muon.takes(n, v.shape))
        self.momentum_buffer = _Moments(self.master)
        # The per-sample clipping accumulator. Empty means the caller clips the batch once;
        # non-empty means `clip_and_accumulate` ran per sample and `step()` uses what it
        # summed rather than clipping a second time. `recipes.py`'s loop fills it, which is
        # upstream's shape. `participation` is their per-parameter active count
        # (`parameter_participation_counts`) and `step()` divides by it, because a parameter
        # absent from most samples must not be averaged by the batch size.
        self.accum: Dict[str, "np.ndarray"] = {}
        self.participation: Dict[str, int] = {}
        self.accum_count = 0

    def hold(self, masks: dict) -> int:
        """Never move the masked elements of each named parameter; returns how many.

        For elements the build filled with constants rather than checkpoint values: the model
        has no such parameter, so their gradient is dropped before the clip norm, as it would
        be for a weight that did not exist, and neither the step nor the decay touches them.
        """
        import numpy as np
        self.held = {n: np.asarray(m, bool).reshape(self.master[n].shape)
                     for n, m in masks.items() if n in self.master}
        return int(sum(m.sum() for m in self.held.values()))

    def _grad(self, name, t):
        import numpy as np
        g = to_host(t.grad).astype(np.float32).reshape(self.master[name].shape)
        if name in self.held:
            g[self.held[name]] = 0.0
        return g

    @staticmethod
    def _narrow(arr, dtype):
        """``arr`` at the parameter's own device dtype, for storage. Lossless by construction.

        Only ever called on the master at construction, which came from the device weight, so
        there is nothing below `dtype` to lose. `_widen` is the inverse.
        """
        import torch
        name = str(getattr(dtype, "name", dtype)).rsplit(".", 1)[-1].lower()
        if name != "bfloat16":
            return arr
        return torch.from_numpy(arr).to(torch.bfloat16)

    @staticmethod
    def _widen(v):
        import numpy as np
        if isinstance(v, np.ndarray):
            return v
        return v.to(__import__("torch").float32).numpy()

    def zero_grad(self) -> None:
        for t in self.params.values():
            t.grad = None

    def step(self, *, replicas=None, disabled=()) -> dict:
        """One update. Returns the per-parameter update magnitudes, for the control.

        ``replicas`` is the per-chip gradient for each parameter,
        ``{name: [g_chip0, g_chip1, ...]}``, when the DP axis is wider than one chip. It is
        reduced here. Omitting it on a wide axis raises
        :class:`~tt_bio.train.mesh.UnreducedGradients` rather than stepping on one replica's
        gradient, because that failure is invisible in every metric a user watches.
        """
        import numpy as np
        if self._sharded():
            return self._step_sharded(replicas, set(disabled))
        self._reduce(replicas)
        disabled = set(disabled)
        per_sample = bool(self.accum)
        if per_sample:
            # Upstream's `_sync_and_average_grads` divides each parameter's accumulated
            # gradient by ITS OWN participation count (`grad_manager.py:225-232`), not by the
            # batch size. The counts genuinely differ: their runner disables the confidence
            # head on any sample whose confidence weight is zero, which `initial_training.yml`
            # does on 4 of its 5 datasets, so a parameter absent from most samples would
            # otherwise be averaged by a divisor it never contributed to. Adam does not cancel
            # it -- a uniform per-tensor scaling it does cancel, and this is not uniform.
            for name, count in self.participation.items():
                if count > 1:
                    self.accum[name] = self.accum[name] / np.float32(count)
        # The schedule is read BEFORE the counter moves, and the order is the whole of it.
        # Upstream steps the optimizer and THEN the scheduler (`runner.py:464-465`), and
        # `AlphaFoldLRScheduler` is built with `last_epoch=-1`, so `_LRScheduler.__init__`
        # steps it once to 0 before training starts. Their k-th update therefore runs at
        # `lr(k-1)` and their first runs at `lr(0)`, which the AF3 warmup makes exactly 0.
        # `self.steps` counts COMPLETED steps, so reading here is that same order. Reading
        # after the increment put every step of ours one rung further along the warmup than
        # theirs and made our first update non-zero where theirs is identically zero.
        lr = self.lr if self.schedule is None else float(self.schedule(self.steps))
        self.steps += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1 = 1.0 - self.beta1_pow
        bc2 = 1.0 - self.beta2_pow
        # Global-norm clipping, computed once over every gradient before any of them is
        # applied. Per-parameter clipping would be a different algorithm: it changes the
        # DIRECTION of the update, not just its length.
        # Under per-sample clipping every sample was already clipped as it arrived, so the
        # norm here is the accumulated one and is reported rather than applied -- clipping the
        # sum again would be a third algorithm, neither theirs nor ours.
        read = {}
        gnorm = (math.sqrt(sum(float(v.ravel() @ v.ravel()) for v in self.accum.values()))
                 if per_sample else self.grad_norm(disabled, keep=read))
        clip = clip_muon = 1.0 if per_sample else self.clip_coef(gnorm)
        if self.muon_names:
            if per_sample:
                raise NotImplementedError(
                    "per-sample clipping with Muon: no upstream trains that way, so there is "
                    "no reference to be right against")
            # One norm per optimizer, as upstream clips each of its two optimizers alone.
            sq = lambda names: math.sqrt(sum(float(v.ravel() @ v.ravel())
                                             for n, v in read.items() if n in names))
            gnorm_muon = sq(self.muon_names)
            gnorm = sq(read.keys() - self.muon_names)
            clip, clip_muon = self.clip_coef(gnorm), self.clip_coef(gnorm_muon)
            self.last_grad_norm_muon = gnorm_muon
            lr_muon = (self.lr if self.muon.schedule is None
                       else float(self.muon.schedule(self.steps - 1)))
        report = {}
        writes_skipped = 0
        for name, t in self.params.items():
            # EVERY parameter takes a step, including one no sample of this step activated.
            # That is upstream's behaviour and it is not incidental: `sync_and_average_grads`
            # assigns `param.grad = self.grad_accumulator[name].clone()` for every parameter
            # it manages (`grad_manager.py:296-297`), the accumulator was zeroed by
            # `reset_accumulator`, and `_sync_and_average_grads` explicitly `zero_()`s a
            # parameter whose global participation count is 0 (`grad_manager.py:237-239`)
            # rather than leaving it out. Their `torch.optim.Adam` (`configure_optimizers`,
            # `runner.py:845-850`) then
            # steps it on a zero gradient, so `m` and `v` DECAY by beta1/beta2 and the weight
            # moves on momentum alone. Skipping it left the moments frozen and the weight
            # where it was -- measured at 1.10e-03 relative against their own optimizer,
            # permanent for the rest of the run, where the arm with no such step reads
            # 2.07e-08 (`perf/of3t_optsem/`).
            #
            # A zero gradient is NOT the same as no update, and it is also not always a
            # different one. With `m` and `v` still zero the update is `0 / (0 + eps)`, so a
            # parameter that has never received a gradient does not move under either
            # optimizer -- measured bit-identical over 20 steps. The divergence needs prior
            # momentum, which is exactly the confidence head's case: it trains on the
            # confidence-weighted dataset and then meets a batch drawn wholly from the four
            # that zero those weights.
            #
            # Nothing else in the step changes. The zeros are synthesised HERE rather than
            # written into `self.accum`, so the reported gradient norm is computed over the
            # same set it always was; adding zeros to a sum of squares would not move it
            # either, but not touching it keeps that true by construction rather than by
            # arithmetic. Clipping is per-sample under `per_sample` and a no-op on zeros in
            # any case. Weight decay is the one term that is not a no-op on a zero gradient,
            # and upstream builds a plain `torch.optim.Adam` with no decay at all, which is
            # why `train_loop` ships `weight_decay=0.0`; at a non-zero decay ours applies it
            # decoupled, as `torch.optim.AdamW` does and `torch.optim.Adam(weight_decay=...)`
            # does not.
            g = None if name in disabled else (
                self.accum.get(name) if per_sample else (
                    None if t.grad is None else read.pop(name)))
            theta = self.master[name]
            new = None
            if name in self.muon_names:
                if g is None:
                    upd, g = np.zeros_like(theta), np.zeros_like(theta)
                else:
                    g = g * clip_muon if clip_muon != 1.0 else g
                    new = self._muon_new(name, g, theta, lr_muon)
                    upd = theta - new
            else:
                if g is None:
                    g = np.zeros_like(self.master[name])
                elif clip != 1.0:
                    g = g * clip
                m = self.exp_avg[name]
                v = self.exp_avg_sq[name]
                m *= self.beta1
                m += (1.0 - self.beta1) * g
                v *= self.beta2
                v += (1.0 - self.beta2) * (g * g)
                upd = lr * ((m / bc1) / (np.sqrt(v / bc2) + self.eps)
                            + self.weight_decay * theta)
            if name in self.held:
                upd[self.held[name]] = 0.0
                if new is not None:
                    new = theta - upd     # a held element keeps its value under Muon too
            # The control the brief demands, measured rather than asserted: the step the
            # weight the FORWARD reads actually took, after rounding to the device dtype.
            # A master update that vanishes in the cast is an expensive no-op.
            #
            # Both sides of it are computed HERE, on the host. `dev_before` is the master
            # rounded down, which is what the card holds because the only writer of
            # `t.value` is this line one step earlier; `dev_after` is the same cast of the
            # updated master. Reading them back cost two PCIe traversals of the whole
            # parameter set every step -- 0.66 s of 5.055 s at OF3T's census -- to learn
            # something the host already knows. The CUMULATIVE control is unaffected and
            # still reads the card: `displacement()` calls `to_host(t.value)`, so a device
            # weight written by something other than this optimizer is still caught, by the
            # control this module says is the real one.
            #
            # A float32 device dtype makes the cast the IDENTITY, and the helper hands back
            # the master array itself rather than a copy -- deliberately, so the common case
            # allocates nothing. `theta -= upd` then moves the array BOTH names point at, so
            # comparing dev_after against dev_before would be comparing theta with itself:
            # every update would read as rounded away, every write would be skipped, and a
            # float32 parameter would silently stop reaching the card while the report said
            # `kept: 0.0`. Identity is what separates the two regimes here, not equality.
            dev_before = round_to_device_dtype(theta, t.value.dtype)
            identity_cast = dev_before is theta
            if new is None:
                theta -= upd
            else:
                theta[...] = new      # torch's own op order, not theta - (theta - new)
            # `norm(upd)` IS `norm(theta - before)`: `theta -= upd` is the only thing that
            # moved it. Taking it off the update drops a full-census copy of every parameter
            # per step -- 1.42 GiB of allocation at OF3T's census, 0.361 s against 0.032 s.
            want = float(np.linalg.norm(upd))
            if dev_before is None:
                # A dtype torch cannot express. Pay the round trip rather than guess.
                dev_before = to_host(t.value).astype(np.float32).reshape(theta.shape)
                _write(t, theta)
                dev_after = to_host(t.value).astype(np.float32).reshape(theta.shape)
                kept = float(np.linalg.norm(dev_after - dev_before))
            elif identity_cast:
                # Nothing can round away, so the device step is the master step exactly and
                # the write always happens.
                _write(t, theta)
                kept = want
            else:
                dev_after = round_to_device_dtype(theta, t.value.dtype)
                if np.array_equal(dev_after, dev_before):
                    # The whole update rounded away. The tensor already on the card is the
                    # one `to_device` would build, bit for bit, so the write is skipped
                    # rather than repeated -- an exactness, not an approximation. This is the
                    # COMMON case during warmup, where `af3_lr` puts every step under bf16
                    # spacing: the module's own 20-step arm measured a cumulative ratio of
                    # 0.810 for exactly this reason.
                    writes_skipped += 1
                else:
                    _write(t, theta)
                kept = float(np.linalg.norm(dev_after - dev_before))
            report[name] = {"master_step": want, "device_step": kept,
                            "kept": (kept / want) if want > 0 else float("nan"),
                            "grad_norm": float(np.linalg.norm(g))}
        self.last_lr, self.last_clip, self.last_grad_norm = lr, clip, gnorm
        self.peak_lr = max(self.peak_lr, float(lr))
        self.last_per_sample = per_sample
        # How many parameters this step did not write back because the update rounded away
        # entirely. Reported rather than silent: a step that skips every write is either a
        # warmup working as designed or an optimizer that has stopped, and the two are told
        # apart by `displacement()`, not by this counter.
        self.last_writes_skipped = writes_skipped
        self.last_report = report
        self.accum, self.participation, self.accum_count = {}, {}, 0
        return report

    def _step_sharded(self, replicas, disabled) -> dict:
        """One step with the optimizer state sharded across the DP ranks (ZeRO-1).

        reduce-scatter the gradient on tensor-aligned edges, so this rank receives the summed
        gradient of exactly the parameters it owns; take the global clip norms (one per
        optimizer group, as the replicated step does) from every rank's sums of squares; step
        the owned masters, Adam or Muon, with state built lazily so only for those; all-gather
        the stepped masters and write every device weight from them. Muon orthogonalises whole
        matrices, and every matrix lives on exactly one rank, so each rank runs Newton-Schulz
        on its own share and none is split. Every rank applies the same gathered bytes, so the
        master-hash invariant still holds. The norms are summed per rank and then across
        ranks, so they can differ from the replicated step's in the last bits: identical
        across ranks, not bit-identical to the unsharded step. Per-sample accumulation is not
        supported here yet and refuses.
        """
        import numpy as np
        ax = self.data_parallel
        if self.accum:
            raise NotImplementedError("shard=True with per-sample clip_and_accumulate")
        if not replicas:
            raise UnreducedGradients(
                f"the data-parallel axis {ax} is {ax.width} chips wide and step() was given "
                f"no per-chip gradients to reduce")
        names, offs, edges, owned = self._shard_layout()
        me = ax.dp_rank
        # Filled in place in the slot the axis publishes, so the send is not another copy.
        # The slot holds an earlier step's bytes, so every element is written, absent ones too.
        flat = ax.send_buffer(int(offs[-1]))
        absent = set()
        for i, n in enumerate(names):
            got = replicas.get(n)
            if got is None or n in disabled:
                absent.add(n)
                flat[offs[i]:offs[i + 1]] = 0.0
                continue
            g = np.asarray(to_host(list(got)[0]), np.float32).reshape(-1)
            if n in self.held:
                g = g.copy(); g[self.held[n].reshape(-1)] = 0.0
            flat[offs[i]:offs[i + 1]] = g
        part = ax.reduce_scatter(flat, edges)
        del flat
        lo = int(edges[me])
        local = {names[i]: part[offs[i] - lo:offs[i + 1] - lo].reshape(self.master[names[i]].shape)
                 for i in owned}
        sq = lambda muon: sum(float(g.ravel() @ g.ravel()) for n, g in local.items()
                              if (n in self.muon_names) == muon and n not in absent)
        sq_adam, sq_muon = ax.sum_scalars([sq(False), sq(True)])
        if self.muon_names:
            gnorm, gnorm_muon = math.sqrt(sq_adam), math.sqrt(sq_muon)
            self.last_grad_norm_muon = gnorm_muon
            lr_muon = (self.lr if self.muon.schedule is None
                       else float(self.muon.schedule(self.steps)))
        else:
            gnorm = math.sqrt(sq_adam + sq_muon)
        clip = self.clip_coef(gnorm)
        clip_muon = self.clip_coef(gnorm_muon) if self.muon_names else clip
        lr = self.lr if self.schedule is None else float(self.schedule(self.steps))
        self.steps += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1, bc2 = 1.0 - self.beta1_pow, 1.0 - self.beta2_pow
        report = {}
        # Whether each owned parameter's update survives the cast to its device dtype is decided
        # once, by its owner, and gathered: every rank holds the same master and the same device
        # copy, so the answer is the same everywhere, and each rank rounds 1/width of the model
        # instead of all of it twice (0.7 of 1.55 s on rank 0 at 4 x 60 M). 1 = rounded away.
        keep = {}
        for n, g in local.items():
            theta = self.master[n]
            before = round_to_device_dtype(theta, self.params[n].value.dtype)
            before = None if before is None else np.array(before, copy=True)
            if n in self.muon_names:
                # The replicated step leaves a Muon matrix with no gradient alone, momentum
                # included (torch.optim.Muon skips it); so does this one.
                if n in absent:
                    keep[n] = float(before is not None)
                    continue
                g = g * clip_muon if clip_muon != 1.0 else g
                new = self._muon_new(n, g, theta, lr_muon)
                upd = theta - new
                if n in self.held:
                    upd[self.held[n]] = 0.0
                    new = theta - upd
                theta[...] = new
            else:
                if clip != 1.0:
                    g = g * clip
                m, v = self.exp_avg[n], self.exp_avg_sq[n]
                m *= self.beta1
                m += (1.0 - self.beta1) * g
                v *= self.beta2
                v += (1.0 - self.beta2) * (g * g)
                upd = lr * ((m / bc1) / (np.sqrt(v / bc2) + self.eps) + self.weight_decay * theta)
                if n in self.held:
                    upd[self.held[n]] = 0.0
                theta -= upd
            after = None if before is None else round_to_device_dtype(theta, self.params[n].value.dtype)
            keep[n] = float(after is not None and np.array_equal(after, before))
            report[n] = {"master_step": float(np.linalg.norm(upd)),
                         "grad_norm": float(np.linalg.norm(g))}
        # Owned parameters are contiguous in `names`, so the flags gather in the same rank order.
        # Gathered BEFORE the weights: a gathered slice is a mapping valid only until this rank's
        # next collective, and the copy out of it below must come last.
        owner_of = self._owners()[3]
        fedges = np.concatenate([[0], np.cumsum(np.bincount(owner_of, minlength=ax.width))])
        skip = ax.all_gather(np.array([keep[names[i]] for i in owned], np.float32), edges=fedges)
        mine = (np.concatenate([self.master[names[i]].ravel() for i in owned]) if owned
                else np.zeros(0, np.float32))
        # Edges are tensor-aligned, so every parameter sits whole in its owner's slice and is
        # copied from that rank's mapping straight into the master.
        parts = ax.all_gather_parts(mine)
        writes_skipped = 0
        for i, n in enumerate(names):
            t, theta = self.params[n], self.master[n]
            r = int(owner_of[i])
            lo = int(offs[i] - edges[r])
            theta[...] = parts[r][lo:lo + theta.size].reshape(theta.shape)
            if skip[i]:
                writes_skipped += 1
            else:
                _write(t, theta)
        self.last_lr, self.last_clip, self.last_grad_norm = lr, clip, gnorm
        self.peak_lr = max(self.peak_lr, float(lr))
        self.last_per_sample = False
        self.last_writes_skipped = writes_skipped
        self.last_report = report
        return report

    def _sharded(self) -> bool:
        return self.shard and self.data_parallel is not None and self.data_parallel.width > 1

    def _shard_layout(self):
        """Names in exchange order, their flat offsets, the tensor-aligned slice edges, and the
        indices of the parameters this rank owns. Identical on every rank by construction."""
        names, offs, e, owner = self._owners()
        me = self.data_parallel.dp_rank
        return names, offs, e, [i for i in range(len(names)) if owner[i] == me]

    def _owners(self):
        """Names in exchange order, flat offsets, slice edges, and each parameter's owning rank.

        A parameter belongs to the one rank whose slice contains its first element; with
        tensor-aligned edges that slice contains all of it, and an empty tensor on a boundary
        still gets exactly one owner.
        """
        import numpy as np
        from .launcher import shard_edges
        names = sorted(self.master)
        sizes = [self.master[n].size for n in names]
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        w = self.data_parallel.width
        if getattr(self, "_edges", None) is None or len(self._edges) != w + 1:
            self._edges = shard_edges(sizes, w)
        e = self._edges
        owner = np.searchsorted(e, offs[:-1], side="right") - 1
        owner = np.minimum(owner, w - 1)
        return names, offs, e, owner

    def state_families(self) -> dict:
        """The optimizer's per-parameter state, ``{family: {name: array}}``, for a checkpoint.

        ``exp_avg`` and ``exp_avg_sq`` cover the Adam parameters, ``momentum_buffer`` the Muon
        ones. A sharded optimizer holds only its own parameters' state, so it gathers the rest
        from the other ranks: a COLLECTIVE, every rank must call it at the same point (the
        checkpointer saves on every rank at the same steps, each into its own directory).
        Every rank then holds the same complete state, so a checkpoint resumes at any width.
        """
        adam = [n for n in sorted(self.master) if n not in self.muon_names]
        fams = {"exp_avg": (self.exp_avg, adam), "exp_avg_sq": (self.exp_avg_sq, adam)}
        if self.muon_names:
            fams["momentum_buffer"] = (self.momentum_buffer, sorted(self.muon_names))
        if not self._sharded():
            return {f: {n: mom[n] for n in which} for f, (mom, which) in fams.items()}
        import numpy as np
        names, _, _, owner = self._owners()
        rank_of = dict(zip(names, owner.tolist()))
        me, w = self.data_parallel.dp_rank, self.data_parallel.width
        out = {}
        for f, (mom, which) in fams.items():
            # `which` is sorted and ownership is contiguous in sorted order, so the family's
            # flat vector is the ranks' owned pieces back to back, rank 0 first.
            counts = np.zeros(w, np.int64)
            for n in which:
                counts[rank_of[n]] += self.master[n].size
            edges = np.concatenate([[0], np.cumsum(counts)])
            mine = [mom[n].ravel() for n in which if rank_of[n] == me]
            full = self.data_parallel.all_gather(
                np.concatenate(mine) if mine else np.zeros(0, np.float32), edges=edges)
            out[f], o = {}, 0
            for n in which:
                k = self.master[n].size
                out[f][n] = full[o:o + k].reshape(self.master[n].shape)
                o += k
        return out

    def drop_unowned(self) -> None:
        """Forget the state of parameters another rank owns (after loading a full checkpoint)."""
        if not self._sharded():
            return
        names, _, _, owned = self._shard_layout()
        keep = {names[i] for i in owned}
        for mom in (self.exp_avg, self.exp_avg_sq, self.momentum_buffer):
            for n in [n for n in dict.keys(mom) if n not in keep]:
                del mom[n]

    def _muon_new(self, name, g, theta, lr):
        """``torch.optim.Muon``'s step for one matrix: the new master, in torch's op order.

        Momentum ``buf <- lerp(buf, g, 1 - mu)``, Nesterov ``u = lerp(g, buf, mu)``,
        ``theta <- theta * (1 - lr * wd) - lr * ratio * NS(u)``. Each lerp is written the way
        torch evaluates it for its weight (``start + w (end - start)`` below 0.5,
        ``end - (end - start)(1 - w)`` from 0.5), because the bf16 cast inside Newton-Schulz
        turns a one-ulp difference in ``u`` into a 2^-8 difference in the update.
        """
        import numpy as np
        mu = self.muon
        f = np.float32
        buf = self.momentum_buffer[name]
        buf += f(1.0 - mu.momentum) * (g - buf)
        u = (buf - (buf - g) * f(1.0 - mu.momentum)) if mu.nesterov else buf
        o = mu.orthogonalize(np.ascontiguousarray(u, np.float32))
        return theta * f(1.0 - lr * mu.weight_decay) + f(-lr * mu.ratio(theta.shape)) * o

    def _reduce(self, replicas) -> None:
        """Sum each parameter's gradient across the DP axis, in place on the tape.

        Deliberately not a mean: the divisor is the global batch the caller pinned, and
        dividing by chip count here is exactly the substitution ``accelerate`` makes at
        ``data_loader.py:347-348`` that turns one recipe into a different one per box.
        """
        import numpy as np
        width = 1 if self.data_parallel is None else self.data_parallel.width
        if width == 1:
            if replicas:
                raise ValueError(
                    "per-chip gradients were passed but the optimizer has no data-parallel "
                    "axis wider than one chip. Hand it data_parallel=mesh.axis('dp')")
            return
        if self.accum:
            # Under per-sample clipping the step's gradient is the ACCUMULATOR, not the tape:
            # the tape holds whichever sample ran last. So the axis sums the accumulator, and
            # the per-parameter participation counts with it, because upstream divides by the
            # GLOBAL count. Averaging each rank's own samples first and summing the averages
            # is a different number the moment the counts differ across ranks, which is the
            # only case the count exists for.
            if replicas:
                raise ValueError(
                    "per-chip gradients were passed alongside a per-sample accumulator. The "
                    "tape holds the last sample, not the step, so step() reduces the "
                    "accumulator itself and replicas must be empty. Call zero_grad() after "
                    "the last clip_and_accumulate()")
            # Every parameter, not only the ones this rank accumulated: the axis flattens in
            # name order, so two ranks with different key sets would exchange vectors of
            # different lengths and sum the wrong bytes into each other's gradients. Upstream
            # keeps a zeros_like entry for every parameter for the same reason.
            # The participation counts ride in the same message: one barrier a step, and the
            # collective's timings and byte count describe the gradient rather than a 16 KB
            # vector of counts.
            both = self.data_parallel.reduce_all({
                **{n: [self.accum[n] if n in self.accum else np.zeros_like(self.master[n])]
                   for n in self.master},
                **{f"{n}\0count": [np.float32([self.participation.get(n, 0)])]
                   for n in self.master}})
            summed = {n: both[n] for n in self.master}
            self.participation = {n: int(both[f"{n}\0count"].ravel()[0]) for n in self.master}
            # A parameter no rank activated is dropped rather than stepped on a zero
            # gradient. Upstream zeroes its grad and lets Adam step it from momentum alone;
            # ours skips it, and that difference is recorded rather than papered over.
            self.accum = {n: summed[n] for n in self.master if self.participation.get(n, 0)}
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
        import numpy as np  # noqa: F401  (displacement() needs it)
        d = self.displacement()
        lo, hi = band
        r = d["ratio"]
        if self.steps == 0:
            raise RuntimeError("nothing has stepped yet; there is no displacement to check")
        if not (r == r) and self.peak_lr == 0.0:
            # Every step ran at lr 0, the first step of the warmup, so nothing was meant to move.
            return {**d, "lr_zero": True,
                    "note": f"all {self.steps} step(s) ran at lr 0 (the warmup's first step), "
                            f"so the weights are unchanged by design; band not asserted"}
        if not (r == r):  # nan: master has not moved at all
            raise AssertionError(
                f"the master has not moved after {self.steps} steps (displacement "
                f"{d['master']:.3e}), so nothing was learned. Check that the loss reached "
                f"the parameters: a frozen tensor accumulates no gradient and a pruned "
                f"branch produces none")
        # A correct short run can sit below the band and this is not a defect: if the master
        # has moved less than the device dtype can represent, every update rounds away by
        # design and the fp32 master is doing its job. Measured on a 20-step OF3 warmup, where
        # lr ~ k * 1.8e-06 puts every step under bf16 spacing and the arm read 0.810 against a
        # (0.9, 1.1) band -- a guard raising on correct behaviour gets disabled by the next
        # caller, which is worse than no guard. Reported, never silently passed.
        if d["master"] < d["resolution"] and r < lo:
            return {**d, "below_resolution": True,
                    "note": (f"master displacement {d['master']:.3e} is under the device's own "
                             f"resolution {d['resolution']:.3e}, so ratio {r:.4f} measures the "
                             f"dtype rather than the optimizer; band not asserted")}
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
            init = self._widen(self.init[n])
            m += float(np.sum((self.master[n] - init) ** 2))
            cur = to_host(t.value).astype(np.float32).reshape(self.master[n].shape)
            d += float(np.sum((cur - init) ** 2))
        m, d = math.sqrt(m), math.sqrt(d)
        # The displacement the device copy CANNOT show. A change smaller than half an ulp of
        # the weight it is applied to rounds away entirely, so below this the ratio is
        # measuring the dtype and not the optimizer.
        res = 0.0
        for n, t in self.params.items():
            u = unit_roundoff(t.value.dtype)
            res += float(np.sum((u * np.abs(self._widen(self.init[n]))) ** 2))
        return {"master": m, "device": d, "ratio": (d / m) if m > 0 else float("nan"),
                "resolution": math.sqrt(res)}

    def grad_norm(self, disabled=(), *, keep=None) -> float:
        """Global L2 norm of the gradients, for clipping and for the trajectory log.

        ``disabled`` is the set of parameter names this sample does not activate, and it is
        excluded from the norm. Theirs does the same and it is not a detail: `_clip_grads`
        builds `params_enabled` by dropping `disabled_params` BEFORE computing the norm
        (`grad_manager.py:147-153`), and their runner disables the confidence head whenever a
        sample's summed confidence weight is zero, which `initial_training.yml` does on 4 of its
        5 datasets. Norming over a set they excluded gives a different clip coefficient for
        every gradient in the step -- measured at 8.368e-01 relative on one small enabled tensor
        beside one large disabled one, which is not a corner case but most of initial training.

        ``keep``, a dict, receives every gradient read, by name. The clip is global, so the
        norm has to see every gradient before any is applied and cannot be folded into the
        update loop; ``step`` and ``clip_and_accumulate`` pass ``keep`` and consume the arrays
        instead of reading each gradient off the card a second time. Dropping that second read
        measured 1.245x on the AdamW phase at the OpenFold3 crop-384 census, bit-identical
        (on the ``wk/of3t-p10optim`` branch). The price is holding the host
        copies between the two passes, one float32 array per parameter, released as the loop
        pops them.
        """
        disabled = set(disabled)
        tot = 0.0
        for name, t in self.params.items():
            if t.grad is not None and name not in disabled:
                gg = self._grad(name, t)
                tot += float(gg.ravel() @ gg.ravel())
                if keep is not None:
                    keep[name] = gg
        return math.sqrt(tot)

    def clip_coef(self, gnorm: float) -> float:
        """``min(1, clip_norm/gnorm)``, their `max_norm / maximum(gnorm, max_norm)`.

        The same rule written two ways, agreeing to 8.51e-08 across eight straddling cases
        against their real `compute_global_norm` -- inside the float32 floor both stacks share.
        Factored out so the per-sample path and the per-batch path cannot drift apart.
        """
        return (min(1.0, self.clip_norm / gnorm)
                if (self.clip_norm > 0 and gnorm > 0) else 1.0)

    def clip_and_accumulate(self, disabled=()) -> dict:
        """Clip THIS SAMPLE's gradients and add them to the accumulator. Their algorithm.

        Per-sample clipping is a different algorithm from per-batch clipping, not a different
        constant: it changes the DIRECTION of the accumulated update, which is the same reason
        this file already refuses per-parameter clipping. Measured at 1.948e-01 relative over 3
        samples with one 200x outlier. `per_sample_clipping: True` at `clip_val 10.0` is their
        shipped default (`clip_and_accumulate`, `grad_manager.py:242`).

        Call it after each sample's backward, then `zero_grad()` before the next sample's. The
        caller owns the per-sample loop because only the caller knows what a sample is; `step()`
        consumes the accumulator when there is one and clips the batch once when there is not.
        `train_loop` is that caller, and it clears the tape after the last sample so the
        gradient the data-parallel axis reduces is the accumulator rather than one sample.
        """
        import numpy as np
        disabled = set(disabled)
        read = {}
        gnorm = self.grad_norm(disabled, keep=read)
        clip = self.clip_coef(gnorm)
        for name, t in self.params.items():
            if t.grad is None or name in disabled:
                continue
            g = read.pop(name)
            if clip != 1.0:
                g = g * clip
            acc = self.accum.get(name)
            self.accum[name] = g if acc is None else acc + g
            self.participation[name] = self.participation.get(name, 0) + 1
        self.accum_count += 1
        return {"grad_norm": gnorm, "clip": clip, "samples": self.accum_count}

    def state_dict(self) -> dict:
        return {"steps": self.steps, "lr": self.lr, "beta1": self.beta1,
                "beta2": self.beta2, "eps": self.eps,
                "weight_decay": self.weight_decay, "clip_norm": self.clip_norm,
                "beta1_pow": self.beta1_pow, "beta2_pow": self.beta2_pow,
                "peak_lr": self.peak_lr}

    def load_state_dict(self, d: dict) -> None:
        for k in ("steps", "lr", "beta1", "beta2", "eps", "weight_decay", "clip_norm",
                  "beta1_pow", "beta2_pow", "peak_lr"):
            if k in d:
                setattr(self, k, d[k])

