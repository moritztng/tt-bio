"""Does a training run match its reference? Three bars, fixed before the data they grade.

A gradient check (``checks.py``) says one op's backward is right. It cannot say the model
trains, or that it trains like upstream. Those are three separate questions and each has a
check here:

* :class:`GradientCoverage` -- every trainable parameter receives a gradient (G2);
* :func:`loss_parity` -- on identical batches, ``t`` and noise, our loss and each of its terms
  match the upstream reference (G3, first half);
* :func:`curve_match` -- a matched-compute run (same data order, seeds and steps on both
  sides) ends at the same loss and never wanders from upstream's curve on the way (G3,
  second half).

**The constants are pre-registered.** They were committed before any curve or loss they grade
existed, and the commit is the proof. A bar chosen by someone who can see the curve is not a
bar, so a change to any number below needs a recorded reason that predates the data.
``tests/test_training_parity_bars.py`` pins every value and shows each check failing.

Host-side, numpy only. The caller produces the numbers; this module owns the verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = ["LOSS_PARITY_BARS", "LOSS_PARITY_MIN_BATCHES", "CURVE_MIN_STEPS", "CURVE_TAIL",
           "CURVE_REL", "CURVE_SEED_FACTOR", "CURVE_WINDOW", "CURVE_DRIFT", "TRAINS_MIN_STEPS",
           "LossParity", "loss_parity", "CurveMatch", "curve_match", "GradientCoverage",
           "zero_or_missing"]


# ---------------------------------------------------------------------------- loss parity
#
# Our loss against upstream's fp32 GPU loss on the same batch, the same t and the same noise,
# passed in, with the same weights. bf16 operands with fp32 accumulation keep ~1e-3 on a
# reduction (``checks.BARS``); a loss is a few reductions deep, so the production arm gets
# 1e-2 and the fp32 arm, which removes the operand rounding, 1e-4.
LOSS_PARITY_BARS = {"bfloat16": 1.0e-2, "float32": 1.0e-4}
LOSS_PARITY_MIN_BATCHES = 8


@dataclass
class LossParity:
    dtype: str
    bar: float
    n: int
    worst_rel: float
    worst_batch: int
    worst_term: str
    failures: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        head = (f"LOSS-PARITY: {'PASS' if self.passed else 'FAIL'} rel={self.worst_rel:.3e} "
                f"tol={self.bar:.1e} n={self.n} dtype={self.dtype} worst=batch "
                f"{self.worst_batch} term {self.worst_term!r}")
        return "\n".join([head] + ["  " + f for f in self.failures])


def loss_parity(ours: Sequence[Mapping[str, float]], ref: Sequence[Mapping[str, float]],
                dtype: str) -> LossParity:
    """Grade per-batch loss terms against the reference's. Every term, every batch.

    ``ours[i]`` and ``ref[i]`` map term name to value for batch ``i`` and must carry the same
    names, the total included. A term one side has and the other lacks is a failure, not a
    skip: a loss that silently drops a term matches on everything it still computes.
    """
    bar = LOSS_PARITY_BARS[dtype]
    fails = []
    if len(ours) != len(ref):
        fails.append(f"{len(ours)} batches against {len(ref)} reference batches")
    if len(ref) < LOSS_PARITY_MIN_BATCHES:
        fails.append(f"n={len(ref)} batches, the bar needs >= {LOSS_PARITY_MIN_BATCHES}")
    worst = (0.0, -1, "")
    for i, (a, b) in enumerate(zip(ours, ref)):
        if set(a) != set(b):
            fails.append(f"batch {i}: terms differ, ours-only {sorted(set(a) - set(b))}, "
                         f"ref-only {sorted(set(b) - set(a))}")
        for k in sorted(set(a) & set(b)):
            x, y = float(a[k]), float(b[k])
            if not (math.isfinite(x) and math.isfinite(y)):
                rel = math.inf
            elif x == y:
                rel = 0.0
            else:
                rel = abs(x - y) / abs(y) if y != 0.0 else math.inf
            if rel > worst[0] or worst[1] < 0:
                worst = (rel, i, k)
            if rel > bar:
                fails.append(f"batch {i} term {k!r}: ours {x!r} ref {y!r} rel {rel:.3e}")
    return LossParity(dtype=dtype, bar=bar, n=len(ref), worst_rel=worst[0],
                      worst_batch=worst[1], worst_term=worst[2], failures=fails)


# ---------------------------------------------------------------------------- the curve
#
# Matched compute: same data order, same seeds, same number of steps, same schedule, both sides.
# The end point is the mean loss over the final CURVE_TAIL steps. It has to sit within
# max(CURVE_REL of upstream's, CURVE_SEED_FACTOR x upstream's own seed-to-seed gap): a cross-arch
# gap is only a finding if it is bigger than what re-seeding upstream on its own hardware does.
# And the shape: after warm-up, our CURVE_WINDOW-step running mean never leaves upstream's by
# more than CURVE_DRIFT relative. ``warmup`` is the leg's learning-rate warm-up step count,
# identical on both sides by construction of a matched leg; the drift check never starts before
# the first full window.
CURVE_MIN_STEPS = 2000
CURVE_TAIL = 200
CURVE_REL = 0.02
CURVE_SEED_FACTOR = 1.5
CURVE_WINDOW = 100
CURVE_DRIFT = 0.10


@dataclass
class CurveMatch:
    steps: int
    tail_ours: float
    tail_a: float
    tail_b: float
    gap: float
    bar: float
    worst_drift: float
    worst_drift_step: int
    failures: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        head = (f"CURVE: {self.steps} steps {'PASS' if self.passed else 'FAIL'} "
                f"tail{CURVE_TAIL}_gap={self.gap:.4g} bar={self.bar:.4g} "
                f"(ours {self.tail_ours:.6g}, gpu A {self.tail_a:.6g}, gpu B {self.tail_b:.6g}) "
                f"drift={self.worst_drift:.3g}@{self.worst_drift_step} bar={CURVE_DRIFT}")
        return "\n".join([head] + ["  " + f for f in self.failures])


def _running_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean; element j is the mean of steps j-w+1..j, defined from j = w-1."""
    c = np.cumsum(np.concatenate([[0.0], x]))
    with np.errstate(invalid="ignore"):  # a non-finite loss is already a reported failure
        return (c[w:] - c[:-w]) / w


def curve_match(ours: Sequence[float], gpu_a: Sequence[float], gpu_b: Sequence[float],
                warmup: int) -> CurveMatch:
    """Grade a matched-compute loss curve against upstream seed A, with seed B as the control.

    All three are per-step losses of equal length. ``gpu_b`` is upstream re-run with a
    different seed on the same hardware; without it a cross-arch gap has nothing to be
    measured against (a seed change has moved an output more than the chip did before).
    """
    o, a, b = (np.asarray(v, dtype=np.float64) for v in (ours, gpu_a, gpu_b))
    fails = []
    n = len(a)
    if not (len(o) == len(a) == len(b)):
        fails.append(f"lengths differ: ours {len(o)}, gpu A {len(a)}, gpu B {len(b)}")
        n = min(len(o), len(a), len(b))
        o, a, b = o[:n], a[:n], b[:n]
    if n < CURVE_MIN_STEPS:
        fails.append(f"{n} steps, the bar needs >= {CURVE_MIN_STEPS}")
    if not (0 <= warmup <= n - CURVE_TAIL):
        fails.append(f"warmup {warmup} outside [0, {n - CURVE_TAIL}]")
    for name, v in (("ours", o), ("gpu A", a), ("gpu B", b)):
        bad = np.flatnonzero(~np.isfinite(v))
        if bad.size:
            fails.append(f"{name}: {bad.size} non-finite losses, first at step {int(bad[0])}")
    if n < max(CURVE_TAIL, CURVE_WINDOW):
        return CurveMatch(n, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, -1,
                          fails or ["too short to grade"])
    to, ta, tb = (float(v[-CURVE_TAIL:].mean()) for v in (o, a, b))
    gap = abs(to - ta)
    bar = max(CURVE_REL * abs(ta), CURVE_SEED_FACTOR * abs(ta - tb))
    if not gap <= bar:
        fails.append(f"final-{CURVE_TAIL} mean gap {gap:.4g} > bar {bar:.4g}")
    ro, ra = _running_mean(o, CURVE_WINDOW), _running_mean(a, CURVE_WINDOW)
    start = max(warmup, CURVE_WINDOW - 1) - (CURVE_WINDOW - 1)  # index into running means
    with np.errstate(divide="ignore", invalid="ignore"):
        drift = np.abs(ro[start:] - ra[start:]) / np.abs(ra[start:])
    drift = np.where(np.isfinite(drift), drift, np.inf)
    if drift.size:
        j = int(np.argmax(drift))
        worst, worst_step = float(drift[j]), start + j + CURVE_WINDOW - 1
        if worst > CURVE_DRIFT:
            over = int((drift > CURVE_DRIFT).sum())
            fails.append(f"running mean left upstream's by {worst:.3g} > {CURVE_DRIFT} at step "
                         f"{worst_step} ({over} steps over)")
    else:
        worst, worst_step = math.nan, -1
    return CurveMatch(n, to, ta, tb, gap, bar, worst, worst_step, fails)


# ---------------------------------------------------------------------------- every parameter
#
# One step is not the test. Upstream zero-initialises residual output projections (ABB3's
# ``init="final"``, RFD4's triangle ``to_out``, adaLN-zero gammas and recycle projections), so
# at step 1 the weights behind them have an exactly zero gradient. That is upstream's behaviour:
# the zero weight itself gets ``x^T g`` and one update switches the branch on. ABB3 read 218 of
# 500 parameters as dead at step 1. What separates that transient from a dead parameter is time,
# so a parameter has to be gradient-free on EVERY one of the first TRAINS_MIN_STEPS steps.
#
# And the check is per parameter because nothing aggregate sees it: ABB3's first leg ran 1,388
# steps with 356 of 436 parameters at an identically zero gradient while the loss, a plausible
# grad_norm of 0.18 and a master digest that changed every step all read healthy.
TRAINS_MIN_STEPS = 5


def zero_or_missing(grads: Mapping[Hashable, object]) -> set:
    """Names whose gradient is ``None`` or identically zero.

    Missing and zero are both "this parameter does not train" for the guard, and neither may
    be papered over: substituting ``zeros_like`` for a missing gradient is how ABB3 trained
    nothing for 5.8 hours.
    """
    dead = set()
    for k, g in grads.items():
        if g is None:
            dead.add(k)
            continue
        if hasattr(g, "detach"):
            g = g.detach().float().cpu().numpy()
        if not np.any(np.asarray(g)):
            dead.add(k)
    return dead


class GradientCoverage:
    """Refuse a run in which a trainable parameter gets no gradient on any of its first steps.

    ``names`` is every trainable parameter. ``exempt`` maps a name to the upstream reason it
    legitimately never trains (an input that a training batch never carries, say); every
    exemption is printed with its reason, and an exemption naming no parameter is an error.
    Call :meth:`observe` once per optimizer step with the names whose gradient was missing or
    zero; on the ``steps``-th call it raises if any non-exempt name was dead on all of them.
    """

    def __init__(self, names: Iterable[Hashable], steps: int = TRAINS_MIN_STEPS,
                 exempt: Optional[Mapping[Hashable, str]] = None, where: str = ""):
        if steps < TRAINS_MIN_STEPS:
            raise ValueError(f"steps={steps}: one or a few steps read upstream's zero-init "
                             f"transient as dead parameters; use >= {TRAINS_MIN_STEPS}")
        self.names = list(names)
        self.exempt = dict(exempt or {})
        unknown = set(self.exempt) - set(self.names)
        if unknown:
            raise ValueError(f"exemptions name no parameter: {sorted(map(str, unknown))}")
        self.steps, self.where = steps, where
        self.seen, self.never = 0, None

    @property
    def done(self) -> bool:
        return self.seen >= self.steps

    def observe(self, dead: Iterable[Hashable]) -> None:
        if self.done:
            return
        dead = set(dead) - set(self.exempt)
        self.never = dead if self.never is None else (self.never & dead)
        self.seen += 1
        if self.done:
            self.check()

    def check(self) -> None:
        if not self.never:
            return
        dead = sorted(self.never, key=str)
        names = ", ".join(str(i) for i in dead[:8])
        raise RuntimeError(
            f"{self.where}{len(dead)} of {len(self.names)} parameters received no gradient on "
            f"any of the first {self.seen} steps ({names}{'...' if len(dead) > 8 else ''}). "
            f"Refusing to continue: a run in this state produces a plausible loss curve and "
            f"trains nothing. Check the starting weights and the tape first.")
