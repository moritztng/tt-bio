"""``gradcheck`` -- Tier 3's first-class call. Three levels of evidence, in order.

Promoted from ``perf/hallgrad/gradcheck.py``, where it was a script with the bars in module
constants. It is in the interface because a user at Tier 3 owns an op AND its backward, and a
backward is the one thing in this stack whose correctness has no other witness: a wrong
backward trains, converges to something, and produces a model that is confidently wrong.

The three levels, in the order that makes each trustworthy:

1. **The float64 reference is checked against central finite differences first.** In float64,
   never touching a device. A reference nobody verified is how a campaign spends weeks
   chasing a confident wrong number.
2. **The device gradient is compared to that verified reference.** Inputs are rounded to the
   device dtype FIRST and the reference is fed the rounded values upcast, so what is measured
   is the op's own error and not the input quantisation.
3. **Controls.** An fp32 arm must show the error collapse -- a wrong formula does not get
   better with precision, only an imprecise one does -- and a deliberately broken arm must
   FAIL, so we know the check can fail at all.

**One bar for all ops is wrong twice over.** The bars below are per op class off measured
floors, not one number off the bf16 mantissa. Two reasons, both measured on qb1 card 3 at
1350 MHz. The matmul and K-ladder figures are
``perf/abb3_port/precision_probe_qb1c3.txt``; the eltwise floor is A1's fp32 arm,
``perf/train_a1_defork/out/gradcheck_fp32.json``, where mul, add, sigmoid and silu land at
2.53e-08 to 7.55e-08:

* A matmul on fp32 operands keeps about 11 mantissa bits whatever the kernel config says --
  1.25e-03 relative at HiFi4 with ``fp32_dest_acc_en``, 7.05e-03 at HiFi2, 2.85e-02 at LoFi
  -- while eltwise subtract and multiply are fp32-exact at 3.0e-07. The K ladder is flat
  (1.04e-03 at K=1, 1.48e-03 at K=512) and K=1 accumulates nothing, so it is input rounding
  and no accumulator setting fixes it. ``ttnn.sum`` carries 1.2e-02 on 96 fp32 terms, so
  reductions round like matmuls. An fp32 arm therefore does NOT collapse a reduction-carrying
  gradient to machine epsilon, and scoring one against an eltwise bar fails a correct op.
* A mantissa-derived bar is a SMOOTH-op bar. The derivation assumes the gradient is a
  continuous function of the input, which it is not at a kink. Measured by A1 on the
  dispatched ops: ``linear``, ``linear_nobias``, ``linear_silu``, ``linear_sigmoid`` and
  ``layernorm`` land at 0.0024-0.0049 against a 0.01 bar while ``linear_relu`` is the single
  failure at 0.0625, and HiFi4 pulls it only to 0.0128. So **a kinked op is scored on
  direction and on its gate mask, and not on rel_L2 at all.** What it must not get is a
  smooth-op bar plus an excuse.

Host-side: numpy and torch. The device half is the caller's ``tt_fn``, because a Tier-3 user
owns the op and this module owns the evidence discipline.
"""

from __future__ import annotations

import math
import zlib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

__all__ = ["gradcheck", "GradcheckReport", "BARS", "MASK_BAR", "FD_BAR", "OP_CLASSES",
           "metrics", "fd_check", "WeightCoverage", "weight_coverage", "COMPOSED_BAR", "composed_verdict", "FLOOR_SITES", "floor_site"]


# The reference's own bar against central differences. It is not a device number: this is
# float64 against float64 and a disagreement above it means the reference is wrong.
FD_BAR = 2e-6
# Two steps must agree to this fraction of FD_BAR before the larger one is taken as smooth. At 0.5 a
# pair that straddled ReLU kinks with a shared error passed a value 3e-5 off (PairUpdate bf16,
# pugc3): agreement within half the bar says nothing about an error the size of the bar.
FD_AGREE = 0.05

# A composed block or the whole network (D28, OF3T section 3d): per-tensor rel_L2 max, median of the
# per-tensor rel_L2, worst cosine. G1 (BARS) stays the bar for a single VJP; a composition stacks
# several device floors, so it is graded on the spread across tensors, not on any one op's floor.
COMPOSED_BAR = (5.0e-02, 2.0e-02, 0.9999)

# fraction of gate coordinates a kinked op's device gradient and reference must agree on
MASK_BAR = 0.99

OP_CLASSES = ("eltwise", "reduction", "kink")

# (rel_l2 bar, cos bar, where the bar comes from). A `None` rel_l2 bar means the class is not
# scored on rel_L2 -- see MASK_BAR.
BARS = {
    ("eltwise", "float32"):    (3.0e-06, 0.9999990,
                                "10x the measured fp32 eltwise floor 3.0e-07"),
    ("eltwise", "bfloat16"):   (1.0e-02, 0.9999,
                                "the bf16 mantissa, sqrt(2)*2^-9 = 2.76e-03"),
    ("reduction", "float32"):  (2.5e-02, 0.9999,
                                "3.5x the measured fp32 HiFi2 matmul floor 7.05e-03"),
    ("reduction", "bfloat16"): (1.0e-02, 0.9999,
                                "the bf16 mantissa, sqrt(2)*2^-9 = 2.76e-03"),
    ("kink", "float32"):       (None, 0.9990,
                                "gradient is discontinuous; rel_L2 is not mantissa-bounded, "
                                "scored on direction and the gate mask instead"),
    ("kink", "bfloat16"):      (None, 0.9990,
                                "gradient is discontinuous; rel_L2 is not mantissa-bounded, "
                                "scored on direction and the gate mask instead"),
}

BF16_FLOOR = math.sqrt(2.0) * 2.0 ** -9


def case_rng(seed: int, name: str) -> "np.random.Generator":
    """The input stream for one named case, reproducible across processes.

    Seeding a case from ``hash(name)`` is the obvious thing and it is wrong: Python salts str
    hashing per process unless PYTHONHASHSEED is set, so ``--seed 7`` gives one set of inputs
    today and a different set tomorrow. Inside a single process it looks correct -- the case
    list stops mattering, which is what the seeding was for -- so the defect hides until
    someone tries to re-run a failing case on the inputs that failed, or to reproduce a
    recorded number, and cannot. CRC32 is stable by specification.
    """
    return np.random.default_rng([seed, zlib.crc32(name.encode()) % (2 ** 31)])


def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    g, r = np.asarray(got, dtype=np.float64).ravel(), np.asarray(ref, dtype=np.float64).ravel()
    diff = g - r
    rn = np.linalg.norm(r)
    big = np.abs(r) > 0.01 * np.abs(r).max() if r.size else np.zeros(0, bool)
    return {
        "rel_l2": float(np.linalg.norm(diff) / rn) if rn > 0 else float("nan"),
        "max_abs": float(np.abs(diff).max()) if diff.size else 0.0,
        "max_rel": float((np.abs(diff[big]) / np.abs(r[big])).max()) if big.any() else 0.0,
        "cos": float(g @ r / (np.linalg.norm(g) * rn))
               if rn > 0 and np.linalg.norm(g) > 0 else 0.0,
    }


def fd_check(loss_fn, params, n_probe=40, h=1e-5, seed=0, mag_floor=1e-6, kinks=None):
    """Central differences in float64 against torch's float64 analytic gradient.

    This validates the REFERENCE, so it runs entirely in float64 and never touches a device.

    Only coordinates carrying real signal are probed. A central difference can resolve a
    component only if its contribution to the loss clears float64 roundoff on the loss itself:
    ``|dL/dx| * 2h`` has to beat ``eps * |L|``, so with ``h = 1e-5`` anything below about
    ``1e-11 * |L|`` measures roundoff rather than a gradient. Softmax is what forces this -- a
    peaked row has components at 1e-12 and probing them returns pure cancellation noise.
    Coordinates under ``mag_floor`` times the largest analytic component are skipped and the
    eligible count is returned, so the skipping is visible in the output rather than buried in
    a tolerance.

    A kink (ReLU) in the reference breaks the difference without the gradient being wrong: a
    step that moves any pre-activation across zero adds an O(1) error, and the chance of that
    grows with the step and with the number of units (PairUpdate's transition has ~3M at n=64,
    and every probe of an upstream weight moves all of them: at h=1e-5 a large share of probes
    cross one). So each coordinate is differenced at ``h`` and ``h/10``; on a smooth point the
    two agree to roundoff. When they disagree with EACH OTHER a boundary sits inside a step, and
    the pair moves down a decade (``h/10`` and ``h/100``); a coordinate still disagreeing there
    is skipped. That test never
    reads the analytic gradient, so it cannot hide a wrong one. Skipped coordinate indices are appended
    to ``kinks`` when a list is given. A skipped coordinate is replaced by another eligible one, up to
    ``4 * n_probe`` draws per tensor, so a heavily kinked tensor still gets its smooth probes instead of
    a coin flip deciding the verdict (net5 fp32 trunk arm: 104 probed, 110 skipped, unverified). If
    fewer than half the requested probes found a smooth coordinate the result is ``inf``: a reference
    that cannot be differenced has not been verified.

    Returns ``(worst relative disagreement, coordinates probed, coordinates eligible)``.
    """
    import torch
    rng = np.random.default_rng(seed)
    for p in params:
        if p.grad is not None:
            p.grad = None
    # Explicitly, because `torch.set_grad_enabled(False)` is process-wide and several modules
    # in this repo call it at import time to keep an inference reference cheap. Nothing
    # restores it, so whether this function works came down to what else the process had
    # imported -- and the failure is `element 0 of tensors does not require grad`, which reads
    # like a caller mistake rather than ambient state. Asking for the gradient we are about to
    # take is free and does not reach outside this block.
    with torch.enable_grad():
        loss = loss_fn()
        loss.backward()
    resolution_floor = 1.0e4 * 2.22e-16 * abs(loss.item()) / (2.0 * h)
    worst, probed, eligible, skipped, wanted = 0.0, 0, 0, 0, 0
    for p in params:
        flat = p.detach().reshape(-1)
        ana = p.grad.detach().reshape(-1).clone()
        scale = ana.abs().max().item()
        if scale == 0.0:
            continue
        floor = max(mag_floor * scale, resolution_floor)
        ok = (ana.abs() >= floor).nonzero().reshape(-1).numpy()
        eligible += int(ok.size)
        if ok.size == 0:
            continue
        want = min(n_probe, ok.size)
        wanted += want
        got = 0
        for i in rng.permutation(ok)[:4 * want]:
            if got == want:
                break
            i = int(i)
            num = _smooth_central(loss_fn, flat, i, h, FD_AGREE * FD_BAR * scale)
            if num is None:
                skipped += 1
                if kinks is not None:
                    kinks.append(i)
                continue
            # Scaled by the gradient's own magnitude, not by this element's: a per-element
            # ratio is dominated by whichever probed coordinate is smallest, whose finite
            # difference is a difference of two nearly equal float64 losses.
            worst = max(worst, abs(num - ana[i].item()) / scale)
            probed += 1
            got += 1
    if 2 * probed < wanted:
        worst = float("inf")
    return worst, probed, eligible


def _smooth_central(loss_fn, flat, i, h, tol, decades=2):
    """The central difference at the largest step in ``h, h/10, ...`` that agrees within ``tol`` with
    the next step down, or None if no pair within ``decades`` shrinks does (a kink inside every step)."""
    prev = _central(loss_fn, flat, i, h)
    for _ in range(decades):
        h /= 10.0
        cur = _central(loss_fn, flat, i, h)
        if abs(prev - cur) <= tol:
            return prev
        prev = cur
    return None


def _central(loss_fn, flat, i, h):
    """One central difference of ``loss_fn`` in coordinate ``i`` of ``flat``, restored after."""
    import torch
    orig = flat[i].item()
    with torch.no_grad():
        flat[i] = orig + h
    lp = loss_fn().item()
    with torch.no_grad():
        flat[i] = orig - h
    lm = loss_fn().item()
    with torch.no_grad():
        flat[i] = orig
    return (lp - lm) / (2.0 * h)


@dataclass
class GradcheckReport:
    """What ``gradcheck`` returns. ``passed`` is the conjunction of every arm that ran.

    ``reference_verified`` is separate from ``passed`` on purpose: a run where the reference
    itself failed finite differences has learned nothing about the device, and reporting that
    as a gradient failure sends the next reader to the wrong file.
    """

    name: str
    op_class: str
    dtype: str
    reference_verified: bool = False
    fd_worst: float = float("nan")
    fd_probed: int = 0
    fd_eligible: int = 0
    fd_kinked: int = 0
    grads: Dict[str, dict] = field(default_factory=dict)
    bar_why: str = ""
    notes: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.reference_verified and self.grads
                    and all(g["pass"] for g in self.grads.values()))

    def __str__(self) -> str:
        head = (f"gradcheck {self.name} [{self.op_class}/{self.dtype}] "
                f"{'PASS' if self.passed else 'FAIL'}")
        verdict = ("ok" if self.reference_verified else
                   "FAILED -- the reference is wrong, not the device")
        ref = (f"reference vs central differences: {self.fd_worst:.2e} against {FD_BAR:.0e} "
               f"({self.fd_probed}/{self.fd_eligible} coords probed"
               + (f", {self.fd_kinked} skipped on a kink" if self.fd_kinked else "") + f") {verdict}")
        rows = []
        for k, g in self.grads.items():
            bits = [f"rel_l2 {g['rel_l2']:.3e}" if g["rel_l2_bar"] is not None
                    else "rel_l2 not scored",
                    f"cos {g['cos']:.7f}"]
            if "mask_agree" in g:
                bits.append(f"gate mask {g['mask_agree']:.4f}")
            rows.append(f"  {k}: {'pass' if g['pass'] else 'FAIL'}  " + ", ".join(bits))
        tail = "".join(f"\n  note: {n}" for n in self.notes)
        return "\n".join([head, "  " + ref, *rows]) + tail


def gradcheck(name: str, *, ref_loss, ref_params, tt_grads: Dict[str, np.ndarray],
              ref_grads: Optional[Dict[str, np.ndarray]] = None,
              op_class: str = "reduction", dtype: str = "float32",
              gate_mask: Optional[Dict[str, tuple]] = None,
              n_probe: int = 40, h: float = 1e-5, seed: int = 0) -> GradcheckReport:
    """Check a device gradient against a finite-difference-verified float64 reference.

    ``ref_loss`` is a zero-argument callable returning a float64 torch scalar built from
    ``ref_params`` (float64 torch tensors with ``requires_grad``). It is differentiated by
    torch and that gradient is validated against central differences before anything is
    compared to it.

    ``tt_grads`` is your device gradient per parameter name, already read back to numpy.
    ``ref_grads`` defaults to the reference's own ``.grad``, which is the normal case.

    ``op_class`` is one of ``eltwise``, ``reduction``, ``kink``, and it is yours to declare
    because only you know whether your backward reduces. Declaring ``eltwise`` for a backward
    that contains a reduction is the one way to make this check too strict and fail a correct
    op; the class table's docstring says which is which.

    ``gate_mask`` is required for a kinked op: ``{param: (device_gate, reference_gate)}`` as
    boolean arrays. A kinked op is scored on direction and on the fraction of gate
    coordinates the two agree on, because its rel_L2 is not mantissa-bounded.
    """
    if op_class not in OP_CLASSES:
        raise ValueError(f"op_class must be one of {OP_CLASSES}, got {op_class!r}")
    key = (op_class, dtype)
    if key not in BARS:
        raise ValueError(f"no bar for {key}; bars exist for {sorted(BARS)}. A dtype without a "
                         f"measured floor has no bar, and inventing one is how a bar stops "
                         f"meaning anything")
    rel_bar, cos_bar, why = BARS[key]
    rep = GradcheckReport(name=name, op_class=op_class, dtype=dtype, bar_why=why)

    # Level 1: the reference, before anything else.
    kinks = []
    worst, probed, eligible = fd_check(ref_loss, ref_params, n_probe=n_probe, h=h, seed=seed, kinks=kinks)
    rep.fd_worst, rep.fd_probed, rep.fd_eligible, rep.fd_kinked = worst, probed, eligible, len(kinks)
    rep.reference_verified = bool(worst < FD_BAR)
    if not rep.reference_verified:
        rep.notes.append(
            f"the float64 reference disagrees with central differences by {worst:.2e}, above "
            f"{FD_BAR:.0e}. Nothing was learned about the device: fix the reference first")
        return rep

    if ref_grads is None:
        ref_grads = {k: p.grad.detach().numpy() for k, p in zip(tt_grads, ref_params)}

    if op_class == "kink" and not gate_mask:
        raise ValueError(
            f"{name} is declared kinked, so it needs gate_mask={{param: (device_gate, "
            f"reference_gate)}}. A kinked gradient's rel_L2 is not mantissa-bounded -- A1 "
            f"measured linear_relu at 0.0625 against a 0.01 bar with HiFi4 reaching only "
            f"0.0128 -- so scoring it on rel_L2 either fails a correct op or passes a wrong "
            f"one depending on the input draw")

    # Level 2: the device against the verified reference.
    for k, got in tt_grads.items():
        if k not in ref_grads:
            raise KeyError(f"no reference gradient for {k!r}; have {sorted(ref_grads)}")
        m = metrics(got, ref_grads[k])
        ok = m["cos"] >= cos_bar
        if rel_bar is not None:
            ok = ok and m["rel_l2"] <= rel_bar
        entry = dict(m, rel_l2_bar=rel_bar, cos_bar=cos_bar, why=why)
        if op_class == "kink":
            dev_gate, ref_gate = gate_mask[k]
            agree = float(np.mean(np.asarray(dev_gate, bool) == np.asarray(ref_gate, bool)))
            entry["mask_agree"] = agree
            entry["mask_bar"] = MASK_BAR
            ok = ok and agree >= MASK_BAR
        entry["pass"] = bool(ok)
        rep.grads[k] = entry
    return rep


# ------------------------------------------------------------------- leaves against a total
#
# A leaf count with no denominator cannot show a shortfall. 2119 looks like a healthy number;
# 2119 of 2531 is a gap you can act on, and the difference between those two ways of reporting
# the same measurement is how a model with no trainable triangle multiplication passed a
# 100 %-call-routing result (LEDGER K29, R22). So the shipped check reports both, always, and
# it names every miss rather than counting them.


@dataclass
class WeightCoverage:
    """How many of a built model's device weights the tape can reach, out of how many exist."""

    total: int
    registered: int
    with_grad: int
    unregistered: list = field(default_factory=list)
    without_grad: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and not self.unregistered

    def __str__(self) -> str:
        out = [f"weights: {self.registered} of {self.total} registered as leaves, "
               f"{self.with_grad} of {self.total} carrying a gradient"]
        for label, names in (("NOT a leaf", self.unregistered),
                             ("no gradient", self.without_grad)):
            if names:
                shown = ", ".join(names[:8])
                out.append(f"  {len(names)} {label}: {shown}"
                           + (" ..." if len(names) > 8 else ""))
        return "\n".join(out)


def weight_coverage(model) -> WeightCoverage:
    """Walk ``model`` and report how much of it the tape reaches. Never a count alone.

    Asked by identity of the handle the model holds, which is the only form of the question
    that means anything: a name-keyed comparison against the optimizer's parameter dict agrees
    with itself for a weight the forward stopped reading. ``unregistered`` is therefore the
    load-bearing list -- a path on it is a tensor the forward multiplies and no optimizer can
    move.

    Take it AFTER a forward. Weights a module fuses lazily do not exist before one, so a
    coverage report on a freshly built model is a report on a smaller model than the one that
    runs.
    """
    from ..autograd import parameter_for
    from ..tenstorrent import device_weights
    found = device_weights(model)
    unregistered, without_grad = [], []
    with_grad = 0
    for name, raw in found.items():
        leaf = parameter_for(raw)
        if leaf is None:
            unregistered.append(name)
            continue
        if getattr(leaf, "grad", None) is None:
            without_grad.append(name)
        else:
            with_grad += 1
    return WeightCoverage(total=len(found), registered=len(found) - len(unregistered),
                          with_grad=with_grad, unregistered=unregistered,
                          without_grad=without_grad)


# D39: the only tensors whose composed bar may widen to the bf16-forward floor. Every gradient here flows
# through PairUpdate's transition ReLU, whose mask a bf16 forward flips; linear_2 and everything else keep D28.
FLOOR_SITES = (r"(^|\.)transition_out\.layer_norm\.", r"(^|\.)transition_out\.linear_1\.")
FLOOR_FACTOR = 1.5


def floor_site(name: str) -> bool:
    return any(re.search(p, name) for p in FLOOR_SITES)


def composed_verdict(rep: "GradcheckReport", floor: dict | None = None) -> dict:
    """Grade a composed block's report on ``COMPOSED_BAR``. Needs a verified reference.

    ``floor`` maps a tensor to {"rel", "cos"} of a bf16-emulated float64 reference against the float64
    one (D39). At ``FLOOR_SITES`` only, the tensor's bars widen to ``FLOOR_FACTOR`` times that distance:
    rel_l2 <= max(D28, 1.5 rel) and 1 - cos <= max(D28, 1.5^2 (1 - cos)), the same factor on the
    difference vector. The median bar never moves."""
    mx_bar, med_bar, cos_bar = COMPOSED_BAR
    if not (rep.reference_verified and rep.grads):
        return {"pass": False, "why": "reference not verified" if rep.grads else "no gradients"}
    rel = {k: g["rel_l2"] for k, g in rep.grads.items()}
    cos = {k: g["cos"] for k, g in rep.grads.items()}
    bars = {}
    for k in rel:
        f = (floor or {}).get(k)
        if f is not None and floor_site(k):
            bars[k] = (max(mx_bar, FLOOR_FACTOR * f["rel"]),
                       min(cos_bar, 1.0 - FLOOR_FACTOR ** 2 * (1.0 - f["cos"])))
    worst = max(rel, key=rel.get)
    low = min(cos, key=cos.get)
    med = float(np.median(list(rel.values())))
    fails = sorted(k for k in rel if rel[k] > bars.get(k, (mx_bar, cos_bar))[0]
                   or cos[k] < bars.get(k, (mx_bar, cos_bar))[1])
    out = {"max": [worst, rel[worst]], "median": med, "min_cos": [low, cos[low]], "fails": fails,
           "floored": {k: list(b) for k, b in bars.items()}}
    out["pass"] = bool(not fails and med <= med_bar)
    return out
