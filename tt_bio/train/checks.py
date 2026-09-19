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
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

__all__ = ["gradcheck", "GradcheckReport", "BARS", "MASK_BAR", "FD_BAR", "OP_CLASSES",
           "metrics", "fd_check"]


# The reference's own bar against central differences. It is not a device number: this is
# float64 against float64 and a disagreement above it means the reference is wrong.
FD_BAR = 2e-6

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


def fd_check(loss_fn, params, n_probe=40, h=1e-5, seed=0, mag_floor=1e-6):
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
    worst, probed, eligible = 0.0, 0, 0
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
        idx = rng.choice(ok, size=min(n_probe, ok.size), replace=False)
        for i in idx:
            i = int(i)
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + h
            lp = loss_fn().item()
            with torch.no_grad():
                flat[i] = orig - h
            lm = loss_fn().item()
            with torch.no_grad():
                flat[i] = orig
            num = (lp - lm) / (2.0 * h)
            # Scaled by the gradient's own magnitude, not by this element's: a per-element
            # ratio is dominated by whichever probed coordinate is smallest, whose finite
            # difference is a difference of two nearly equal float64 losses.
            worst = max(worst, abs(num - ana[i].item()) / scale)
            probed += 1
    return worst, probed, eligible


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
               f"({self.fd_probed}/{self.fd_eligible} coords probed) {verdict}")
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
    worst, probed, eligible = fd_check(ref_loss, ref_params, n_probe=n_probe, h=h, seed=seed)
    rep.fd_worst, rep.fd_probed, rep.fd_eligible = worst, probed, eligible
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
