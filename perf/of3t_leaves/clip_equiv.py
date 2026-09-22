#!/usr/bin/env python3
"""R26's two clipping divergences, measured against THEIR grad_manager after the fix.

`of3t-orchestrator` measured both divergences and handed them here. This is the other half:
our side is `tt_bio.train.optim.AdamW`'s real methods, theirs is
`openfold3.core.utils.grad_manager` imported and EXECUTED, and the question is whether the two
now agree on the quantity R26 said they did not.

Their module is loaded with `pytorch_lightning` and `torchmetrics` stubbed, and that is
disclosed rather than hidden: `grad_manager.py:18` and `:22` import them at module scope, and
neither `compute_global_norm` nor the clip coefficient at `:183-187` touches either -- the
metrics are `PerSampleGradManager`'s logging counters. openfold3 0.5.0 is installed source-only
on this host (`pip install --no-deps --target`). The code under test is theirs, unmodified.

Bar: the float32 floor both stacks share, 6e-07 (PROTOCOL SS9/A7, K33) -- 2^-24 times sqrt of
the element count, because both sides reduce a sum of squares in float32 with different
association order. SS4's original 1e-12 is reported alongside so the mis-specification stays
visible rather than being swapped out.

    python3 perf/of3t_leaves/clip_equiv.py
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

OF3PKG = "/tmp/of3t/of3t-leaves/of3pkg"
CLIP = 10.0                  # their shipped `clip_val`
BAR = 1e-12                  # PROTOCOL SS4 as written, kept so the miss stays visible
F32_FLOOR = 6e-07            # 2^-24 * sqrt(~100), derived from the dtype, not from results
#: Their module-scope imports that nothing under test reads. `pytorch_lightning` is the trainer
#: handle; `MaxMetric`/`MeanMetric` are `PerSampleGradManager`'s logging counters, touched by
#: `log_unclipped_grad_metrics` and by nothing in `compute_global_norm` or in the clip
#: coefficient. Stubbing them is what lets their real file import on a host that has neither;
#: it is listed in the artifact so a reader can check the claim rather than take it.
STUBBED = ("pytorch_lightning", "torchmetrics")


def _their_grad_manager():
    """Their real module, with the one unused module-scope import stubbed."""
    sys.path.insert(0, OF3PKG)
    for name in STUBBED:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__getattr__ = lambda _n: object            # nothing below is read
            sys.modules[name] = m
    from openfold3.core.utils.grad_manager import compute_global_norm
    return compute_global_norm


def _ours(grads, clip_norm=CLIP, disabled=()):
    """`AdamW.grad_norm` and `AdamW.clip_coef`, the real methods, on a CPU-only instance."""
    from tt_bio.train.optim import AdamW

    class _P:
        def __init__(self, g):
            self.grad = None if g is None else np.asarray(g, dtype=np.float32)
            self.value = None

    opt = AdamW.__new__(AdamW)
    opt.params = {k: _P(v) for k, v in grads.items()}
    opt.clip_norm = float(clip_norm)
    gnorm = opt.grad_norm(disabled)
    return gnorm, opt.clip_coef(gnorm), opt


def _theirs(grads, clip_norm=CLIP, disabled=(), compute_global_norm=None):
    """Their `_clip_grads` body, `grad_manager.py:141-181`, on their own norm."""
    disabled = set(disabled)
    named = []
    for k, v in grads.items():
        if k in disabled:
            continue
        p = torch.nn.Parameter(torch.zeros(np.asarray(v).shape if v is not None else (1,)))
        p.grad = None if v is None else torch.tensor(np.asarray(v), dtype=torch.float32)
        named.append((k, p))
    gnorm, with_grad = compute_global_norm([p for _, p in named])
    if not with_grad:
        return 0.0, 1.0, {}
    mx = torch.tensor(float(clip_norm))
    coef = mx / torch.maximum(gnorm, mx)                          # grad_manager.py:183-187
    out = {}
    for name, p in named:
        if p.grad is None:
            continue
        p.grad.mul_(coef.to(p.dtype))                             # grad_manager.py:180-181
        out[name] = p.grad.numpy().copy()
    return float(gnorm), float(coef), out


def _rel(a, b):
    return abs(a - b) / (abs(b) + 1e-300)


def _rel_vec(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def main() -> int:
    cgn = _their_grad_manager()
    rng = np.random.default_rng(20260919)
    out, fails = {"instrument": "R26 -- our clipping against their grad_manager, after the fix",
                  "protocol_bar": BAR, "f32_floor": F32_FLOOR, "clip_norm": CLIP,
                  "their_source": OF3PKG, "stubbed_imports": list(STUBBED)}, []

    # ---- DIVERGENCE 1, R26(1): a disabled parameter must leave the global norm ------------
    g = {"a": rng.standard_normal(64).astype(np.float32) * 2,
         "conf": rng.standard_normal(256).astype(np.float32) * 6}
    gn_before, cl_before, _ = _ours(g)                             # what ours did before
    gn_o, cl_o, _ = _ours(g, disabled={"conf"})                    # what ours does now
    gn_t, cl_t, _ = _theirs(g, disabled={"conf"}, compute_global_norm=cgn)
    rel_now, rel_before = _rel(cl_o, cl_t), _rel(cl_before, cl_t)
    ok = rel_now <= F32_FLOOR
    out["disabled_params"] = {
        "ours_norm_before_the_fix": gn_before, "ours_clip_before_the_fix": cl_before,
        "relative_before_the_fix": rel_before,
        "ours_norm": gn_o, "theirs_norm": gn_t, "ours_clip": cl_o, "theirs_clip": cl_t,
        "relative_now": rel_now, "within_f32_floor": ok,
        "within_protocol_s4_bar": rel_now <= BAR}
    print(f"[{'PASS' if ok else 'FAIL'}] R26(1) disabled params: norm {gn_o:.6f} vs "
          f"{gn_t:.6f}, clip {cl_o:.9f} vs {cl_t:.9f}, rel {rel_now:.3e} "
          f"(was {rel_before:.3e} before the fix)")
    if not ok:
        fails.append(f"R26(1) reads {rel_now:.3e}, over the float32 floor")

    # ---- DIVERGENCE 2, R26(2): clip each sample, THEN accumulate -------------------------
    samples = [rng.standard_normal(64).astype(np.float32) * s for s in (0.2, 0.2, 40.0)]
    # theirs, per sample
    theirs_acc = np.zeros(64, np.float32)
    for s in samples:
        _, _, scaled = _theirs({"a": s}, compute_global_norm=cgn)
        theirs_acc += scaled["a"]
    # ours, per sample, through the real `clip_and_accumulate`
    from tt_bio.train.optim import AdamW

    class _P:
        def __init__(self):
            self.grad, self.value = None, None

    opt = AdamW.__new__(AdamW)
    p = _P()
    opt.params, opt.clip_norm = {"a": p}, CLIP
    opt.master = {"a": np.zeros(64, np.float32)}
    opt.accum, opt.participation, opt.accum_count = {}, {}, 0
    for s in samples:
        p.grad = s.copy()
        opt.clip_and_accumulate()
        p.grad = None
    ours_acc = opt.accum["a"]
    rel_ps = _rel_vec(ours_acc, theirs_acc)
    # and the per-BATCH arm, which is the algorithm R26 says is a different one
    _, cl_b, _ = _ours({"a": np.sum(samples, axis=0)})
    batch_acc = np.sum(samples, axis=0) * cl_b
    rel_algo = _rel_vec(batch_acc, theirs_acc)
    ok2 = rel_ps <= F32_FLOOR
    out["per_sample"] = {"samples": len(samples), "outlier_scale": 40.0,
                         "ours_per_sample_vs_theirs": rel_ps,
                         "ours_per_batch_vs_theirs": rel_algo,
                         "participation": opt.participation,
                         "within_f32_floor": ok2}
    print(f"[{'PASS' if ok2 else 'FAIL'}] R26(2) per-sample clip+accumulate: ours vs theirs "
          f"rel {rel_ps:.3e}; the per-BATCH arm reads {rel_algo:.3e} against the same "
          f"reference, which is the algorithm difference R26 measured")
    if not ok2:
        fails.append(f"R26(2) reads {rel_ps:.3e}, over the float32 floor")

    # ---- the negative control: both checks must be able to fail --------------------------
    _, cl_ref, _ = _theirs(g, disabled={"conf"}, compute_global_norm=cgn)
    _, cl_bad, _ = _ours(g, clip_norm=CLIP * 1.01, disabled={"conf"})
    c1 = _rel(cl_bad, cl_ref)
    # and for arm 2: drop ONE sample's clipping, which is the mistake the arm exists to catch
    wrong = np.zeros(64, np.float32)
    for i, s in enumerate(samples):
        if i == 2:
            wrong += s
        else:
            _, _, sc = _theirs({"a": s}, compute_global_norm=cgn)
            wrong += sc["a"]
    c2 = _rel_vec(wrong, theirs_acc)
    out["negative_control"] = {
        "arm1_perturbation": "clip_norm x1.01", "arm1_rel": c1,
        "arm1_rejected": c1 > F32_FLOOR, "arm1_margin": c1 / F32_FLOOR,
        "arm2_perturbation": "one sample left unclipped", "arm2_rel": c2,
        "arm2_rejected": c2 > F32_FLOOR, "arm2_margin": c2 / F32_FLOOR}
    print(f"[{'PASS' if c1 > F32_FLOOR and c2 > F32_FLOOR else 'FAIL'}] negative control: a 1 % "
          f"clip_norm error reads {c1:.3e} ({c1 / F32_FLOOR:.0f}x the floor); one sample left "
          f"unclipped reads {c2:.3e} ({c2 / F32_FLOOR:.0f}x). Both rejected.")
    if not (c1 > F32_FLOOR and c2 > F32_FLOOR):
        fails.append("a negative control did not fire")

    out["verdict"] = "PASS" if not fails else "FAIL"
    out["failures"] = fails
    p = Path("perf/of3t_leaves/clip_equiv.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nVERDICT: {out['verdict']}  ->  {p}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
