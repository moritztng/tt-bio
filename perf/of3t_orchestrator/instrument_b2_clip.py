"""PROTOCOL SS4, the clipping half: our global-norm clipping against THEIR grad_manager.

SS4 was finished for the LR schedule and left open for `grad_manager`. This closes it, on CPU,
with their real code on one side.

Their side is imported and executed, not transcribed: `openfold3.core.utils.grad_manager.
compute_global_norm` plus the clip coefficient at `grad_manager.py:185-187`,

    clip_coef = self._max_norm_tensor / torch.maximum(global_norm, self._max_norm_tensor)

Our side calls the real `tt_bio.train.optim.AdamW.grad_norm()` -- it runs on CPU because
`train/tensors.to_host` returns a numpy array untouched -- and the two-line clip decision it
feeds, quoted verbatim from `optim.py:171-172`:

    clip = (min(1.0, self.clip_norm / gnorm)
            if (self.clip_norm > 0 and gnorm > 0) else 1.0)

Only those two lines are transcribed rather than called, because the surrounding `step()` needs
`to_device` and therefore a card. They are quoted so the transcription is checkable by eye.

Bar, fixed by PROTOCOL SS4 before any of this ran: relative 1e-12 in float64 on the scaling
factor. THAT BAR IS MIS-SPECIFIED FOR THIS QUANTITY and the run reports against both it and the
right one, rather than quietly swapping them (see SS9/A7).

Why it is mis-specified, derived from the dtype and not from what was measured: both stacks
compute the global norm in FLOAT32 -- theirs at `grad_manager.py:49` (`p.grad.float()`), ours at
`optim.py` `grad_norm()` -- and they associate the sum differently, theirs as a norm over stacked
per-tensor norms and ours as a running sum of per-tensor dot products. Float32 unit roundoff is
2^-24 = 5.96e-08, and a sum of squares over N elements carries about sqrt(N) of it, so for the
~100-element cases here the floor is of order 6e-07. No pair of float32 implementations with
different association order can agree to 1e-12, so the original bar could only ever have been
failed. `f32_floor` below is that dtype-derived figure; `BAR` is what SS4 says.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from openfold3.core.utils.grad_manager import compute_global_norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.train.optim import AdamW  # noqa: E402

CLIP = 10.0
BAR = 1e-12              # PROTOCOL SS4 as written, kept so the miss stays visible
F32_FLOOR = 6e-07        # 2^-24 * sqrt(~100), derived from the dtype, not from the results


class _P:
    """The minimum an `AdamW` parameter needs for `grad_norm()`: a `.grad` numpy array."""

    def __init__(self, g):
        self.grad = None if g is None else np.asarray(g, dtype=np.float32)
        self.value = None


def _ours(grads: dict, clip_norm=CLIP):
    opt = AdamW.__new__(AdamW)
    opt.params = {k: _P(v) for k, v in grads.items()}
    opt.clip_norm = clip_norm
    gnorm = opt.grad_norm()                                   # the real method
    # ...and the real coefficient. This line used to TRANSCRIBE `optim.py:171-172`, which made
    # our side of the comparison a copy of the code under test: a change to `clip_coef` would
    # have left this instrument agreeing with a rule nothing ships. `clip_coef` was factored
    # out exactly so the per-sample and per-batch paths cannot drift, and calling it is the
    # only way this arm can notice if it does. If a branch has no `clip_coef`, this raises,
    # which is the right answer for a branch whose clipping this instrument cannot reach.
    clip = opt.clip_coef(gnorm)
    return gnorm, clip


def _theirs(grads: dict, clip_norm=CLIP, disabled=frozenset()):
    ps = []
    for k, v in grads.items():
        if k in disabled:
            continue
        p = torch.nn.Parameter(torch.zeros(np.asarray(v).shape if v is not None else (1,)))
        p.grad = None if v is None else torch.tensor(np.asarray(v), dtype=torch.float32)
        ps.append(p)
    gnorm, with_grad = compute_global_norm(ps)
    if not with_grad:
        return 0.0, 1.0
    mx = torch.tensor(float(clip_norm))
    coef = mx / torch.maximum(gnorm, mx)                      # grad_manager.py:185-187
    return float(gnorm), float(coef)


def _rel(a, b):
    return abs(a - b) / (abs(b) + 1e-300)


def main() -> int:
    rng = np.random.default_rng(20260919)
    out, fails = {"instrument": "PROTOCOL SS4 -- clipping", "protocol_bar": BAR,
                  "f32_floor": F32_FLOOR, "clip_norm": CLIP}, []

    # --- straddling cases, constructed so each lands on a named side of the threshold
    cases = {}
    for name, target in [("just_under", CLIP * 0.999), ("exactly_at", CLIP),
                         ("just_over", CLIP * 1.001), ("far_over", CLIP * 37.0),
                         ("tiny", CLIP * 1e-6)]:
        a = rng.standard_normal(64).astype(np.float32)
        b = rng.standard_normal(32).astype(np.float32)
        s = target / math.sqrt(float(a @ a) + float(b @ b))
        cases[name] = {"a": a * s, "b": b * s}
    cases["zero_grad"] = {"a": np.zeros(16, np.float32), "b": np.zeros(8, np.float32)}
    cases["some_absent"] = {"a": rng.standard_normal(16).astype(np.float32) * 9, "b": None}
    cases["all_absent"] = {"a": None, "b": None}

    rows = {}
    for name, g in cases.items():
        gn_o, cl_o = _ours(g)
        gn_t, cl_t = _theirs(g)
        r_gn, r_cl = _rel(gn_o, gn_t), _rel(cl_o, cl_t)
        worst = max(r_gn, r_cl)
        ok_f32, ok_s4 = worst <= F32_FLOOR, worst <= BAR
        rows[name] = {"ours_norm": gn_o, "theirs_norm": gn_t, "rel_norm": r_gn,
                      "ours_clip": cl_o, "theirs_clip": cl_t, "rel_clip": r_cl,
                      "within_f32_floor": ok_f32, "within_protocol_s4_bar": ok_s4}
        if not ok_f32:
            fails.append(f"{name}: worst rel {worst:.3e} exceeds the float32 floor")
        print(f"[{'PASS' if ok_f32 else 'FAIL'}] {name:12s} norm {gn_o:12.6f} vs {gn_t:12.6f} "
              f"(rel {r_gn:.2e})  clip {cl_o:.9f} vs {cl_t:.9f} (rel {r_cl:.2e})"
              f"{'' if ok_s4 else '   [over SS4 1e-12, float32 floor applies]'}")
    out["straddling"] = rows

    # --- negative control: the check must be able to fail
    g = {"a": rng.standard_normal(64).astype(np.float32) * 5,
         "b": rng.standard_normal(32).astype(np.float32) * 5}
    _, cl_ref = _theirs(g)
    _, cl_bad = _ours(g, clip_norm=CLIP * 1.01)
    ctrl = _rel(cl_bad, cl_ref)
    out["negative_control"] = {"perturbation": "clip_norm x1.01", "rel": ctrl,
                               "rejected": ctrl > F32_FLOOR,
                               "margin_over_floor": ctrl / F32_FLOOR}
    print(f"[{'PASS' if ctrl > F32_FLOOR else 'FAIL'}] negative control: a 1 % clip_norm error "
          f"reads rel {ctrl:.3e}, {ctrl / F32_FLOOR:.0f}x the float32 floor, "
          f"{'rejected' if ctrl > F32_FLOOR else 'NOT REJECTED'}")
    if ctrl <= F32_FLOOR:
        fails.append("negative control did not fire")

    # --- DIVERGENCE 1: disabled parameters are excluded from THEIR norm and not from ours
    g = {"a": rng.standard_normal(64).astype(np.float32) * 2,
         "conf": rng.standard_normal(256).astype(np.float32) * 6}
    gn_o, cl_o = _ours(g)
    gn_t, cl_t = _theirs(g, disabled={"conf"})
    out["disabled_params"] = {"ours_norm_all": gn_o, "theirs_norm_excluding": gn_t,
                              "ours_clip": cl_o, "theirs_clip": cl_t,
                              "rel_clip": _rel(cl_o, cl_t)}
    print(f"\n[DIVERGENCE] disabled params: their norm excludes them ({gn_t:.6f}) where ours "
          f"includes them ({gn_o:.6f}); clip {cl_o:.6f} vs {cl_t:.6f}, "
          f"rel {_rel(cl_o, cl_t):.3e}")

    # --- DIVERGENCE 2: per-sample clipping is a different algorithm, not a different constant
    samples = [rng.standard_normal(64).astype(np.float32) * s for s in (0.2, 0.2, 40.0)]
    per_sample = np.zeros(64, np.float32)
    for s in samples:
        _, c = _theirs({"a": s})
        per_sample += s * c
    accum = np.sum(samples, axis=0)
    _, c_batch = _ours({"a": accum})
    batch = accum * c_batch
    rel = float(np.linalg.norm(per_sample - batch) / (np.linalg.norm(per_sample) + 1e-300))
    out["per_sample_vs_batch"] = {"samples": 3, "outlier_scale": 40.0,
                                  "relative_difference": rel}
    print(f"[DIVERGENCE] per-sample vs per-batch clipping over 3 samples with one 200x "
          f"outlier: the accumulated gradients differ by {rel:.3e} relative")

    out["verdict"] = "PASS" if not fails else "FAIL"
    out["failures"] = fails
    p = Path(__file__).with_suffix(".json")
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nVERDICT: {out['verdict']}  ->  {p}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
