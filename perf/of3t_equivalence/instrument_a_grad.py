#!/usr/bin/env python3
"""PROTOCOL SS3 -- per-parameter gradient equivalence, in THEIR parameter space.

One module: the INCOMING triangle multiplication of `pairformer_stack.blocks.0`, built from
the real OF3 preview2 checkpoint on both sides. It is chosen because it is where SS3a's
fusion lives -- their `linear_a_g`/`linear_b_g` are one `g_in.weight` on our side -- so it is
the case where a comparison made in OUR space could let one half's agreement mask the other
half's error. Ten of their tensors against eight of ours.

THE REFERENCE (SS3c). Upstream's own `TriangleMultiplicationIncoming`, in float64, with the
checkpoint's weights. Validated BEFORE it is used, twice:
  * its forward is checked against our device forward, so the reference is known to
    differentiate the function we actually compute rather than a plausible neighbour. PTX's
    SDPA trap is the precedent: a conventionally-written reference agrees with a WRONG
    gradient while both disagree with the forward;
  * its analytic gradient is checked against float64 CENTRAL FINITE DIFFERENCES along a
    random unit direction per parameter.

THE COMPARISON (SS3a/SS3d). Our fused gradient is split back along the fusion axis and each
half compared with its own tensor of theirs. Metric is relative L2 in float64,
||g_ours - g_ref|| / (||g_ref|| + 1e-30). Bars, fixed in the protocol before any number
existed: per-tensor 5.0e-02, median over tensors 2.0e-02. A tensor over the bar is the
finding and is reported, never smoothed.

SS3b: a missing gradient and a zero gradient are different. Any parameter whose gradient is
absent on either side is reported as absent. Nothing is filled with zeros to make the
comparison run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

CKPT = "/home/ttuser/.boltz/of3-p2-155k.pt"
OUT = Path(__file__).resolve().parent / "instrument_a_grad.json"

C_Z, C_HIDDEN, N = 128, 128, 64
PER_TENSOR_BAR, MEDIAN_BAR = 5.0e-02, 2.0e-02
FD_BAR = 1e-6          # the reference's own check, float64 against float64

# SS3a, for this module: their tensor -> (our tensor, rows in OUR pre-transpose layout).
# Our g_in/p_in are torch.cat([a, b], dim=0) of their two (c_hidden, c_z) projections.
FUSION = {
    "linear_a_g.weight": ("g_in.weight", slice(0, C_HIDDEN)),
    "linear_b_g.weight": ("g_in.weight", slice(C_HIDDEN, 2 * C_HIDDEN)),
    "linear_a_p.weight": ("p_in.weight", slice(0, C_HIDDEN)),
    "linear_b_p.weight": ("p_in.weight", slice(C_HIDDEN, 2 * C_HIDDEN)),
}
DIRECT = {
    "linear_g.weight": "g_out.weight",
    "linear_z.weight": "p_out.weight",
    "layer_norm_in.weight": "norm_in.weight",
    "layer_norm_in.bias": "norm_in.bias",
    "layer_norm_out.weight": "norm_out.weight",
    "layer_norm_out.bias": "norm_out.bias",
}


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def load_their_module(sd):
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming)
    pre = "pairformer_stack.blocks.0.pair_stack.tri_mul_in."
    sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if not sub:
        raise SystemExit(f"nothing under {pre!r}")
    m = TriangleMultiplicationIncoming(c_z=C_Z, c_hidden=C_HIDDEN)
    missing, unexpected = m.load_state_dict(
        {k: v.to(torch.float32) for k, v in sub.items()}, strict=False)
    return m.to(torch.float64), sub, list(missing), list(unexpected)


def their_grads(m, z64, cot64):
    for p in m.parameters():
        p.grad = None
    out = m(z64)
    (out * cot64).sum().backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in m.named_parameters()}, out.detach()


def fd_validate(m, z64, cot64, grads, rng, eps=1e-5):
    """Central differences along a random unit direction, per parameter. float64 throughout."""
    def value(sd_override=None):
        with torch.no_grad():
            return float((m(z64) * cot64).sum())

    out = {}
    for n, p in m.named_parameters():
        g = grads.get(n)
        if g is None:
            out[n] = {"skipped": "no analytic gradient"}
            continue
        d = torch.from_numpy(rng.standard_normal(tuple(p.shape))).to(torch.float64)
        d /= d.norm()
        base = p.detach().clone()
        with torch.no_grad():
            p.copy_(base + eps * d)
            f_plus = value()
            p.copy_(base - eps * d)
            f_minus = value()
            p.copy_(base)
        fd = (f_plus - f_minus) / (2 * eps)
        an = float((g.to(torch.float64) * d).sum())
        out[n] = {"analytic": an, "finite_difference": fd,
                  "rel": abs(fd - an) / (abs(an) + 1e-30)}
    return out


def main() -> int:
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import tenstorrent as tt
    from tt_bio import openfold3_weights as of3w

    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    report = {"instrument": "PROTOCOL SS3 -- per-parameter gradient equivalence",
              "module": "TriangleMultiplicationIncoming, pairformer_stack.blocks.0",
              "checkpoint": CKPT, "shape": {"N": N, "c_z": C_Z, "c_hidden": C_HIDDEN},
              "bars": {"per_tensor": PER_TENSOR_BAR, "median": MEDIAN_BAR,
                       "reference_fd": FD_BAR}}

    rng = np.random.default_rng(3)
    z = torch.from_numpy(rng.standard_normal((1, N, N, C_Z)) * 0.05)
    cot = torch.from_numpy(rng.standard_normal((1, N, N, C_Z)) * 0.05)
    z64, cot64 = z.to(torch.float64), cot.to(torch.float64)

    # ---------------- reference ----------------------------------------------------------
    them, sub, missing, unexpected = load_their_module(sd)
    report["reference"] = {"their_param_count": sum(1 for _ in them.named_parameters()),
                           "checkpoint_keys": sorted(sub),
                           "missing_on_load": missing, "unexpected_on_load": unexpected}
    g_ref, out_ref = their_grads(them, z64, cot64)
    report["reference"]["grad_present"] = {n: (v is not None) for n, v in g_ref.items()}

    fd = fd_validate(them, z64, cot64, g_ref, rng)
    fd_worst = max(((v.get("rel", 0.0), n) for n, v in fd.items()), key=lambda x: x[0])
    report["reference"]["finite_difference"] = fd
    report["reference"]["fd_worst"] = {"rel": fd_worst[0], "tensor": fd_worst[1]}
    report["reference"]["fd_pass"] = fd_worst[0] <= FD_BAR

    # ---------------- ours -----------------------------------------------------------------
    device = tt.get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    stack = of3w.remap_pairformer_stack(sd, "pairformer_stack")
    pre = "layers.0.tri_mul_in."
    flat = {k[len(pre):]: v for k, v in stack.items() if k.startswith(pre)}

    # Record every weight the module loads, the way of3t-tape's coverage harness does, and
    # register each as a trainable leaf. Without this the backward completes and produces
    # nothing for the parameters -- measured on origin/main, `grad_reach.json`.
    loaded, orig = [], tt.Module.torch_to_tt

    def recording(self, key, *a, **kw):
        t = orig(self, key, *a, **kw)
        loaded.append((key, t))
        return t

    tt.Module.torch_to_tt = recording
    try:
        mod = tt.TriangleMultiplication(True, flat, ckc)
    finally:
        tt.Module.torch_to_tt = orig

    # The fused input projections are built in __init__ from `flat` directly, not through
    # torch_to_tt, so they are registered explicitly by name.
    named = dict(loaded)
    fused_kind = {k: type(getattr(mod, k)).__name__ for k in ("_g_in_t", "_p_in_t")}
    for k in ("_g_in_t", "_p_in_t"):
        v = getattr(mod, k)
        if isinstance(v, torch.Tensor):
            continue          # not a device handle; see the note above
        named[k] = v
    leaves = {k: ag.parameter(t) for k, t in named.items()}
    report["ours"] = {"registered": sorted(leaves),
                      "registered_count": len(leaves),
                      "loaded_via_torch_to_tt": [k for k, _ in loaded],
                      "fused_input_projection_type": fused_kind,
                      "fused_note": "g_in/p_in are torch tensors on the module until "
                                    "_gp_in_chunks converts them per chunk width, so they "
                                    "are not device handles at registration time and no "
                                    "ttnn op ever presents them to the tape"}

    zt = ttnn.from_torch(z.to(torch.float32), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)
    cott = ttnn.from_torch(cot.to(torch.float32), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
    x = ag.Tensor(zt, requires_grad=True)
    with ag.tape():
        out = mod(x)
    out_ours = ttnn.to_torch(out.value).to(torch.float64)
    ag.backward([out], [cott])

    # Forward agreement first: a reference that differentiates a different function agrees
    # with a wrong gradient (SS3c). bf16 device against float64 host, so this is a sanity
    # bar, not a parity bar.
    report["forward_rel"] = rel_l2(out_ours.numpy(), out_ref.numpy())

    g_ours = {}
    for k, leaf in leaves.items():
        g_ours[k] = (ttnn.to_torch(leaf.grad).to(torch.float64).numpy()
                     if leaf.grad is not None else None)
    report["ours"]["grad_present"] = {k: (v is not None) for k, v in g_ours.items()}

    # ---------------- compare, in THEIR space ----------------------------------------------
    def our_tensor(name):
        """Our gradient for one of their keys, transposed back to their (out, in) layout."""
        if name in FUSION:
            ours_key, rows = FUSION[name]
            src = {"g_in.weight": "_g_in_t", "p_in.weight": "_p_in_t"}[ours_key]
            g = g_ours.get(src)
            if g is None:
                return None, (f"{src} is not a registered device parameter: the fused input "
                              f"projection stays a torch tensor until _gp_in_chunks converts "
                              f"it, so the tape never sees it")
            # `_g_in_t` is g_in.weight.t(), i.e. (c_z, 2*c_hidden); undo, then slice rows.
            return np.asarray(g).T[rows], None
        ours_key = DIRECT[name]
        g = g_ours.get(ours_key)
        if g is None:
            return None, f"{ours_key} carries no gradient"
        g = np.asarray(g)
        return (g.T if g.ndim == 2 else g), None

    rows, absent = [], []
    for name, ref in g_ref.items():
        mine, why = our_tensor(name)
        if ref is None or mine is None:
            absent.append({"their_tensor": name, "reference_present": ref is not None,
                           "ours_present": mine is not None, "reason": why})
            continue
        r = np.asarray(ref.numpy(), np.float64)
        if mine.shape != r.shape:
            absent.append({"their_tensor": name, "shape_mismatch":
                           [list(mine.shape), list(r.shape)]})
            continue
        rows.append({"their_tensor": name, "our_tensor": FUSION.get(name, (DIRECT.get(name),))[0],
                     "rel_l2": rel_l2(mine, r),
                     "max_abs_rel": float(np.max(np.abs(mine - r)) /
                                          (np.max(np.abs(r)) + 1e-30)),
                     "ref_norm": float(np.linalg.norm(r))})

    rows.sort(key=lambda d: -d["rel_l2"])
    rel = [d["rel_l2"] for d in rows]
    report["per_parameter"] = rows
    report["absent"] = absent
    report["summary"] = {
        "compared": len(rows), "absent": len(absent),
        "worst": rows[0] if rows else None,
        "median": float(np.median(rel)) if rel else None,
        "per_tensor_pass": bool(rows) and max(rel) <= PER_TENSOR_BAR,
        "median_pass": bool(rel) and float(np.median(rel)) <= MEDIAN_BAR,
    }

    # ---------------- negative control (SS3e) ----------------------------------------------
    # SS3e says "perturb exactly one parameter's gradient by 1 % and confirm the check
    # fails". Run literally against SS3d's 5.0e-02 per-tensor bar, that control CANNOT fire,
    # and the arithmetic is not subtle: scaling a gradient by 1.01 moves its relative L2 by
    # about 1e-02, so unless the tensor already disagrees by more than 4e-02 the perturbed
    # value stays under a 5e-02 bar. Measured here: the best-agreeing tensor sits at
    # 5.381e-03 and x1.01 takes it to 1.090e-02, comfortably inside the bar. A 1 %
    # perturbation against a 5 % bar is not a control, it is a tautology.
    #
    # So the control is run at a perturbation the bar can actually resolve -- x(1 + 2*bar),
    # which must exceed the bar by construction -- and the x1.01 result is KEPT and reported
    # as the evidence that the protocol's own figure needs amending. This changes no
    # PROTOCOL tolerance: the bars are untouched and the comparison above was scored against
    # them unmodified. It changes only this instrument's own probe amplitude, and it is
    # disclosed here rather than left in a diff.
    if rows:
        victim = rows[-1]["their_tensor"]          # the best-agreeing tensor, hardest case
        ref = np.asarray(g_ref[victim].numpy(), np.float64)
        mine, _ = our_tensor(victim)
        mine = np.asarray(mine)
        weak = rel_l2(mine * 1.01, ref)
        amp = 1.0 + 2 * PER_TENSOR_BAR
        strong = rel_l2(mine * amp, ref)
        zeroed = rel_l2(np.zeros_like(mine), ref)
        others = [d["rel_l2"] for d in rows if d["their_tensor"] != victim]
        report["negative_control"] = {
            "tensor": victim, "baseline_rel": rows[-1]["rel_l2"], "bar": PER_TENSOR_BAR,
            "protocol_1pct": {"perturbation": "x1.01", "rel_after": weak,
                              "fires": weak > PER_TENSOR_BAR,
                              "finding": "SS3e's 1 % perturbation cannot exceed SS3d's 5 % "
                                         "bar; the protocol's control is unfireable as "
                                         "written and needs amending to a perturbation "
                                         "larger than the bar"},
            "calibrated": {"perturbation": f"x{amp:.2f}", "rel_after": strong,
                           "fires": strong > PER_TENSOR_BAR},
            "zeroed": {"perturbation": "our gradient replaced by zeros", "rel_after": zeroed,
                       "fires": zeroed > PER_TENSOR_BAR,
                       "question": "which check fails if our model is replaced by zeros?"},
            "others_unchanged_and_under_bar": all(v <= PER_TENSOR_BAR for v in others),
            "pass": strong > PER_TENSOR_BAR and zeroed > PER_TENSOR_BAR}

    s = report["summary"]
    ok = (report["reference"]["fd_pass"]
          and s["per_tensor_pass"] and s["median_pass"]
          and report.get("negative_control", {}).get("pass", False))
    report["complete"] = not absent
    report["scope_note"] = ("PASS covers only the parameters that carry a gradient on both "
                            "sides. Any tensor in `absent` was NOT compared, and SS3b forbids "
                            "filling it with zeros to make the comparison run.")
    report["verdict"] = "PASS" if ok else "FAIL"
    OUT.write_text(json.dumps(report, indent=2, default=str))

    print(f"reference: {report['reference']['their_param_count']} params, "
          f"FD worst {fd_worst[0]:.3e} on {fd_worst[1]} "
          f"({'PASS' if report['reference']['fd_pass'] else 'FAIL'} vs {FD_BAR:.0e})")
    print(f"forward (bf16 device vs float64 host): {report['forward_rel']:.3e}")
    print(f"registered leaves: {len(leaves)}; "
          f"with gradient: {sum(1 for v in g_ours.values() if v is not None)}")
    print(f"compared {len(rows)} of their tensors, {len(absent)} absent")
    for d in rows:
        print(f"   {d['rel_l2']:.3e}  {d['their_tensor']}  -> {d['our_tensor']}")
    for a in absent:
        print(f"   ABSENT {a}")
    if rows:
        print(f"worst {s['worst']['rel_l2']:.3e} on {s['worst']['their_tensor']}, "
              f"median {s['median']:.3e}")
        nc = report["negative_control"]
        print(f"[{'PASS' if nc['pass'] else 'FAIL'}] negative control on {nc['tensor']} "
              f"(baseline {nc['baseline_rel']:.3e}, bar {nc['bar']:.1e}): "
              f"x1.01 -> {nc['protocol_1pct']['rel_after']:.3e} "
              f"(fires={nc['protocol_1pct']['fires']}), "
              f"{nc['calibrated']['perturbation']} -> {nc['calibrated']['rel_after']:.3e} "
              f"(fires={nc['calibrated']['fires']}), "
              f"zeroed -> {nc['zeroed']['rel_after']:.3e} "
              f"(fires={nc['zeroed']['fires']})")
    print(f"\nVERDICT: {report['verdict']}  ->  {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
