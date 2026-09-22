#!/usr/bin/env python3
"""PROTOCOL SS7 -- the N-step weight trajectory, at module scope.

With SS4 and SS5 verifying the schedule and the optimizer exactly, this has one job left:
catch WIRING -- a component applied in the wrong order, fed stale state, or not called. It
runs on the same module as instrument A (incoming triangle multiplication, block 0) and over
the six parameters that carry a gradient on both sides; the four fused input projections are
excluded for the reason instrument A records, not quietly dropped.

SS7a (amendment A1): the compared quantity is the UPDATE d_k = w_k - w_0, not w_k. Over a
warmup the weights barely move, so a relative L2 on w_k is dominated by a w_0 that is
identical BY CONSTRUCTION and would read as a pass for any implementation at all. k = 1 and
k = 2 are marked structurally non-discriminating and the growth law is fitted over k = 3..20.

THE BAR IS THE GROWTH LAW. Per-step divergence inside the SS3d bars is bf16 noise
accumulating. Divergence growing linearly or sub-linearly in k passes. Super-linear or
geometric growth fails AT ANY MAGNITUDE, including inside the bars, because that signature is
a different update rule rather than a rounding difference.

Both sides step their own optimizer on their own gradient from their own forward, from
identical weights and an identical input sequence. Theirs is torch.optim.Adam exactly as
`configure_optimizers` builds it; ours is tt_bio's AdamW configured to BE Adam
(weight_decay 0, clipping off), over an fp32 master, writing the updated weight back into the
module each step -- without that write-back the module would keep reading its original
handles and the run would silently measure nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

CKPT = "/home/ttuser/.boltz/of3-p2-155k.pt"
OUT = Path(__file__).resolve().parent / "instrument_t_traj.json"
C_Z = C_HIDDEN = 128
N_TOK, STEPS = 64, 20
OF3 = dict(lr=1.8e-3, beta1=0.9, beta2=0.95, eps=1e-8)

# their parameter -> (our registry key, our module attribute)
PAIRS = {
    "linear_g.weight": ("g_out.weight", "g_out_weight"),
    "linear_z.weight": ("p_out.weight", "out_p_weight"),
    "layer_norm_in.weight": ("norm_in.weight", "in_norm_weight"),
    "layer_norm_in.bias": ("norm_in.bias", "in_norm_bias"),
    "layer_norm_out.weight": ("norm_out.weight", "out_norm_weight"),
    "layer_norm_out.bias": ("norm_out.bias", "out_norm_bias"),
}


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (d + 1e-30)), float(d)


def main() -> int:
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import tenstorrent as tt
    from tt_bio import openfold3_weights as of3w
    from tt_bio.train.optim import AdamW, af3_lr
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming)

    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    rng = np.random.default_rng(23)
    batches = [(torch.from_numpy(rng.standard_normal((1, N_TOK, N_TOK, C_Z)) * 0.05),
                torch.from_numpy(rng.standard_normal((1, N_TOK, N_TOK, C_Z)) * 0.05))
               for _ in range(STEPS)]

    results = {}
    for tag, warmup in (("shipped_warmup_1000", 1000), ("scaled_warmup_5", 5)):
        sched = lambda s, w=warmup: af3_lr(s, OF3["lr"], warmup_steps=w,
                                           decay_every_n_steps=50000, decay_factor=0.95,
                                           base_lr=0.0, plateau_until=50000)

        # ---- theirs ---------------------------------------------------------------------
        pre = "pairformer_stack.blocks.0.pair_stack.tri_mul_in."
        sub = {k[len(pre):]: v.to(torch.float32) for k, v in sd.items() if k.startswith(pre)}
        them = TriangleMultiplicationIncoming(c_z=C_Z, c_hidden=C_HIDDEN)
        them.load_state_dict(sub, strict=False)
        tp = dict(them.named_parameters())
        w0_them = {n: p.detach().numpy().copy() for n, p in tp.items() if n in PAIRS}
        opt_t = torch.optim.Adam([p for n, p in tp.items()], lr=OF3["lr"],
                                 betas=(OF3["beta1"], OF3["beta2"]), eps=OF3["eps"])
        traj_them = []
        for k, (z, cot) in enumerate(batches, start=1):
            for g in opt_t.param_groups:
                g["lr"] = sched(k)
            opt_t.zero_grad()
            (them(z.to(torch.float32)) * cot.to(torch.float32)).sum().backward()
            opt_t.step()
            traj_them.append({n: p.detach().numpy().copy() for n, p in tp.items()
                              if n in PAIRS})

        # ---- ours -----------------------------------------------------------------------
        device = tt.get_device()
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        stack = of3w.remap_pairformer_stack(sd, "pairformer_stack")
        p2 = "layers.0.tri_mul_in."
        flat = {k[len(p2):]: v for k, v in stack.items() if k.startswith(p2)}
        ag.forget_parameters()
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
        named = dict(loaded)
        leaves = {ours: ag.parameter(named[ours]) for ours, _ in PAIRS.values()}
        opt_o = AdamW(leaves, lr=OF3["lr"], betas=(OF3["beta1"], OF3["beta2"]),
                      eps=OF3["eps"], weight_decay=0.0, clip_norm=0.0, schedule=sched)
        w0_ours = {k: opt_o.master[k].copy() for k in leaves}

        traj_ours = []
        for k, (z, cot) in enumerate(batches, start=1):
            opt_o.zero_grad()
            zt = ttnn.from_torch(z.to(torch.float32), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
            ct = ttnn.from_torch(cot.to(torch.float32), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
            with ag.tape():
                out = mod(ag.Tensor(zt, requires_grad=True))
            ag.backward([out], [ct])
            opt_o.step()
            # Re-key the registry onto the new handle AND point the module at it. Skipping
            # either is silent: the tape would accumulate into a leaf the optimizer no longer
            # steps, or the forward would keep reading the original weight.
            for ours, attr in PAIRS.values():
                ag.parameter(leaves[ours])
                setattr(mod, attr, leaves[ours].value)
            traj_ours.append({k2: opt_o.master[k2].copy() for k2 in leaves})

        # ---- compare the UPDATE, in their space -----------------------------------------
        per_step = []
        for k in range(STEPS):
            row = {"step": k + 1, "tensors": {}}
            for their, (ours, _) in PAIRS.items():
                d_them = traj_them[k][their] - w0_them[their]
                mine = traj_ours[k][ours]
                m0 = w0_ours[ours]
                d_ours = (mine - m0)
                if d_ours.ndim == 2:
                    d_ours = d_ours.T            # torch_to_tt transposes 2-D weights
                r, norm = rel_l2(d_ours, d_them)
                row["tensors"][their] = {"rel_l2": r, "update_norm": norm}
            vals = [(v["rel_l2"], n) for n, v in row["tensors"].items()]
            worst = max(vals, key=lambda x: x[0])
            row["worst_rel"], row["worst_tensor"] = worst[0], worst[1]
            per_step.append(row)

        # growth law over k = 3..20 (k = 1, 2 are structurally non-discriminating, SS7a)
        ks = np.arange(3, STEPS + 1, dtype=np.float64)
        ys = np.array([per_step[int(k) - 1]["worst_rel"] for k in ks], dtype=np.float64)
        ok_fit = np.all(ys > 0)
        slope = (float(np.polyfit(np.log(ks), np.log(ys), 1)[0]) if ok_fit else None)
        step1 = per_step[0]["worst_rel"]
        first_move = next((r["step"] for r in per_step[2:] if r["worst_rel"] > step1), None)

        results[tag] = {
            "warmup_no_steps": warmup,
            "lr_at_steps": {str(k): sched(k) for k in (1, 2, 3, 10, 20)},
            "per_step": per_step,
            "k1_k2_note": "structurally non-discriminating per SS7a; excluded from the fit",
            "growth_exponent_log_log_k3_k20": slope,
            "growth_verdict": (None if slope is None else
                               ("sub-linear or linear: PASS" if slope <= 1.0
                                else "SUPER-LINEAR: FAIL")),
            "first_tensor_to_move_beyond_step1": {
                "step": first_move,
                "tensor": (per_step[first_move - 1]["worst_tensor"] if first_move else None)},
            "worst_overall": max((r["worst_rel"], r["step"], r["worst_tensor"])
                                 for r in per_step),
        }
        print(f"[{tag}] lr(1)={sched(1):.3e} lr(20)={sched(20):.3e}")
        for r in per_step:
            print(f"   k={r['step']:2d}  worst {r['worst_rel']:.3e}  {r['worst_tensor']}")
        print(f"   growth exponent (log-log, k=3..20): {slope}")

    passed = all(v["growth_exponent_log_log_k3_k20"] is not None
                 and v["growth_exponent_log_log_k3_k20"] <= 1.0 for v in results.values())
    report = {"instrument": "PROTOCOL SS7 -- N-step trajectory (module scope)",
              "module": "TriangleMultiplicationIncoming, pairformer_stack.blocks.0",
              "N": STEPS, "compared_parameters": sorted(PAIRS),
              "excluded": ["linear_a_g.weight", "linear_b_g.weight",
                           "linear_a_p.weight", "linear_b_p.weight"],
              "excluded_reason": "the fused input projections carry no gradient on our side; "
                                 "see instrument_a_grad.json. Excluded, not zero-filled.",
              "runs": results,
              "bar": "growth law: linear or sub-linear in k passes; super-linear fails at "
                     "any magnitude",
              "verdict": "PASS" if passed else "FAIL"}
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nVERDICT: {report['verdict']}  ->  {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
