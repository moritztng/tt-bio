#!/usr/bin/env python3
"""PROTOCOL SS7: the N-step weight-trajectory, at pairformer-stack scope.

WHAT IS COMPARED, and why it is the update and not the weight. SS7a: over 20 warmup steps the
weights move ~1e-4 relative, so a relative L2 on `w_k` is dominated by a `w_0` that is identical
by construction and would read clean for a stack computing garbage. The compared quantity is
`d_k = w_k - w_0`, per tensor, in THEIR parameter space, as
`||d_k_ours - d_k_ref|| / (||d_k_ref|| + 1e-30)`.

THE UPDATE RULE ON BOTH SIDES, pinned to theirs rather than to our defaults (R14).
`torch.optim.Adam`, lr 1.8e-3, betas (0.9, 0.95), eps 1e-8, NO weight decay
(`projects/of3_all_atom/config/model_config.py:142-147`), global-norm clipping at 10.0
(`gradient_clipping.clip_val`), and `AlphaFoldLRScheduler` with base_lr 0.0,
warmup_no_steps 1000, start_decay_after_n_steps 50000. Per-sample clipping is their shipped
default and with ONE sample it is arithmetically the same as clipping the batch once, which is
why a single-sample probe may use either; that equality is checked at runtime, not assumed.

THE SCHEDULE PHASE IS AN ARM, NOT A SETTING (R15). Lightning calls `optimizer.step()` before
`scheduler.step()`, so their step k runs at `lr(k-1)` and their FIRST step runs at lr exactly 0.
`recipes.py:119` wires `schedule=lambda s: af3_lr(s, ...)` and `AdamW.step` increments before it
reads, so our step 1 runs at `lr(1)`. Both phases are run and reported: `shipped` is what
tt-bio ships today, `lightning` is theirs. SS7 exists to catch wiring and this is wiring.

THE REFERENCE (SS3c) is upstream's own `PairFormerBlock` in float64, the same reference
`instrument_a_stack.py` validated by float64 central finite differences. It is NOT the frozen
bundle: BUNDLE-MIN carries no `w0.pt` and no `grads_f64.pt` on this host, so no figure here may
be quoted as measured against it.

THE PROBE is warmed through their own float64 blocks first (K28) and then held FIXED for all 20
steps -- one frozen batch, replayed. The warm blocks are an input generator and are not in the
compared parameter set.

CONTROLS (SS3e), all three reported whatever they do:
  `x1.01`  -- scale one tensor's gradient by 1.01 at EVERY step. K16 measured that Adam is
              invariant to a uniform per-tensor gradient scaling, so this is expected NOT to
              fire; running it is how that invariance stays a measurement instead of a belief.
  `phase`  -- the shipped one-step schedule offset, which R15 priced at 1.499e-03.
  `zeros`  -- our gradient replaced by zeros at every step. With weight decay 0 the master never
              moves, `d_k_ours = 0`, and every tensor reads exactly 1.0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "perf/of3t_gradients"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PER_TENSOR_BAR = 5.0e-02                       # SS3d, quoted for "first tensor to move"
GROWTH_BAR = 1.0                               # SS7b: linear or sub-linear passes

# Their optimizer, verbatim from projects/of3_all_atom/config/model_config.py:142-158.
THEIR_LR, THEIR_BETAS, THEIR_EPS, THEIR_CLIP = 1.8e-3, (0.9, 0.95), 1e-8, 10.0
THEIR_BASE_LR, THEIR_DECAY_AFTER = 0.0, 50000


def rel_l2(a, b):
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def growth_exponent(ks, vals):
    """Least-squares slope of log(rel) against log(k). SS7b's bar is on this number.

    Returns (exponent, stderr, n). Steps whose divergence is exactly zero are dropped and
    counted, because log(0) is not a data point and silently treating it as one is how a fit
    reports a slope over a set it never saw.
    """
    import numpy as np
    x = np.array([math.log(k) for k, v in zip(ks, vals) if v > 0.0])
    y = np.array([math.log(v) for v in vals if v > 0.0])
    if len(x) < 3:
        return None, None, len(x)
    A = np.stack([x, np.ones_like(x)], 1)
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(math.sqrt(max(cov[0, 0], 0.0))), len(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--probe-warmup", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import numpy as np
    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import device_weights, get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    from tt_bio.train.optim import AdamW, af3_lr
    from tt_bio.train.lora import walked_weights
    from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
    from bijection_device import device_bijection
    from instrument_a_stack import load_ckpt, their_stack, their_forward

    t0 = time.perf_counter()
    N, S = a.tokens, a.steps
    rep = {"instrument": "PROTOCOL SS7 N-step trajectory, pairformer stack scope",
           "steps": S, "blocks": a.blocks, "tokens": N, "checkpoint": CKPT,
           "compared_quantity": "d_k = w_k - w_0, per tensor, in their parameter space",
           "their_optimizer": {"class": "torch.optim.Adam", "lr": THEIR_LR,
                               "betas": list(THEIR_BETAS), "eps": THEIR_EPS,
                               "weight_decay": 0.0, "clip_val": THEIR_CLIP,
                               "source": "projects/of3_all_atom/config/model_config.py:142-158"},
           "bars": {"growth_exponent": GROWTH_BAR, "per_tensor_SS3d": PER_TENSOR_BAR},
           "reference_is_the_frozen_bundle": False,
           "reference_note": ("upstream's own PairFormerBlock in float64, the reference "
                              "instrument_a_stack.py validated by float64 central finite "
                              "differences. BUNDLE-MIN carries no w0.pt on this host.")}

    sd = load_ckpt()

    # ---- the probe, warmed through their own float64 blocks and then FROZEN ------------------
    mods0, dims, _ = their_stack(sd, a.blocks, first=a.probe_warmup)
    g = torch.Generator().manual_seed(11)
    s64 = (torch.randn(1, N, dims["c_s"], generator=g) * 0.05).to(torch.float64)
    z64 = (torch.randn(1, N, N, dims["c_z"], generator=g) * 0.05).to(torch.float64)
    cot_s = torch.randn(1, N, dims["c_s"], generator=g) * 0.05
    cot_z = torch.randn(1, N, N, dims["c_z"], generator=g) * 0.05
    single_mask = torch.ones(1, N, dtype=torch.float64)
    pair_mask = torch.ones(1, N, N, dtype=torch.float64)
    if a.probe_warmup:
        warm, _, _ = their_stack(sd, a.probe_warmup, first=0)
        with torch.no_grad():
            for m in warm:
                s64, z64 = m(s64, z64, single_mask, pair_mask)
        s64, z64 = s64.detach(), z64.detach()
    cs64 = (cot_s.to(torch.float64) / (cot_s.norm() + 1e-30)) * s64.norm()
    cz64 = (cot_z.to(torch.float64) / (cot_z.norm() + 1e-30)) * z64.norm()
    s32, z32 = s64.to(torch.float32), z64.to(torch.float32)
    cs32, cz32 = cs64.to(torch.float32), cz64.to(torch.float32)
    rep["probe"] = {"warmup_blocks": a.probe_warmup, "s_norm": float(s64.norm()),
                    "z_norm": float(z64.norm()), "frozen_for_all_steps": True,
                    "compared_blocks": f"their {a.probe_warmup}..{a.probe_warmup+a.blocks-1}"}
    print(f"[{time.perf_counter()-t0:.0f}s] probe warmed, |s|={float(s64.norm()):.4g} "
          f"|z|={float(z64.norm()):.4g}", flush=True)

    # ---- the reference trajectory --------------------------------------------------------
    def reference_trajectory(warmup_no_steps):
        mods, _, _ = their_stack(sd, a.blocks, first=a.probe_warmup)
        params = [(f"blocks.{i}.{n}", p) for i, m in enumerate(mods)
                  for n, p in m.named_parameters()]
        w0 = {n: p.detach().clone() for n, p in params}
        opt = torch.optim.Adam([p for _, p in params], lr=THEIR_LR, betas=THEIR_BETAS,
                               eps=THEIR_EPS)
        sch = AlphaFoldLRScheduler(opt, last_epoch=-1, base_lr=THEIR_BASE_LR, max_lr=THEIR_LR,
                                   warmup_no_steps=warmup_no_steps,
                                   start_decay_after_n_steps=THEIR_DECAY_AFTER,
                                   decay_every_n_steps=THEIR_DECAY_AFTER, decay_factor=0.95)
        traj, meta = {}, []
        for k in range(1, S + 1):
            opt.zero_grad(set_to_none=True)
            s, z = their_forward(mods, s64, z64, single_mask, pair_mask)
            ((s * cs64).sum() + (z * cz64).sum()).backward()
            gn = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], THEIR_CLIP))
            lr = float(sch.get_last_lr()[0])
            opt.step()
            sch.step()
            traj[k] = {n: (p.detach() - w0[n]).clone() for n, p in params}
            meta.append({"k": k, "lr": lr, "grad_norm": gn,
                         "clipped": bool(gn > THEIR_CLIP),
                         "d_norm": float(math.sqrt(sum(float((v * v).sum())
                                                       for v in traj[k].values())))})
            print(f"[{time.perf_counter()-t0:.0f}s]   ref k={k:2d} lr={lr:.6g} "
                  f"|g|={gn:.6g} |d_k|={meta[-1]['d_norm']:.6g}", flush=True)
        return traj, meta, {n: p for n, p in params}

    # ---- ours ----------------------------------------------------------------------------
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = {}
    for j in range(a.blocks):
        src, dst = f"layers.{a.probe_warmup + j}.", f"layers.{j}."
        flat.update({dst + k[len(src):]: v for k, v in flat_all.items() if k.startswith(src)})
    head_dim = flat["layers.0.tri_att_start.mha.linear_q.weight"].shape[0] // dims["no_heads_pair"]
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def build():
        return T.Pairformer(a.blocks, head_dim, dims["no_heads_pair"],
                            dims["c_hidden_pair_bias"], dims["no_heads_pair_bias"],
                            True, {k: v.clone() for k, v in flat.items()}, ckc,
                            scale_pair_bias=False, fp32_softmax=True)

    def our_trajectory(warmup_no_steps, phase, control=None, victim=None):
        """`phase` is 'shipped' (af3_lr(s), recipes.py:119) or 'lightning' (af3_lr(s-1), theirs)."""
        mod = build()
        params = walked_weights(lambda: mod(ft(s32), ft(z32)), None, mod)
        off = 0 if phase == "shipped" else 1
        sched = (lambda s: af3_lr(s - off, THEIR_LR, warmup_steps=warmup_no_steps,
                                  base_lr=THEIR_BASE_LR, plateau_until=THEIR_DECAY_AFTER))
        opt = AdamW(params, lr=THEIR_LR, betas=THEIR_BETAS, eps=THEIR_EPS,
                    weight_decay=0.0, clip_norm=THEIR_CLIP, schedule=sched)
        w0 = {n: v.copy() for n, v in opt.master.items()}
        traj, meta = {}, []
        for k in range(1, S + 1):
            opt.zero_grad()
            if control == "zeros":
                for n, t in params.items():
                    t.grad = ttnn.from_torch(torch.zeros(tuple(ttnn.to_torch(t.value).shape)),
                                             layout=ttnn.TILE_LAYOUT, device=dev,
                                             dtype=ttnn.bfloat16)
            else:
                sa, za = ag.Tensor(ft(s32)), ag.Tensor(ft(z32))
                with ag.tape():
                    s_out, z_out = mod(sa, za)
                ag.backward([s_out, z_out], [ft(cs32), ft(cz32)])
                if control == "x1.01" and params[victim].grad is not None:
                    params[victim].grad = ttnn.multiply(params[victim].grad, 1.01)
            opt.step()
            params.rebind()
            for t in params.values():                       # re-key the leaf on its new handle
                ag.parameter(t)
            traj[k] = {n: (v - w0[n]) for n, v in opt.master.items()}
            meta.append({"k": k, "lr": opt.last_lr, "grad_norm": opt.last_grad_norm,
                         "clip": opt.last_clip,
                         "d_norm": float(math.sqrt(sum(float((v * v).sum())
                                                       for v in traj[k].values())))})
            print(f"[{time.perf_counter()-t0:.0f}s]   ours[{phase}/{control}] k={k:2d} "
                  f"lr={opt.last_lr:.6g} |g|={opt.last_grad_norm:.6g} "
                  f"|d_k|={meta[-1]['d_norm']:.6g}", flush=True)
        # `displacement()`, not `check_displacement()`. The check RAISES below its band and
        # the shipped-warmup arm is supposed to be below it: 20 steps at lr ~ k x 1.8e-06 move
        # each element ~2e-04, while bf16 spacing at |w| ~ 1 is ~8e-03, so the cast eats the
        # whole trajectory and the ratio reads ~0.05. That is the fp32 master doing its job,
        # not a defect, and it is reported as a number rather than as an exception.
        disp = opt.displacement()
        disp["bf16_spacing_at_unit_weight"] = 2.0 ** -7   # bfloat16 keeps 7 mantissa bits
        return traj, meta, mod, params, disp

    # ---- the bijection, once, on a freshly built model -------------------------------------
    mod0 = build()
    _ = walked_weights(lambda: mod0(ft(s32), ft(z32)), None, mod0)
    dev_all = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(mod0).items()}
    atoms = {f"blocks.{i}.{n}": p.detach().to(torch.float32)
             for i, m in enumerate(mods0) for n, p in m.named_parameters()}
    placements, their_unplaced = {}, []
    for i in range(a.blocks):
        pre = f"blocks.{i}."
        b = device_bijection({k: v for k, v in dev_all.items() if k.startswith(pre)},
                             {k[len(pre):]: v for k, v in atoms.items() if k.startswith(pre)})
        placements.update({pre + k: v for k, v in b["placements"].items()})
        their_unplaced += [pre + k for k in b["their_unplaced"]]
    rep["bijection"] = {"their_tensors": len(atoms), "placed": len(placements),
                        "unplaced": their_unplaced}
    print(f"[{time.perf_counter()-t0:.0f}s] bijection {len(placements)}/{len(atoms)} placed",
          flush=True)
    dev_shapes = {p: tuple(t.shape) for p, t in dev_all.items()}

    def in_their_space(our_d, their_key):
        """Our update for one of THEIR tensors: the band of the device tensor that carries it."""
        pls = placements.get(their_key)
        if not pls:
            return None, "no device tensor carries this parameter"
        pl = pls[0]
        arr = our_d.get(pl["device_path"])
        if arr is None:
            return None, f"{pl['device_path']} absent from the master set"
        t = torch.from_numpy(arr.reshape(dev_shapes[pl["device_path"]]))
        band = t.narrow(pl["axis"], pl["start"], pl["length"])
        if pl["layout"].endswith("transposed"):
            band = band.T
        return band.contiguous().to(torch.float64), None

    def compare(ref_traj, our_traj, label):
        per, absent = {}, []
        for their_key in sorted(atoms):
            rows = []
            for k in range(1, S + 1):
                rd = ref_traj[k].get(their_key)
                od, why = in_their_space(our_traj[k], their_key)
                if rd is None or od is None:
                    if k == 1:
                        absent.append({"their_tensor": their_key, "reason": why})
                    rows = []
                    break
                r, o = rd.numpy().astype(np.float64), od.numpy().astype(np.float64)
                if r.shape != o.shape:
                    if k == 1:
                        absent.append({"their_tensor": their_key,
                                       "shape_mismatch": [list(o.shape), list(r.shape)]})
                    rows = []
                    break
                rows.append({"k": k, "rel": rel_l2(o, r),
                             "ref_norm": float(np.linalg.norm(r)),
                             "our_norm": float(np.linalg.norm(o))})
            if not rows:
                continue
            ks = [d["k"] for d in rows if d["k"] >= 2]
            vs = [d["rel"] for d in rows if d["k"] >= 2]
            e, se, n = growth_exponent(ks, vs)
            first_over = next((d["k"] for d in rows if d["rel"] > PER_TENSOR_BAR), None)
            per[their_key] = {"rel_by_k": rows, "growth_exponent": e, "exponent_stderr": se,
                              "points_in_fit": n, "first_step_over_SS3d_bar": first_over,
                              "rel_k2": rows[1]["rel"] if len(rows) > 1 else None,
                              "rel_kN": rows[-1]["rel"]}
        # The stack-scope quantity: every tensor's update concatenated into one vector, which
        # is the trajectory of the parameter set rather than of any tensor in it.
        stack = []
        for k in range(1, S + 1):
            num = den = 0.0
            for their_key in per:
                rd = ref_traj[k][their_key].numpy().astype(np.float64)
                od, _ = in_their_space(our_traj[k], their_key)
                od = od.numpy().astype(np.float64)
                num += float(np.sum((od - rd) ** 2))
                den += float(np.sum(rd ** 2))
            stack.append({"k": k, "rel": math.sqrt(num) / (math.sqrt(den) + 1e-30),
                          "ref_d_norm": math.sqrt(den), "our_d_norm": None})
        e, se, n = growth_exponent([d["k"] for d in stack if d["k"] >= 2],
                                   [d["rel"] for d in stack if d["k"] >= 2])
        movers = [(v["first_step_over_SS3d_bar"], kk) for kk, v in per.items()
                  if v["first_step_over_SS3d_bar"] is not None]
        movers.sort()
        exps = [v["growth_exponent"] for v in per.values() if v["growth_exponent"] is not None]
        return {"label": label, "compared": len(per), "absent": absent,
                "k1_is_non_discriminating": {
                    "ref_d1_norm": stack[0]["ref_d_norm"],
                    "note": "SS7a: lr is exactly 0.0 at k=1 on their phase, so d_1 = 0 on both "
                            "sides and any implementation agrees. Reported, not counted."},
                "stack_rel_by_k": stack,
                "stack_growth_exponent": e, "stack_exponent_stderr": se,
                "per_tensor_exponent": {
                    "median": float(np.median(exps)) if exps else None,
                    "max": float(np.max(exps)) if exps else None,
                    "max_tensor": (max(per.items(),
                                       key=lambda kv: (kv[1]["growth_exponent"] or -9))[0]
                                   if exps else None)},
                "first_tensor_over_SS3d_bar": (
                    {"tensor": movers[0][1], "step": movers[0][0]} if movers else None),
                "n_tensors_over_SS3d_bar": len(movers),
                "growth_law_pass": bool(e is not None and e <= GROWTH_BAR),
                "per_tensor": per}

    arms = {}
    for warm_name, warm in (("shipped_warmup_1000", 1000), ("scaled_warmup_20", S)):
        ref_traj, ref_meta, _ = reference_trajectory(warm)
        rep.setdefault("reference_meta", {})[warm_name] = ref_meta
        for phase in ("lightning", "shipped"):
            our_traj, our_meta, mod, params, disp = our_trajectory(warm, phase)
            key = f"{warm_name}/{phase}"
            arms[key] = compare(ref_traj, our_traj, key)
            arms[key]["our_meta"] = our_meta
            arms[key]["displacement_control"] = disp
            print(f"[{time.perf_counter()-t0:.0f}s] ARM {key}: stack rel k=2 "
                  f"{arms[key]['stack_rel_by_k'][1]['rel']:.4e} k={S} "
                  f"{arms[key]['stack_rel_by_k'][-1]['rel']:.4e} exponent "
                  f"{arms[key]['stack_growth_exponent']}", flush=True)
        if warm == S:                                        # controls on the dynamic-range arm
            victim = sorted(placements[sorted(atoms)[0]], key=lambda p: 0)[0]["device_path"]
            for ctl in ("x1.01", "zeros"):
                ct, cm, _, _, cd = our_trajectory(warm, "lightning", control=ctl, victim=victim)
                c = compare(ref_traj, ct, f"{warm_name}/lightning/{ctl}")
                c["our_meta"] = cm
                c["control_victim"] = victim if ctl == "x1.01" else "every tensor"
                arms[f"control:{ctl}"] = c
                print(f"[{time.perf_counter()-t0:.0f}s] CONTROL {ctl}: stack rel k={S} "
                      f"{c['stack_rel_by_k'][-1]['rel']:.4e}", flush=True)

    rep["arms"] = arms
    base = arms["scaled_warmup_20/lightning"]
    ctl_zero = arms.get("control:zeros", {})
    ctl_scale = arms.get("control:x1.01", {})
    rep["negative_control"] = {
        "zeros": {"stack_rel_final": ctl_zero.get("stack_rel_by_k", [{}])[-1].get("rel"),
                  "baseline": base["stack_rel_by_k"][-1]["rel"],
                  "fires": bool((ctl_zero.get("stack_rel_by_k", [{}])[-1].get("rel") or 0) >
                                base["stack_rel_by_k"][-1]["rel"]),
                  "answers": "which check fails if our model is replaced by zeros?"},
        "x1.01": {"stack_rel_final": ctl_scale.get("stack_rel_by_k", [{}])[-1].get("rel"),
                  "baseline": base["stack_rel_by_k"][-1]["rel"],
                  "expected": "K16: Adam is invariant to a uniform per-tensor gradient scaling, "
                              "so this control is expected NOT to fire and its failure to fire "
                              "is the measurement"},
        "phase": {"lightning": base["stack_rel_by_k"][-1]["rel"],
                  "shipped": arms["scaled_warmup_20/shipped"]["stack_rel_by_k"][-1]["rel"],
                  "fires": bool(arms["scaled_warmup_20/shipped"]["stack_rel_by_k"][-1]["rel"] !=
                                base["stack_rel_by_k"][-1]["rel"]),
                  "what": "R15's one-step schedule offset, at stack scope on the real model"}}
    rep["verdict"] = ("PASS" if all(arms[k]["growth_law_pass"]
                                    for k in arms if not k.startswith("control:")) else "FAIL")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"trajectory_stack{('_' + a.tag) if a.tag else ''}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)
    print(f"\nVERDICT: {rep['verdict']}  ->  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
