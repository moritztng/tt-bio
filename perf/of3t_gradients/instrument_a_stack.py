#!/usr/bin/env python3
"""PROTOCOL SS3 instrument A, at STACK scope: every parameter of a pairformer stack.

`of3t-equivalence` ran SS3 on one module, ten of their tensors against eight of ours, and that
was the right first step. This runs the same protocol over a whole B-block pairformer stack --
57 of their tensors per block against 53 of ours -- which is where the campaign's headline
number has to come from, because a per-parameter claim is a claim about every parameter.

WHAT IS COMPARED, and where it is compared (SS3a). In THEIR parameter space. Our flat remap
already fuses `linear_a_g`/`linear_b_g` into one `g_in.weight`, and the built model fuses again
and transposes on the way to the card, so the chain is two steps and both are derived rather
than transcribed: K22's published manifest for their key -> (flat key, dim-0 rows), and
`bijection_device.py` for flat key -> (device path, axis, band, layout), matched by exact bf16
value with every candidate confirmed elementwise.

THE REFERENCE (SS3c), and what it is NOT. Upstream's own `PairFormerBlock`, float64, loaded
strict from the same checkpoint, validated before use by float64 central finite differences --
per tensor, and jointly along one random unit direction through the whole parameter set. The
forward is checked first, because a reference that differentiates a plausible neighbour of our
function agrees with a WRONG gradient (PTX's mask-before-scale).

It is NOT the frozen bundle. `of3t-reference`'s BUNDLE-MIN is the only thing a published
equivalence claim may be measured against, and at the time of writing its directory on qb2 is
empty. This instrument exists so that when the bundle lands the only new thing is the data, and
its verdict line says so.

SS3b: a missing gradient and a zero gradient are different. Presence is compared as a set,
per parameter, BEFORE any magnitude. Nothing is filled with `zeros_like`.

K29: leaves are reported against total, never alone.

SS3e: the negative control runs and its result is part of the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "perf/of3t_gradients"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
K22 = "perf/of3t_equivalence/bijection_manifest.json"
PER_TENSOR_BAR, MEDIAN_BAR = 5.0e-02, 2.0e-02
FD_BAR = 1e-6


def rel_l2(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def load_ckpt():
    import torch
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def their_stack(sd, blocks, first=0):
    """Their `PairFormerBlock` x B from block `first`, float64, strict load."""
    import torch
    from openfold3.core.model.latent.pairformer import PairFormerBlock

    pre = "pairformer_stack.blocks."
    c_s = sd[pre + "0.attn_pair_bias.layer_norm_a.weight"].shape[0]
    c_z = sd[pre + "0.pair_stack.tri_mul_in.layer_norm_in.weight"].shape[0]
    no_heads_pair_bias = sd[pre + "0.attn_pair_bias.linear_z.weight"].shape[0]
    no_heads_pair = sd[pre + "0.pair_stack.tri_att_start.linear_z.weight"].shape[0]
    c_hidden_mul = sd[pre + "0.pair_stack.tri_mul_in.linear_a_p.weight"].shape[0]
    c_hidden_pair_att = (sd[pre + "0.pair_stack.tri_att_start.mha.linear_q.weight"].shape[0]
                         // no_heads_pair)
    transition_n = sd[pre + "0.pair_stack.pair_transition.swiglu.linear_a.weight"].shape[0] // c_z
    dims = dict(c_s=c_s, c_z=c_z, c_hidden_pair_bias=c_s // no_heads_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias, c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att, no_heads_pair=no_heads_pair,
                transition_type="swiglu", transition_n=transition_n, pair_dropout=0.25,
                fuse_projection_weights=False, inf=1e9)
    mods, report = [], []
    for i in range(first, first + blocks):
        sub = {k[len(f"{pre}{i}."):]: v.to(torch.float64)
               for k, v in sd.items() if k.startswith(f"{pre}{i}.")}
        m = PairFormerBlock(**dims).to(torch.float64)
        missing, unexpected = m.load_state_dict(sub, strict=True)
        m.eval()
        mods.append(m)
        report.append({"their_block": i, "their_tensors": len(sub),
                       "missing": list(missing), "unexpected": list(unexpected)})
    return mods, dims, report


def their_forward(mods, s, z, single_mask, pair_mask):
    for m in mods:
        s, z = m(s, z, single_mask, pair_mask)
    return s, z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2,
                    help="their float64 blocks run first, to put the probe in the regime a "
                         "trunk block actually sees (K28). 0 reproduces the invalid probe.")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--per-tensor-fd", action="store_true",
                    help="central differences per tensor as well as jointly")
    ap.add_argument("--fd-eps", type=float, default=1e-5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import numpy as np
    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import device_weights, get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    from tt_bio.train.checks import weight_coverage
    from tt_bio.train.lora import walked_weights
    from bijection_device import device_bijection

    t0 = time.perf_counter()
    tag = a.tag or f"b{a.blocks}_n{a.tokens}"
    rep = {"instrument": "PROTOCOL SS3 instrument A, pairformer stack scope",
           "blocks": a.blocks, "tokens": a.tokens, "checkpoint": CKPT,
           "bars": {"per_tensor": PER_TENSOR_BAR, "median": MEDIAN_BAR,
                    "reference_finite_difference": FD_BAR},
           "reference_is_the_frozen_bundle": False,
           "reference_note": ("self-generated float64 reference, upstream's own PairFormerBlock. "
                              "of3t-reference's BUNDLE-MIN is the only reference a PUBLISHED "
                              "equivalence claim may use; this run validates the harness so that "
                              "when the bundle lands the only new thing is the data.")}

    sd = load_ckpt()
    N = a.tokens

    # ---- the reference ---------------------------------------------------------------------
    mods, dims, load_report = their_stack(sd, a.blocks, first=a.warmup)
    rep["their_dims"], rep["their_load"] = dims, load_report
    n_their = sum(1 for m in mods for _ in m.named_parameters())
    rep["their_tensor_count"] = n_their
    print(f"[{time.perf_counter()-t0:.0f}s] reference: {a.blocks} PairFormerBlock, "
          f"{n_their} tensors, float64", flush=True)

    g = torch.Generator().manual_seed(11)
    s0 = torch.randn(1, N, dims["c_s"], generator=g) * 0.05
    z0 = torch.randn(1, N, N, dims["c_z"], generator=g) * 0.05
    cot_s = torch.randn(1, N, dims["c_s"], generator=g) * 0.05
    z_shape_out = (1, N, N, dims["c_z"])
    cot_z = torch.randn(*z_shape_out, generator=g) * 0.05
    single_mask = torch.ones(1, N, dtype=torch.float64)
    pair_mask = torch.ones(1, N, N, dtype=torch.float64)
    s64, z64 = s0.to(torch.float64), z0.to(torch.float64)
    cs64, cz64 = cot_s.to(torch.float64), cot_z.to(torch.float64)

    # K28, and it cost this instrument a run. A shape-correct input is not a valid input: on a
    # 0.05-scale unit-normal draw their block 0 reads d|z|/|z| = 1.000 -- the residual
    # contributes NOTHING and the block output is its own update, 148x the input norm. Every
    # module downstream then runs far outside the regime it was trained in and small relative
    # errors compound multiplicatively, which is how a composition of five sub-modules that
    # each agree to between 3.7e-03 and 4.5e-02 (`bisect_forward.json`) produced a stack
    # reading 5.1e-01. Warming the probe through their own float64 blocks puts it in the
    # regime: d|z|/|z| settles to 0.10-0.19 from block 1 on.
    warm = []
    if a.warmup:
        warm, _, _ = their_stack(sd, a.warmup, first=0)
        with torch.no_grad():
            for m in warm:
                s64, z64 = m(s64, z64, single_mask, pair_mask)
        s64, z64 = s64.detach(), z64.detach()
        s0, z0 = s64.to(torch.float32), z64.to(torch.float32)
    rep["probe"] = {"warmup_blocks": a.warmup,
                    "s_norm": float(s64.norm()), "z_norm": float(z64.norm()),
                    "note": "the compared stack is their blocks "
                            f"{a.warmup}..{a.warmup + a.blocks - 1}"}
    # The cotangent is drawn at the probe's own scale, so the backward is driven by a signal of
    # the size the loss actually produces rather than one 1e5 times too small.
    cs64 = (cot_s.to(torch.float64) / (cot_s.norm() + 1e-30)) * s64.norm()
    cz64 = (cot_z.to(torch.float64) / (cot_z.norm() + 1e-30)) * z64.norm()
    cot_s, cot_z = cs64.to(torch.float32), cz64.to(torch.float32)

    def loss_value():
        with torch.no_grad():
            s, z = their_forward(mods, s64, z64, single_mask, pair_mask)
            return float((s * cs64).sum() + (z * cz64).sum())

    for m in mods:
        for p in m.parameters():
            p.grad = None
    s_ref, z_ref = their_forward(mods, s64, z64, single_mask, pair_mask)
    ((s_ref * cs64).sum() + (z_ref * cz64).sum()).backward()
    g_ref, their_name = {}, {}
    for i, m in enumerate(mods):
        for n, p in m.named_parameters():
            g_ref[f"blocks.{i}.{n}"] = (p.grad.detach().clone()
                                        if p.grad is not None else None)
            their_name[f"blocks.{i}.{n}"] = f"pairformer_stack.blocks.{a.warmup + i}.{n}"
    s_ref, z_ref = s_ref.detach(), z_ref.detach()
    rep["reference_grad_present"] = {k: (v is not None) for k, v in g_ref.items()}
    print(f"[{time.perf_counter()-t0:.0f}s] reference backward done, "
          f"{sum(1 for v in g_ref.values() if v is not None)}/{len(g_ref)} with a gradient",
          flush=True)

    # ---- validate the reference against central finite differences (SS3c) --------------------
    rng = np.random.default_rng(5)
    params = [(f"blocks.{i}.{n}", p) for i, m in enumerate(mods) for n, p in m.named_parameters()]
    eps = a.fd_eps
    dirs = {n: torch.from_numpy(rng.standard_normal(tuple(p.shape))).to(torch.float64)
            for n, p in params}
    nrm = torch.sqrt(sum((d * d).sum() for d in dirs.values()))
    for d in dirs.values():
        d /= nrm
    base = {n: p.detach().clone() for n, p in params}
    with torch.no_grad():
        for n, p in params:
            p.copy_(base[n] + eps * dirs[n])
        f_plus = loss_value()
        for n, p in params:
            p.copy_(base[n] - eps * dirs[n])
        f_minus = loss_value()
        for n, p in params:
            p.copy_(base[n])
    joint_fd = (f_plus - f_minus) / (2 * eps)
    joint_an = float(sum((g_ref[n] * dirs[n]).sum() for n, _ in params if g_ref[n] is not None))
    rep["reference_fd_joint"] = {
        "analytic": joint_an, "finite_difference": joint_fd,
        "rel": abs(joint_fd - joint_an) / (abs(joint_an) + 1e-30),
        "direction": "one random unit direction through the whole parameter set",
        "eps": eps}
    print(f"[{time.perf_counter()-t0:.0f}s] joint FD {rep['reference_fd_joint']['rel']:.3e}",
          flush=True)

    fd_per = {}
    if a.per_tensor_fd:
        for n, p in params:
            if g_ref[n] is None:
                fd_per[n] = {"skipped": "no analytic gradient"}
                continue
            d = dirs[n] / (dirs[n].norm() + 1e-300)
            b = base[n]
            with torch.no_grad():
                p.copy_(b + eps * d)
                fp = loss_value()
                p.copy_(b - eps * d)
                fm = loss_value()
                p.copy_(b)
            an = float((g_ref[n] * d).sum())
            fd = (fp - fm) / (2 * eps)
            fd_per[n] = {"analytic": an, "finite_difference": fd,
                         "rel": abs(fd - an) / (abs(an) + 1e-30)}
        worst = max(((v.get("rel", 0.0), k) for k, v in fd_per.items()), key=lambda x: x[0])
        rep["reference_fd_per_tensor"] = {"worst_rel": worst[0], "worst_tensor": worst[1],
                                          "count": len(fd_per), "detail": fd_per}
        print(f"[{time.perf_counter()-t0:.0f}s] per-tensor FD worst {worst[0]:.3e} on {worst[1]}",
              flush=True)
    fd_rel = max([rep["reference_fd_joint"]["rel"]]
                 + ([rep["reference_fd_per_tensor"]["worst_rel"]] if fd_per else []))
    rep["reference_fd_pass"] = bool(fd_rel <= FD_BAR)

    # ---- ours ---------------------------------------------------------------------------------
    loaded, orig = [], T.Module.torch_to_tt

    def recording(self, key, *ar, **kw):
        t = orig(self, key, *ar, **kw)
        loaded.append(key)
        return t

    T.Module.torch_to_tt = recording
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = {}
    for j in range(a.blocks):
        src, dst = f"layers.{a.warmup + j}.", f"layers.{j}."
        flat.update({dst + k[len(src):]: v for k, v in flat_all.items() if k.startswith(src)})
    head_dim = flat["layers.0.tri_att_start.mha.linear_q.weight"].shape[0] // dims["no_heads_pair"]
    mod = T.Pairformer(a.blocks, head_dim, dims["no_heads_pair"],
                       dims["c_hidden_pair_bias"], dims["no_heads_pair_bias"],
                       True, flat, ckc, scale_pair_bias=False, fp32_softmax=True)
    T.Module.torch_to_tt = orig

    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    # A FRESH upload per forward, and this is load-bearing rather than tidy. `PairformerLayer`
    # accumulates its residuals with `ttnn.add_`, so the block MUTATES the tensor it was handed:
    # reuse the discovery forward's input for the taped one and the taped forward starts from
    # block 0's output. It does not raise and it does not look like a wiring bug -- it reads as
    # a forward that disagrees with the reference by 5.6e-01, which is how it was found here.
    before = device_weights(mod)
    ours = walked_weights(lambda: mod(ft(s0), ft(z0)), None, mod)
    after = device_weights(mod)
    rep["ours_discovery"] = {"from_loader": len(loaded),
                             "reachable_before_forward": len(before),
                             "reachable_after_forward": len(after),
                             "registered": len(ours),
                             "materialised_by_the_forward":
                                 sorted(set(after) - set(before))[:16]}
    print(f"[{time.perf_counter()-t0:.0f}s] ours: loader {len(loaded)}, before {len(before)}, "
          f"after {len(after)}, registered {len(ours)}", flush=True)

    err = None
    try:
        sa = ag.Tensor(ft(s0), requires_grad=True)
        za = ag.Tensor(ft(z0), requires_grad=True)
        with ag.tape():
            s_out, z_out = mod(sa, za)
        s_ours = ttnn.to_torch(s_out.value).to(torch.float64)
        z_ours = ttnn.to_torch(z_out.value).to(torch.float64)
        ag.backward([s_out, z_out], [ft(cot_s), ft(cot_z)])
    except Exception as e:                       # a gap is a finding, not a crash
        import traceback
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
        s_ours = z_ours = None
    rep["taped_backward_error"] = err

    cov = weight_coverage(mod)
    rep["leaves_against_total"] = {"total": cov.total, "registered": cov.registered,
                                   "with_grad": cov.with_grad,
                                   "unregistered": list(cov.unregistered),
                                   "without_grad": list(cov.without_grad)}
    print(f"[{time.perf_counter()-t0:.0f}s] LEAVES AGAINST TOTAL: {cov.registered}/{cov.total} "
          f"registered, {cov.with_grad}/{cov.total} with a gradient", flush=True)

    # SS3c: the forward first. bf16 device against float64 host, so this is a sanity reading
    # that the reference differentiates OUR function, not a parity bar.
    if s_ours is not None:
        rep["forward_rel"] = {"s": rel_l2(s_ours.numpy(), s_ref.numpy()),
                              "z": rel_l2(z_ours.numpy(), z_ref.numpy())}
        print(f"[{time.perf_counter()-t0:.0f}s] forward s {rep['forward_rel']['s']:.3e} "
              f"z {rep['forward_rel']['z']:.3e}", flush=True)

    # ---- the bijection, second half ----------------------------------------------------------
    dev_all = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(mod).items()}
    atoms_all = {f"blocks.{i}.{n}": p.detach().to(torch.float32)
                 for i, m in enumerate(mods) for n, p in m.named_parameters()}
    placements, per_device, dev_unmatched, their_unplaced = {}, {}, [], []
    for i in range(a.blocks):
        pre = f"blocks.{i}."
        dv = {k: v for k, v in dev_all.items() if k.startswith(pre)}
        at = {k[len(pre):]: v for k, v in atoms_all.items() if k.startswith(pre)}
        b = device_bijection(dv, at)
        for k, v in b["placements"].items():
            placements[pre + k] = v
        per_device.update(b["per_device"])
        dev_unmatched += b["device_unmatched"]
        their_unplaced += [pre + k for k in b["their_unplaced"]]
    rep["bijection_device"] = {
        "device_tensors": len(dev_all), "their_tensors": len(atoms_all),
        "their_placed": len(placements), "their_unplaced": their_unplaced,
        "device_unmatched": dev_unmatched,
        "kinds": {k: sum(1 for m in per_device.values() if m["kind"] == k)
                  for k in sorted({m["kind"] for m in per_device.values()})},
        "approx_placements": sorted({p["their"] for v in placements.values() for p in v
                                     if p["layout"].startswith("approx")}),
        "padding": {k: m["padding"] for k, m in per_device.items() if m["padding"]}}
    print(f"[{time.perf_counter()-t0:.0f}s] device bijection: {len(placements)}/{len(atoms_all)} "
          f"of their tensors placed, {len(dev_unmatched)} device tensors unmatched, "
          f"kinds {rep['bijection_device']['kinds']}", flush=True)

    # Independent corroboration of K22, which was built by a tracer through the CPU remap while
    # this was built by value against the BUILT model. Two different derivations of the same
    # fusion boundary; a disagreement is a finding for both.
    k22 = json.load(open(K22))["manifest"]
    agree = disagree = 0
    for their_key, pls in placements.items():
        full = their_name.get(their_key, "")
        spans = k22.get(full)
        if not spans or len(spans) != 1:
            continue
        flat_key = spans[0]["our_tensor"].split(".", 1)[1]
        fused = len(k22[full]) == 1 and (spans[0]["rows"][1] - spans[0]["rows"][0]) < \
            flat.get(flat_key, torch.zeros(1)).shape[0]
        if fused == any(p["length"] < max(p["device_shape"]) for p in pls):
            agree += 1
        else:
            disagree += 1
    rep["k22_crosscheck"] = {"agree_on_fused_or_not": agree, "disagree": disagree,
                             "note": "K22 by tracer through the CPU remap, this by value "
                                     "against the built model"}

    grads = {}
    for name, leaf in ours.items():
        grads[name] = (ttnn.to_torch(leaf.grad).to(torch.float64)
                       if leaf.grad is not None else None)

    def our_grad_for(their_key):
        """Their tensor's gradient in their layout, or (None, why)."""
        pls = placements.get(their_key)
        if not pls:
            return None, "no device tensor carries this parameter"
        chosen, why = None, []
        for pl in pls:
            gd = grads.get(pl["device_path"])
            why.append(f"{pl['device_path']}={'grad' if gd is not None else 'no grad'}")
            if gd is not None and chosen is None:
                chosen = (pl, gd)
        if chosen is None:
            return None, "; ".join(why)
        pl, gd = chosen
        band = gd.narrow(pl["axis"], pl["start"], pl["length"])
        if pl["layout"].endswith("transposed"):
            band = band.T
        return band.contiguous(), None

    rows, absent = [], []
    for their_key, ref in g_ref.items():
        full = their_name[their_key]
        mine, why = our_grad_for(their_key)
        if ref is None or mine is None:
            absent.append({"their_tensor": full,
                           "reference_present": ref is not None,
                           "ours_present": mine is not None, "reason": why})
            continue
        if tuple(mine.shape) != tuple(ref.shape):
            absent.append({"their_tensor": full, "shape_mismatch":
                           [list(mine.shape), list(ref.shape)]})
            continue
        r = ref.numpy()
        m = mine.numpy()
        rows.append({"their_tensor": full, "key": their_key,
                     "device": [p["device_path"] for p in placements[their_key]],
                     "placement": placements[their_key][0],
                     "rel_l2": rel_l2(m, r),
                     "max_abs_rel": float(np.max(np.abs(m - r)) / (np.max(np.abs(r)) + 1e-30)),
                     "ref_norm": float(np.linalg.norm(r))})
    rows.sort(key=lambda d: -d["rel_l2"])
    rel = [d["rel_l2"] for d in rows]
    rep["per_parameter"] = rows
    rep["absent"] = absent
    rep["presence"] = {
        "their_with_gradient": sum(1 for v in g_ref.values() if v is not None),
        "their_total": len(g_ref),
        "compared": len(rows), "absent": len(absent),
        "rule": "SS3b: None matches None, zero matches zero. Nothing was zero-filled."}
    rep["summary"] = {
        "compared": len(rows), "absent": len(absent),
        "worst": rows[0] if rows else None,
        "best": rows[-1] if rows else None,
        "median": float(np.median(rel)) if rel else None,
        "over_bar": [d for d in rows if d["rel_l2"] > PER_TENSOR_BAR],
        "per_tensor_pass": bool(rows) and max(rel) <= PER_TENSOR_BAR,
        "median_pass": bool(rel) and float(np.median(rel)) <= MEDIAN_BAR}

    # ---- SS3e negative control ---------------------------------------------------------------
    # SS3e asks for a 1 % perturbation. Against SS3d's 5 % bar that cannot fire, and
    # `of3t-equivalence` already measured and disclosed it: x1.01 moves a relative L2 by about
    # 1e-02, so it stays inside the bar unless the tensor already disagreed by >4e-02. Both
    # amplitudes are run and both are reported; no PROTOCOL bar is touched.
    if rows:
        victim = rows[-1]["their_tensor"]
        short = rows[-1]["key"]
        ref = g_ref[short].numpy()
        mine, _ = our_grad_for(short)
        mine = mine.numpy()
        amp = 1.0 + 2 * PER_TENSOR_BAR
        weak, strong = rel_l2(mine * 1.01, ref), rel_l2(mine * amp, ref)
        zeroed = rel_l2(np.zeros_like(mine), ref)
        others = [d["rel_l2"] for d in rows if d["their_tensor"] != victim]
        rep["negative_control"] = {
            "tensor": victim, "baseline_rel": rows[-1]["rel_l2"], "bar": PER_TENSOR_BAR,
            "protocol_1pct": {"rel_after": weak, "fires": bool(weak > PER_TENSOR_BAR)},
            "calibrated": {"perturbation": f"x{amp:.2f}", "rel_after": strong,
                           "fires": bool(strong > PER_TENSOR_BAR)},
            "zeroed": {"rel_after": zeroed, "fires": bool(zeroed > PER_TENSOR_BAR),
                       "answers": "which check fails if our model is replaced by zeros?"},
            "others_still_under_bar": bool(all(v <= PER_TENSOR_BAR for v in others)),
            "pass": bool(strong > PER_TENSOR_BAR and zeroed > PER_TENSOR_BAR)}

    ok = (rep["reference_fd_pass"] and rep["summary"]["per_tensor_pass"]
          and rep["summary"]["median_pass"] and not absent
          and rep.get("negative_control", {}).get("pass", False) and err is None)
    rep["verdict"] = "PASS" if ok else "FAIL"
    rep["scope_note"] = ("A PASS covers only the tensors in `per_parameter`. Anything in "
                         "`absent` was NOT compared and SS3b forbids filling it to make the "
                         "comparison run. This is the pairformer stack, not the whole model.")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"instrument_a_stack_{tag}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)

    print(f"\ncompared {len(rows)} of their tensors, {len(absent)} absent")
    for d in rows[:12]:
        print(f"   {d['rel_l2']:.3e}  {d['their_tensor']}")
    for x in absent[:20]:
        print(f"   ABSENT {x}")
    if rows:
        print(f"worst {rows[0]['rel_l2']:.3e} on {rows[0]['their_tensor']}")
        print(f"median {rep['summary']['median']:.3e} over {len(rows)} tensors")
        nc = rep["negative_control"]
        print(f"control on {nc['tensor']}: baseline {nc['baseline_rel']:.3e}, "
              f"x1.01 -> {nc['protocol_1pct']['rel_after']:.3e} "
              f"(fires={nc['protocol_1pct']['fires']}), "
              f"{nc['calibrated']['perturbation']} -> {nc['calibrated']['rel_after']:.3e} "
              f"(fires={nc['calibrated']['fires']}), zeroed -> "
              f"{nc['zeroed']['rel_after']:.3e} (fires={nc['zeroed']['fires']})")
    print(f"\nVERDICT: {rep['verdict']}  ->  {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
