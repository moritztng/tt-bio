#!/usr/bin/env python3
"""The pairformer stack's REAL boundary and REAL incoming cotangent, taken from the reference's
own full-model run on batch_step003.

WHY THIS EXISTS (D237). `perf/of3t_modelboundary/runarm.sh` drives the trunk with
`block47_boundary.pt`, a capture-driven crop-384 walk, while the model reference
`grads_f64_043.pt` is the full-model backward on `batch_step003`. Different cotangent, therefore
a different trunk gradient: the two float64 references disagree at 1.8416532191 at cos 0.36514
with no device op on either side, and 82.93 % of the graded artifact's `pairformer_stack` error
mass is that mismatch rather than our port. So the campaign's one failing clause is currently a
statement about an artifact. This script produces the two tensors that fix it.

WHAT IT DOES NOT DO. It does not reimplement the reference step. The model build, the dropout
policy, the cast policy, the draw replay and the deterministic-kernel pin are
`perf/of3t_reference/bundle_min.py`'s own functions, imported. The only thing added is two hooks:
a forward pre-hook on `model.pairformer_stack` that records what the stack was CALLED with, and
a tensor hook on each of its two outputs that records the gradient that FLOWS BACK into them.
Neither consumes randomness, changes a dtype or touches the graph, and the control for that claim
is not an argument: `--expect-loss` requires the float64 loss to be BIT-IDENTICAL to the value
the published bundle recorded, and `--ref-grads` scores this run's own trunk parameter gradients
against the published reference in the same process. If anything about the build, the load or the
draws differed, neither would hold.

The checkpoint load below is the one block that is a copy of `bundle_min.main()` rather than a
call into it, because extracting a helper would edit the producer every other row runs. The key
gate is copied with it: a reference built on a checkpoint whose tensors have nowhere to go is
not a reference (D23/R126).

  capture_model_frame.py --batch .../batch_step003.pt --batch-sha256 3c32... \
      --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
      --replay-draws .../draws_recycles0.pt --ref-grads .../grads_f64_043.pt \
      --expect-loss 1.267624369070698 --out-dir /home/ttuser/of3t_modelframe --threads 14
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "of3t_reference"))
import bundle_min as bm  # noqa: E402  the reference producer, imported rather than copied

PRE = "pairformer_stack."


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def norms(t):
    return None if t is None else float(torch.linalg.vector_norm(t.double()))


def score(mine, ref, prefix=PRE):
    """Mass-weighted relative L2, norm ratio and cosine of two per-tensor gradient dicts, plus
    the best single scalar a* = argmin_a ||a*mine - ref|| and the residual that survives it.

    The residual answers of3t-frameself step 0 directly: a near-constant ratio at 48
    independently-parameterised blocks is either one scalar applied once, in which case the
    residual collapses, or an average of 48 unrelated numbers, in which case it does not.
    """
    names = sorted(n for n in ref if n.startswith(prefix) and ref[n] is not None)
    ref_sq = err_sq = arm_sq = dot = 0.0
    worst, worst_name = -1.0, None
    n_bit = n_absent = 0
    for n in names:
        r = ref[n].to(torch.float64).reshape(-1)
        m = mine.get(n)
        if m is None:
            n_absent += 1
            ref_sq += float(torch.linalg.vector_norm(r)) ** 2
            continue
        m = m.to(torch.float64).reshape(-1)
        if torch.equal(m, r):
            n_bit += 1
        rn = float(torch.linalg.vector_norm(r))
        en = float(torch.linalg.vector_norm(m - r))
        ref_sq += rn ** 2
        err_sq += en ** 2
        arm_sq += float(torch.linalg.vector_norm(m)) ** 2
        dot += float(torch.dot(m, r))
        rel = en / rn if rn > 0 else (0.0 if en == 0 else float("inf"))
        if rel > worst:
            worst, worst_name = rel, n
    a_star = dot / arm_sq if arm_sq > 0 else None
    # ||a*m - r||^2 = a*^2 ||m||^2 - 2 a* <m,r> + ||r||^2, and at the optimum that is
    # ||r||^2 - <m,r>^2/||m||^2 up to round-off. Formed that way to keep it positive.
    res_sq = max(ref_sq - (dot ** 2 / arm_sq if arm_sq > 0 else 0.0), 0.0)
    return {
        "n_tensors": len(names), "n_absent_from_arm": n_absent, "n_bit_identical": n_bit,
        "mass_weighted_rel_l2": (err_sq / ref_sq) ** 0.5 if ref_sq else None,
        "norm_ratio_arm_over_reference": (arm_sq / ref_sq) ** 0.5 if ref_sq else None,
        "cos": dot / (arm_sq * ref_sq) ** 0.5 if arm_sq > 0 and ref_sq > 0 else None,
        "best_scalar_a_star": a_star,
        "residual_after_best_scalar_frac_of_reference": (res_sq / ref_sq) ** 0.5 if ref_sq
        else None,
        "reference_squared_norm": ref_sq, "arm_squared_norm": arm_sq,
        "worst_rel_l2": worst if worst >= 0 else None, "worst_tensor": worst_name,
    }


def per_block(mine, ref, n_blocks=48):
    out = {}
    for i in range(n_blocks):
        pre = f"{PRE}blocks.{i}."
        if any(n.startswith(pre) for n in ref):
            out[str(i)] = score(mine, ref, pre)
    return out


def tensor_pair(name, mine, theirs, mask=None):
    """Elementwise agreement of two tensors, and again over the real token rows only. The
    padded frame is 384 tokens with 56 real ones, so a norm over the whole tensor is 97.9 %
    pad in z and a match there is not a match on the part that carries the answer."""
    if mine is None or theirs is None:
        return {"name": name, "present": False}
    m, t = mine.to(torch.float64), theirs.to(torch.float64)
    if m.shape != t.shape:
        return {"name": name, "present": True, "shape_mismatch": [list(m.shape), list(t.shape)]}
    d = {"name": name, "present": True, "shape": list(m.shape),
         "bit_identical": bool(torch.equal(m, t)),
         "norm_mine": float(torch.linalg.vector_norm(m)),
         "norm_theirs": float(torch.linalg.vector_norm(t)),
         "rel_l2": float(torch.linalg.vector_norm(m - t))
         / max(float(torch.linalg.vector_norm(t)), 1e-300)}
    if mask is not None:
        real = mask.reshape(-1) > 0
        if m.dim() >= 3 and m.shape[1] == real.numel() and m.shape[2] == real.numel():
            mm, tt = m[:, real][:, :, real], t[:, real][:, :, real]
        elif m.dim() >= 2 and m.shape[1] == real.numel():
            mm, tt = m[:, real], t[:, real]
        else:
            mm = tt = None
        if mm is not None:
            d["real_tokens_only"] = {
                "bit_identical": bool(torch.equal(mm, tt)),
                "norm_theirs": float(torch.linalg.vector_norm(tt)),
                "rel_l2": float(torch.linalg.vector_norm(mm - tt))
                / max(float(torch.linalg.vector_norm(tt)), 1e-300),
                "share_of_full_squared_norm": (float(torch.linalg.vector_norm(tt)) ** 2
                                               / max(float(torch.linalg.vector_norm(t)) ** 2,
                                                     1e-300)),
            }
    return d


def run_selftest(model, b, c, grads, a, mode):
    """Differentiate the REAL `model.pairformer_stack`, in this process, on the captured pair.

    Same module object, same parameter tensors, same call kwargs the model itself used, same
    boundary, same cotangent. The only thing that changes against the full-model backward is
    that the cotangent is injected at the stack's outputs instead of arriving from the loss.
    If THAT reproduces `grads_f64_043.pt`'s trunk section, the captured pair is sound and the
    defect is in `ref_grad.py`'s reconstructed 48-block loop. If it does not, the pair itself
    is insufficient and no reconstruction can be blamed.
    """
    stack = model.pairformer_stack
    kw_taped = dict(b.get("call_kwargs_nontensor") or {})
    # Only the plain arguments: the pre-hook stores a repr() for anything else, and a repr
    # passed back as a string is not the argument. Dropping one silently would replay a
    # DIFFERENT call while reporting agreement, so this refuses instead. On this frame nothing
    # is dropped -- all seven of the taped call's non-tensor kwargs are plain.
    kw = {k: v for k, v in kw_taped.items()
          if isinstance(v, (int, float, bool, type(None)))}
    dropped = sorted(set(kw_taped) - set(kw))
    if dropped:
        raise SystemExit(
            f"--selftest cannot replay the taped call: {dropped} came back from the pre-hook as "
            f"a repr() and would be replaced by the module's default. Widen the pre-hook to "
            f"keep the object rather than replaying a different call.")
    bpc_real = getattr(stack, "blocks_per_ckpt", None)
    bpc_used = bpc_real if bpc_real is not None else 1
    note_bpc = ("as the model had it" if bpc_real is not None else
                "the model had None; forced to 1 because a non-checkpointed 48-block float64 "
                "backward at 384 tokens does not fit in host memory. Recompute of a block with "
                "dropout pinned to r=0 is deterministic, and of3t-twoside measured the identity "
                "at 4 blocks: 0.034117901729881786 plain against 0.0341179017298818 "
                "checkpointed, which agrees to 1e-16 and is not bit-identical")

    model.zero_grad(set_to_none=True)

    s_in = b["s_in"].detach().clone().requires_grad_(True)
    z_in = b["z_in"].detach().clone().requires_grad_(True)
    sm = b["single_mask"].detach().clone()
    pm = b["pair_mask"].detach().clone()
    cot_s, cot_z = c["cot_s"], c["cot_z"]

    prev = stack.blocks_per_ckpt
    stack.blocks_per_ckpt = bpc_used
    t0 = time.time()
    policy2 = bm.cast_policy(mode, "cpu")
    with policy2:
        s_out, z_out = stack(s=s_in, z=z_in, single_mask=sm, pair_mask=pm, **kw)
        loss2 = (s_out * cot_s).sum() + (z_out * cot_z).sum()
    t_f = time.time() - t0
    t0 = time.time()
    loss2.backward()
    t_b = time.time() - t0
    stack.blocks_per_ckpt = prev

    mine = {}
    for name, p in model.named_parameters():
        if name.startswith(PRE):
            mine[name] = p.grad.detach().to(torch.float64).clone() if p.grad is not None else None
    n_none = sum(1 for v in mine.values() if v is None)

    ref = None
    if a.ref_grads:
        r = torch.load(a.ref_grads, map_location="cpu", weights_only=False, mmap=True)
        ref = {k: r[k] for k in r if k.startswith(PRE)}

    out = {
        "what": "the REAL model.pairformer_stack object, differentiated on the captured pair in "
                "the same process as the capture",
        "module_state_at_the_taped_call": b.get("module_state"),
        "call_kwargs_replayed": kw,
        "call_kwargs_nontensor_at_the_taped_call": b.get("call_kwargs_nontensor"),
        "blocks_per_ckpt_used": bpc_used, "blocks_per_ckpt_note": note_bpc,
        "cotangent_hook_firings_outputs": c.get("out_fired"),
        "cotangent_hook_firings_inputs": b.get("in_fired"),
        "injected_loss": float(loss2),
        "seconds_forward": t_f, "seconds_backward": t_b,
        "n_trunk_tensors": len(mine), "n_without_gradient": n_none,
        "forward_reproduces_the_taped_outputs": {
            "s": tensor_pair("s_out", s_out.detach().to(torch.float64), c.get("s_out"), sm),
            "z": tensor_pair("z_out", z_out.detach().to(torch.float64), c.get("z_out"), sm),
        },
        "input_cotangent_vs_the_real_backward": {
            "s": tensor_pair("ds_in", s_in.grad, b.get("cot_in_s"), sm),
            "z": tensor_pair("dz_in", z_in.grad, b.get("cot_in_z"), sm),
            "caveat": "the real backward's dL/ds_in and dL/dz_in include every consumer of "
                      "those two tensors, not only the stack. At num_recycles 0 the stack is "
                      "their only taped consumer inside run_trunk, but this is a read and not "
                      "a control.",
        },
    }
    if ref is not None:
        out["vs_published_reference"] = score(mine, ref)
        out["vs_published_reference_per_block"] = per_block(mine, ref)
        out["bar"] = 1e-12
        out["verdict"] = ("H-B: the captured pair reproduces the reference through the real "
                          "module, so ref_grad.py's reconstructed stack is the defect"
                          if (out["vs_published_reference"]["mass_weighted_rel_l2"] or 1.0) <= 1e-12
                          else "H-A: the captured pair does not reproduce the reference even "
                               "through the real module, so the pair is insufficient")
    out["vs_this_runs_own_full_model_backward"] = score(
        mine, {k: v for k, v in grads.items() if k.startswith(PRE) and v is not None})

    if a.selftest_compare and a.selftest_compare.exists():
        d = torch.load(a.selftest_compare, map_location="cpu", weights_only=False)
        rg = d.get("grads", {})
        cmp_ = {
            "path": str(a.selftest_compare),
            "what": "ref_grad.py's reconstructed 48-block loop on the same pair",
            "recorded_loss": d.get("loss"), "policy": d.get("policy"), "tree": d.get("tree"),
            "its_forward_vs_the_real_module": {
                "s": tensor_pair("s", d.get("s"), s_out.detach().to(torch.float64), sm),
                "z": tensor_pair("z", d.get("z"), z_out.detach().to(torch.float64), sm),
            },
            "its_input_gradient_vs_the_real_module": {
                "s": tensor_pair("ds_in", d.get("ds_in"), s_in.grad, sm),
                "z": tensor_pair("dz_in", d.get("dz_in"), z_in.grad, sm),
            },
            "its_parameter_gradient_vs_the_real_module": score(rg, mine),
        }
        if ref is not None:
            cmp_["its_parameter_gradient_vs_published_reference"] = score(rg, ref)
        out["ref_grad_reconstruction"] = cmp_

    print("SELFTEST " + json.dumps(
        {k: out.get(k) for k in ("verdict", "injected_loss", "blocks_per_ckpt_used")}
        | {"vs_reference": out.get("vs_published_reference")}, indent=1), flush=True)
    return out


def run_cotprobe(cots, loss, published_cot, out_dir, tag):
    """--cotprobe only: read the stack's output cotangent with a SECOND instrument.

    The published pair's cotangent comes from a tensor hook. `of3t-frameself` has shown the
    768 trunk tensors reachable only through `cot_s` reproduce `grads_f64_043.pt` exactly while
    the 1968 reachable through `cot_z` do not, so the hook's `cot_z` is the one term left under
    suspicion. `torch.autograd.grad(loss, z_out)` answers the same question without a hook and
    without touching the trunk's backward, which is what makes this cheap: it stops at z_out.

    It also enumerates every graph node that consumes s_out and z_out, which is H-A's "second
    path" read directly off the tape instead of inferred.
    """
    taped = [c for c in cots if c["grad_enabled"]]
    if len(taped) != 1:
        raise SystemExit(f"the trunk was taped {len(taped)} times, not once")
    c = taped[0]
    s_out, z_out = c["s_out_t"], c["z_out_t"]

    # every node on the tape that takes s_out or z_out as an input
    targets = {"s_out": s_out.grad_fn, "z_out": z_out.grad_fn}
    consumers = {k: [] for k in targets}
    seen, stack, n_nodes = set(), [loss.grad_fn], 0
    while stack:
        n = stack.pop()
        if n is None or id(n) in seen:
            continue
        seen.add(id(n)); n_nodes += 1
        for nf, idx in getattr(n, "next_functions", ()):
            if nf is None:
                continue
            for k, t in targets.items():
                if nf is t:
                    consumers[k].append({"consumer": type(n).__name__, "input_index": int(idx)})
            stack.append(nf)

    t0 = time.time()
    g_s, g_z = torch.autograd.grad(loss, [s_out, z_out], retain_graph=True, allow_unused=True)
    t_probe = time.time() - t0

    pub = torch.load(published_cot, map_location="cpu", weights_only=False)
    pub_s, pub_z = pub["cot"]
    rec = {}
    for key, mine, theirs in (("s", g_s, pub_s), ("z", g_z, pub_z)):
        if mine is None:
            rec[key] = {"present": False}
            continue
        m = mine.detach().to(torch.float64).reshape(-1)
        t = theirs.to(torch.float64).reshape(-1)
        tn = float(torch.linalg.vector_norm(t))
        d = float(torch.linalg.vector_norm(m - t))
        rec[key] = {"present": True, "bit_identical": bool(torch.equal(m, t)),
                    "norm_autograd_grad": float(torch.linalg.vector_norm(m)),
                    "norm_published_hook": tn,
                    "rel_l2_vs_published_hook": (d / tn) if tn else d,
                    "ratio_autograd_over_hook":
                        float(torch.linalg.vector_norm(m)) / tn if tn else None}
    return {
        "what": "the stack's output cotangent read by torch.autograd.grad instead of by a "
                "tensor hook, and every tape node that consumes s_out / z_out",
        "host": socket.gethostname(), "row": "of3t-frameself", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
        "published_cotangent": {"path": str(published_cot),
                                "sha256": sha256_file(published_cot)},
        "loss": float(loss),
        "tape_nodes_walked_from_the_loss": n_nodes,
        "consumers_of_the_stack_outputs": {k: {"n": len(v), "nodes": v}
                                           for k, v in consumers.items()},
        "autograd_grad_vs_the_published_hook": rec,
        "seconds_autograd_grad": t_probe,
        "verdict": {
            "s": ("the hook and autograd.grad agree bit for bit" if rec["s"].get("bit_identical")
                  else "the hook and autograd.grad DISAGREE on cot_s"),
            "z": ("the hook and autograd.grad agree bit for bit" if rec["z"].get("bit_identical")
                  else "the hook and autograd.grad DISAGREE on cot_z"),
        },
    }


def run_blockprobe(cots, loss, model, blocks, ref_grads, injected, out_dir, tag,
                   blk=None):
    """--blockprobe only: read the REFERENCE's own trunk gradient with a second instrument.

    Every premise that would force the injected replay to reproduce `grads_f64_043.pt` has now
    been measured and holds -- same module object, boundary reproducing the outputs bit for
    bit, one hook firing, cotangent confirmed by `torch.autograd.grad`, no trunk parameter
    object registered at a second name -- and the replay still disagrees on the 1,968 tensors
    `cot_z` reaches. The one quantity never read twice is the reference gradient itself.

    `torch.autograd.grad(loss, <one block's parameters>)` prunes the graph to that block, so the
    last block costs ONE block's backward instead of forty-eight. Block 47 is where the injected
    replay's residual peaks (0.30026 against a 0.0895 median), which makes it the cheapest place
    to ask whether `p.grad` and `autograd.grad` agree with each other.
    """
    taped = [c for c in cots if c["grad_enabled"]]
    c = taped[0]
    names, params = [], []
    for i in blocks:
        for n, prm in model.named_parameters():
            if n.startswith(f"pairformer_stack.blocks.{i}."):
                names.append(n); params.append(prm)
    extra = ([blk["in_s"], blk["in_z"]] if blk and "in_s" in blk else [])
    t0 = time.time()
    got = torch.autograd.grad(loss, [c["s_out_t"], c["z_out_t"]] + extra + params,
                              retain_graph=False, allow_unused=True)
    t_probe = time.time() - t0
    g_s, g_z = got[0], got[1]
    real_in = got[2:2 + len(extra)]
    gp = got[2 + len(extra):]
    mine = {n: (g.detach().to(torch.float64) if g is not None else None)
            for n, g in zip(names, gp)}

    def load(path):
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return d["grads"] if isinstance(d, dict) and "grads" in d else d

    def fit(arm, ref, keys):
        dot = a2 = r2 = e2 = 0.0
        worst, worst_name, n_bit = -1.0, None, 0
        for n in keys:
            r = ref[n].to(torch.float64).reshape(-1)
            m = arm[n]
            if m is None:
                continue
            m = m.reshape(-1)
            rn2 = float(torch.dot(r, r)); r2 += rn2
            dot += float(torch.dot(m, r)); a2 += float(torch.dot(m, m))
            e = float(torch.dot(m - r, m - r)); e2 += e
            if torch.equal(m, r):
                n_bit += 1
            rel = (e / rn2) ** 0.5 if rn2 > 0 else (0.0 if e == 0 else float("inf"))
            if rel > worst:
                worst, worst_name = rel, n
        return {"n_tensors": len(keys), "n_bit_identical": n_bit,
                "rel_l2_as_is": (e2 / r2) ** 0.5 if r2 else None,
                "norm_ratio_arm_over_ref": (a2 / r2) ** 0.5 if r2 else None,
                "cos": dot / (a2 * r2) ** 0.5 if a2 > 0 and r2 > 0 else None,
                "best_scalar_a_star": dot / a2 if a2 > 0 else None,
                "residual_after_best_scalar_frac_of_ref":
                    (max(r2 - dot * dot / a2, 0.0) / r2) ** 0.5 if a2 > 0 and r2 else None,
                "ref_squared_norm": r2, "arm_squared_norm": a2,
                "worst_rel_l2": worst, "worst_tensor": worst_name}

    # ONE BLOCK, standalone, in this same process. Everything that could differ between the
    # reference's backward and a replay is held fixed here: the same block object, the input the
    # real forward handed it, the cotangent the real backward put on its output. If this
    # reproduces the real backward, no single block is the defect.
    oneblock = None
    if blk and "in_s" in blk:
        bi = blk["index"]
        block = model.pairformer_stack.blocks[bi]
        bn = [n for n in names if n.startswith(f"pairformer_stack.blocks.{bi}.")]
        bp = [prm for n, prm in model.named_parameters() if n in set(bn)]
        keep = [prm.grad for prm in bp]
        for prm in bp:
            prm.grad = None
        si = blk["in_s"].detach().clone().requires_grad_(True)
        zi = blk["in_z"].detach().clone().requires_grad_(True)
        t1 = time.time()
        so, zo = block(si, zi, *blk["args_rest"], **blk["kwargs"])
        fwd = {"s": tensor_pair("block_out_s", so, blk["out_s"]),
               "z": tensor_pair("block_out_z", zo, blk["out_z"])}
        ((so * c["cot_s"]).sum() + (zo * c["cot_z"]).sum()).backward()
        t_one = time.time() - t1
        mine_one = {n: (prm.grad.detach().to(torch.float64) if prm.grad is not None else None)
                    for n, prm in zip(bn, bp)}
        real_one = {n: g for n, g in zip(names, gp) if n in set(bn)}
        oneblock = {
            "what": "the probed block replayed on its own, on the boundary the real forward "
                    "handed it and the cotangent the real backward put on its output, in this "
                    "same process",
            "block": bi, "n_tensors": len(bn),
            "seconds": t_one,
            "forward_reproduces_the_real_block_output": fwd,
            "standalone_replay_vs_autograd_grad": fit(mine_one, real_one, bn),
            "standalone_replay_vs_grads_f64_043": fit(mine_one, load(ref_grads), bn),
            "input_cotangent": {
                k: tensor_pair("d" + k, t.grad, r)
                for k, t, r in (("in_s", si, real_in[0] if real_in else None),
                                ("in_z", zi, real_in[1] if len(real_in) > 1 else None))},
        }
        for prm, g in zip(bp, keep):
            prm.grad = g
        # the whole defect, as one block's boundary. D242 now has a 13-second reproducer that
        # needs no model forward: load this, run the block, and the 1.679 is back.
        dp = out_dir / f"ONEBLOCK_{tag}.pt"
        torch.save({"block": bi,
                    "in_s": blk["in_s"].detach().to(torch.float64).clone(),
                    "in_z": blk["in_z"].detach().to(torch.float64).clone(),
                    "kwargs_tensor": {k: v.detach().clone()
                                      for k, v in blk["kwargs"].items() if torch.is_tensor(v)},
                    "kwargs_other": {k: v for k, v in blk["kwargs"].items()
                                     if not torch.is_tensor(v)},
                    "cot_s": c["cot_s"], "cot_z": c["cot_z"],
                    "out_s": blk["out_s"].detach().to(torch.float64).clone(),
                    "out_z": blk["out_z"].detach().to(torch.float64).clone(),
                    "real_grad": {n: g.clone() for n, g in real_one.items() if g is not None},
                    "real_din_s": real_in[0].detach().to(torch.float64).clone(),
                    "real_din_z": real_in[1].detach().to(torch.float64).clone()}, dp)
        oneblock["reproducer"] = {"path": str(dp), "bytes": dp.stat().st_size,
                                  "sha256": sha256_file(dp)}

    rg = load(ref_grads)
    inj = load(injected)
    per = {}
    for i in blocks:
        keys = [n for n in names if n.startswith(f"pairformer_stack.blocks.{i}.")]
        per[str(i)] = {
            "autograd_grad_vs_grads_f64_043": fit(mine, rg, keys),
            "injected_replay_vs_grads_f64_043": fit({n: inj[n] for n in keys}, rg, keys),
            "autograd_grad_vs_injected_replay": fit(mine, inj, keys),
        }
    return {
        "what": "the reference's own trunk gradient read by torch.autograd.grad on one block's "
                "parameters instead of by p.grad after a full backward",
        "host": socket.gethostname(), "row": "of3t-frameself", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
        "blocks": list(blocks), "loss": float(loss),
        "reference": {"path": str(ref_grads), "sha256": sha256_file(ref_grads),
                      "is": "the full-model float64 backward on batch_step003, num_recycles 0"},
        "injected_arm": {"path": str(injected), "sha256": sha256_file(injected),
                         "is": "of3t-twoside's injected float64 replay through ref_grad.py"},
        "cotangent_reread": {
            "cot_s_norm": norms(g_s), "cot_z_norm": norms(g_z),
            "bit_identical_to_the_hook": {
                "s": bool(torch.equal(g_s.detach().to(torch.float64), c["cot_s"])),
                "z": bool(torch.equal(g_z.detach().to(torch.float64), c["cot_z"]))}},
        "seconds_autograd_grad": t_probe,
        "one_block_standalone": oneblock,
        "per_block": per,
    }


def run_graphdrive(entries, cots, loss, model, ref_grads, injected, zonly, out_dir,
                   tag, blocks, published_cot=None):
    """--graphdrive only: drive the ORIGINAL graph with the captured cotangent.

    Amendment 4/5. Every premise that would force the replay to reproduce `grads_f64_043.pt`
    has been measured and holds, so the premise nobody wrote down is the one under test here: a
    bit-exact FORWARD does not imply an identical BACKWARD GRAPH. `--selftest` clones the
    boundary and re-runs the forward, so it differentiates a RECONSTRUCTION and cannot see a
    defect in reconstructing a graph. `--blockprobe` drives the original graph but from the
    LOSS, which confirms the reference and not the injection.

    This is the missing cell: the original graph, driven by the captured cotangent.

        torch.autograd.grad(outputs=(s_out, z_out), grad_outputs=(cot_s, cot_z),
                            inputs=list(trunk_params) + [s_in, z_in])

    The falsifier is pre-registered and both its values were banked before this ran:
    ||dL/dz_in|| at 0.000848887340907281 means the injection is EXACT on the original graph and
    the replay's fresh forward is the defect; at 0.0014907294032500784 means the injection
    overcounts on the original graph too.
    """
    e = [x for x in entries if x["grad_enabled"]][0]
    c = [x for x in cots if x["grad_enabled"]][0]
    s_out, z_out = c["s_out_t"], c["z_out_t"]
    s_in, z_in = e["s_in_t"], e["z_in_t"]
    # The value hooks only fire during a backward and --graphdrive runs none, so the cotangent
    # is read here with autograd.grad from the loss. That traversal stops at the stack's
    # outputs and never enters the trunk, and the object it returns has already been shown
    # bit-identical to the published hook capture three times.
    cot_s, cot_z = c["cot_s"], c["cot_z"]
    cot_provenance = "the taped tensor hook"
    if cot_s is None or cot_z is None:
        t0 = time.time()
        cot_s, cot_z = torch.autograd.grad(loss, [s_out, z_out], retain_graph=True)
        cot_s = cot_s.detach().to(torch.float64).clone()
        cot_z = cot_z.detach().to(torch.float64).clone()
        cot_provenance = ("torch.autograd.grad(loss, [s_out, z_out]) in this process, %.1f s; "
                          "no backward has run, so the value hooks never fired" %
                          (time.time() - t0))
    cot_check = None
    if published_cot is not None:
        pub_s, pub_z = torch.load(published_cot, map_location="cpu",
                                  weights_only=False)["cot"]
        cot_check = {
            "published": str(published_cot), "sha256": sha256_file(published_cot),
            "s": tensor_pair("cot_s", cot_s, pub_s.to(torch.float64)),
            "z": tensor_pair("cot_z", cot_z, pub_z.to(torch.float64))}
        print("GRAPHDRIVE cotangent vs published: s bit-identical %s, z bit-identical %s"
              % (cot_check["s"]["bit_identical"], cot_check["z"]["bit_identical"]), flush=True)

    names = [n for n, _ in model.named_parameters() if n.startswith("pairformer_stack.")]
    nameset = set(names)
    params = [prm for n, prm in model.named_parameters() if n in nameset]

    def load(path):
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return d["grads"] if isinstance(d, dict) and "grads" in d else d

    def fit(arm, ref, keys):
        dot = a2 = r2 = e2 = 0.0
        worst, worst_name, n_bit, n_absent = -1.0, None, 0, 0
        for n in keys:
            r = ref[n].to(torch.float64).reshape(-1)
            rn2 = float(torch.dot(r, r)); r2 += rn2
            m = arm.get(n)
            if m is None:
                n_absent += 1; e2 += rn2; continue
            m = m.reshape(-1)
            dot += float(torch.dot(m, r)); a2 += float(torch.dot(m, m))
            d = float(torch.dot(m - r, m - r)); e2 += d
            if torch.equal(m, r):
                n_bit += 1
            rel = (d / rn2) ** 0.5 if rn2 > 0 else (0.0 if d == 0 else float("inf"))
            if rel > worst:
                worst, worst_name = rel, n
        return {"n_tensors": len(keys), "n_absent_from_arm": n_absent,
                "n_bit_identical": n_bit,
                "rel_l2_as_is": (e2 / r2) ** 0.5 if r2 else None,
                "norm_ratio_arm_over_ref": (a2 / r2) ** 0.5 if r2 else None,
                "cos": dot / (a2 * r2) ** 0.5 if a2 > 0 and r2 > 0 else None,
                "best_scalar_a_star": dot / a2 if a2 > 0 else None,
                "residual_after_best_scalar_frac_of_ref":
                    (max(r2 - dot * dot / a2, 0.0) / r2) ** 0.5 if a2 > 0 and r2 else None,
                "ref_squared_norm": r2, "arm_squared_norm": a2,
                "worst_rel_l2": worst, "worst_tensor": worst_name}

    out = {
        "what": "the ORIGINAL graph driven by the captured cotangent, not a replay of it",
        "host": socket.gethostname(), "row": "of3t-frameself", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened",
        "drive": "grad_outputs=(cot_s, cot_z) on (s_out, z_out) of the taped call",
        "graph": "the original one built by the capture's own forward; nothing is re-run",
        "cotangent_provenance": cot_provenance,
        "cotangent_vs_the_published_capture": cot_check,
        "prereg_falsifier": {
            "quantity": "norm of dL/dz_in from this call",
            "exact_means": 0.000848887340907281,
            "overcounts_means": 0.0014907294032500784,
            "both_banked_before_this_ran": True},
    }

    # cheap arm first, so a kill after it still leaves a reading: one block, pruned.
    bn = [n for n in names if any(n.startswith(f"pairformer_stack.blocks.{i}.")
                                  for i in blocks)]
    bp = [prm for n, prm in model.named_parameters() if n in set(bn)]
    t0 = time.time()
    gb = torch.autograd.grad(outputs=(s_out, z_out), grad_outputs=(cot_s, cot_z),
                             inputs=bp, retain_graph=True, allow_unused=True)
    t_b = time.time() - t0
    mine_b = {n: (g.detach().to(torch.float64) if g is not None else None)
              for n, g in zip(bn, gb)}
    rg = load(ref_grads)
    inj = load(injected)
    out["pruned_to_blocks"] = {
        "blocks": list(blocks), "seconds": t_b,
        "vs_grads_f64_043": fit(mine_b, rg, bn),
        "vs_the_injected_replay": fit(mine_b, {n: inj[n] for n in bn}, bn)}
    print("GRAPHDRIVE pruned " + json.dumps(out["pruned_to_blocks"], indent=1), flush=True)
    del gb, mine_b

    # the full arm: every trunk parameter and both stack inputs
    t0 = time.time()
    got = torch.autograd.grad(outputs=(s_out, z_out), grad_outputs=(cot_s, cot_z),
                              inputs=params + [s_in, z_in], retain_graph=False,
                              allow_unused=True)
    t_f = time.time() - t0
    gp, ds_in, dz_in = got[:len(params)], got[len(params)], got[len(params) + 1]
    out["seconds_full"] = t_f
    out["input_cotangent_on_the_original_graph"] = {
        "ds_in_norm": norms(ds_in), "dz_in_norm": norms(dz_in),
        "ds_in_vs_the_real_backwards_own_hook": tensor_pair(
            "ds_in", ds_in, e.get("cot_in_s")),
        "dz_in_vs_the_real_backwards_own_hook": tensor_pair(
            "dz_in", dz_in, e.get("cot_in_z"))}
    print("GRAPHDRIVE dz_in " + json.dumps(
        {"ds_in_norm": norms(ds_in), "dz_in_norm": norms(dz_in)}), flush=True)
    mine = {n: (g.detach().to(torch.float64) if g is not None else None)
            for n, g in zip(names, gp)}
    out["vs_grads_f64_043"] = fit(mine, rg, names)
    out["vs_the_injected_replay"] = fit(mine, inj, names)
    if zonly is not None:
        zg = load(zonly)
        out["vs_the_banked_z_only_arm"] = fit(mine, zg, names)
    dz = out["input_cotangent_on_the_original_graph"]["dz_in_norm"]
    out["verdict"] = (
        "EXACT on the original graph: the injection reproduces the real backward's own dL/dz_in, "
        "so the replay's fresh forward is the defect"
        if dz is not None and abs(dz - 0.000848887340907281) < 1e-12
        else ("OVERCOUNTS on the original graph too: the injection is not a replay artefact"
              if dz is not None and abs(dz - 0.0014907294032500784) < 1e-12
              else "a third answer: dL/dz_in is %r, neither pre-registered value" % dz))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--batch-sha256", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--replay-draws", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--ref-grads", type=Path,
                    help="the published model reference. Its pairformer_stack tensors are scored "
                         "against this run's own, in this process, as the witness that the "
                         "captured pair belongs to THAT backward and not to a plausible "
                         "neighbour of it.")
    ap.add_argument("--expect-loss", type=float,
                    help="the published float64 loss. Required to be bit-identical.")
    ap.add_argument("--expect-grad-norm", type=float,
                    help="the published global gradient norm, checked to 1e-12 relative.")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--tag", default="model_n384")
    ap.add_argument("--selftest", action="store_true",
                    help="of3t-frameself, D242. After the backward and the witness, differentiate "
                         "THE SAME module object on the captured pair -- zero the grads, call "
                         "model.pairformer_stack on the recorded boundary with the recorded call "
                         "kwargs, backward <cot,out>, and score against --ref-grads. The replay "
                         "in perf/of3t_trunkg043/ref_grad.py reconstructs 48 standalone "
                         "PairFormerBlocks instead, and overshoots this reference by 1.7584x at "
                         "a bit-exact forward. This holds everything fixed but the "
                         "differentiator. Opt-in: without it nothing below runs and the file "
                         "produces byte for byte what it produced before.")
    ap.add_argument("--graphdrive", metavar="I,J,...", default=None,
                    help="opt-in. Drive the ORIGINAL graph with the captured cotangent: "
                         "torch.autograd.grad(outputs=(s_out, z_out), "
                         "grad_outputs=(cot_s, cot_z), inputs=trunk_params + [s_in, z_in]). "
                         "The one cell --selftest (replayed graph, cotangent drive) and "
                         "--blockprobe (original graph, loss drive) leave empty. The named "
                         "blocks are scored first as a cheap pruned arm. Stops before the full "
                         "backward. Needs --ref-grads and --graphdrive-injected.")
    ap.add_argument("--graphdrive-injected", type=Path, default=None,
                    help="the injected replay's gradient file --graphdrive is scored against")
    ap.add_argument("--graphdrive-zonly", type=Path, default=None,
                    help="optional: the banked z-only arm, scored as a third comparison")
    ap.add_argument("--blockprobe", metavar="I,J,...", default=None,
                    help="opt-in. Read the reference's own trunk gradient for these blocks with "
                         "torch.autograd.grad instead of p.grad after a full backward, and score "
                         "it against --ref-grads and against --blockprobe-injected. autograd "
                         "prunes the graph to the named blocks, so the last block costs one "
                         "block's backward, not forty-eight. Stops before the full backward.")
    ap.add_argument("--blockprobe-injected", type=Path, default=None,
                    help="the injected replay's gradient file the --blockprobe reading is also "
                         "scored against")
    ap.add_argument("--cotprobe", action="store_true",
                    help="opt-in. Read the stack's output cotangent a second time with "
                         "torch.autograd.grad instead of a tensor hook, enumerate every tape "
                         "node that consumes s_out/z_out, and stop before the full backward. "
                         "Cheap: it never traverses the trunk. Needs --cotprobe-compare.")
    ap.add_argument("--cotprobe-compare", type=Path, default=None,
                    help="the published cotangent file the --cotprobe reading is scored against")
    ap.add_argument("--selftest-compare", type=Path,
                    help="ref_grad.py's own output .pt for this frame (of3t_twoside/ctrl_f64.pt). "
                         "Its s, z, ds_in, dz_in and grads are scored against the real module's "
                         "in this process, so the forward claim is checked elementwise on real "
                         "tokens rather than through a norm the pads dominate.")
    a = ap.parse_args()

    t_start = time.time()
    deterministic = bm.pin_deterministic_kernels(True)
    torch.set_num_threads(a.threads)
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import openfold3
    tree = str(Path(openfold3.__file__).resolve().parents[1])
    print(f"REF_TREE resolved: {tree}", flush=True)

    got = sha256_file(a.batch)
    if got != a.batch_sha256:
        raise SystemExit(f"batch sha256 {got} != {a.batch_sha256}")

    raw_batch = torch.load(a.batch, weights_only=False)
    cfg, model, loss_fn, dropout = bm.build(dtype, a.seed, "cpu", num_recycles=0)

    # --- bundle_min.main()'s checkpoint block, with its key gate ------------------------------
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    ckpt_info = {"file": a.checkpoint.name, "sha256": sha256_file(a.checkpoint),
                 "n_loaded": len(sd), "n_missing": len(inc.missing_keys),
                 "n_unexpected": len(inc.unexpected_keys),
                 "missing_keys": sorted(inc.missing_keys)[:8],
                 "unexpected_all": sorted(inc.unexpected_keys)}
    print(f"REFBUILD: missing {ckpt_info['n_missing']} {ckpt_info['missing_keys']} "
          f"unexpected {ckpt_info['n_unexpected']}", flush=True)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} checkpoint tensors have "
                         f"nowhere to go, e.g. {sorted(inc.unexpected_keys)[:4]}")
    del ck, sd

    batch = bm.move(raw_batch, "cpu", dtype)
    pinned = bm.rng_state(model)
    replay = torch.load(a.replay_draws, map_location="cpu", weights_only=False)
    replay_info = {"file": a.replay_draws.name, "sha256": sha256_file(a.replay_draws),
                   "n_torch_randn_recorded": len(replay["torch_randn"]),
                   "n_python_random_recorded": len(replay["python_random"]),
                   "recorded_num_recycles": int(replay["num_recycles"])}

    # --- the two hooks ------------------------------------------------------------------------
    # A pre-hook records the call, so the boundary is what the stack was HANDED and not a
    # reconstruction of it. A tensor hook on each output records the cotangent, which is the
    # gradient the rest of the model sends back: the diffusion module, the confidence heads and
    # the auxiliary heads all consume s and z, so this is their sum and no term of it is modelled.
    entries, cots = [], []

    def pre_hook(mod, args_, kwargs):
        entries.append({
            "grad_enabled": torch.is_grad_enabled(),
            "s_in": kwargs["s"].detach().to(torch.float64).clone(),
            "z_in": kwargs["z"].detach().to(torch.float64).clone(),
            "single_mask": kwargs["single_mask"].detach().to(torch.float64).clone(),
            "pair_mask": kwargs["pair_mask"].detach().to(torch.float64).clone(),
        })
        if a.graphdrive:
            # --graphdrive only, and ABOVE the --selftest early return: the graph tensors
            # themselves, not the detached clones saved above. The difference between those two
            # is the whole object under test.
            entries[-1]["s_in_t"] = kwargs["s"]
            entries[-1]["z_in_t"] = kwargs["z"]
        if not a.selftest:
            return
        # --selftest only. None of this touches the graph: the extra tensor hooks return None,
        # so they cannot change a gradient, and everything else is a read.
        e = entries[-1]
        e["call_kwargs_nontensor"] = {k: (v if isinstance(v, (int, float, bool, str, type(None)))
                                          else repr(v))
                                      for k, v in kwargs.items() if not torch.is_tensor(v)}
        e["call_kwargs_tensor"] = sorted(k for k, v in kwargs.items() if torch.is_tensor(v))
        e["n_positional_args"] = len(args_)
        e["module_state"] = {
            "class": type(mod).__name__,
            "training": bool(mod.training),
            "n_blocks": len(mod.blocks),
            "blocks_per_ckpt": getattr(mod, "blocks_per_ckpt", "<absent>"),
            "use_reentrant": getattr(mod, "use_reentrant", "<absent>"),
            "clear_cache_between_blocks": getattr(mod, "clear_cache_between_blocks", "<absent>"),
            "tune_chunk_size": getattr(mod, "tune_chunk_size", "<absent>"),
            "chunk_size_tuner": repr(getattr(mod, "chunk_size_tuner", "<absent>")),
            "block0_class": type(mod.blocks[0]).__name__,
            "block0_training": bool(mod.blocks[0].training),
        }
        e["in_fired"] = {"s": 0, "z": 0}
        if torch.is_grad_enabled():
            for key in ("s", "z"):
                t = kwargs[key]
                if t.requires_grad:
                    def ih(g, e=e, key=key):
                        e["in_fired"][key] += 1
                        e["cot_in_" + key] = g.detach().to(torch.float64).clone()
                    t.register_hook(ih)

    def post_hook(mod, args_, kwargs, output):
        s_out, z_out = output
        slot = {"grad_enabled": torch.is_grad_enabled(), "cot_s": None, "cot_z": None,
                "s_out_norm": norms(s_out), "z_out_norm": norms(z_out),
                "s_requires_grad": bool(s_out.requires_grad),
                "z_requires_grad": bool(z_out.requires_grad)}
        if s_out.requires_grad:
            s_out.register_hook(
                lambda g, sl=slot: sl.__setitem__("cot_s", g.detach().to(torch.float64).clone()))
        if z_out.requires_grad:
            z_out.register_hook(
                lambda g, sl=slot: sl.__setitem__("cot_z", g.detach().to(torch.float64).clone()))
        if a.cotprobe or a.blockprobe or a.graphdrive:
            # --cotprobe only. The graph tensors themselves, so autograd.grad can be pointed at
            # them. Holding a reference does not change the graph.
            slot["s_out_t"], slot["z_out_t"] = s_out, z_out
        if a.selftest:
            # --selftest only. A counting hook (the existing one overwrites, so a second firing
            # would be invisible) and the real outputs, kept for the elementwise forward check.
            slot["out_fired"] = {"s": 0, "z": 0}
            for key, t in (("s", s_out), ("z", z_out)):
                if t.requires_grad:
                    t.register_hook(
                        lambda g, sl=slot, key=key: sl["out_fired"].__setitem__(
                            key, sl["out_fired"][key] + 1))
            slot["s_out"] = s_out.detach().to(torch.float64).clone()
            slot["z_out"] = z_out.detach().to(torch.float64).clone()
        cots.append(slot)

    h1 = model.pairformer_stack.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = model.pairformer_stack.register_forward_hook(post_hook, with_kwargs=True)

    # --blockprobe only: the first named block's own boundary, so ONE block can be replayed
    # standalone against the real backward with everything else held fixed.
    blk = {}
    hb = []
    if a.blockprobe:
        bi = int(a.blockprobe.split(",")[0])
        blk["index"] = bi

        def blk_pre(mod, args_, kwargs, _b=blk):
            if not torch.is_grad_enabled() or "in_s" in _b:
                return
            kw = dict(kwargs)
            if len(args_) >= 2:
                _b["in_s"], _b["in_z"] = args_[0], args_[1]
                _b["args_rest"] = args_[2:]
            else:
                _b["in_s"], _b["in_z"] = kw.pop("s"), kw.pop("z")
                _b["args_rest"] = tuple(args_)
            _b["kwargs"] = kw

        def blk_post(mod, args_, kwargs, output, _b=blk):
            if not torch.is_grad_enabled() or "out_s" in _b:
                return
            _b["out_s"], _b["out_z"] = output

        tgt = model.pairformer_stack.blocks[bi]
        hb = [tgt.register_forward_pre_hook(blk_pre, with_kwargs=True),
              tgt.register_forward_hook(blk_post, with_kwargs=True)]

    bm.set_rng_state(pinned, model)
    rec = bm.DrawRecorder(replay)
    mode = "removed" if dtype is torch.float64 else "upstream"
    policy = bm.cast_policy(mode, "cpu")

    t0 = time.time()
    loss, breakdown, out = bm.forward_loss(model, loss_fn, batch, rec, cast_ctx=policy)
    t_fwd = time.time() - t0
    print(f"forward {t_fwd:.1f}s loss {float(loss)!r} stack entered {len(entries)}x", flush=True)
    if a.expect_loss is not None and float(loss) != a.expect_loss:
        raise SystemExit(f"loss {float(loss)!r} is not bit-identical to the published "
                         f"{a.expect_loss!r} -- this is not the reference's own step, so its "
                         f"boundary and cotangent are not the reference's either")

    if a.graphdrive:
        if a.graphdrive_injected is None or a.ref_grads is None:
            raise SystemExit("--graphdrive needs --ref-grads and --graphdrive-injected")
        gd = run_graphdrive(entries, cots, loss, model, a.ref_grads, a.graphdrive_injected,
                            a.graphdrive_zonly, a.out_dir, a.tag,
                            [int(x) for x in a.graphdrive.split(",") if x.strip()],
                            a.cotprobe_compare)
        h1.remove(); h2.remove()
        for h in hb:
            h.remove()
        a.out_dir.mkdir(parents=True, exist_ok=True)
        rp = a.out_dir / f"GRAPHDRIVE_{a.tag}.json"
        rp.write_text(json.dumps(gd, indent=1))
        print("GRAPHDRIVE " + json.dumps(
            {k: gd[k] for k in ("verdict", "input_cotangent_on_the_original_graph",
                                "vs_grads_f64_043", "vs_the_injected_replay")}, indent=1),
              flush=True)
        print("report " + str(rp), flush=True)
        return 0

    if a.blockprobe:
        if a.blockprobe_injected is None or a.ref_grads is None:
            raise SystemExit("--blockprobe needs --ref-grads and --blockprobe-injected")
        blocks = [int(x) for x in a.blockprobe.split(",") if x.strip() != ""]
        rb = run_blockprobe(cots, loss, model, blocks, a.ref_grads, a.blockprobe_injected,
                            a.out_dir, a.tag, blk)
        h1.remove(); h2.remove()
        for h in hb:
            h.remove()
        a.out_dir.mkdir(parents=True, exist_ok=True)
        rp = a.out_dir / f"BLOCKPROBE_{a.tag}.json"
        rp.write_text(json.dumps(rb, indent=1))
        print("BLOCKPROBE " + json.dumps(
            {k: rb[k] for k in ("cotangent_reread", "per_block", "seconds_autograd_grad")},
            indent=1), flush=True)
        print("report " + str(rp), flush=True)
        return 0

    if a.cotprobe:
        if a.cotprobe_compare is None:
            raise SystemExit("--cotprobe needs --cotprobe-compare <published cot_*.pt>")
        rc = run_cotprobe(cots, loss, a.cotprobe_compare, a.out_dir, a.tag)
        h1.remove(); h2.remove()
        a.out_dir.mkdir(parents=True, exist_ok=True)
        rp = a.out_dir / f"COTPROBE_{a.tag}.json"
        rp.write_text(json.dumps(rc, indent=1))
        print("COTPROBE " + json.dumps(
            {k: rc[k] for k in ("verdict", "autograd_grad_vs_the_published_hook",
                                "consumers_of_the_stack_outputs",
                                "tape_nodes_walked_from_the_loss")}, indent=1), flush=True)
        print("report " + str(rp), flush=True)
        return 0

    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0
    h1.remove(); h2.remove()

    taped_in = [e for e in entries if e["grad_enabled"]]
    taped_out = [c for c in cots if c["grad_enabled"]]
    if len(taped_in) != 1 or len(taped_out) != 1:
        raise SystemExit(f"the trunk was taped {len(taped_in)} / {len(taped_out)} times, not "
                         f"once -- at num_recycles 0 exactly one cycle is taped and a second "
                         f"firing means the captured pair is ambiguous")
    b, c = taped_in[0], taped_out[0]
    if c["cot_s"] is None or c["cot_z"] is None:
        raise SystemExit("no cotangent reached the stack's outputs")

    # --- the global gradient, and the trunk's share of it -------------------------------------
    grads, presence = {}, {}
    for name, p in model.named_parameters():
        presence[name] = p.grad is not None
        grads[name] = (p.grad.detach().to(torch.float64).clone()
                       if p.grad is not None else None)
    per_tensor = [torch.linalg.vector_norm(g) for g in grads.values() if g is not None]
    global_norm = float(torch.linalg.vector_norm(torch.stack(per_tensor)))
    if a.expect_grad_norm is not None:
        rel = abs(global_norm - a.expect_grad_norm) / a.expect_grad_norm
        print(f"global gradient norm {global_norm!r} vs published {a.expect_grad_norm!r} "
              f"rel {rel:.3e}", flush=True)
        if rel > 1e-12:
            raise SystemExit(f"global gradient norm differs by {rel:.3e} from the published "
                             f"reference -- this run is not the reference's backward")

    # --- write the pair, in the formats ref_grad.py and dev_cot.py already read ---------------
    bpath = a.out_dir / f"boundary_{a.tag}.pt"
    cpath = a.out_dir / f"cot_{a.tag}.pt"
    torch.save({"s_in": b["s_in"], "z_in": b["z_in"],
                "single_mask": b["single_mask"], "pair_mask": b["pair_mask"]}, bpath)
    torch.save({"cot": (c["cot_s"], c["cot_z"])}, cpath)

    probe = {
        "s_in_norm": norms(b["s_in"]), "z_in_norm": norms(b["z_in"]),
        "s_out_norm": c["s_out_norm"], "z_out_norm": c["z_out_norm"],
        "cot_s_norm": norms(c["cot_s"]), "cot_z_norm": norms(c["cot_z"]),
        "real_tokens": int(b["single_mask"].sum()), "tokens": int(b["single_mask"].shape[-1]),
        "shapes": {k: list(v.shape) for k, v in
                   (("s_in", b["s_in"]), ("z_in", b["z_in"]),
                    ("single_mask", b["single_mask"]), ("pair_mask", b["pair_mask"]),
                    ("cot_s", c["cot_s"]), ("cot_z", c["cot_z"]))},
    }
    print("probe " + json.dumps(probe["shapes"]), flush=True)

    # --- the witness: this run's trunk gradient against the published reference ---------------
    # A boundary and a cotangent cannot be checked directly against anything. What CAN be checked
    # is that the backward they were taken from is the one the reference published, per tensor
    # over the 2,736 the trunk owns.
    witness = None
    if a.ref_grads:
        ref = torch.load(a.ref_grads, map_location="cpu", weights_only=False, mmap=True)
        names = sorted(k for k in grads if k.startswith(PRE))
        n_bit, worst, worst_name, ref_sq, err_sq = 0, 0.0, None, 0.0, 0.0
        n_absent = 0
        for n in names:
            mine, theirs = grads[n], ref.get(n)
            if theirs is None or mine is None:
                n_absent += 1
                continue
            theirs = theirs.to(torch.float64)
            if torch.equal(mine, theirs):
                n_bit += 1
            rn = float(torch.linalg.vector_norm(theirs))
            en = float(torch.linalg.vector_norm(mine - theirs))
            ref_sq += rn ** 2
            err_sq += en ** 2
            r = en / rn if rn > 0 else (0.0 if en == 0 else float("inf"))
            if r > worst:
                worst, worst_name = r, n
        witness = {
            "what": "this run's own pairformer_stack gradient against the published reference, "
                    "per tensor, float64 on both sides",
            "reference": str(a.ref_grads), "reference_sha256": sha256_file(a.ref_grads),
            "compared": len(names) - n_absent, "of_total": len(names), "absent": n_absent,
            "n_bit_identical": n_bit,
            "mass_weighted_rel_l2": (err_sq / ref_sq) ** 0.5 if ref_sq else None,
            "reference_squared_norm": ref_sq,
            "worst_rel_l2": worst, "worst_tensor": worst_name,
        }
        print("WITNESS " + json.dumps({k: witness[k] for k in
                                       ("compared", "of_total", "n_bit_identical",
                                        "mass_weighted_rel_l2", "worst_rel_l2", "worst_tensor")}),
              flush=True)
        del ref

    selftest = None
    if a.selftest:
        selftest = run_selftest(model, b, c, grads, a, mode)

    rep = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(),
        "device_involved": False,
        "why_no_aiclk": "CPU only, no card is opened; the device arm this pair drives records "
                        "its own DURING-sampled AICLK",
        "tree": tree, "openfold3_file": openfold3.__file__,
        "dtype": a.dtype,
        "why_float64": "the cotangent is part of the reference's identity. grads_f64_043.pt is "
                       "the float64 backward, so a narrower cotangent would swap this row's "
                       "known 1.8417 frame mismatch for an unknown one of its own and the arm "
                       "would again be measuring an artifact. Cost is one backward, already "
                       "paid once by the published bundle.",
        "threads": a.threads, "seed": a.seed, "num_recycles_pinned": 0,
        "deterministic_kernels": deterministic,
        "dropout": dropout,
        "cast_policy": policy.report() if hasattr(policy, "report") else mode,
        "inputs": {
            "batch": {"path": str(a.batch), "sha256": got,
                      "bytes": a.batch.stat().st_size},
            "checkpoint": ckpt_info,
            "replayed_draws": replay_info,
        },
        "outputs": {
            "boundary": {"path": str(bpath), "sha256": sha256_file(bpath),
                         "bytes": bpath.stat().st_size,
                         "keys": ["s_in", "z_in", "single_mask", "pair_mask"]},
            "cotangent": {"path": str(cpath), "sha256": sha256_file(cpath),
                          "bytes": cpath.stat().st_size, "keys": ["cot"]},
        },
        "loss": float(loss),
        "loss_bit_identical_to_published": (a.expect_loss is None
                                            or float(loss) == a.expect_loss),
        "published_loss": a.expect_loss,
        "global_gradient_norm": global_norm,
        "published_global_gradient_norm": a.expect_grad_norm,
        "n_parameters": len(grads),
        "n_absent_gradients": sum(1 for v in grads.values() if v is None),
        "trunk_tensors": sum(1 for k in grads if k.startswith(PRE)),
        "trunk_squared_gradient_norm": sum(
            float(torch.linalg.vector_norm(v)) ** 2
            for k, v in grads.items() if k.startswith(PRE) and v is not None),
        "stack_entered_times": len(entries),
        "stack_taped_times": len(taped_in),
        "probe": probe,
        "witness_trunk_gradient_vs_published_reference": witness,
        "selftest_real_module_differentiated": selftest,
        "seconds_forward": t_fwd, "seconds_backward": t_bwd,
        "seconds_total": time.time() - t_start,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
    }
    rpath = a.out_dir / f"CAPTURE_{a.tag}.json"
    rpath.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in
                      ("host", "loss", "loss_bit_identical_to_published",
                       "global_gradient_norm", "trunk_squared_gradient_norm",
                       "seconds_forward", "seconds_backward", "peak_rss_gb")}, indent=1))
    print(f"report {rpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
