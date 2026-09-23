#!/usr/bin/env python3
"""PROTOCOL SS3 instrument A, magnitude half, against the FROZEN BUNDLE.

Every earlier run of this instrument measured against a self-generated float64 reference and said
so in its own verdict line. This one measures against `of3t-reference`'s BUNDLE-MIN:
`grads_f64_recycles0.pt`, trained weights, `num_recycles` pinned to 0, 4138 of 4147 tensors
non-zero, validated by float64 central finite differences at 2.18e-03 worst against the 5.0e-02
bar it is used under.

HOW A WHOLE-MODEL REFERENCE BECOMES A BLOCK-SCOPE MEASUREMENT. tt-bio wires no OF3 training
forward, so the loss their gradient came from cannot be computed on our side at all. It does not
have to be. Pairformer block i's parameters appear in exactly one place in their graph, and at
`num_recycles` 0 the trunk runs once, so

    dL/d(theta_i)  =  d/d(theta_i) [ <cot_s, s_out_i> + <cot_z, z_out_i> ]

with (s_in, z_in, masks) and the cotangents taken at that block's own boundary. Those five
tensors are what `capture_trunk_boundary.py` records from a float64 CPU replay of their step, and
that replay is validated by recomputing the same block's gradients and comparing them to the
bundle entry-by-entry. So the reference used below is the bundle's own numbers, and the probe is
their own activations at their own scale -- not a synthetic draw, which is the trap K28 cost this
campaign a run over.

SS3a: compared in THEIR parameter space, fused gradients split back along the fusion boundary.
SS3b: presence before magnitude, as a set. Nothing is filled with `zeros_like`.
SS3c: the forward is checked against their captured block output before any gradient is read.
SS3e: the negative control runs and its result is part of the verdict.
K29: leaves are reported against total, never alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "perf/of3t_gradients"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BUNDLE = "/home/ttuser/of3t/bundle_min"
REF_BRANCH = "origin/wk/of3t-reference"
MANIFEST_GIT = "perf/of3t_reference/bundle_min/MANIFEST.json"
#: durable; /tmp/of3t/<slug>/ is swept and this capture costs 26 minutes of CPU.
CAP = "/home/ttuser/of3t_gradients/cap"
PER_TENSOR_BAR, MEDIAN_BAR = 5.0e-02, 2.0e-02


def rel_l2(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def manifest_from_git():
    out = subprocess.run(["git", "show", f"{REF_BRANCH}:{MANIFEST_GIT}"],
                         capture_output=True, check=True)
    return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=0, help="which pairformer block")
    ap.add_argument("--stack", type=int, default=0, metavar="N",
                    help="run N consecutive blocks from --block as ONE taped stack instead "
                         "of one block. The composition is the point: every sub-module of "
                         "the pair track passes alone and the assembled block does not "
                         "(D8), so a per-block result cannot be extrapolated over 48 of "
                         "them. The boundary is still theirs -- the first block's captured "
                         "inputs in, the LAST block's captured output cotangent back -- "
                         "which is exact because the stack's parameters appear once in "
                         "their graph and at num_recycles 0 the trunk runs once.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scale-pair-bias", default="shipped", choices=("shipped", "on", "off"),
                    help="the ATTENTION pair bias scale. `shipped` is True, with the triangle "
                         "route's own tri_att_scale_pair_bias pinned False, which is what "
                         "openfold3_trunk.py:137 builds since the flag was split.")
    ap.add_argument("--transpose-bias", default="shipped", choices=("shipped", "on", "off"),
                    help="tri_att_end's bias orientation. `shipped` is `not is_openbind(sd)`. "
                         "The discriminator: the ending node is the one that transposes the "
                         "pair, and it is where the weight-gradient disagreement concentrates "
                         "while the forward agrees. If flipping this moves the FORWARD and "
                         "leaves the gradient ordering, the orientation is not the cause.")
    ap.add_argument("--crop", type=int, default=0, metavar="N",
                    help="crop the captured boundary to the first N token positions. The batch is "
                         "384 tokens of which 56 are real, and a layer-norm WEIGHT gradient sums "
                         "over every position including the 328 padded ones. Their block zeroes "
                         "those through single_mask/pair_mask; if ours does not, the forward can "
                         "agree on the real tokens while the weight gradient does not. Cropping "
                         "to 64 leaves 8 padded positions instead of 328, so a disagreement that "
                         "collapses under it is padded-position contamination and one that does "
                         "not is arithmetic.")
    ap.add_argument("--capture-report",
                    default="perf/of3t_gradients/capture_trunk_boundary_nodropout.json",
                    help="the capture this boundary came from, read for its own self-check")
    ap.add_argument("--reference", default="bundle", choices=("bundle", "block-eval"),
                    help="`bundle` is grads_f64_recycles0.pt as published, taped in train mode "
                         "at dropout r = 0.25 with a mask nobody recorded. `block-eval` "
                         "differentiates the same block on the same captured boundary with the "
                         "Dropout modules in eval, which is the r = 0 function our tape can "
                         "compute at all and the only one that is bit-reproducible "
                         "(dropout_floor_block0.json: 0.000e+00 worst across two seeds).")
    ap.add_argument("--fp32-softmax", default="on", choices=("on", "off"),
                    help="the shipped setting is on. `off` is the D8 arm.")
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
    i = a.block
    nb = a.stack or 1
    last = i + nb - 1
    tag = a.tag or (f"stack{i}_{last}" if a.stack else f"block{i}") + \
        ("" if a.fp32_softmax == "on" else "_nofp32softmax")
    rep = {"instrument": "PROTOCOL SS3 instrument A, magnitude, against BUNDLE-MIN",
           "block": i, "n_blocks": nb, "last_block": last, "checkpoint": CKPT,
           "bars": {"per_tensor": PER_TENSOR_BAR, "median": MEDIAN_BAR},
           "reference_is_the_frozen_bundle": True,
           "fp32_softmax": a.fp32_softmax == "on", "crop": a.crop}

    man = manifest_from_git()
    decl = {x["file"]: x for x in man["artifacts"] if "sha256" in x}
    gfile = man["validated_gradient"]["file"]
    got = sha256_file(os.path.join(BUNDLE, gfile))
    if got != decl[gfile]["sha256"]:
        raise SystemExit(f"{gfile}: sha256 {got} != manifest {decl[gfile]['sha256']}")
    rep["bundle"] = {"file": gfile, "sha256": got, "verified": True,
                     "manifest": f"{REF_BRANCH}:{MANIFEST_GIT}",
                     "num_recycles": man["validated_gradient"]["num_recycles"],
                     "weights": "of3-p2-155k.pt (trained)",
                     "reference_fd_max_rel": man["validated_gradient"]["finite_difference"]["max_rel_err"],
                     "n_nonzero_of_total":
                         f"{man['validated_gradient']['n_nonzero_gradient_tensors']} of "
                         f"{man['validated_gradient']['n_parameters']}"}
    print(f"[{time.perf_counter()-t0:.0f}s] bundle hash verified", flush=True)

    cap = torch.load(os.path.join(CAP, f"block{i}_boundary.pt"),
                     map_location="cpu", weights_only=False)
    #: In stack mode the cotangent belongs to the LAST block's output, not the first's. Taking
    #: it from the wrong end would drive the backward with a cotangent for a different tensor
    #: and produce a number that looks like a gradient and is not one.
    cap_out = cap if not a.stack else torch.load(
        os.path.join(CAP, f"block{last}_boundary.pt"), map_location="cpu", weights_only=False)
    caprep = json.load(open(a.capture_report))
    rep["capture"] = {"forward_loss_rel_vs_bundle": caprep["forward"]["rel"],
                      "global_norm_rel": caprep["global_norm_rel"],
                      "block_grad_vs_bundle": {b: caprep["capture_vs_bundle"][b]
                                               for b in ({str(i), str(last)}
                                                         & set(caprep["capture_vs_bundle"]))},
                      "cotangent_from_block": last,
                      "inputs_from_block": i,
                      "replay_mismatches": caprep["replay"]["n_mismatch"]}

    rep["reference_mode"] = a.reference
    args_in = cap["args"]
    s_in, z_in = args_in[0].to(torch.float64), args_in[1].to(torch.float64)
    single_mask = args_in[2].to(torch.float64) if len(args_in) > 2 else None
    pair_mask = args_in[3].to(torch.float64) if len(args_in) > 3 else None
    for k, v in (cap["kwargs"] or {}).items():
        if torch.is_tensor(v):
            if "single" in k:
                single_mask = v.to(torch.float64)
            elif "pair" in k or "mask" in k:
                pair_mask = v.to(torch.float64)
    s_ref_out = cap_out["out"][0].to(torch.float64)
    z_ref_out = cap_out["out"][1].to(torch.float64)
    cot_s, cot_z = cap_out["cot"][0], cap_out["cot"][1]
    if cot_s is None or cot_z is None:
        raise SystemExit("captured cotangent missing -- the boundary is unusable")
    cot_s, cot_z = cot_s.to(torch.float64), cot_z.to(torch.float64)
    if a.crop:
        c = a.crop
        s_in, z_in = s_in[:, :c].contiguous(), z_in[:, :c, :c].contiguous()
        s_ref_out, z_ref_out = s_ref_out[:, :c].contiguous(), z_ref_out[:, :c, :c].contiguous()
        cot_s, cot_z = cot_s[:, :c].contiguous(), cot_z[:, :c, :c].contiguous()
        if single_mask is not None:
            single_mask = single_mask[:, :c].contiguous()
        if pair_mask is not None:
            pair_mask = pair_mask[:, :c, :c].contiguous()
    N = int(z_in.shape[1])
    rep["probe"] = {"tokens": N, "s_norm": float(s_in.norm()), "z_norm": float(z_in.norm()),
                    "cot_s_norm": float(cot_s.norm()), "cot_z_norm": float(cot_z.norm()),
                    "single_mask_sum": None if single_mask is None else float(single_mask.sum()),
                    "source": "their own activations at this block, from the bundle's own step"}
    print(f"[{time.perf_counter()-t0:.0f}s] probe N={N} |s|={float(s_in.norm()):.4g} "
          f"|z|={float(z_in.norm()):.4g} |cot_z|={float(cot_z.norm()):.4g}", flush=True)

    # ---- the reference: the bundle's own entries for this block -----------------------------
    pre = "pairformer_stack." if a.stack else f"pairformer_stack.blocks.{i}."
    keep = (lambda k: k.startswith(pre) and int(k[len(pre) + len("blocks."):].split(".")[0])
            in range(i, last + 1)) if a.stack else (lambda k: k.startswith(pre))
    if a.reference == "bundle":
        ref_all = torch.load(os.path.join(BUNDLE, gfile), map_location="cpu", weights_only=False)
        g_ref = {k[len(pre):]: (v.to(torch.float64) if v is not None else None)
                 for k, v in ref_all.items() if keep(k)}
        del ref_all
    else:
        from instrument_a_stack import their_stack, load_ckpt
        blk = their_stack(load_ckpt(), 1, first=i)[0][0]
        blk.eval()
        for q in blk.parameters():
            q.grad = None
        s_r, z_r = blk(s_in, z_in, single_mask, pair_mask)
        ((s_r * cot_s).sum() + (z_r * cot_z).sum()).backward()
        g_ref = {n: (q.grad.detach().clone() if q.grad is not None else None)
                 for n, q in blk.named_parameters()}
        s_ref_out, z_ref_out = s_r.detach(), z_r.detach()
        del blk, s_r, z_r
    rep["reference_tensor_count"] = len(g_ref)
    rep["reference_grad_present"] = {k: (v is not None) for k, v in g_ref.items()}
    print(f"[{time.perf_counter()-t0:.0f}s] reference: {len(g_ref)} tensors for block {i}, "
          f"{sum(1 for v in g_ref.values() if v is not None)} with a gradient", flush=True)

    # ---- ours --------------------------------------------------------------------------------
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    atoms = {k[len(pre):]: v.detach().to(torch.float32) for k, v in sd.items() if keep(k)}
    #: the shape probes below read block `i`'s entries whatever the scope, so they are keyed
    #: through the block-local prefix rather than through `pre`.
    bp = f"pairformer_stack.blocks.{i}."
    no_heads_pair = sd[bp + "pair_stack.tri_att_start.linear_z.weight"].shape[0]
    no_heads_pair_bias = sd[bp + "attn_pair_bias.linear_z.weight"].shape[0]
    c_s = sd[bp + "attn_pair_bias.layer_norm_a.weight"].shape[0]

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
    for j in range(i, last + 1):
        src = f"layers.{j}."
        flat.update({f"layers.{j - i}." + k[len(src):]: v
                     for k, v in flat_all.items() if k.startswith(src)})
    head_dim = flat["layers.0.tri_att_start.mha.linear_q.weight"].shape[0] // no_heads_pair
    # The SHIPPED configuration, not a plausible neighbour of it. An earlier run of this
    # instrument pinned `scale_pair_bias` and `fp32_softmax` and left `transpose_bias` and
    # `accurate_softmax` at their library defaults, which is a different kernel selection from
    # the one `openfold3_trunk.py:137` builds -- and the tri_att_END ordering it reported is
    # exactly where `transpose_bias` acts, so the reading could not be attributed.
    from tt_bio.openfold3_weights import is_openbind
    from tt_bio.tenstorrent import accurate_softmax_site
    transpose_bias = (not is_openbind(sd)) if a.transpose_bias == "shipped" \
        else (a.transpose_bias == "on")
    acc = accurate_softmax_site("openfold3.trunk")
    # `of3t-pairbias` SPLIT this flag after this instrument was written: the shipped trunk now
    # passes `scale_pair_bias=True, tri_att_scale_pair_bias=False`, where a single
    # `scale_pair_bias=False` used to cover both. An instrument still passing the old single
    # value runs the attention pair bias at the wrong scale -- and `attn_pair_bias` is the group
    # this instrument reports as failing, so the attribution was unsafe until this was pinned.
    # Same class as transpose_bias and accurate_softmax, found the same way: by reading the
    # shipped construction rather than trusting a default.
    SHIPPED_SPB, SHIPPED_TRI_SPB = True, False
    spb = SHIPPED_SPB if a.scale_pair_bias == "shipped" else (a.scale_pair_bias == "on")
    rep["shipped_config"] = {"scale_pair_bias": spb,
                             "tri_att_scale_pair_bias": SHIPPED_TRI_SPB,
                             "scale_pair_bias_arm": a.scale_pair_bias,
                             "fp32_softmax": (a.fp32_softmax == "on"),
                             "transpose_bias": bool(transpose_bias),
                             "accurate_softmax": acc,
                             "arm": a.transpose_bias,
                             "source": "openfold3_trunk.py:137"}
    mod = T.Pairformer(nb, head_dim, no_heads_pair, c_s // no_heads_pair_bias,
                       no_heads_pair_bias, True, flat, ckc,
                       scale_pair_bias=spb, tri_att_scale_pair_bias=SHIPPED_TRI_SPB,
                       fp32_softmax=(a.fp32_softmax == "on"),
                       transpose_bias=transpose_bias, accurate_softmax=acc)
    T.Module.torch_to_tt = orig

    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    # Their masks in our convention: `pair_mask` is the multiplicative outer product the
    # triangle multiplications contract against, `attn_mask` the additive -1e9 companion the
    # softmaxes take. Built from THEIR mask tensors rather than from a token count, so a padded
    # batch cannot silently become a dense one.
    pm = pair_mask if pair_mask is not None else torch.ones(1, N, N, dtype=torch.float64)
    sm = single_mask if single_mask is not None else torch.ones(1, N, dtype=torch.float64)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
    rep["masks"] = {"pair_mask_sum": float(pm.sum()), "single_mask_sum": float(sm.sum()),
                    "real_tokens": int(sm.sum().item())}

    before = device_weights(mod)
    ours = walked_weights(lambda: mod(ft(s_in), ft(z_in), ft(pm), ft(attn), ft(attn)), None, mod)
    after = device_weights(mod)
    rep["ours_discovery"] = {"from_loader": len(loaded),
                             "reachable_before_forward": len(before),
                             "reachable_after_forward": len(after),
                             "registered": len(ours)}
    print(f"[{time.perf_counter()-t0:.0f}s] ours: loader {len(loaded)}, before {len(before)}, "
          f"after {len(after)}, registered {len(ours)}", flush=True)

    err = None
    try:
        sa = ag.Tensor(ft(s_in), requires_grad=True)
        za = ag.Tensor(ft(z_in), requires_grad=True)
        with ag.tape():
            s_out, z_out = mod(sa, za, ft(pm), ft(attn), ft(attn))
        s_ours = ttnn.to_torch(s_out.value).to(torch.float64)
        z_ours = ttnn.to_torch(z_out.value).to(torch.float64)
        ag.backward([s_out, z_out], [ft(cot_s), ft(cot_z)])
    except Exception as e:
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

    # SS3c: the forward, against THEIR captured output for this block, before any gradient.
    if s_ours is not None:
        msk = sm.reshape(1, N, 1)
        pmk = pm.reshape(1, N, N, 1)
        rep["forward_rel"] = {
            "s": rel_l2(s_ours.numpy(), s_ref_out.numpy()),
            "z": rel_l2(z_ours.numpy(), z_ref_out.numpy()),
            "s_masked": rel_l2((s_ours * msk).numpy(), (s_ref_out * msk).numpy()),
            "z_masked": rel_l2((z_ours * pmk).numpy(), (z_ref_out * pmk).numpy())}
        print(f"[{time.perf_counter()-t0:.0f}s] forward s {rep['forward_rel']['s']:.3e} "
              f"z {rep['forward_rel']['z']:.3e} (masked s {rep['forward_rel']['s_masked']:.3e} "
              f"z {rep['forward_rel']['z_masked']:.3e})", flush=True)

    # ---- the bijection, by value against the built model -------------------------------------
    dev_all = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(mod).items()}
    if not a.stack:
        b = device_bijection(dev_all, atoms)
        placements, per_device = b["placements"], b["per_device"]
        rep["bijection_scope"] = "one block, matched by value over the whole built module"
    else:
        # PER BLOCK, deliberately, and this is not a micro-optimisation. `device_bijection`
        # confirms a candidate elementwise, so over 48 blocks at once any two tensors that are
        # elementwise EQUAL are interchangeable candidates -- and OF3 zero-initialises the gate
        # and output projection of nearly every residual branch, which puts thousands of
        # identical all-zero tensors in the pool. A whole-stack match could pair their block 3
        # with our block 17 and produce a comparison that is either garbage or an accidental
        # pass, with nothing in the output to say so. Restricting each match to one block's
        # tensors makes the block index structural instead of something the values have to
        # carry, and their block i+j is matched against our block j by construction.
        placements, per_device, amb = {}, {}, []
        for j in range(i, last + 1):
            tp, dp = f"blocks.{j}.", f"blocks.{j - i}."
            atoms_j = {k[len(tp):]: v for k, v in atoms.items() if k.startswith(tp)}
            dev_j = {k[len(dp):]: v for k, v in dev_all.items() if k.startswith(dp)}
            bj = device_bijection(dev_j, atoms_j)
            for k, pls in bj["placements"].items():
                placements[tp + k] = [dict(pl, device_path=dp + pl["device_path"]) for pl in pls]
            for k, m in bj["per_device"].items():
                per_device[dp + k] = m
            if bj["their_unplaced"]:
                amb.append({"block": j, "unplaced": bj["their_unplaced"]})
        b = {"their_unplaced": [x for e in amb for x in e["unplaced"]],
             "device_unmatched": sorted(set(dev_all) - {pl["device_path"]
                                                        for pls in placements.values()
                                                        for pl in pls})}
        rep["bijection_scope"] = ("per block, %d blocks matched independently so an all-zero "
                                  "tensor cannot pair across blocks" % nb)
        rep["bijection_unplaced_by_block"] = amb
    rep["bijection_device"] = {
        "device_tensors": len(dev_all), "their_tensors": len(atoms),
        "their_placed": len(placements), "their_unplaced": b["their_unplaced"],
        "device_unmatched": b["device_unmatched"],
        "kinds": {k: sum(1 for m in per_device.values() if m["kind"] == k)
                  for k in sorted({m["kind"] for m in per_device.values()})}}
    print(f"[{time.perf_counter()-t0:.0f}s] bijection: {len(placements)}/{len(atoms)} placed, "
          f"{len(b['device_unmatched'])} device tensors unmatched", flush=True)

    grads = {n: (ttnn.to_torch(l.grad).to(torch.float64) if l.grad is not None else None)
             for n, l in ours.items()}

    def our_grad_for(key):
        pls = placements.get(key)
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
    for key, ref in g_ref.items():
        full = pre + key
        mine, why = our_grad_for(key)
        if ref is None or mine is None:
            absent.append({"their_tensor": full, "reference_present": ref is not None,
                           "ours_present": mine is not None, "reason": why})
            continue
        if tuple(mine.shape) != tuple(ref.shape):
            absent.append({"their_tensor": full,
                           "shape_mismatch": [list(mine.shape), list(ref.shape)]})
            continue
        r, m = ref.numpy(), mine.numpy()
        rows.append({"their_tensor": full, "key": key,
                     "device": [p["device_path"] for p in placements[key]],
                     "rel_l2": rel_l2(m, r),
                     "max_abs_rel": float(np.max(np.abs(m - r)) / (np.max(np.abs(r)) + 1e-30)),
                     "ref_norm": float(np.linalg.norm(r)),
                     "ref_is_zero": bool(np.max(np.abs(r)) == 0.0)})
    rows.sort(key=lambda d: -d["rel_l2"])
    rel = [d["rel_l2"] for d in rows]
    rep["per_parameter"] = rows
    rep["absent"] = absent
    rep["presence"] = {"their_with_gradient": sum(1 for v in g_ref.values() if v is not None),
                       "their_total": len(g_ref), "compared": len(rows), "absent": len(absent),
                       "their_zero_valued": sum(1 for d in rows if d["ref_is_zero"]),
                       "rule": "SS3b: None matches None, zero matches zero. Nothing zero-filled."}
    rep["summary"] = {"compared": len(rows), "absent": len(absent),
                      "worst": rows[0] if rows else None, "best": rows[-1] if rows else None,
                      "median": float(np.median(rel)) if rel else None,
                      "over_bar": [d["their_tensor"] for d in rows if d["rel_l2"] > PER_TENSOR_BAR],
                      "per_tensor_pass": bool(rows) and max(rel) <= PER_TENSOR_BAR,
                      "median_pass": bool(rel) and float(np.median(rel)) <= MEDIAN_BAR}

    if rows:
        victim = rows[-1]
        r = g_ref[victim["key"]].numpy()
        m, _ = our_grad_for(victim["key"])
        m = m.numpy()
        amp = 1.0 + 2 * PER_TENSOR_BAR
        weak, strong = rel_l2(m * 1.01, r), rel_l2(m * amp, r)
        zeroed = rel_l2(np.zeros_like(m), r)
        rep["negative_control"] = {
            "tensor": victim["their_tensor"], "baseline_rel": victim["rel_l2"],
            "bar": PER_TENSOR_BAR,
            "protocol_1pct": {"rel_after": weak, "fires": bool(weak > PER_TENSOR_BAR)},
            "calibrated": {"perturbation": f"x{amp:.2f}", "rel_after": strong,
                           "fires": bool(strong > PER_TENSOR_BAR)},
            "zeroed": {"rel_after": zeroed, "fires": bool(zeroed > PER_TENSOR_BAR),
                       "answers": "which check fails if our model is replaced by zeros?"},
            "pass": bool(strong > PER_TENSOR_BAR and zeroed > PER_TENSOR_BAR)}

    ok = (rep["summary"]["per_tensor_pass"] and rep["summary"]["median_pass"] and not absent
          and rep.get("negative_control", {}).get("pass", False) and err is None)
    rep["verdict"] = "PASS" if ok else "FAIL"
    rep["scope_note"] = (
        ("Pairformer blocks %d..%d as ONE taped stack, driven by the bundle's own boundary: "
         "block %d's captured inputs in, block %d's captured output cotangent back. A PASS "
         "covers the tensors in `per_parameter` and nothing in `absent`." % (i, last, i, last))
        if a.stack else
        ("Pairformer block %d only, driven by the bundle's own boundary. A PASS covers the "
         "tensors in `per_parameter` and nothing in `absent`." % i))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"instrument_a_bundle_{tag}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)
    print(f"\ncompared {len(rows)} of their tensors, {len(absent)} absent")
    for d in rows[:12]:
        print(f"   {d['rel_l2']:.3e}  {d['their_tensor']}")
    for x in absent[:20]:
        print(f"   ABSENT {x}")
    if rows:
        print(f"worst {rows[0]['rel_l2']:.3e} on {rows[0]['their_tensor']}")
        print(f"median {rep['summary']['median']:.3e} over {len(rows)} tensors")
    print(f"\nVERDICT: {rep['verdict']}  ->  {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
