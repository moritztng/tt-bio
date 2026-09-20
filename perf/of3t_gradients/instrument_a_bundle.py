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
    # D23/R126: the published bundle is upstream 0.5.0 running a checkpoint 0.5.0's own registry
    # declares incompatible, so which bundle an arm was taken against is part of the arm. Bundle,
    # manifest, capture directory and output directory are arguments rather than constants.
    # Defaults are the published ones, so every arm already on the branch reads the same bytes.
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--manifest-json", default=None,
                    help=f"manifest to hash against. Default is {REF_BRANCH}:{MANIFEST_GIT}, "
                         f"read out of git and never from a working tree.")
    ap.add_argument("--cap", default=CAP, help="captured block boundaries to drive the arm with")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--s-fp32-residual", default="shipped", choices=("shipped", "on", "off"),
                    help="the single-track residual accumulation dtype. `tenstorrent.py` "
                         "implements this and its own comment gives the mechanism: a track "
                         "whose residual is much larger than its per-block update quantises "
                         "that update away, because bf16's resolution is relative to what the "
                         "accumulator already holds. It is True in exactly one place, the "
                         "CONFIDENCE Pairformer (openfold3_confidence.py:101); the trunk does "
                         "not pass it and runs the default False, and the trunk is the stack "
                         "whose single track grows 367x from block 0 to block 47. RELEASE "
                         "GATED: this changes accuracy and costs one [B, L, c_s] fp32 tensor. "
                         "The measurement is the deliverable, not the flag.")
    ap.add_argument("--nan-pad", action="store_true",
                    help="D28 discriminator. Poison the PAD token positions of the captured "
                         "input with NaN and report how many parameter gradients come back NaN "
                         "on each side. A crop-64 boundary carries 56 real tokens and 8 pad "
                         "rows, and the forward disagrees far more unmasked than masked, so the "
                         "device writes something different into the pad and the mask hides it "
                         "in the forward. A mask-clean forward does not imply a mask-clean "
                         "backward. NaN in the parameter gradients means the pad reaches the "
                         "weights and every crop-64 figure inherits it.")
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

    if a.manifest_json:
        man = json.load(open(a.manifest_json))
        man_src = a.manifest_json
    else:
        man = manifest_from_git()
        man_src = f"{REF_BRANCH}:{MANIFEST_GIT}"
    decl = {x["file"]: x for x in man["artifacts"] if "sha256" in x}
    gfile = man["validated_gradient"]["file"]
    got = sha256_file(os.path.join(a.bundle, gfile))
    if got != decl[gfile]["sha256"]:
        raise SystemExit(f"{gfile}: sha256 {got} != manifest {decl[gfile]['sha256']}")
    rep["bundle"] = {"file": gfile, "sha256": got, "verified": True,
                     "dir": a.bundle, "manifest": man_src,
                     "upstream_revision": man.get("upstream", {}).get("version", "0.5.0"),
                     "num_recycles": man["validated_gradient"]["num_recycles"],
                     "weights": "of3-p2-155k.pt (trained)",
                     "reference_fd_max_rel": man["validated_gradient"]["finite_difference"]["max_rel_err"],
                     "n_nonzero_of_total":
                         f"{man['validated_gradient']['n_nonzero_gradient_tensors']} of "
                         f"{man['validated_gradient']['n_parameters']}"}
    print(f"[{time.perf_counter()-t0:.0f}s] bundle hash verified", flush=True)

    cap = torch.load(os.path.join(a.cap, f"block{i}_boundary.pt"),
                     map_location="cpu", weights_only=False)
    #: In stack mode the cotangent belongs to the LAST block's output, not the first's. Taking
    #: it from the wrong end would drive the backward with a cotangent for a different tensor
    #: and produce a number that looks like a gradient and is not one.
    cap_out = cap if not a.stack else torch.load(
        os.path.join(a.cap, f"block{last}_boundary.pt"), map_location="cpu", weights_only=False)
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
    if a.nan_pad:
        if single_mask is None:
            raise SystemExit("--nan-pad needs the single mask to know which rows are pad")
        pad = (single_mask.reshape(-1) <= 0)
        if not bool(pad.any()):
            raise SystemExit("--nan-pad: this boundary has no pad rows, nothing to poison")
        s_in = s_in.clone(); z_in = z_in.clone()
        s_in[:, pad] = float("nan")
        z_in[:, pad, :] = float("nan")
        z_in[:, :, pad] = float("nan")
        rep["nan_pad"] = {"pad_rows": int(pad.sum()), "real_rows": int((~pad).sum()),
                          "poisoned": ["s_in rows", "z_in rows", "z_in cols"]}
        print(f"[{time.perf_counter()-t0:.0f}s] D28: poisoned {int(pad.sum())} pad rows of "
              f"{int(pad.numel())} with NaN", flush=True)

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
        ref_all = torch.load(os.path.join(a.bundle, gfile), map_location="cpu", weights_only=False)
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
    # The shipped trunk does not pass s_fp32_residual, so `shipped` is the class default.
    s_fp32 = {"shipped": False, "on": True, "off": False}[a.s_fp32_residual]
    rep["shipped_config"]["s_fp32_residual"] = s_fp32
    rep["shipped_config"]["s_fp32_residual_arm"] = a.s_fp32_residual
    mod = T.Pairformer(nb, head_dim, no_heads_pair, c_s // no_heads_pair_bias,
                       no_heads_pair_bias, True, flat, ckc,
                       scale_pair_bias=spb, tri_att_scale_pair_bias=SHIPPED_TRI_SPB,
                       fp32_softmax=(a.fp32_softmax == "on"),
                       transpose_bias=transpose_bias, accurate_softmax=acc,
                       s_fp32_residual=s_fp32)
    T.Module.torch_to_tt = orig

    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    # `_s_compute`/`_s_residual` are no-ops unless the single track ARRIVES in fp32:
    # `_s_compute` returns `s` unchanged when `s.dtype == ttnn.bfloat16`, and `_s_residual`'s
    # fp32 branch adds into whatever dtype `s` already is. So an arm that feeds everything in
    # bf16 cannot make this lever fire, and would report a null result for the wrong reason.
    s_dtype = ttnn.float32 if s_fp32 else ttnn.bfloat16
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)
    rep["shipped_config"]["s_input_dtype"] = str(s_dtype)
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
    ours = walked_weights(lambda: mod(fts(s_in), ft(z_in), ft(pm), ft(attn), ft(attn)), None, mod)
    after = device_weights(mod)
    rep["ours_discovery"] = {"from_loader": len(loaded),
                             "reachable_before_forward": len(before),
                             "reachable_after_forward": len(after),
                             "registered": len(ours)}
    print(f"[{time.perf_counter()-t0:.0f}s] ours: loader {len(loaded)}, before {len(before)}, "
          f"after {len(after)}, registered {len(ours)}", flush=True)

    err = None
    try:
        sa = ag.Tensor(fts(s_in), requires_grad=True)
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

    if a.nan_pad:
        # The whole point of the run: does NaN in the PAD reach the WEIGHTS? Counted on both
        # sides, because "both leak" and "only ours leaks" are different findings. In
        # `--reference bundle` mode theirs is a saved tensor and never saw the poison, so this
        # is only a statement about our side unless `--reference block-eval` re-ran it.
        def nan_count(d):
            n_nan = sum(1 for v in d.values()
                        if v is not None and bool(torch.isnan(v).any()))
            return {"tensors": sum(1 for v in d.values() if v is not None), "nan": n_nan,
                    "nan_tensors": sorted(k for k, v in d.items()
                                          if v is not None and bool(torch.isnan(v).any()))[:12]}
        rep["nan_pad"]["ours"] = nan_count(grads)
        rep["nan_pad"]["theirs"] = nan_count(g_ref)
        rep["nan_pad"]["theirs_recomputed_on_poisoned_input"] = (a.reference == "block-eval")
        rep["nan_pad"]["forward_out_nan"] = {
            "ours_s": bool(torch.isnan(s_out_t).any()) if "s_out_t" in dir() else None}
        print(f"[{time.perf_counter()-t0:.0f}s] D28: ours {rep['nan_pad']['ours']['nan']} of "
              f"{rep['nan_pad']['ours']['tensors']} gradients NaN; theirs "
              f"{rep['nan_pad']['theirs']['nan']} of {rep['nan_pad']['theirs']['tensors']} "
              f"(recomputed: {a.reference == 'block-eval'})", flush=True)

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
        # PROTOCOL A20. An absent tensor carries its REFERENCE NORM out with it, so the reach
        # below can be stated against the full reference mass instead of against the tensors
        # that happened to place. An unplaceable tensor is UNREACHED, not not-there: at block 47
        # `attn_pair_bias.linear_z.weight` is a quarter of the block's gradient mass and it goes
        # absent under the shipped `scale_pair_bias`, because the device folds the scale into
        # the weight and the by-value bijection stops recognising it. Reporting reach over the
        # compared set alone hides exactly the tensor the lever acts on.
        ref_n = None if ref is None else float(np.linalg.norm(ref.numpy()))
        if ref is None or mine is None:
            absent.append({"their_tensor": full, "reference_present": ref is not None,
                           "ours_present": mine is not None, "reason": why,
                           "ref_norm": ref_n})
            continue
        if tuple(mine.shape) != tuple(ref.shape):
            absent.append({"their_tensor": full,
                           "shape_mismatch": [list(mine.shape), list(ref.shape)],
                           "ref_norm": ref_n})
            continue
        r, m = ref.numpy(), mine.numpy()
        # PROTOCOL A16/D35. A single rel_l2 cannot say WHICH WAY we are wrong. With
        # rel^2 = 1 + r^2 - 2*r*c it only bounds the norm ratio to [1-rel, 1+rel], and a zero
        # gradient gives r = 0 and rel = 1 exactly, so every rel is scored against a 1.0
        # ceiling. Two more floats, already in memory and free, separate the cases outright:
        #   r << 1, c ~ 1 -> a SHRUNK copy (what a quantised-away residual update looks like)
        #   r  > 1, c ~ 1 -> an inflated copy
        #   c ~ 0         -> noise, and the gradient carries no signal at all
        rn, mn = float(np.linalg.norm(r)), float(np.linalg.norm(m))
        rows.append({"their_tensor": full, "key": key,
                     "device": [p["device_path"] for p in placements[key]],
                     "rel_l2": rel_l2(m, r),
                     "max_abs_rel": float(np.max(np.abs(m - r)) / (np.max(np.abs(r)) + 1e-30)),
                     "ref_norm": rn,
                     "device_norm": mn,
                     "norm_ratio": (mn / rn) if rn else None,
                     "cos": (float((m * r).sum() / (mn * rn)) if (mn and rn) else None),
                     "zero_model_rel": 1.0,
                     "ref_is_zero": bool(np.max(np.abs(r)) == 0.0)})
    rows.sort(key=lambda d: -d["rel_l2"])
    rel = [d["rel_l2"] for d in rows]
    rep["per_parameter"] = rows
    rep["absent"] = absent
    # Reach, over the FULL reference mass rather than over what placed (A20).
    sq_cmp = sum(d["ref_norm"] ** 2 for d in rows)
    sq_abs = sum((x.get("ref_norm") or 0.0) ** 2 for x in absent)
    sq_all = sq_cmp + sq_abs
    rep["reach"] = {
        "rule": "PROTOCOL A20: an unplaceable tensor is UNREACHED, not absent from the "
                "denominator. Reach is over the full reference gradient mass of this scope.",
        "compared_tensors": len(rows), "absent_tensors": len(absent),
        "squared_norm_compared": sq_cmp, "squared_norm_absent": sq_abs,
        "squared_norm_full_scope": sq_all,
        "reach_over_full_scope": (sq_cmp / sq_all) if sq_all else None,
        # Two different questions, and the campaign has quoted them as one. `reach` is how much
        # of the scope's gradient mass was MEASURED; `passing` is how much of it is inside the
        # per-tensor bar. Unreached mass counts against both, which is the point of A20.
        "passing_share_of_full_scope":
            (sum(d["ref_norm"] ** 2 for d in rows if d["rel_l2"] <= PER_TENSOR_BAR) / sq_all)
            if sq_all else None,
        "passing_share_of_compared":
            (sum(d["ref_norm"] ** 2 for d in rows if d["rel_l2"] <= PER_TENSOR_BAR) / sq_cmp)
            if sq_cmp else None,
        "unreached_ranked": sorted(
            ({"tensor": x["their_tensor"], "ref_norm": x.get("ref_norm"),
              "share_of_full_scope": ((x.get("ref_norm") or 0.0) ** 2 / sq_all) if sq_all else None,
              "reason": x.get("reason") or "shape mismatch"} for x in absent),
            key=lambda d: -(d["share_of_full_scope"] or 0.0)),
    }
    rep["presence"] = {"their_with_gradient": sum(1 for v in g_ref.values() if v is not None),
                       "their_total": len(g_ref), "compared": len(rows), "absent": len(absent),
                       "their_zero_valued": sum(1 for d in rows if d["ref_is_zero"]),
                       "rule": "SS3b: None matches None, zero matches zero. Nothing zero-filled."}
    # A16: the zero-model answer beside the median, so "how much better than emitting zeros"
    # can be read off this file instead of derived by a reader.
    _kept = [d for d in rows if d["ref_norm"] >= 1e-12]
    _rel = sorted(d["rel_l2"] for d in _kept)
    _q = lambda f: _rel[min(len(_rel) - 1, int(f * (len(_rel) - 1)))] if _rel else None
    _med = float(np.median(_rel)) if _rel else None
    _rat = [d["norm_ratio"] for d in _kept if d["norm_ratio"] is not None]
    _cos = [d["cos"] for d in _kept if d["cos"] is not None]
    rep["identifiability"] = {
        "rule": "PROTOCOL A16/D35: rel alone bounds the norm ratio to [1-rel, 1+rel] and is "
                "scored against a zero model's 1.0. norm_ratio and cos separate a shrunk copy, "
                "an inflated copy and noise.",
        "n_kept_a14": len(_kept),
        "zero_model_median": 1.0,
        "median_rel": _med,
        "better_than_zeros_pct": (100.0 * (1.0 - _med)) if _med is not None else None,
        "iqr_over_median": ((_q(0.75) - _q(0.25)) / _med) if _med else None,
        "norm_ratio_bound_from_median_rel": ([1 - _med, 1 + _med] if _med is not None else None),
        "norm_ratio": {"median": float(np.median(_rat)) if _rat else None,
                       "min": min(_rat) if _rat else None, "max": max(_rat) if _rat else None},
        "cos": {"median": float(np.median(_cos)) if _cos else None,
                "min": min(_cos) if _cos else None, "max": max(_cos) if _cos else None},
    }
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
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"instrument_a_bundle_{tag}.json")
    json.dump(rep, open(path, "w"), indent=1, default=str)
    print(f"\ncompared {len(rows)} of their tensors, {len(absent)} absent; reach over the FULL "
          f"reference mass of this scope "
          f"{100*(rep['reach']['reach_over_full_scope'] or 0):.3f} %, of which "
          f"{100*(rep['reach']['passing_share_of_full_scope'] or 0):.3f} % passes the bar")
    for u in rep["reach"]["unreached_ranked"][:4]:
        if u["share_of_full_scope"]:
            print(f"   UNREACHED {100*u['share_of_full_scope']:6.2f} %  {u['tensor']}")
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
