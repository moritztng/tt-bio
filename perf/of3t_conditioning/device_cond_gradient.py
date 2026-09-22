#!/usr/bin/env python3
"""Instrument A at conditioning scope: our device conditioning against the reference's float64.

`diffusion_module.diffusion_conditioning` is 36.9462 % of OpenFold3's squared gradient norm over
26 tensors, and four of those 26 are 33.9354 % of the whole model over 1,985 scalars. This runs
OUR `tt_bio.openfold3_diffusion.OF3DiffusionConditioning` on the card at the boundary
`capture_cond_boundary.py` took from BUNDLE-MIN-043, seeds it with THEIR cotangent at the
conditioning's own output, and compares all 26 parameter gradients against `grad_f64`.

D19 cancels here by construction: the 6.735e-03 forward gap lives upstream of these inputs, so
the floor under this comparison is zero.

Two protocol rules shape what it does, not just what it prints.

PROTOCOL A18 -- the forward gates the gradient. `--forward-only` runs the discriminator alone:
our `si` and our `zij` against theirs, separately, on the captured inputs. A disagreeing forward
invalidates a gradient taken at it. An agreeing one clears a NECESSARY condition and nothing
more (A18 addendum, D9: a 3.2x gradient shift hid under a 12 % forward shift).

PROTOCOL A23 -- the headline is mass-weighted. `rel_l2` over the concatenated 26 is the number;
the median over 26 tensors is printed beside it and never instead of it, because the median
tensor of this section holds 0.0002 % of the model. The four heavy tensors are reported
individually with `rel_l2`, the norm ratio `r = ||g_dev|| / ||g_ref||` and the cosine, since rel
alone bounds r to [1-rel, 1+rel] and says nothing about direction.

The reference drew `use_conditioning=False` at this step, so their own forward zeroes `si_trunk`
and `zij_trunk` before the concat. We feed zeros for the same reason: it is the function the
captured cotangent and the published gradient were taken at. Nothing in our module changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
CAP = "/home/ttuser/of3t_cond_cap/cond_boundary.pt"
OUT = "perf/of3t_conditioning"
# The model's own squared gradient norm on the 0.4.3 reference, exhaustive over 4,170 tensors
# (perf/of3t_orchestrator/WHERE_THE_GRADIENT_MASS_LIVES.json). Every share below divides by it.
MODEL_SQ_NORM = 10.279642678524985
HEAVY = ("transition_s.0.layer_norm.bias", "transition_s.1.layer_norm.bias",
         "transition_s.0.layer_norm.weight", "layer_norm_s.weight")
A14_NORM_CUT = 1e-12


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cap", default=CAP)
    p.add_argument("--out-dir", default=OUT)
    p.add_argument("--tag", default="")
    p.add_argument("--samples", default="all",
                   help="'all' for the 48 noise levels the reference gradient sums, or a "
                        "comma list. The pair branch has no sample axis and runs once.")
    p.add_argument("--forward-only", action="store_true",
                   help="A18's discriminator alone, before any gradient is taken")
    p.add_argument("--negative-control", default="none", dest="negctl",
                   choices=["none", "reverse-cot"],
                   help="`reverse-cot` seeds sample k with sample 47-k's cotangent. The forward, "
                        "the weights, the bijection and the arithmetic are all untouched, so a "
                        "run that still passes is measuring something other than the gradient. "
                        "A zero-model baseline cannot catch that: it breaks OUR side, and the "
                        "question is whether the check reads THEIR seed at all.")
    p.add_argument("--act", default="fp32", choices=["fp32", "bf16"],
                   help="device activation dtype for the taped arm")
    a = p.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    from of3_coverage import _device_weights

    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion import OF3DiffusionConditioning
    from tt_bio.openfold3_weights import _sub

    B = torch.load(a.cap, map_location="cpu", weights_only=False)
    ref_grad = B["grad_f64"]
    si_ref, zij_ref = B["si_ref"], B["zij_ref"]
    si_cot, zij_cot = B["si_cot"], B["zij_cot"]
    n_sample, n_token = int(si_ref.shape[1]), int(si_ref.shape[-2])
    c_s, c_z = int(si_ref.shape[-1]), int(zij_ref.shape[-1])
    use_cond = bool(B["use_conditioning"])
    # A forced arm is a DIFFERENT function from the one the model's squared gradient norm was
    # measured over, so its section mass is not a share of that norm and the "% of model"
    # denominators do not apply to it. They are emitted as null rather than as a number that
    # reads like a model share (a stale field that is merely labelled still gets read).
    forced = bool(B.get("forced", False))
    tok = B["token_mask"].reshape(n_token).double()
    print(f"[{time.perf_counter()-t0:.0f}s] boundary: {n_token} tokens ({int(tok.sum())} real), "
          f"{n_sample} noise levels, c_s={c_s} c_z={c_z}, use_conditioning={use_cond}, "
          f"{len(ref_grad)} reference gradients", flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dcsd = _sub(_sub(sd, "diffusion_module"), "diffusion_conditioning")
    shape_by_name = {k: tuple(v.shape) for k, v in dcsd.items() if torch.is_tensor(v)}

    # ---- the bijection, by transpose-invariant fingerprint ----------------------------------
    # `_w_tt` hands `ttnn.from_torch` a fresh `w.t().contiguous()`, so an id map over the
    # checkpoint catches nothing. Recording every `from_torch` for the duration of construction
    # cannot miss a load path by construction (device_gradient.py's amendment item 1).
    def fingerprint(x):
        x = x.double()
        return (tuple(sorted(x.shape)), round(float(x.sum()), 9),
                round(float(x.abs().max()), 9), x.numel())

    fp_name, fp_clash = {}, set()
    for k, v in dcsd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        f = fingerprint(v)
        if f in fp_name:
            fp_clash.add(f)
        fp_name[f] = k

    reg = {}
    orig_from_torch = ttnn.from_torch

    def recording_from_torch(tensor, *args, **kw):
        v = orig_from_torch(tensor, *args, **kw)
        try:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                f = fingerprint(tensor)
                if f in fp_name and f not in fp_clash:
                    reg[id(v)] = fp_name[f]
        except Exception:
            pass
        return v

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32 if a.act == "fp32" else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)

    ttnn.from_torch = recording_from_torch
    try:
        with device_dtype_override(act):
            dc = OF3DiffusionConditioning(dcsd, cfg)
    finally:
        ttnn.from_torch = orig_from_torch
    walked = _device_weights(dc)
    named = {id(t): reg.get(id(t)) for t in walked.values()}
    n_named = sum(1 for v in named.values() if v)
    print(f"[{time.perf_counter()-t0:.0f}s] module built: {len(walked)} device weights, "
          f"{n_named} carry a checkpoint name ({len(fp_clash)} fingerprint clashes)", flush=True)

    # ---- the captured inputs, on device ------------------------------------------------------
    # `use_conditioning=False` is the reference's own draw at this step: their forward zeroes
    # both trunk inputs before the concat, so ours takes zeros. Everything else is theirs.
    z_trunk = B["zij_trunk"].reshape(1, n_token, n_token, c_z).float()
    s_trunk = B["si_trunk"].reshape(1, n_token, c_s).float()
    if not use_cond:
        z_trunk = torch.zeros_like(z_trunk)
        s_trunk = torch.zeros_like(s_trunk)
    relpos_d = ft(B["relpos"].reshape(1, n_token, n_token, -1))
    z_trunk_d, s_trunk_d = ft(z_trunk), ft(s_trunk)
    s_input_d = ft(B["si_input"].reshape(1, n_token, -1))
    tokm = tok.float()
    tok_d = ft(tokm.reshape(1, n_token, 1))
    pair_d = ft((tokm[:, None] * tokm[None, :]).reshape(1, n_token, n_token, 1))
    n_emb = B["n_emb"].reshape(n_sample, -1).float()
    which = list(range(n_sample)) if a.samples == "all" else \
        [int(x) for x in a.samples.split(",")]

    def _t(x):
        return ttnn.to_torch(x.value if hasattr(x, "value") else x).double()

    def _rel(x, y):
        x, y = x.reshape(-1), y.reshape(-1)
        return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

    # ---- A18: the forward discriminator, before any gradient ---------------------------------
    with device_dtype_override(act):
        zij_ours = _t(dc.pair(z_trunk_d, relpos_d, pair_d)).reshape(zij_ref[0, 0].shape)
        si_ours = torch.stack(
            [_t(dc.single(s_trunk_d, s_input_d, ft(n_emb[k].reshape(1, 1, -1)), tok_d)
                ).reshape(si_ref[0, 0].shape) for k in which])
    si_theirs = torch.stack([si_ref[0, k].double() for k in which])
    fwd = {"si": _rel(si_ours, si_theirs), "zij": _rel(zij_ours, zij_ref[0, 0].double()),
           "si_per_sample": [_rel(si_ours[i], si_theirs[i]) for i in range(len(which))],
           "act": a.act, "bar": 5.0e-02}
    fwd["si_worst_sample"] = max(fwd["si_per_sample"])
    print(f"[{time.perf_counter()-t0:.0f}s] A18 discriminator ({a.act}): si {fwd['si']:.6e} "
          f"(worst sample {fwd['si_worst_sample']:.6e}), zij {fwd['zij']:.6e}, "
          f"bar 5.0e-02", flush=True)
    if a.forward_only:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, f"forward_discriminator{a.tag}.json")
        json.dump({"forward": fwd, "tokens": n_token, "real_tokens": int(tok.sum()),
                   "samples": which, "use_conditioning": use_cond},
                  open(path, "w"), indent=1, sort_keys=True)
        print("wrote", path, flush=True)
        return 0

    # ---- instrument A: tape, seed with their cotangent, accumulate ---------------------------
    for t in walked.values():
        ag.parameter(t)
    probe_name = "transition_s.0.layer_norm.bias"          # the section's heaviest tensor
    probe_t = next((t for t in walked.values() if named.get(id(t)) == probe_name), None)
    probe, err = [], None
    try:
        with device_dtype_override(act), ag.tape():
            z = dc.pair(ag.Tensor(z_trunk_d), ag.Tensor(relpos_d), ag.Tensor(pair_d))
            ag.backward([z], [ft(zij_cot[0, 0])])
        print(f"[{time.perf_counter()-t0:.0f}s] pair branch taped and seeded", flush=True)
        for i, k in enumerate(which):
            with device_dtype_override(act), ag.tape():
                s = dc.single(ag.Tensor(s_trunk_d), ag.Tensor(s_input_d),
                              ag.Tensor(ft(n_emb[k].reshape(1, 1, -1))), ag.Tensor(tok_d))
                kk = (n_sample - 1 - k) if a.negctl == "reverse-cot" else k
                ag.backward([s], [ft(si_cot[0, kk])])
            # The 48 noise levels are summed by 48 tapes accumulating into the same leaves,
            # which is exact only if `backward` ADDS across tape contexts. A probe norm that
            # stays flat means the run measured the last sample alone.
            #
            # Watch a SINGLE-branch tensor by name. The first pass watched whichever leaf had
            # a gradient first, which is `layer_norm_z.weight` -- a pair-branch tensor that the
            # pair tape fills once and no single tape touches. It read the same value 48 times
            # and would have read the same value had accumulation been broken, so the check
            # could not fail. A probe that cannot fail is not a probe.
            lf = ag._PARAMS.get(id(probe_t)) if probe_t is not None else None
            g = getattr(lf, "grad", None)
            probe.append(None if g is None else float(_t(g).norm()))
            if i % 12 == 0 or i == len(which) - 1:
                print(f"[{time.perf_counter()-t0:.0f}s]   sample {k} ({i+1}/{len(which)}), "
                      f"probe grad norm {probe[-1]}", flush=True)
    except Exception as e:
        err = e
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)

    # ---- compare, transposing back by shape --------------------------------------------------
    ours = {}
    for dev_t in walked.values():
        nm = named.get(id(dev_t))
        if nm is None:
            continue
        leaf = ag._PARAMS.get(id(dev_t))
        g = getattr(leaf, "grad", None)
        if g is None:
            continue
        gt = _t(g)
        want = shape_by_name[nm]
        gt = gt.reshape(gt.shape[-len(want):]) if gt.dim() > len(want) else gt
        if tuple(gt.shape) != want and tuple(gt.shape)[::-1] == want:
            gt = gt.t().contiguous()
        ours[nm] = gt

    # A20: the denominator is the FULL reference mass in scope, not the compared subset.
    scope_sq = sum(float(v.double().pow(2).sum()) for v in ref_grad.values())
    rows, unreached, excluded = [], [], []
    for nm, r in sorted(ref_grad.items()):
        rd = r.double()
        rn = float(torch.linalg.vector_norm(rd))
        if rn < A14_NORM_CUT:                       # A14: divide-by-zero guard only
            excluded.append(nm)
            continue
        gt = ours.get(nm)
        if gt is None or tuple(gt.shape) != tuple(rd.shape):
            unreached.append(nm)
            continue
        dn = float(torch.linalg.vector_norm(gt))
        rows.append({"tensor": nm, "rel_l2": float(torch.linalg.vector_norm(gt - rd) / rn),
                     "ref_norm": rn, "device_norm": dn, "norm_ratio": dn / rn,
                     "cos": (float((gt * rd).sum() / (dn * rn)) if dn else 0.0),
                     "ref_sq": rn * rn,
                     "share_of_model_pct": (None if forced
                                            else 100.0 * rn * rn / MODEL_SQ_NORM),
                     "share_of_section_pct": 100.0 * rn * rn / scope_sq,
                     "numel": int(rd.numel())})

    def mass_weighted(zero_model=False):
        """A23 rule 2: `rel_l2` over the CONCATENATED set. Mass-weighted by construction --
        each tensor enters with its own norm, so the four that hold 91.85 % of the section
        decide the number and the 22 that hold 8.15 % move it by what they are worth."""
        num = sum((r["ref_norm"] ** 2) if zero_model
                  else (r["rel_l2"] * r["ref_norm"]) ** 2 for r in rows)
        den = sum(r["ref_norm"] ** 2 for r in rows)
        return (num / den) ** 0.5 if den else None

    med = sorted(r["rel_l2"] for r in rows)
    sort_key = (lambda r: -r["ref_sq"])
    reached_sq = sum(r["ref_sq"] for r in rows)
    heavy = {r["tensor"]: {k: r[k] for k in
                           ("rel_l2", "norm_ratio", "cos", "ref_norm", "device_norm",
                            "share_of_model_pct", "share_of_section_pct", "numel")}
             for r in rows if r["tensor"] in HEAVY}
    worst = max(rows, key=lambda r: r["rel_l2"]) if rows else None
    heavy_sq = sum(r["ref_sq"] for r in rows if r["tensor"] in HEAVY)

    rep = {
        "forward_A18": fwd,
        "gradient_A23": {
            "mass_weighted_rel_l2": mass_weighted(),
            "zero_model_mass_weighted_rel_l2": mass_weighted(zero_model=True),
            "median_over_tensors": (med[len(med) // 2] if med else None),
            "zero_model_median_over_tensors": (1.0 if med else None),
            "bar_mass_weighted": 2.0e-02, "bar_per_tensor": 5.0e-02,
            "over_per_tensor_bar": sum(1 for r in rows if r["rel_l2"] > 5.0e-02),
            "worst": worst, "compared": len(rows),
            "reach_pct_of_section": 100.0 * reached_sq / scope_sq if scope_sq else None,
            "reach_pct_of_model": (None if forced
                                   else 100.0 * reached_sq / MODEL_SQ_NORM),
            "section_pct_of_model": (None if forced
                                     else 100.0 * scope_sq / MODEL_SQ_NORM),
        },
        "heavy_four": heavy,
        "heavy_four_pct_of_model": (None if forced else 100.0 * heavy_sq / MODEL_SQ_NORM),
        "arm": ("use_conditioning=True, SYNTHETIC BRANCH: the reference drew False at this "
                "step, so this arm's float64 reference was recomputed on their code at the "
                "same inputs with the same cotangent. The model-share denominators are null "
                "here because the model norm was measured on the drawn branch."
                if forced else
                "use_conditioning=False, as the reference drew it. The published bundle "
                "gradient is the reference."),
        "heavy_four_pct_of_section": 100.0 * heavy_sq / scope_sq if scope_sq else None,
        "a14_excluded": excluded, "a14_norm_cut": A14_NORM_CUT,
        "unreached": unreached,
        "accumulation_probe": probe, "accumulation_probe_tensor": probe_name,
        "accumulation_probe_grows": (
            None if len(probe) < 2 or probe[0] in (None, 0.0)
            else probe[-1] / probe[0]),
        "reference_tensors": len(ref_grad), "device_weights": len(walked),
        "device_weights_named": n_named,
        "negative_control": a.negctl,
        "tokens": n_token, "real_tokens": int(tok.sum()), "samples": which,
        "use_conditioning": use_cond, "act": a.act,
        "model_sq_norm": MODEL_SQ_NORM, "section_sq_norm": scope_sq,
        "per_tensor": sorted(rows, key=sort_key),
        "error": None if err is None else f"{type(err).__name__}: {err}",
    }
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"device_cond_gradient{a.tag}.json")
    json.dump(rep, open(path, "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("per_tensor", "accumulation_probe")},
                     indent=1, default=str), flush=True)
    print("wrote", path, flush=True)
    return 1 if err is not None else 0


if __name__ == "__main__":
    sys.exit(main())
