#!/usr/bin/env python3
"""Our port's trunk gradient over the captured boundary, in THEIR parameter space.

The device side does not depend on which upstream revision the reference came from, so this
writes tensors rather than ratios: one run per arm, scored afterwards against any reference.

ARMS. `shipped` is the configuration read off `OF3Trunk`'s own construction by spying on the
`Pairformer` constructor -- not a hardcoded copy of somebody's default, which is how an earlier
instrument came to call `scale_pair_bias=True` "shipped" while the trunk ships False. `flipped`
is that configuration with `scale_pair_bias=True` and nothing else moved. `--permute-cot` is the
break control: the captured cotangent paired with the wrong real token positions, weights, masks,
flags and kernels untouched.

THE FUSED FIVE. `device_bijection` matches by value and confirms elementwise, which is the right
default and cannot place the five `attn_pair_bias` leaves the port fuses or rescales:
`mha.linear_{q,k,v}.weight` and `mha.linear_q.bias` go into one padded `qkv_weight`/`qkv_bias`,
and `linear_z.weight` is multiplied by `_bias_scale` on the way to the card. Those 240 of 2,736
tensors are a quarter of a block's gradient mass at the far end of the stack, so dropping them
would leave the headline unreadable in exactly the direction this row is about. They are placed
DERIVED instead, from the construction in `tenstorrent.AttentionPairBias.__init__`, and every
derived placement is verified by rebuilding THEIR weight out of the device weight and comparing
it to the checkpoint. A placement that cannot reproduce the weight does not get to carry the
gradient.

PAD INVARIANCE. `--pad-scale K` multiplies the boundary's PAD rows (and, in z, the pad rows and
columns) by K and changes nothing else. Under correct masking a real output cannot depend on a
pad position and the captured output cotangent is exactly zero on the pads, so every parameter
gradient must be invariant to K. It is a control on both sides: a reference that moves means the
pads genuinely participate, and an arm that moves while the reference does not has a masking
leak feeding its weight gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_gradients"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PRE = "pairformer_stack.blocks."


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def apb_inverse(dev_t, leaf, n_heads, head_dim, padded_head_dim, c_s):
    """Their tensor's slot inside a fused/padded device tensor, as a pure selection.

    The forward map is `tenstorrent.AttentionPairBias.__init__`:
        qkv_weight = cat([q, k, v], 0).reshape(3H, d, c_s).pad(d -> D).reshape(3HD, c_s).t()
        qkv_bias   = cat([q_bias.reshape(H, d).pad(d -> D).reshape(HD), zeros(2HD)])
    Both are selections of the source entries into a larger zero-padded array, so the same
    selection applied to the device GRADIENT is the gradient of the source -- no scaling and no
    sum, which is what makes this safe to derive rather than fit.
    """
    import torch
    H, d, D = n_heads, head_dim, padded_head_dim
    if leaf.endswith("linear_q.bias"):
        return dev_t[:H * D].reshape(H, D)[:, :d].reshape(H * d).contiguous()
    x = dev_t.t().reshape(3 * H, D, c_s)
    off = {"linear_q.weight": 0, "linear_k.weight": H, "linear_v.weight": 2 * H}[
        leaf.split("mha.")[-1]]
    return x[off:off + H, :d, :].reshape(H * d, c_s).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--cap-last", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--arm", default="shipped", choices=("shipped", "flipped"))
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--pad-scale", type=float, default=1.0, metavar="K",
                    help="multiply the boundary's pad rows/columns by K, nothing else")
    ap.add_argument("--permute-cot", type=int, default=0, metavar="SEED")
    a = ap.parse_args()
    t0 = time.perf_counter()

    import numpy as np
    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import device_weights, get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    from tt_bio.train.checks import weight_coverage
    from tt_bio.train.lora import walked_weights
    import tt_bio.openfold3_trunk as OT
    from bijection_device import device_bijection

    # The checkout, not an installed package. `parity-gate-scores-installed-package-not-checkout`
    # is a standing fleet defect: the venv this runs in belongs to another tree, and a silent
    # resolution to it would measure that tree's trunk instead of this branch's.
    import tt_bio
    _want = os.path.join(os.getcwd(), "tt_bio")
    if not os.path.dirname(os.path.abspath(tt_bio.__file__)).startswith(_want):
        raise SystemExit(f"tt_bio resolved to {tt_bio.__file__}, not {_want} -- put the "
                         f"worktree on PYTHONPATH")

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # ---- the SHIPPED configuration, read off the shipped construction site -------------------
    spy = {}

    class _Stop(Exception):
        pass

    def _spy(*ar, **kw):
        spy["n_blocks"] = ar[0]
        spy["dims"] = list(ar[1:5])
        spy["transform_s"] = ar[5]
        spy["kwargs"] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v))
                         for k, v in kw.items()}
        raise _Stop()

    real_pf = OT.Pairformer
    OT.Pairformer = _spy
    try:
        OT.OF3Trunk(sd, ckc)
    except _Stop:
        pass
    finally:
        OT.Pairformer = real_pf
    if "kwargs" not in spy:
        raise SystemExit("the spy never reached Pairformer -- a hardcoded default would be the "
                         "only alternative and this row refuses to keep one")
    shipped_kw = dict(spy["kwargs"])
    kw = dict(shipped_kw)
    if a.arm == "flipped":
        kw["scale_pair_bias"] = True
    print(f"[{time.perf_counter()-t0:.0f}s] arm={a.arm} config {kw}", flush=True)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s_in, z_in = b["s_in"], b["z_in"]
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(z_in.shape[1])
    pad_rep = None
    if a.pad_scale != 1.0:
        pad = (sm.reshape(-1) <= 0)
        s_in = s_in.clone(); z_in = z_in.clone()
        s_in[:, pad] *= a.pad_scale
        z_in[:, pad, :] *= a.pad_scale
        z_in[:, :, pad] *= a.pad_scale
        pad_rep = {"pad_scale": a.pad_scale, "pad_rows": int(pad.sum()),
                   "real_rows": int((~pad).sum()),
                   "s_in_norm_after": float(s_in.norm()),
                   "z_in_norm_after": float(z_in.norm())}

    cap = torch.load(a.cap_last, map_location="cpu", weights_only=False)
    cot_s, cot_z = cap["cot"][0], cap["cot"][1]
    if cot_s is None or cot_z is None:
        raise SystemExit("captured cotangent missing -- the boundary is unusable")
    c = a.crop
    cot_s = cot_s.to(torch.float64)[:, :c].contiguous() if c else cot_s.to(torch.float64)
    cot_z = cot_z.to(torch.float64)[:, :c, :c].contiguous() if c else cot_z.to(torch.float64)
    perm_rep = None
    if a.permute_cot:
        real = torch.nonzero(sm.reshape(-1) > 0).reshape(-1)
        g = torch.Generator().manual_seed(a.permute_cot)
        order = real[torch.randperm(int(real.numel()), generator=g)]
        idx = torch.arange(int(sm.shape[-1]))
        idx[real] = order
        cot_s = cot_s[:, idx].contiguous()
        cot_z = cot_z[:, idx][:, :, idx].contiguous()
        perm_rep = {"seed": a.permute_cot, "real_positions_permuted": int(real.numel()),
                    "fixed_points": int((order == real).sum()),
                    "what": "the same cotangent, paired with the wrong token positions"}
        print(f"[{time.perf_counter()-t0:.0f}s] BREAK CONTROL: {perm_rep}", flush=True)

    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = (flat_all if a.blocks == spy["n_blocks"]
            else {k: v for k, v in flat_all.items() if int(k.split(".")[1]) < a.blocks})
    mod = T.Pairformer(a.blocks, *spy["dims"], spy["transform_s"], flat, ckc, **kw)

    s_fp32 = bool(kw.get("s_fp32_residual", False))
    s_dtype = ttnn.float32 if s_fp32 else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9

    rep = {"what": __doc__.strip().splitlines()[0], "arm": a.arm, "blocks": a.blocks,
           "tokens": N, "real_tokens": int(sm.sum()),
           "boundary": a.boundary, "boundary_sha256": sha256_file(a.boundary),
           "cotangent_from": a.cap_last, "cotangent_sha256": sha256_file(a.cap_last),
           "permuted_cotangent": perm_rep, "pad_perturbation": pad_rep,
           "shipped_config": {"source": "tt_bio/openfold3_trunk.py OF3Trunk.__init__, read by "
                                        "spying on the Pairformer constructor",
                              "kwargs": shipped_kw},
           "arm_config": kw, "s_input_dtype": str(s_dtype),
           "probe": {"s_in_norm": float(s_in.norm()), "z_in_norm": float(z_in.norm()),
                     "cot_s_norm": float(cot_s.norm()), "cot_z_norm": float(cot_z.norm())}}

    # A fresh upload per forward, and this is load-bearing: `PairformerLayer` accumulates its
    # residuals with `ttnn.add_`, so a reused input tensor makes the taped forward start from
    # the discovery forward's output (instrument_a_stack's note, which cost that row a run).
    before = device_weights(mod)
    ours = walked_weights(lambda: mod(fts(s_in), ft(z_in), ft(pm), ft(attn), ft(attn)), None, mod)
    after = device_weights(mod)
    rep["ours_discovery"] = {"reachable_before_forward": len(before),
                             "reachable_after_forward": len(after), "registered": len(ours)}
    print(f"[{time.perf_counter()-t0:.0f}s] discovery: {len(ours)} leaves registered", flush=True)

    err = None
    try:
        sa = ag.Tensor(fts(s_in), requires_grad=True)
        za = ag.Tensor(ft(z_in), requires_grad=True)
        with ag.tape():
            s_out, z_out = mod(sa, za, ft(pm), ft(attn), ft(attn))
        s_ours = ttnn.to_torch(s_out.value).to(torch.float64)
        z_ours = ttnn.to_torch(z_out.value).to(torch.float64)
        d_cot_s, d_cot_z = ft(cot_s), ft(cot_z)
        rep["cotangent_on_device"] = {
            "reference_cot_s_norm": float(cot_s.norm()),
            "device_cot_s_norm": float(torch.linalg.vector_norm(
                ttnn.to_torch(d_cot_s).to(torch.float64))),
            "reference_cot_z_norm": float(cot_z.norm()),
            "device_cot_z_norm": float(torch.linalg.vector_norm(
                ttnn.to_torch(d_cot_z).to(torch.float64))),
            "note": "a ratio away from 1 would mean the harness rescales a track on the way in"}
        ag.backward([s_out, z_out], [d_cot_s, d_cot_z])
    except Exception as e:                       # a gap is a finding, not a crash
        import traceback
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
        s_ours = z_ours = None
    rep["taped_backward_error"] = err
    if err:
        raise SystemExit(f"taped backward failed: {err}")

    cov = weight_coverage(mod)
    rep["leaves_against_total"] = {"total": cov.total, "registered": cov.registered,
                                   "with_grad": cov.with_grad,
                                   "without_grad": list(cov.without_grad)[:24]}
    print(f"[{time.perf_counter()-t0:.0f}s] LEAVES AGAINST TOTAL: {cov.registered}/{cov.total} "
          f"registered, {cov.with_grad}/{cov.total} with a gradient", flush=True)

    rep["input_grad_ours"] = {
        "ds_in_norm": (float(torch.linalg.vector_norm(ttnn.to_torch(sa.grad).to(torch.float64)))
                       if sa.grad is not None else None),
        "dz_in_norm": (float(torch.linalg.vector_norm(ttnn.to_torch(za.grad).to(torch.float64)))
                       if za.grad is not None else None)}

    # ---- placement: by value where the port keeps the tensor whole, derived where it fuses ----
    dev_all = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(mod).items()}
    atoms = {k[len("pairformer_stack."):]: v.detach().to(torch.float32)
             for k, v in sd.items() if k.startswith(PRE)
             and int(k[len(PRE):].split(".")[0]) < a.blocks}
    placements, per_device, unplaced_byvalue = {}, {}, []
    for j in range(a.blocks):
        tp = f"blocks.{j}."
        atoms_j = {k[len(tp):]: v for k, v in atoms.items() if k.startswith(tp)}
        dev_j = {k[len(tp):]: v for k, v in dev_all.items() if k.startswith(tp)}
        bj = device_bijection(dev_j, atoms_j)
        for k, pls in bj["placements"].items():
            placements[tp + k] = [dict(pl, device_path=tp + pl["device_path"]) for pl in pls]
        for k, m in bj["per_device"].items():
            per_device[tp + k] = m
        unplaced_byvalue += [tp + k for k in bj["their_unplaced"]]
    rep["bijection_by_value"] = {
        "their_tensors": len(atoms), "placed": len(placements),
        "unplaced": len(unplaced_byvalue),
        "unplaced_leaves": sorted({k.split(".", 2)[2] for k in unplaced_byvalue}),
        "device_unmatched": sorted(set(dev_all) - {pl["device_path"] for pls in
                                                   placements.values() for pl in pls})[:12],
        "scope": "per block, so an all-zero tensor cannot pair across blocks"}
    print(f"[{time.perf_counter()-t0:.0f}s] by value: {len(placements)}/{len(atoms)} placed, "
          f"{len(unplaced_byvalue)} unplaced", flush=True)

    # the derived half, and its own verification
    apb = mod.blocks[0].attention_pair_bias
    H, d = int(apb.n_heads), int(apb.head_dim)
    D = int(getattr(apb, "padded_head_dim", d))
    c_s = int(atoms["blocks.0.attn_pair_bias.layer_norm_a.weight"].shape[0])
    bias_scale = float(apb._bias_scale)
    # The scale the card ACTUALLY applies, measured here rather than assumed. `ttnn.multiply_`
    # by a python float on a BFLOAT16 tensor truncates the scalar's mantissa to bf16 instead of
    # rounding it: sqrt(24) = 4.898979 is applied as 4.875, 0.4903 % low. Round-to-nearest bf16
    # would be 4.90625, so this is truncation and not storage width. On a float32 tensor the
    # same call keeps the scalar to fp32. Measured on a ones tile in this process, in the
    # weight's own dtype, so the placement check and the gradient map both use the map the
    # shipped code applied rather than the one it asked for.
    _ones = ttnn.from_torch(torch.ones(32, 32), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=apb.z_weight.dtype)
    eff_scale = float(ttnn.to_torch(ttnn.multiply_(_ones, bias_scale)).to(torch.float64)[0, 0])
    derived, derived_check = {}, []
    FUSED = ["attn_pair_bias.mha.linear_q.weight", "attn_pair_bias.mha.linear_k.weight",
             "attn_pair_bias.mha.linear_v.weight", "attn_pair_bias.mha.linear_q.bias"]
    for j in range(a.blocks):
        dp = f"blocks.{j}.attention_pair_bias."
        for leaf in FUSED:
            key = f"blocks.{j}.{leaf}"
            if key in placements:
                continue           # by value already has it; do not shadow a verified placement
            src = dev_all[dp + ("qkv_bias" if leaf.endswith("bias") else "qkv_weight")]
            got = apb_inverse(src, leaf, H, d, D, c_s)
            want = atoms[key].to(torch.bfloat16).to(torch.float32)
            ok = bool(torch.equal(got, want))
            derived[key] = {"device_path": dp + ("qkv_bias" if leaf.endswith("bias")
                                                 else "qkv_weight"),
                            "rule": "apb_inverse", "leaf": leaf, "scale": 1.0}
            derived_check.append({"their_tensor": "pairformer_stack." + key, "bit_identical": ok,
                                  "rel": float(torch.linalg.vector_norm(got - want)
                                               / (torch.linalg.vector_norm(want) + 1e-30))})
        zk = f"blocks.{j}.attn_pair_bias.linear_z.weight"
        if zk not in placements:
            src = dev_all[dp + "z_weight"]
            # Forward, not inverse, and deliberately: the card computes
            # `bf16(bf16(W^T) * s)`, so unscaling in fp32 cannot come back bit-exact when
            # s != 1 -- it reads 5.2e-03, which is bf16's own 2^-8 and not a misplacement.
            # Rebuilding the DEVICE tensor from the checkpoint reproduces the rounding and is
            # exact in both arms, so the placement is checked rather than excused.
            want_dev = (atoms[zk].to(torch.bfloat16).t().contiguous().to(torch.float32)
                        * eff_scale).to(torch.bfloat16).to(torch.float32)
            derived[zk] = {"device_path": dp + "z_weight", "rule": "transpose_and_unscale",
                           "leaf": "attn_pair_bias.linear_z.weight", "scale": eff_scale}
            derived_check.append({
                "their_tensor": "pairformer_stack." + zk,
                "bit_identical": bool(torch.equal(src, want_dev)),
                "rel": float(torch.linalg.vector_norm(src - want_dev)
                             / (torch.linalg.vector_norm(want_dev) + 1e-30)),
                "direction": "checkpoint -> device tensor, the rounding reproduced on host",
                "unscale_rel_to_checkpoint": float(
                    torch.linalg.vector_norm((src / bias_scale).t().contiguous()
                                             - atoms[zk].to(torch.bfloat16).to(torch.float32))
                    / (torch.linalg.vector_norm(atoms[zk]) + 1e-30))})
    bad = [x for x in derived_check if not x["bit_identical"]]
    rep["ttnn_scalar_truncation"] = {
        "requested": bias_scale, "applied": eff_scale,
        "bf16_round_to_nearest_would_be": float(torch.tensor(bias_scale, dtype=torch.bfloat16)),
        "shortfall_pct": 100.0 * (1.0 - eff_scale / bias_scale) if bias_scale else 0.0,
        "what": "ttnn.multiply_ by a python float on a bfloat16 tensor truncates the scalar's "
                "mantissa to bf16 rather than rounding it; on a float32 tensor it does not. "
                "Measured on a ones tile in this process."}
    rep["derived_placements"] = {
        "count": len(derived), "verified_by_rebuilding_their_weight": len(derived_check),
        "bit_identical": sum(1 for x in derived_check if x["bit_identical"]),
        "bias_scale_requested": bias_scale, "bias_scale_applied_by_the_card": eff_scale,
        "bias_scale_shortfall_pct": 100.0 * (1.0 - eff_scale / bias_scale) if bias_scale else 0.0,
        "head_dim": d, "padded_head_dim": D, "n_heads": H,
        "worst_rel": max([x["rel"] for x in derived_check], default=0.0),
        "over_4e-3": bad[:8],
        "rule": "each derived placement is checked by rebuilding the WEIGHT the placement "
                "claims to address and requiring bit identity -- the fused qkv in their layout "
                "against the checkpoint, the scaled z_weight in the device's layout with the "
                "bf16 rounding reproduced on host. Bit identity is required in both arms; a "
                "placement that cannot reproduce the weight does not carry the gradient"}
    if bad:
        raise SystemExit(f"derived placement failed to rebuild {len(bad)} weights: {bad[:3]}")
    print(f"[{time.perf_counter()-t0:.0f}s] derived: {len(derived)} placed, "
          f"{rep['derived_placements']['bit_identical']} bit-identical, "
          f"worst rel {rep['derived_placements']['worst_rel']:.3e}", flush=True)

    grads = {n: (ttnn.to_torch(l.grad).to(torch.float64) if l.grad is not None else None)
             for n, l in ours.items()}
    out, absent = {}, []
    for key in sorted(atoms):
        full = "pairformer_stack." + key
        if key in placements:
            chosen = None
            for pl in placements[key]:
                gd = grads.get(pl["device_path"])
                if gd is not None:
                    chosen = (pl, gd)
                    break
            if chosen is None:
                absent.append({"their_tensor": full, "why": "placed but no gradient on the "
                                                            "device tensor", "how": "by_value"})
                continue
            pl, gd = chosen
            band = gd.narrow(pl["axis"], pl["start"], pl["length"])
            if pl["layout"].endswith("transposed"):
                band = band.T
            out[full] = band.contiguous()
        elif key in derived:
            dsc = derived[key]
            gd = grads.get(dsc["device_path"])
            if gd is None:
                absent.append({"their_tensor": full, "why": "no gradient on the fused device "
                                                            "tensor", "how": "derived"})
                continue
            if dsc["rule"] == "apb_inverse":
                out[full] = apb_inverse(gd, dsc["leaf"], H, d, D, c_s)
            else:
                out[full] = (gd * dsc["scale"]).t().contiguous()
        else:
            absent.append({"their_tensor": full, "why": "no device tensor carries this "
                                                        "parameter", "how": "none"})
    rep["placed"] = {"total_their_tensors": len(atoms), "with_our_gradient": len(out),
                     "absent": len(absent), "absent_detail": absent[:24],
                     "by_value": len(placements), "derived": len(derived)}
    print(f"[{time.perf_counter()-t0:.0f}s] OUR GRADIENT: {len(out)} of {len(atoms)} of their "
          f"tensors carry one, {len(absent)} absent", flush=True)

    torch.save({"grads": out, "s": s_ours, "z": z_ours, "arm": a.arm, "config": kw,
                "permute_cot": a.permute_cot, "pad_scale": a.pad_scale}, a.out)
    rep["out"] = a.out
    rep["seconds"] = time.perf_counter() - t0
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({"arm": a.arm, "placed": len(out), "absent": len(absent),
                      "s_norm": float(s_ours.norm()), "z_norm": float(z_ours.norm()),
                      "seconds": round(rep["seconds"], 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
