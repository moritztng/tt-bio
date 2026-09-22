#!/usr/bin/env python3
"""Does the trunk forward disagreement reach SHIPPED INFERENCE?

`of3t-pairformer` measured the 48-block pairformer forward at 2.796859e-01 on the masked pair
track (D87), UNDER THE TRAINING TAPE, and its gradient result inherits that. Three readings were
open: a product defect, a training-only defect, or the instrument. This runs the three arms that
separate them, all over the SAME captured boundary `of3t_gradients/cap`, all against the SAME
float64 reference (upstream's own float64 forward, captured by
`perf/of3t_gradients/capture_trunk_boundary.py`).

  SHIPPED      the ordinary inference path -- no tape, no `ag.Tensor` -- over the boundary.
  TAPED        the same inputs under `ag.tape()`. Taped-against-shipped on one input separates
               the tape from the composition.
  COMPOSITION  block k fed block k's CAPTURED input instead of our own block k-1 output. If
               per-block agreement is good while the 48-block composition is 28 % off, the error
               compounds and the per-block arms were never wrong.

The shipped configuration is READ OFF THE SHIPPED CONSTRUCTION SITE, not hardcoded. `OF3Trunk`
is built with `tt_bio.openfold3_trunk.Pairformer` replaced by a spy that records the resolved
keyword arguments and stops construction there. That is deliberate: `instrument_a_bundle.py`
hardcodes `SHIPPED_SPB, SHIPPED_TRI_SPB = True, False`, and the shipped trunk has built
`scale_pair_bias=False` since `8d8dbb14e` ("hold the OF3 trunk pair-bias default at False, and
guard it"), so every figure that instrument labels `shipped` is of a configuration inference does
not run. A hardcoded copy of somebody else's default is the failure mode this row was sent to
check for, so this arm refuses to keep one.

Reported per comparison: relative L2, the norm ratio and the cosine, masked and unmasked. Relative
L2 alone cannot say whether we are too large, too small, or pointing elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
CAP = "/home/ttuser/of3t_gradients/cap"
CAPALL = "/home/ttuser/of3t_trunkfwd/capall/blocks_all_crop64.pt"


def metrics(ours, ref):
    """rel_l2, norm ratio and cosine in float64. A rel alone cannot name the direction."""
    import torch
    a = ours.to(torch.float64).reshape(-1)
    b = ref.to(torch.float64).reshape(-1)
    nb = float(torch.linalg.vector_norm(b))
    na = float(torch.linalg.vector_norm(a))
    den = nb if nb > 0 else 1e-300
    cos = float((a @ b) / ((na * nb) or 1e-300))
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / den),
            "norm_ratio": na / den, "cos": cos, "ours_norm": na, "ref_norm": nb}


def compare(s_ours, z_ours, s_ref, z_ref, sm, pm):
    import torch
    msk = sm.reshape(1, -1, 1)
    n = pm.shape[-1]
    pmk = pm.reshape(1, n, n, 1)
    f = lambda x: x.to(torch.float64)
    return {"s": metrics(f(s_ours), f(s_ref)),
            "z": metrics(f(z_ours), f(z_ref)),
            "s_masked": metrics(f(s_ours) * msk, f(s_ref) * msk),
            "z_masked": metrics(f(z_ours) * pmk, f(z_ref) * pmk)}


def line(tag, c):
    return (f"{tag:<26} s {c['s_masked']['rel_l2']:.6e} (nr {c['s_masked']['norm_ratio']:.4f} "
            f"cos {c['s_masked']['cos']:.4f})   z {c['z_masked']['rel_l2']:.6e} "
            f"(nr {c['z_masked']['norm_ratio']:.4f} cos {c['z_masked']['cos']:.4f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--cap", default=CAP)
    ap.add_argument("--capall", default=CAPALL)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--out", default="perf/of3t_trunkfwd/TRUNK_FORWARD.json")
    ap.add_argument("--skip-composition", action="store_true")
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack

    t0 = time.perf_counter()
    rep = {"what": __doc__.strip().splitlines()[0], "crop": a.crop, "n_blocks": a.blocks,
           "reference": "upstream float64 forward, captured at the pairformer block boundary of "
                        "BUNDLE-MIN own step (perf/of3t_gradients/capture_trunk_boundary.py, "
                        "--no-dropout). A float64 reference, not another approximation."}

    # ---- the SHIPPED configuration, read off the shipped construction site -------------------
    import tt_bio.openfold3_trunk as OT
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

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    _real_pf = OT.Pairformer
    OT.Pairformer = _spy
    try:
        OT.OF3Trunk(sd, ckc)
    except _Stop:
        pass
    finally:
        OT.Pairformer = _real_pf
    if "kwargs" not in spy:
        raise SystemExit("the shipped construction never reached Pairformer -- the spy read "
                         "nothing and a hardcoded default would be the only alternative")
    shipped_kw = dict(spy["kwargs"])
    rep["shipped_config"] = {"source": "tt_bio/openfold3_trunk.py, OF3Trunk.__init__, read by "
                                       "spying on the Pairformer constructor",
                             "n_blocks": spy["n_blocks"], "dims": spy["dims"],
                             "transform_s": spy["transform_s"], "kwargs": shipped_kw}
    # The contradiction this row exists to check, recorded as a fact rather than an argument.
    rep["instrument_a_bundle_disagreement"] = {
        "instrument_a_bundle.py": {"SHIPPED_SPB": True, "SHIPPED_TRI_SPB": False},
        "shipped_now": {"scale_pair_bias": shipped_kw.get("scale_pair_bias"),
                        "tri_att_scale_pair_bias": shipped_kw.get("tri_att_scale_pair_bias")},
        "agree": (shipped_kw.get("scale_pair_bias") is True
                  and shipped_kw.get("tri_att_scale_pair_bias") is False)}
    print(f"[{time.perf_counter()-t0:.0f}s] shipped config {shipped_kw}", flush=True)

    # ---- the boundary ------------------------------------------------------------------------
    cap0 = torch.load(os.path.join(a.cap, "block0_boundary.pt"), map_location="cpu",
                      weights_only=False)
    last = a.blocks - 1
    capL = torch.load(os.path.join(a.cap, f"block{last}_boundary.pt"), map_location="cpu",
                      weights_only=False)
    c = a.crop

    def cs(x):
        return x[:, :c].to(torch.float64).contiguous() if c else x.to(torch.float64)

    def cz(x):
        return x[:, :c, :c].to(torch.float64).contiguous() if c else x.to(torch.float64)

    s_in, z_in = cs(cap0["args"][0]), cz(cap0["args"][1])
    sm = cs(cap0["kwargs"]["single_mask"])
    pm = cz(cap0["kwargs"]["pair_mask"])
    s_ref, z_ref = cs(capL["out"][0]), cz(capL["out"][1])
    N = int(z_in.shape[1])
    rep["probe"] = {"tokens": N, "real_tokens": int(sm.sum()),
                    "s_in_norm": float(s_in.norm()), "z_in_norm": float(z_in.norm()),
                    "s_ref_norm": float(s_ref.norm()), "z_ref_norm": float(z_ref.norm())}
    print(f"[{time.perf_counter()-t0:.0f}s] boundary N={N} real={int(sm.sum())}", flush=True)

    # ---- the module, in the shipped configuration ---------------------------------------------
    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = (flat_all if a.blocks == spy["n_blocks"]
            else {k: v for k, v in flat_all.items() if int(k.split(".")[1]) < a.blocks})

    def build(kw):
        return T.Pairformer(a.blocks, *spy["dims"], spy["transform_s"], flat, ckc, **kw)

    mod = build(shipped_kw)
    s_fp32 = bool(shipped_kw.get("s_fp32_residual", False))
    s_dtype = ttnn.float32 if s_fp32 else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
    rep["masks"] = {"pair_mask_sum": float(pm.sum()), "single_mask_sum": float(sm.sum()),
                    "s_input_dtype": str(s_dtype)}

    def run_untaped(m, s, z):
        so, zo = m(fts(s), ft(z), ft(pm), ft(attn), ft(attn))
        return (ttnn.to_torch(so).to(torch.float64), ttnn.to_torch(zo).to(torch.float64))

    def run_taped(m, s, z):
        sa = ag.Tensor(fts(s), requires_grad=True)
        za = ag.Tensor(ft(z), requires_grad=True)
        with ag.tape():
            so, zo = m(sa, za, ft(pm), ft(attn), ft(attn))
        return (ttnn.to_torch(so.value).to(torch.float64),
                ttnn.to_torch(zo.value).to(torch.float64))

    # ---- ARM 1: SHIPPED ------------------------------------------------------------------------
    s_ship, z_ship = run_untaped(mod, s_in, z_in)
    rep["SHIPPED"] = compare(s_ship, z_ship, s_ref, z_ref, sm, pm)
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("SHIPPED", rep["SHIPPED"]), flush=True)

    # ---- ARM 2: TAPED --------------------------------------------------------------------------
    s_tape, z_tape = run_taped(mod, s_in, z_in)
    rep["TAPED"] = compare(s_tape, z_tape, s_ref, z_ref, sm, pm)
    rep["TAPED_vs_SHIPPED"] = compare(s_tape, z_tape, s_ship, z_ship, sm, pm)
    rep["TAPED_vs_SHIPPED"]["bit_identical"] = {
        "s": bool(torch.equal(s_tape, s_ship)), "z": bool(torch.equal(z_tape, z_ship))}
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("TAPED", rep["TAPED"]), flush=True)
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("TAPED vs SHIPPED",
                                                     rep["TAPED_vs_SHIPPED"]), flush=True)

    # ---- the of3t-pairformer configuration, so the D87 number has a home -----------------------
    pf_kw = dict(shipped_kw)
    pf_kw["scale_pair_bias"] = True
    pf_kw["tri_att_scale_pair_bias"] = False
    mod_pf = build(pf_kw)
    s_pf, z_pf = run_untaped(mod_pf, s_in, z_in)
    rep["PAIRFORMER_ARM_CONFIG"] = {"kwargs": pf_kw,
                                    "why": "what instrument_a_bundle.py calls shipped. Untaped, "
                                           "so it isolates the flag from the tape.",
                                    **compare(s_pf, z_pf, s_ref, z_ref, sm, pm)}
    s_pft, z_pft = run_taped(mod_pf, s_in, z_in)
    rep["PAIRFORMER_ARM_CONFIG_TAPED"] = compare(s_pft, z_pft, s_ref, z_ref, sm, pm)
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("spb=True untaped",
                                                     rep["PAIRFORMER_ARM_CONFIG"]), flush=True)
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("spb=True TAPED",
                                                     rep["PAIRFORMER_ARM_CONFIG_TAPED"]),
          flush=True)
    del mod_pf

    # ---- LOCALISE: the one shipped flag that moves the pair track ------------------------------
    # `of3t-gradients` item (14) found `transpose_bias` moves the per-block pair forward
    # 7.811e-03 -> 3.548e-03 and said it is visible to inference. It is a SHIPPED flag
    # (`openfold3_trunk.py`, `tri_att_end_bias_follows_pair = not is_openbind(sd)`), so if it
    # carries the composed disagreement the finding has a named site rather than a magnitude.
    tb_kw = dict(shipped_kw)
    tb_kw["transpose_bias"] = not bool(shipped_kw.get("transpose_bias", True))
    mod_tb = build(tb_kw)
    s_tb, z_tb = run_untaped(mod_tb, s_in, z_in)
    rep["LEVER_transpose_bias_flipped"] = {
        "kwargs": tb_kw,
        "why": "the shipped configuration with ONE flag moved, untaped. Everything else, "
               "including the checkpoint, the boundary and the masks, is the SHIPPED arm.",
        **compare(s_tb, z_tb, s_ref, z_ref, sm, pm)}
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("transpose_bias flipped",
                                                     rep["LEVER_transpose_bias_flipped"]),
          flush=True)

    # ---- CONTROLS ------------------------------------------------------------------------------
    zero_s, zero_z = torch.zeros_like(s_ref), torch.zeros_like(z_ref)
    rep["CONTROL_zero_model"] = {
        "why": "A16. What a model emitting nothing scores at this boundary, measured.",
        **compare(zero_s, zero_z, s_ref, z_ref, sm, pm)}
    # A break control has to break what the comparison READS. Permuting the INPUT over the real
    # token positions leaves every weight, mask and kernel alone and mispairs the tokens.
    real = torch.nonzero(sm.reshape(-1) > 0).reshape(-1)
    g = torch.Generator().manual_seed(20260920)
    order = real[torch.randperm(int(real.numel()), generator=g)]
    idx = torch.arange(N)
    idx[real] = order
    s_perm = s_in[:, idx].contiguous()
    z_perm = z_in[:, idx][:, :, idx].contiguous()
    s_bk, z_bk = run_untaped(mod, s_perm, z_perm)
    rep["CONTROL_break_permuted_input"] = {
        "why": "the same inputs paired with the wrong token positions; weights, masks, flags and "
               "kernels untouched. If this does not move the headline the comparison is "
               "saturated and reads nothing.",
        "seed": 20260920, "real_positions_permuted": int(real.numel()),
        "fixed_points": int((order == real).sum()),
        **compare(s_bk, z_bk, s_ref, z_ref, sm, pm)}
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("CONTROL zero",
                                                     rep["CONTROL_zero_model"]), flush=True)
    print(f"[{time.perf_counter()-t0:.0f}s] " + line("CONTROL break",
                                                     rep["CONTROL_break_permuted_input"]),
          flush=True)

    # ---- D86: the transpose fingerprint, on the tensor that has an orientation ----------------
    # D86 cost this campaign a published figure: a shape test that cannot fire on a square
    # weight scored 87 tensors against their own transposes. The pair track IS square in its two
    # token axes and the flag under suspicion is literally an orientation, so the fingerprint is
    # checked directly rather than left to a shape test. rel ~ sqrt(2) with cos ~ 0 against the
    # reference, and a SMALL rel against the transposed reference, is what a transposed output
    # looks like.
    pmk = pm.reshape(1, N, N, 1)
    pmk_t = pm.transpose(1, 2).reshape(1, N, N, 1)
    rep["D86_transpose_fingerprint"] = {
        "sqrt2": 2.0 ** 0.5,
        "shipped_z_vs_ref": rep["SHIPPED"]["z_masked"],
        "shipped_z_vs_ref_TRANSPOSED": metrics(z_ship * pmk_t,
                                               z_ref.transpose(1, 2) * pmk_t),
        "tb_flipped_z_vs_ref_TRANSPOSED": metrics(z_tb * pmk_t,
                                                  z_ref.transpose(1, 2) * pmk_t),
        "ref_is_symmetric": metrics(z_ref.transpose(1, 2) * pmk_t, z_ref * pmk),
        "why": "if our pair output were their pair output transposed, rel against the reference "
               "would read near sqrt(2) with cos near 0 and rel against the TRANSPOSED reference "
               "would read near 0. ref_is_symmetric says how much of that test the reference "
               "itself could pass, since a symmetric z makes the check vacuous."}
    print(f"[{time.perf_counter()-t0:.0f}s] D86 z vs ref^T "
          f"{rep['D86_transpose_fingerprint']['shipped_z_vs_ref_TRANSPOSED']['rel_l2']:.6e} "
          f"(ref^T vs ref "
          f"{rep['D86_transpose_fingerprint']['ref_is_symmetric']['rel_l2']:.6e})", flush=True)

    # ---- ARM 3: COMPOSITION --------------------------------------------------------------------
    if not a.skip_composition and os.path.isfile(a.capall):
        allb = torch.load(a.capall, map_location="cpu", weights_only=False)
        if int(allb["crop"]) != c:
            raise SystemExit(f"capall crop {allb['crop']} != --crop {c}")

        def per_block(m, label):
            """Two curves per block, on one input set.

            alone: block k fed THEIR captured input. composed: the same block fed OUR running
            output. Both scored against THEIR block k output, so the gap between the curves IS
            the compounding.
            """
            out = []
            s_run, z_run = s_in, z_in
            for k in range(a.blocks):
                s_cap = allb["s_in"][k].to(torch.float64)
                z_cap = allb["z_in"][k].to(torch.float64)
                s_ref_k = allb["s_out"][k].to(torch.float64)
                z_ref_k = allb["z_out"][k].to(torch.float64)
                blk = m.blocks[k]
                so, zo = blk(fts(s_cap), ft(z_cap), ft(pm), ft(attn), ft(attn))
                alone = compare(ttnn.to_torch(so).to(torch.float64),
                                ttnn.to_torch(zo).to(torch.float64), s_ref_k, z_ref_k, sm, pm)
                so2, zo2 = blk(fts(s_run), ft(z_run), ft(pm), ft(attn), ft(attn))
                s_run = ttnn.to_torch(so2).to(torch.float64)
                z_run = ttnn.to_torch(zo2).to(torch.float64)
                composed = compare(s_run, z_run, s_ref_k, z_ref_k, sm, pm)
                out.append({"block": k,
                            "alone_from_captured_input": {kk: alone[kk] for kk in
                                                          ("s_masked", "z_masked")},
                            "composed_running": {kk: composed[kk] for kk in
                                                 ("s_masked", "z_masked")}})
                if k % 8 == 0 or k == a.blocks - 1:
                    print(f"[{time.perf_counter()-t0:.0f}s] {label} block {k:2d} alone z "
                          f"{alone['z_masked']['rel_l2']:.3e} s "
                          f"{alone['s_masked']['rel_l2']:.3e} | composed z "
                          f"{composed['z_masked']['rel_l2']:.3e} s "
                          f"{composed['s_masked']['rel_l2']:.3e}", flush=True)
            return out, s_run, z_run

        per, s_run, z_run = per_block(mod, "SHIPPED")
        rep["COMPOSITION"] = {
            "per_block": per,
            "capall": {"file": a.capall,
                       "n_blocks": int(allb["n_blocks"]), "crop": int(allb["crop"])},
            "why": "alone_from_captured_input feeds block k THEIR input; composed_running feeds "
                   "it OUR block k-1 output. Both are scored against THEIR block k output."}
        # the per-block loop composed to the end must reproduce the module call, or the
        # per-block figures describe a different composition from the SHIPPED arm
        rep["COMPOSITION"]["loop_vs_module_call"] = compare(s_run, z_run, s_ship, z_ship, sm, pm)
        worst = max(per, key=lambda x: x["alone_from_captured_input"]["z_masked"]["rel_l2"])
        rep["COMPOSITION"]["worst_alone_block"] = worst["block"]
        rep["COMPOSITION"]["worst_alone_z_masked"] = \
            worst["alone_from_captured_input"]["z_masked"]["rel_l2"]
        rep["COMPOSITION"]["median_alone_z_masked"] = sorted(
            x["alone_from_captured_input"]["z_masked"]["rel_l2"] for x in per)[len(per) // 2]
        per_tb, s_tbr, z_tbr = per_block(mod_tb, "tb_flip")
        rep["COMPOSITION_transpose_bias_flipped"] = {
            "per_block": per_tb,
            "why": "the same two curves with the one shipped flag moved, so the per-block and "
                   "the composed effect of the flag are on the same axes.",
            "worst_alone_z_masked": max(
                x["alone_from_captured_input"]["z_masked"]["rel_l2"] for x in per_tb),
            "median_alone_z_masked": sorted(
                x["alone_from_captured_input"]["z_masked"]["rel_l2"] for x in per_tb)[
                len(per_tb) // 2],
            "composed_end": compare(s_tbr, z_tbr, s_ref, z_ref, sm, pm)}
    else:
        rep["COMPOSITION"] = {"ran": False, "why": f"{a.capall} absent"}

    rep["seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
