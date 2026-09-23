#!/usr/bin/env python3
"""of3t-lnreduce steps 2b and 3: price the affine reduction at ONE op, and the K ladder.

Input is `capln.py`'s capture: the real `x` and `g` the shipped backward consumed at
`transition_z.norm_weight` (upstream's `pair_stack.pair_transition.layer_norm`, 32.41 % of the
frame-matched trunk's error mass at 4.5815x upstream's own bf16 level), in the dtype the card
held them in, so these are the exact values and not a re-rounding.

FIVE READINGS of dW and dB, all against the same float64 reference:

  ours            the shipped device path replayed op for op: mean, centered, var, rstd, norm,
                  prod = g * norm, then `ttnn.sum(dim=0, precise_config())`
  ours_fp32in     the same with the reduction operand typecast to fp32, which is the only way
                  to get an fp32 OUTPUT out of this build's `ttnn.sum` -- it takes no `dtype=`
  torch_bf16      `F.layer_norm`'s own backward with x, g, gamma in bfloat16: upstream's level
  host_f64_prod   the DEVICE's own `prod` tensor, summed in float64 ON THE HOST. Same numbers
                  the card reduced, a different accumulator, so the difference from `ours` is
                  the accumulator and nothing else
  host_f64_all    xhat recomputed in float64 from the same x, summed in float64: the reduction
                  AND its summands exact

The reference is the float64 backward of the same operands, and it is validated by float64
CENTRAL FINITE DIFFERENCES: dW and dB are linear in W and B, so the difference quotient there
checks the assembly (the masking, xhat, the reduction) rather than curvature, and the
non-trivial check is dx -- the analytic activation gradient against the central difference of
the float64 loss in x. Both are reported.

THE K LADDER masks the cotangent to K real positions at FIXED padded width, so every rung
reduces over the same 147,456 accumulation slots and differs only in how many carry signal.
The pads carry exact zero, which is what the model's own cotangent carries there.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-rows", type=int, default=56)
    ap.add_argument("--site", type=int, default=-1,
                    help="index into the capture's sites; -1 picks the chunk carrying the real rows")
    ap.add_argument("--eps", type=float, default=1e-5)
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    torch.set_num_threads(8)
    dev = get_device()
    cap = torch.load(a.cap, map_location="cpu", weights_only=False)
    sites = cap["sites"]

    def card(t, dtype):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def rel(got, ref):
        n = torch.linalg.vector_norm(ref).item()
        return (torch.linalg.vector_norm(got - ref).item() / n) if n > 0 else float("nan")

    def cos(got, ref):
        a_ = torch.linalg.vector_norm(got).item(); b_ = torch.linalg.vector_norm(ref).item()
        return (torch.dot(got, ref).item() / (a_ * b_)) if a_ > 0 and b_ > 0 else float("nan")

    # ---- the shipped device backward for the affine pair, replayed op for op -------------
    def ours(x_b, g_b, fp32_in=False):
        xv = card(x_b, ttnn.bfloat16)
        gv = card(g_b, ttnn.bfloat16)
        bwcfg = ag.precise_config()
        mean = ttnn.mean(xv, dim=-1, keepdim=True)
        centered = ttnn.subtract(xv, mean)
        var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                        compute_kernel_config=bwcfg)
        rstd = ttnn.rsqrt(ttnn.add(var, a.eps))
        norm = ttnn.multiply(centered, rstd)
        prod = ttnn.multiply(gv, norm)
        prod_host = ttnn.to_torch(prod).to(torch.float64)      # what the card handed the sum
        if fp32_in:
            flat = ag._flat2d(ttnn.typecast(prod, ttnn.float32))
            dW = ttnn.sum(flat, dim=0, keepdim=True, compute_kernel_config=ag.precise_config())
            flatb = ag._flat2d(ttnn.typecast(gv, ttnn.float32))
            dB = ttnn.sum(flatb, dim=0, keepdim=True, compute_kernel_config=ag.precise_config())
        else:
            dW = ag._sum_leading(prod, [prod.shape[-1]])
            dB = ag._sum_leading(gv, [gv.shape[-1]])
        r = (ttnn.to_torch(dW).to(torch.float64).reshape(-1),
             ttnn.to_torch(dB).to(torch.float64).reshape(-1),
             prod_host, str(dW.dtype),
             ttnn.to_torch(norm).to(torch.float64))
        for t in (xv, gv, prod):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        return r

    def torch_ln(x_b, g_b, gamma, beta, param_dtype):
        """`F.layer_norm`'s own backward. bf16 ACTIVATIONS either way; `param_dtype` is what
        upstream keeps its parameters in. OpenFold3 trains bf16-mixed with fp32 parameters, so
        the fp32-parameter reading is upstream's own level and the bf16-parameter one is the
        floor a fully-bf16 step would sit at. They are reported separately because they differ
        by an order of magnitude and a single `torch-bf16` headline would hide which one the
        bar is built from."""
        xb = x_b.to(torch.bfloat16).to(param_dtype).clone().requires_grad_(True)
        W = gamma.to(param_dtype).clone().requires_grad_(True)
        B = beta.to(param_dtype).clone().requires_grad_(True)
        out = torch.nn.functional.layer_norm(xb, (x_b.shape[-1],), W, B, a.eps)
        out.backward(g_b.to(torch.bfloat16).to(param_dtype))
        return (W.grad.to(torch.float64).reshape(-1), B.grad.to(torch.float64).reshape(-1),
                xb.grad.to(torch.float64))

    def ref_f64(x64, g64):
        mean = x64.mean(dim=-1, keepdim=True)
        cen = x64 - mean
        var = (cen * cen).mean(dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + a.eps)
        xhat = cen * rstd
        flat_shape = (-1, x64.shape[-1])
        dW = (g64 * xhat).reshape(flat_shape).sum(dim=0)
        dB = g64.reshape(flat_shape).sum(dim=0)
        return dW, dB, xhat, rstd

    res = {"what": "the affine reduction priced at one real op, and the pre-registered K ladder",
           "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "capture": a.cap, "eps": a.eps, "real_rows": a.real_rows,
           "sites_in_capture": [s["gamma_path"] for s in sites],
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # The pair-track Transition is CHUNKED: the capture holds one call per row block, and
    # only the block covering rows 0..R-1 carries real signal. Pick it by measurement, and
    # report the scan so the choice is not an assertion.
    scan = []
    for i, st in enumerate(sites):
        gg = st["g"].to(torch.float32)
        real = _mask(gg, a.real_rows, gg.dim() == 4)
        scan.append({"i": i, "gamma_path": st["gamma_path"], "g_shape": list(gg.shape),
                     "g_l2_real": torch.linalg.vector_norm(real).item(),
                     "g_l2_total": torch.linalg.vector_norm(gg).item()})
    res["chunk_scan"] = scan
    if a.site < 0:
        a.site = max(range(len(sites)), key=lambda i: scan[i]["g_l2_real"])
    res["site_chosen"] = a.site
    print("chunk scan: " + json.dumps(scan), flush=True)

    s = sites[a.site]
    x_b = s["x"].to(torch.float32)
    g_b = s["g"].to(torch.float32)
    gamma = s["gamma"].reshape(-1)
    beta = (s["beta"].reshape(-1) if s.get("beta") is not None else torch.zeros_like(gamma))
    C = int(x_b.shape[-1])
    res["site"] = {"gamma_path": s["gamma_path"], "x_shape": list(x_b.shape),
                   "g_shape": list(g_b.shape), "x_dtype_on_card": s["x_dtype"],
                   "g_dtype_on_card": s["g_dtype"], "channel": C,
                   "padded_positions": int(x_b.numel() // C)}

    # the device's own dW at this site, from the run that produced the capture
    if s.get("gamma_after") is not None:
        before = (s["gamma_before"] if s.get("gamma_before") is not None
                  else torch.zeros_like(s["gamma_after"]))
        res["site"]["dW_from_the_arm"] = (s["gamma_after"] - before).reshape(-1).tolist()[:4]

    # ---- FD validation of the float64 reference -----------------------------------------
    # Run it on a small real corner so the difference quotients are cheap and exact-ish.
    n = min(24, a.real_rows)
    xs = x_b[..., :n, :n, :].to(torch.float64) if x_b.dim() == 4 else \
        x_b[..., :n, :].to(torch.float64)
    gs = g_b[..., :n, :n, :].to(torch.float64) if g_b.dim() == 4 else \
        g_b[..., :n, :].to(torch.float64)
    dW_s, dB_s, xhat_s, rstd_s = ref_f64(xs, gs)
    g64 = gamma.to(torch.float64)

    def loss(W, B, X):
        mean = X.mean(dim=-1, keepdim=True)
        cen = X - mean
        var = (cen * cen).mean(dim=-1, keepdim=True)
        rs = 1.0 / torch.sqrt(var + a.eps)
        return (gs * (cen * rs * W + B)).sum()

    W0 = g64.clone(); B0 = beta.to(torch.float64).clone()
    fd_W, fd_B = [], []
    for c in range(0, C, max(1, C // 8)):
        h = 1e-6 * max(1.0, abs(W0[c].item()))
        Wp = W0.clone(); Wp[c] += h
        Wm = W0.clone(); Wm[c] -= h
        fd_W.append(((loss(Wp, B0, xs) - loss(Wm, B0, xs)) / (2 * h)).item())
        Bp = B0.clone(); Bp[c] += h
        Bm = B0.clone(); Bm[c] -= h
        fd_B.append(((loss(W0, Bp, xs) - loss(W0, Bm, xs)) / (2 * h)).item())
    idx = list(range(0, C, max(1, C // 8)))
    fdW = torch.tensor(fd_W, dtype=torch.float64)
    fdB = torch.tensor(fd_B, dtype=torch.float64)
    # the non-trivial half: the analytic dx against the central difference in x
    dnorm = gs * g64
    dn_mean = dnorm.mean(dim=-1, keepdim=True)
    dn_norm_mean = (dnorm * xhat_s).mean(dim=-1, keepdim=True)
    dx_an = (dnorm - dn_mean - xhat_s * dn_norm_mean) * rstd_s
    fd_x, an_x, where = [], [], []
    flatx = xs.reshape(-1, C)
    step = max(1, flatx.shape[0] // 12)
    for p in range(0, flatx.shape[0], step):
        for c in (0, C // 2, C - 1):
            h = 1e-5 * max(1.0, abs(flatx[p, c].item()))
            Xp = xs.clone().reshape(-1, C); Xp[p, c] += h
            Xm = xs.clone().reshape(-1, C); Xm[p, c] -= h
            fd_x.append(((loss(W0, B0, Xp.reshape(xs.shape))
                          - loss(W0, B0, Xm.reshape(xs.shape))) / (2 * h)).item())
            an_x.append(dx_an.reshape(-1, C)[p, c].item())
            where.append([p, c])
    fdx = torch.tensor(fd_x, dtype=torch.float64)
    anx = torch.tensor(an_x, dtype=torch.float64)
    res["fd_validation"] = {
        "corner": "%d x %d real positions of the same site" % (n, n),
        "channels_probed": idx,
        "dW_rel_l2_analytic_vs_central_fd": rel(dW_s[idx], fdW),
        "dB_rel_l2_analytic_vs_central_fd": rel(dB_s[idx], fdB),
        "dx_samples": len(fd_x),
        "dx_rel_l2_analytic_vs_central_fd": rel(anx, fdx),
        "note": "dW and dB are LINEAR in W and B, so their difference quotient validates the "
                "assembly -- the masking, xhat and the reduction -- rather than curvature. dx "
                "is not linear in x and is the check that can fail.",
    }

    # ---- the K ladder --------------------------------------------------------------------
    R = a.real_rows
    pair = (x_b.dim() == 4)
    rungs = []
    KS = [1, 2, 4, 8, 16, 28, 56] if pair else [1, 2, 4, 8, 16, 28, 56]
    for side in KS:
        gm = torch.zeros_like(g_b)
        if pair:
            gm[..., :side, :side, :] = g_b[..., :side, :side, :]
            K = side * side
        else:
            gm[..., :side, :] = g_b[..., :side, :]
            K = side
        x64 = x_b.to(torch.float64)
        g64m = gm.to(torch.float64)
        dW64, dB64, xhat64, _ = ref_f64(x64, g64m)
        o_dW, o_dB, prod_host, out_dtype, norm_dev = ours(x_b, gm)
        f_dW, f_dB, _, _, _ = ours(x_b, gm, fp32_in=True)
        t_dW, t_dB, _ = torch_ln(x_b, gm, gamma, beta, torch.bfloat16)
        p_dW, p_dB, _ = torch_ln(x_b, gm, gamma, beta, torch.float32)
        hp_dW = prod_host.reshape(-1, C).sum(dim=0)
        hp_dB = g64m.reshape(-1, C).sum(dim=0)
        row = {"K_real_positions": K, "side": side,
               "padded_positions": int(x_b.numel() // C),
               "ref_l2_dW": torch.linalg.vector_norm(dW64).item(),
               "cancellation_dW": ((g64m * xhat64).abs().reshape(-1, C).sum(dim=0).norm().item()
                                   / torch.linalg.vector_norm(dW64).item()),
               "out_dtype": out_dtype,
               "dW": {"ours": rel(o_dW, dW64), "ours_fp32in": rel(f_dW, dW64),
                      "torch_bf16": rel(t_dW, dW64),
                      "torch_bf16act_fp32param": rel(p_dW, dW64),
                      "host_f64_prod": rel(hp_dW, dW64)},
               "dB": {"ours": rel(o_dB, dB64), "ours_fp32in": rel(f_dB, dB64),
                      "torch_bf16": rel(t_dB, dB64),
                      "torch_bf16act_fp32param": rel(p_dB, dB64),
                      "host_f64_prod": rel(hp_dB, dB64)},
               "cos_dW_ours": cos(o_dW, dW64), "cos_dW_torch": cos(t_dW, dW64)}
        for k in ("dW", "dB"):
            tb = row[k]["torch_bf16"]
            up = row[k]["torch_bf16act_fp32param"]
            row[k]["ratio_ours_over_torch_bf16"] = (row[k]["ours"] / tb) if tb > 0 else None
            row[k]["ratio_ours_over_upstream_config"] = (row[k]["ours"] / up) if up > 0 else None
            row[k]["ratio_hostf64prod_over_torch_bf16"] = (
                (row[k]["host_f64_prod"] / tb) if tb > 0 else None)
        rungs.append(row)
        print("K=%7d canc=%8.2f ours=%.4e t_bf16=%.4e t_upstream=%.4e r_bf16=%.4f "
              "r_upstream=%.4f fp32in=%.4e hostf64=%.4e"
              % (K, row["cancellation_dW"], row["dW"]["ours"], row["dW"]["torch_bf16"],
                 row["dW"]["torch_bf16act_fp32param"],
                 row["dW"]["ratio_ours_over_torch_bf16"] or float("nan"),
                 row["dW"]["ratio_ours_over_upstream_config"] or float("nan"),
                 row["dW"]["ours_fp32in"], row["dW"]["host_f64_prod"]), flush=True)
    res["ladder"] = rungs

    # ---- the full site, unmasked real set, and the A/A floor ------------------------------
    full = rungs[-1]
    res["op_full_real_set"] = full
    gm_full = _mask(g_b, R, pair)
    dW64f, dB64f, _, _ = ref_f64(x_b.to(torch.float64), gm_full.to(torch.float64))
    oW, oB, _, _, _ = ours(x_b, gm_full)
    tW, tB, _ = torch_ln(x_b, gm_full, gamma, beta, torch.bfloat16)
    uW, uB, _ = torch_ln(x_b, gm_full, gamma, beta, torch.float32)
    wc = int(torch.argmax((oW - dW64f).abs() / dW64f.abs().clamp_min(1e-300)).item())
    res["worst_case"] = {
        "tensor": s["gamma_path"] + " (upstream pairformer_stack.blocks.47.pair_stack."
                                    "pair_transition.layer_norm.weight)",
        "worst_channel": wc,
        "worst_channel_rel_error_ours": abs((oW[wc] - dW64f[wc]).item()
                                            / dW64f[wc].item()) if dW64f[wc] != 0 else None,
        "worst_channel_rel_error_torch_bf16": abs((tW[wc] - dW64f[wc]).item()
                                                  / dW64f[wc].item()) if dW64f[wc] != 0 else None,
        "worst_channel_abs_error_ours": abs((oW[wc] - dW64f[wc]).item()),
        "max_abs_error_ours": (oW - dW64f).abs().max().item(),
        "max_abs_error_torch_bf16": (tW - dW64f).abs().max().item(),
        "max_abs_error_upstream_config": (uW - dW64f).abs().max().item(),
    }
    o1 = ours(x_b, _mask(g_b, R, pair))
    o2 = ours(x_b, _mask(g_b, R, pair))
    res["aa_floor"] = {
        "what": "the same device reduction on the same operands, twice, in one process",
        "dW_bit_identical": bool(torch.equal(o1[0], o2[0])),
        "dB_bit_identical": bool(torch.equal(o1[1], o2[1])),
        "dW_rel_difference": rel(o1[0], o2[0]), "dB_rel_difference": rel(o1[1], o2[1])}

    # ---- the masking control: the same statistic unmasked is pad junk ---------------------
    x64 = x_b.to(torch.float64)
    dW_all, dB_all, _, _ = ref_f64(x64, g_b.to(torch.float64))
    dW_real, dB_real, _, _ = ref_f64(x64, _mask(g_b, R, pair).to(torch.float64))
    res["mask_control"] = {
        "dW_rel_l2_unmasked_against_masked": rel(dW_all, dW_real),
        "g_l2_on_pad_positions": torch.linalg.vector_norm(
            (g_b - _mask(g_b, R, pair)).to(torch.float64)).item(),
        "g_l2_on_real_positions": torch.linalg.vector_norm(
            _mask(g_b, R, pair).to(torch.float64)).item(),
        "note": "the model's own cotangent is exactly zero on the pads; a nonzero left column "
                "here would mean the pads participate and every rung above would be pad junk."}

    # ---- what an EXACT reduction does to a wrong cotangent -------------------------------
    # dW is LINEAR in g, so a cotangent error maps to a dW error through the same reduction.
    # of3t-apbleaf measured a cotangent residue of 1.1745x becoming a gradient residue of
    # 2.4044x with 99.72 % of the dW damage lying ACROSS the reference cotangent. If that
    # amplification is a property of the EXACT reduction, it needs no accumulator defect to
    # explain it, and this measures the two amplification factors in float64 on the real
    # operands: along the cotangent, and across it.
    x64f = x_b.to(torch.float64)
    _, _, xhat64f, _ = ref_f64(x64f, gm_full.to(torch.float64))
    gr = gm_full.to(torch.float64)

    def RW(gg):
        return (gg * xhat64f).reshape(-1, C).sum(dim=0)

    def RB(gg):
        return gg.reshape(-1, C).sum(dim=0)

    nW = torch.linalg.vector_norm(RW(gr)).item()
    nB = torch.linalg.vector_norm(RB(gr)).item()
    ng = torch.linalg.vector_norm(gr).item()
    amp = {"along": {}, "across": {}}
    amp["along"]["dW"] = (torch.linalg.vector_norm(RW(gr)).item() / nW) / 1.0
    amp["along"]["dB"] = (torch.linalg.vector_norm(RB(gr)).item() / nB) / 1.0
    aw, ab = [], []
    for seed in range(8):
        gen = torch.Generator().manual_seed(4242 + seed)
        d = torch.randn(gr.shape, generator=gen, dtype=torch.float64)
        d = _mask(d, R, pair)
        d = d - gr * (torch.sum(d * gr) / torch.sum(gr * gr))      # across the cotangent
        d = d * (ng / torch.linalg.vector_norm(d).item())          # same norm as g
        aw.append(torch.linalg.vector_norm(RW(d)).item() / nW)
        ab.append(torch.linalg.vector_norm(RB(d)).item() / nB)
    amp["across"]["dW_draws"] = aw
    amp["across"]["dB_draws"] = ab
    amp["across"]["dW"] = sum(aw) / len(aw)
    amp["across"]["dB"] = sum(ab) / len(ab)
    amp["across"]["dW_min"], amp["across"]["dW_max"] = min(aw), max(aw)
    amp["note"] = ("amplification = (relative dW error) / (relative cotangent error), computed "
                   "in float64 on the real operands, so no accumulator is involved. ALONG the "
                   "cotangent it is 1 by construction: a scale error passes a linear reduction "
                   "at its own size. ACROSS it, the reduction's own cancellation multiplies it.")
    amp["apbleaf_check"] = {
        "cotangent_residue_measured_there": 1.1745,
        "gradient_residue_measured_there": 2.4044,
        "implied_amplification": 2.4044 / 1.1745,
        "predicted_here_across": amp["across"]["dW"],
        "seed_floor_across": {"min": min(aw), "max": max(aw), "draws": len(aw)},
    }
    res["amplification"] = amp
    print("amplification across=%.4f (min %.4f max %.4f over %d draws), apbleaf implies %.4f"
          % (amp["across"]["dW"], min(aw), max(aw), len(aw), 2.4044 / 1.1745), flush=True)

    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote " + a.out)
    return 0


def _mask(g, R, pair):
    import torch
    gm = torch.zeros_like(g)
    if pair:
        gm[..., :R, :R, :] = g[..., :R, :R, :]
    else:
        gm[..., :R, :] = g[..., :R, :]
    return gm


if __name__ == "__main__":
    raise SystemExit(main())
