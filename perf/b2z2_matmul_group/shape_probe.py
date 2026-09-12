#!/usr/bin/env python3
"""Screen the three shape questions the matmul census raises, off-fold, before any model edit.

W2  54 of the 387 matmul programs are `[1,140,W,128] x [1,1,128,N]` -- 140 windows against ONE
    shared B -- and they run at 4.6-4.7 TFLOP/s where the 2D `[1,1,4480,128] x [128,768]` in the
    same step runs 8.6. Merging the leading dims into M is a tile-aligned reshape. Does it pay,
    is the reshape free, and is it bit-exact?

N   the 193 [512,768]x[768,768] programs come 8 to a token-DiT layer. Stacking two of them along
    N costs one wide matmul plus two slices. The capture prices the matmul step (31.38 -> 49.38 us
    for 2x N) and a slice (10.70 us for 1 MB), which says the trade is dead. Measure both ends.

LN  the same capture shows 100 LayerNorm programs on [1,1,512,768] at 30.21 us on 16 CORES of 72.
    48 of them are `layer_norm(s, weight=s_norm.weight)` for the 48 AdaLNs, all of the same `s`.
    Confirm the per-call cost on this build, since collapsing 47 of them is W1's whole claim.

Every arm is interleaved with its partner inside one process, n>=5 blocks, ratio + A/A floor.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def interleaved(ttnn, dev, arms, reps=20, blocks=7):
    """arms: {name: fn}. One block runs every arm in turn, so drift hits all arms alike."""
    for fn in arms.values():
        for _ in range(4):
            fn()
    ttnn.synchronize_device(dev)
    per = {k: [] for k in arms}
    for _ in range(blocks):
        for name, fn in arms.items():
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            ttnn.synchronize_device(dev)
            per[name].append((time.perf_counter() - t0) / reps * 1e6)
    return {k: {"us": round(st.median(v), 3),
                "spread_pct": round(100 * (max(v) - min(v)) / st.median(v), 2)}
            for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    # The exact config the diffusion stack builds (DiffusionModule.__init__): HiFi4, fp32 dest
    # accumulate, packer L1 accumulate. The capture confirms all 387 matmuls run HiFi4.
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "arch": str(dev.arch()), "grid": str(dev.compute_with_storage_grid_size()),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip()}}
    tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)

    # ---------------- W2: batched vs 2D on the four atom buckets ----------------
    w2 = {}
    for (K, W, Din, N, n_prog) in [(140, 32, 128, 128, 24), (140, 32, 128, 256, 18),
                                   (140, 128, 128, 256, 6), (140, 32, 256, 128, 6)]:
        tag = f"[1,{K},{W},{Din}]x[{Din},{N}]"
        A = tt(torch.randn(1, K, W, Din))
        B = tt(torch.randn(1, 1, Din, N))
        A2 = ttnn.reshape(A, (1, 1, K * W, Din))
        # The model passes core_grid=CORE_GRID_MAIN at every one of these sites, and that is
        # what the capture timed. A bare ttnn.matmul here is a different program.
        cg = T.CORE_GRID_MAIN
        mm = lambda x: ttnn.matmul(x, B, compute_kernel_config=ckc)
        lin = lambda x: ttnn.linear(x, B, compute_kernel_config=ckc, core_grid=cg)
        r = interleaved(ttnn, dev, {
            "batched_cg": lambda: lin(A), "flat_cg": lambda: lin(A2),
            "batched_cg_aa": lambda: lin(A),
            "batched_nocg": lambda: mm(A), "flat_nocg": lambda: mm(A2),
            "reshape": lambda: ttnn.reshape(A, (1, 1, K * W, Din))})
        ref = ttnn.to_torch(lin(A)).reshape(1, 1, K * W, N)
        got = ttnn.to_torch(lin(A2))
        r["bit_exact"] = bool(torch.equal(ref, got))
        r["max_abs"] = float((ref - got).abs().max())
        r["ratio"] = round(r["batched_cg"]["us"] / r["flat_cg"]["us"], 4)
        r["aa_floor"] = round(r["batched_cg"]["us"] / r["batched_cg_aa"]["us"], 4)
        r["ratio_with_reshape"] = round(
            r["batched_cg"]["us"] / (r["flat_cg"]["us"] + r["reshape"]["us"]), 4)
        r["n_programs_in_step"] = n_prog
        w2[tag] = r
        print(tag, json.dumps(r))
        for t in (A, B, A2):
            ttnn.deallocate(t)
    out["w2_batched_vs_flat"] = w2

    # ---------------- N-fusion: 2x[768,768] vs [768,1536]+2 slices ----------------
    X = tt(torch.randn(1, 1, 512, 768))
    W1a, W1b = tt(torch.randn(1, 1, 768, 768)), tt(torch.randn(1, 1, 768, 768))
    W2w = tt(torch.randn(1, 1, 768, 1536))

    def two():
        ttnn.matmul(X, W1a, compute_kernel_config=ckc)
        ttnn.matmul(X, W1b, compute_kernel_config=ckc)

    def wide_split():
        y = ttnn.matmul(X, W2w, compute_kernel_config=ckc)
        y[..., :768]
        y[..., 768:]

    r = interleaved(ttnn, dev, {
        "two_768": two, "wide_1536_plus_2_slices": wide_split,
        "wide_1536_only": lambda: ttnn.matmul(X, W2w, compute_kernel_config=ckc),
        "one_768": lambda: ttnn.matmul(X, W1a, compute_kernel_config=ckc),
        "two_768_aa": two})
    r["ratio_fused_over_two"] = round(r["two_768"]["us"] / r["wide_1536_plus_2_slices"]["us"], 4)
    r["aa_floor"] = round(r["two_768"]["us"] / r["two_768_aa"]["us"], 4)
    r["slice_us_each"] = round((r["wide_1536_plus_2_slices"]["us"] - r["wide_1536_only"]["us"]) / 2, 3)
    out["n_fusion_pair"] = r
    print("n_fusion", json.dumps(r))

    # ---------------- LN: layer_norm of [1,1,512,768], weighted vs not ----------------
    wv = tt(torch.randn(1, 1, 1, 768)).reshape((768,)) if False else ttnn.from_torch(
        torch.randn(768), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    r = interleaved(ttnn, dev, {
        "layer_norm_weighted": lambda: ttnn.layer_norm(X, weight=wv, epsilon=1e-5,
                                                       compute_kernel_config=ckc),
        "layer_norm_plain": lambda: ttnn.layer_norm(X, epsilon=1e-5, compute_kernel_config=ckc),
        "layer_norm_weighted_aa": lambda: ttnn.layer_norm(X, weight=wv, epsilon=1e-5,
                                                          compute_kernel_config=ckc)})
    r["aa_floor"] = round(r["layer_norm_weighted"]["us"] / r["layer_norm_weighted_aa"]["us"], 4)
    out["layer_norm_512x768"] = r
    print("layer_norm", json.dumps(r))

    # ---------------- the fold identity, off-fold, at the production shape -------------
    xt = torch.randn(1, 1, 512, 768)
    wt = torch.randn(768)
    Wt = torch.randn(768, 768)
    Xf, wf = tt(xt), ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    Wf = tt(Wt)
    Wfold = tt(wt[:, None] * Wt)
    unfused = ttnn.to_torch(ttnn.matmul(
        ttnn.layer_norm(Xf, weight=wf, epsilon=1e-5, compute_kernel_config=ckc), Wf,
        compute_kernel_config=ckc))
    folded = ttnn.to_torch(ttnn.matmul(
        ttnn.layer_norm(Xf, epsilon=1e-5, compute_kernel_config=ckc), Wfold,
        compute_kernel_config=ckc))
    # The control that makes the comparison mean something: an fp32 host reference. The
    # question is not "do the two device paths differ" (they must, bf16 rounds) but "is the
    # folded path FURTHER from the truth than the shipped one".
    xn = (xt - xt.mean(-1, keepdim=True)) / (xt.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    ref32 = (xn * wt) @ Wt
    den = ref32.abs().mean().item()
    err = lambda y: float((y.float() - ref32).abs().mean() / den)
    out["fold_identity"] = {
        "bit_exact": bool(torch.equal(unfused, folded)),
        "max_abs": float((unfused - folded).abs().max()),
        "mean_abs": float((unfused - folded).abs().mean()),
        "rel_mean_between_paths": float((unfused - folded).abs().mean() / den),
        "rel_err_vs_fp32_shipped": err(unfused),
        "rel_err_vs_fp32_folded": err(folded),
        "scale_mean_abs": den}
    print("fold_identity", json.dumps(out["fold_identity"]))

    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
