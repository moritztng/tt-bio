#!/usr/bin/env python3
"""A MEASURED roof for this card and this kernel config, and where the step's matmuls sit on it.

The census says the diffusion step's 387 matmul programs cost 18.5366 ms and that 193 of them are
one shape, [1,1,512,768] x [1,1,768,768] at 31.38 us. Whether that is slow depends on a roof, and
the campaign's rule is that a roof is measured, not asserted (`roofline-roof-must-be-measured-not
-asserted`). Nobody has measured one on whglx for the diffusion stack's own compute kernel config
(HiFi4, fp32 dest accumulate, packer L1 accumulate), which is NOT the bf16 LoFi peak that gets
quoted.

Three things here:
  roof_flop   square matmuls from 1k to 6k under the model's own config -> achievable TFLOP/s
  roof_byte   a pure-movement op on a large DRAM tensor -> achievable GB/s
  sites       every distinct step matmul shape, at its own size, placed against both roofs

Plus the one knob question the census raises: 96 of the 193 [768,768] programs are the AdaLN
s_scale / s_bias projections, and those are the only matmuls in the diffusion stack that do NOT
pass core_grid -- `tenstorrent.py` has it commented out with "CAUSES ACCURACY ISSUE", against a
CoreGrid(y=10, x=11) that is a Blackhole grid. Time and score CORE_GRID_MAIN there.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def timed(ttnn, dev, fn, reps=20, blocks=5):
    for _ in range(4):
        fn()
    ttnn.synchronize_device(dev)
    w = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        w.append((time.perf_counter() - t0) / reps * 1e6)
    return round(st.median(w), 3), round(100 * (max(w) - min(w)) / st.median(w), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    lofi = kernel_cls(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                      fp32_dest_acc_en=False, packer_l1_acc=True)
    cg = T.CORE_GRID_MAIN
    tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch()),
                   "grid": str(dev.compute_with_storage_grid_size()),
                   "core_grid_main": f"{cg.y}x{cg.x}",
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()}}

    # ---- FLOP roof -----------------------------------------------------------------
    roof = {}
    for n in (1024, 2048, 3072, 4096, 6144):
        A, Bm = tt(torch.randn(1, 1, n, n)), tt(torch.randn(1, 1, n, n))
        for tag, cfg in (("hifi4_fp32acc", ckc), ("lofi", lofi)):
            us, sp = timed(ttnn, dev, lambda: ttnn.linear(A, Bm, compute_kernel_config=cfg,
                                                          core_grid=cg), reps=5)
            roof[f"{n}_{tag}"] = {"us": us, "spread_pct": sp,
                                  "tflops": round(2 * n ** 3 / (us * 1e-6) / 1e12, 3)}
        ttnn.deallocate(A); ttnn.deallocate(Bm)
    out["roof_flop"] = roof
    best = {k: max((v["tflops"] for k2, v in roof.items() if k2.endswith(k)))
            for k in ("hifi4_fp32acc", "lofi")}
    out["roof_flop_best_tflops"] = best
    print("roof_flop", json.dumps(best), flush=True)

    # ---- byte roof -----------------------------------------------------------------
    big = tt(torch.randn(1, 1, 8192, 8192))          # 128 MiB
    nbytes = 8192 * 8192 * 2
    us, sp = timed(ttnn, dev, lambda: ttnn.clone(big), reps=5)
    out["roof_byte"] = {"clone_us": us, "spread_pct": sp,
                        "gbps_rw": round(2 * nbytes / (us * 1e-6) / 1e9, 2)}
    ttnn.deallocate(big)
    print("roof_byte", json.dumps(out["roof_byte"]), flush=True)

    # ---- the step's own shapes on that roof -----------------------------------------
    SHAPES = [((1, 1, 512, 768), (768, 768), 193), ((1, 1, 512, 768), (768, 1536), 76),
              ((1, 1, 512, 768), (768, 3072), 24), ((1, 140, 128, 128), (128, 256), 6),
              ((1, 140, 32, 128), (128, 256), 18), ((1, 1, 512, 1536), (1536, 768), 26),
              ((1, 140, 32, 128), (128, 128), 24), ((1, 140, 32, 256), (256, 128), 6)]
    sites = []
    for ash, bsh, n_prog in SHAPES:
        A, Bm = tt(torch.randn(*ash)), tt(torch.randn(1, 1, *bsh))
        us, sp = timed(ttnn, dev, lambda: ttnn.linear(A, Bm, compute_kernel_config=ckc,
                                                      core_grid=cg))
        M = 1
        for d in ash[:-1]:
            M *= d
        K, N = bsh
        flop = 2 * M * K * N
        byts = (M * K + K * N + M * N) * 2
        sites.append({"a": list(ash), "b": list(bsh), "n_programs": n_prog, "us": us,
                      "spread_pct": sp,
                      "tflops": round(flop / (us * 1e-6) / 1e12, 3),
                      "pct_of_flop_roof": round(100 * flop / (us * 1e-6) / 1e12
                                                / best["hifi4_fp32acc"], 1),
                      "gbps": round(byts / (us * 1e-6) / 1e9, 2),
                      "pct_of_byte_roof": round(100 * byts / (us * 1e-6) / 1e9
                                                / out["roof_byte"]["gbps_rw"], 1)})
        print("site", json.dumps(sites[-1]), flush=True)
        ttnn.deallocate(A); ttnn.deallocate(Bm)
    out["sites"] = sites

    # ---- the AdaLN core_grid question ------------------------------------------------
    X = tt(torch.randn(1, 1, 512, 768))
    Wt = torch.randn(768, 768)
    Wd = tt(Wt)
    bias = tt(torch.randn(1, 1, 1, 768))
    arms = {}
    for tag, kw in (("no_core_grid", {}), ("core_grid_main", {"core_grid": cg})):
        us, sp = timed(ttnn, dev, lambda: ttnn.linear(X, Wd, bias=bias,
                                                      compute_kernel_config=ckc, **kw))
        arms[tag] = {"us": us, "spread_pct": sp}
    r0 = ttnn.to_torch(ttnn.linear(X, Wd, bias=bias, compute_kernel_config=ckc))
    r1 = ttnn.to_torch(ttnn.linear(X, Wd, bias=bias, compute_kernel_config=ckc, core_grid=cg))
    arms["bit_exact"] = bool(torch.equal(r0, r1))
    arms["max_abs"] = float((r0 - r1).abs().max())
    arms["ratio"] = round(arms["no_core_grid"]["us"] / arms["core_grid_main"]["us"], 4)
    out["adaln_core_grid"] = arms
    print("adaln_core_grid", json.dumps(arms), flush=True)

    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)


main()
