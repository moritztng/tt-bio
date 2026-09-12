#!/usr/bin/env python3
"""Why the atom-level matmuls sit at 24 % of the byte roof, and whether L1 residency moves them.

`roof_probe` measured this card's roofs under the diffusion stack's own kernel config: 54.675
TFLOP/s and 226.72 GB/s. The token-DiT matmuls land at 35-47 % of both. The 54 atom-level ones do
not: 6.2-8.9 % of the FLOP roof and 22.8-25.7 % of the byte roof, 3.967 ms of the 41.60 ms step.

Their arithmetic intensity is ~63 FLOP/byte against a 241 FLOP/byte crossover, so the byte roof is
the one that binds them and they reach a quarter of it. `AttentionPairBias`'s atom branch forces
its input to DRAM (`to_memory_config(s, DRAM_MEMORY_CONFIG)`) and the operand is only 1.15 MB, so
the obvious question is whether the DRAM round trip is what they are paying.

Arms, interleaved, at the four production shapes:
  dram    A and the output interleaved in DRAM      (what the model does)
  l1_in   A interleaved in L1, output in DRAM
  l1_both A and output both in L1
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_matmul_group"))
from shape_probe import interleaved  # noqa: E402


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
    cg = T.CORE_GRID_MAIN
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch()),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()},
           "roofs_from_roof_probe": {"tflops_hifi4_fp32acc": 54.675, "gbps_rw": 226.72}}
    res = {}
    for (K, W, Din, N, n_prog) in [(140, 32, 128, 128, 24), (140, 32, 128, 256, 18),
                                   (140, 128, 128, 256, 6), (140, 32, 256, 128, 6)]:
        tag = f"[1,{K},{W},{Din}]x[{Din},{N}]"
        At = torch.randn(1, K, W, Din)
        mk = lambda mc: ttnn.from_torch(At, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                        device=dev, memory_config=mc)
        A_d, A_l = mk(ttnn.DRAM_MEMORY_CONFIG), mk(ttnn.L1_MEMORY_CONFIG)
        Bw = ttnn.from_torch(torch.randn(1, 1, Din, N), layout=ttnn.TILE_LAYOUT,
                             dtype=ttnn.bfloat16, device=dev)
        lin = lambda x, mc: ttnn.linear(x, Bw, compute_kernel_config=ckc, core_grid=cg,
                                        memory_config=mc)
        r = interleaved(ttnn, dev, {
            "dram": lambda: lin(A_d, ttnn.DRAM_MEMORY_CONFIG),
            "l1_in": lambda: lin(A_l, ttnn.DRAM_MEMORY_CONFIG),
            "l1_both": lambda: lin(A_l, ttnn.L1_MEMORY_CONFIG),
            "dram_aa": lambda: lin(A_d, ttnn.DRAM_MEMORY_CONFIG)})
        ref = ttnn.to_torch(lin(A_d, ttnn.DRAM_MEMORY_CONFIG))
        r["l1_both_bit_exact"] = bool(torch.equal(ref, ttnn.to_torch(
            lin(A_l, ttnn.L1_MEMORY_CONFIG))))
        r["aa_floor"] = round(r["dram"]["us"] / r["dram_aa"]["us"], 4)
        for k in ("l1_in", "l1_both"):
            r[f"ratio_{k}"] = round(r["dram"]["us"] / r[k]["us"], 4)
        r["n_programs_in_step"] = n_prog
        r["ms_in_step_dram"] = round(n_prog * r["dram"]["us"] / 1e3, 4)
        r["ms_saved_if_l1_both"] = round(
            n_prog * (r["dram"]["us"] - r["l1_both"]["us"]) / 1e3, 4)
        res[tag] = r
        print(tag, json.dumps(r), flush=True)
        for t in (A_d, A_l, Bw):
            ttnn.deallocate(t)
    out["atom_sites"] = res
    out["total_ms_saved_if_l1_both"] = round(
        sum(v["ms_saved_if_l1_both"] for v in res.values()), 4)
    print("TOTAL ms/step saved if L1 both:", out["total_ms_saved_if_l1_both"], flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
