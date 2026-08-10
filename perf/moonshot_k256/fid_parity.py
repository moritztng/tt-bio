#!/usr/bin/env python3
"""Is HiFi2 lossless for bf16 operands, or does it cost mantissa bits?

tt-metal's matrix engine multiplies a fixed number of mantissa bits per pass. HiFi4 spends 4
passes, HiFi2 spends 2, LoFi 1. bfloat16 carries 8 mantissa bits including the implicit one, so
the standing tt-metal guidance (knowledgebase kernels/tt-metal-kernel-api.md) is that bf16
operands need HiFi2 and that HiFi4 buys nothing on them. tt-bio runs HiFi4 everywhere
(tenstorrent.py:3910). nscale_l1out.py measures that choice at 1.23x on the dominant trunk
matmul shape, so whether HiFi2 is lossless is worth a real answer.

The reference is torch fp32 on the SAME bf16 operands the device sees, so operand rounding is
not part of the error. A fidelity level is lossless for bf16 if its error against that reference
is no worse than HiFi4's.
"""
import json
import sys

import torch
import ttnn
from tt_bio.tenstorrent import get_device

MM = ttnn.experimental.minimal_matmul
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

dev = get_device()
res = {"shapes": []}
FIDS = [("HiFi4", ttnn.MathFidelity.HiFi4), ("HiFi3", ttnn.MathFidelity.HiFi3),
        ("HiFi2", ttnn.MathFidelity.HiFi2), ("LoFi", ttnn.MathFidelity.LoFi)]

# M kept modest so the fp32 reference is cheap; the numerics do not depend on M.
SHAPES = [("trimul.in_proj  K=256 N=128", 8192, 256, 128),
          ("trimul.out_proj K=256 N=256", 8192, 256, 256),
          ("triatt.qkv      K=256 N=768", 8192, 256, 768)]

for label, M, K, N in SHAPES:
    torch.manual_seed(0)
    # bf16 operands, then an fp32 reference computed from exactly those bf16 values
    ah = (torch.randn(M, K) * 0.05).bfloat16()
    bh = (torch.randn(K, N) * 0.05).bfloat16()
    ref = (ah.float() @ bh.float())
    at = ttnn.from_torch(ah, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=DRAM)
    bt = ttnn.from_torch(bh, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=DRAM)
    row = {"label": label, "m": M, "k": K, "n": N, "arms": {}}
    print(f"\n== {label}  M={M} ==", flush=True)
    print("  arm                    max|err| vs fp32   rel RMS      PCC          "
          "bit-exact vs HiFi4+acc", flush=True)
    base = None
    for name, fid in FIDS:
        for acc in (True, False):
            c = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=fid, fp32_dest_acc_en=acc, packer_l1_acc=True)
            out = ttnn.to_torch(MM(at, bt, memory_config=L1, dtype=ttnn.bfloat16,
                                   compute_kernel_config=c)).float()
            if base is None:
                base = out.clone()
            err = (out - ref).abs()
            rms = float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            pcc = float(torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1])
            eq = bool(torch.equal(out, base))
            print(f"  {name:6s} fp32acc={str(acc):5s}   {float(err.max()):.6e}   "
                  f"{rms:.3e}   {pcc:.9f}   {eq}", flush=True)
            row["arms"][f"{name}_acc{acc}"] = {
                "max_abs_err_vs_fp32": float(err.max()), "rel_rms": rms, "pcc_vs_fp32": pcc,
                "bit_exact_vs_hifi4_acc": eq}
    # bf16 round-trip floor: the best any device arm could do is the fp32 ref rounded to bf16
    floor = (ref.bfloat16().float() - ref).abs()
    print(f"  [floor] fp32 ref rounded to bf16       {float(floor.max()):.6e}", flush=True)
    row["bf16_rounding_floor_max_abs"] = float(floor.max())
    res["shapes"].append(row)
    ttnn.deallocate(at); ttnn.deallocate(bt)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("\nwrote", sys.argv[1], flush=True)
