#!/usr/bin/env python3
"""Where does the fused gate disagree with `multiply_(p, g, [SIGMOID])` -- the sigmoid or the multiply?

The fused kernel is 1-ulp-class off, not wrong, so the sweep over math fidelity cannot find it. This
splits the two-op reference into its parts and compares each against the kernel's own product,
recovered by un-permuting the kernel's output (the move is a pure index reordering, so the
un-permute is exact and the comparison is at the arithmetic, not at the layout).

References, all on device unless marked host:
  R0  ttnn.multiply_(p, g, [SIGMOID])                 the thing production runs
  R1  ttnn.multiply(p, ttnn.sigmoid(g))               same two steps, unfused
  R2  host fp32: p * sigmoid(g), rounded to bf16      the accurate answer
  R3  host: p * bf16(sigmoid(g))                      accurate sigmoid, rounded like the kernel is
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import get_device


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return {"equal": bool(torch.equal(a, b)), "max_abs": float(d.max()),
            "frac_diff": round(float((d > 0).float().mean()), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--pone", action="store_true", help="force the value slice to 1.0, isolating the sigmoid")
    ap.add_argument("--acc", default="", help="override GATE_FP32_ACC: 0 or 1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = get_device()
    torch.manual_seed(0)
    N, C = a.n, a.c
    h = torch.randn(1, N, N, 4 * C, dtype=torch.bfloat16)
    if a.pone:
        # 1.0 is exact in bf16 and x*1.0 is exact in every rounding mode, so the product carries
        # the sigmoid unchanged and the comparison is of the SFPU op alone.
        h[..., 2 * C:3 * C] = 1.0
    if a.acc:
        RB.GATE_FP32_ACC = a.acc == "1"
    xw = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    g_h = h[..., 0:C]
    p_h = h[..., 2 * C:3 * C]

    g_a, g_b, p_a, p_b = ttnn.chunk(xw, chunks=4, dim=-1)
    r0 = ttnn.to_torch(ttnn.multiply_(
        p_a, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
    g_a2, g_b2, p_a2, p_b2 = ttnn.chunk(xw, chunks=4, dim=-1)
    sig_dev = ttnn.sigmoid(g_a2)
    r1 = ttnn.to_torch(ttnn.multiply(p_a2, sig_dev))
    sig_dev_h = ttnn.to_torch(sig_dev)

    r2 = (p_h.float() * torch.sigmoid(g_h.float())).bfloat16()
    sig_bf = torch.sigmoid(g_h.float()).bfloat16()
    r3 = (p_h.float() * sig_bf.float()).bfloat16()

    got = ttnn.to_torch(RB.reblock_permute_gated(xw, 2 * C, 0, C, ttnn.DRAM_MEMORY_CONFIG))
    prod = got.permute(0, 2, 3, 1).contiguous()

    # The sigmoid is settled by --pone; this is the multiply on its own. `m_exact` is the correctly
    # rounded bf16 product of the two operands the DEVICE actually multiplied, so whichever of the
    # kernel and ttnn matches it is the one rounding, and the other is doing something else.
    m_exact = (p_h.float() * sig_dev_h.float()).bfloat16()

    res = {"N": N, "C": C,
           "fidelity": str(RB.GATE_FIDELITY), "fp32_dest_acc": RB.GATE_FP32_ACC,
           "kernel_vs_R0_production": cmp(prod, r0),
           "kernel_vs_R1_unfused": cmp(prod, r1),
           "kernel_vs_R2_host_fp32": cmp(prod, r2),
           "kernel_vs_R3_host_bf16_sigmoid": cmp(prod, r3),
           "R0_vs_R1": cmp(r0, r1),
           "R0_vs_R2": cmp(r0, r2),
           "R0_vs_R3": cmp(r0, r3),
           "devsigmoid_vs_host_bf16_sigmoid": cmp(sig_dev_h, sig_bf),
           "kernel_vs_m_exact": cmp(prod, m_exact),
           "R0_vs_m_exact": cmp(r0, m_exact)}
    a.out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"  {k:34s} {v}", flush=True)


if __name__ == "__main__":
    main()
