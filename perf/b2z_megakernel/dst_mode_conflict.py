#!/usr/bin/env python3
"""Can ONE compute kernel hold both halves of the trimul core?

`fp32_dest_acc_en` is a compile-time property of a compute kernel -- it sets the DST register
format and, with it, how many DST tiles exist. A fused kernel therefore has exactly one DST mode
for every op inside it, and the two halves of the trimul core disagree about which one:

  * the gated channel move runs `GATE_FP32_ACC = False` ON PURPOSE. The repo measured why
    (`perf/trimul_f2/e6_diag.py`): `calculate_sigmoid` branches on the flag, ttnn takes the cheap
    16-bit branch, and a kernel compiled with fp32 DST computes the MORE accurate sigmoid and so
    stops being bit-exact with production on ~10.4 % of elements.
  * the triangle matmul is issued under the model's compute kernel config, HiFi4 with
    `fp32_dest_acc_en=True`, and contracts K = 512.

So the fusion has to give up one of them. This measures the cost of each surrender rather than
asserting it, against a float32 torch reference on the same inputs:

  A. fp32 DST everywhere -> what the sigmoid gate loses (bit-exactness with the shipped path)
  B. 16-bit DST everywhere -> what the K=512 triangle matmul loses (accuracy)

Both arms are run through plain ttnn ops, which is enough: the question is what the DST mode does
to the arithmetic, not what a hand-written kernel's scheduling does to the clock.
"""
import argparse, json
from pathlib import Path
import torch, ttnn
from tt_bio import tenstorrent as TT


def ckc_for(dev, fp32):
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    return kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=fp32, packer_l1_acc=True)


def stats(got, ref):
    d = (got.float() - ref).abs()
    rel = d / ref.abs().clamp_min(1e-6)
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "max_rel": rel.max().item(), "mean_rel": rel.mean().item(),
            "exact_frac": (got.float() == ref).float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = TT.get_device()
    torch.manual_seed(0)
    res = {"arch": str(dev.arch()), "n": a.n, "c": a.c}

    # --- B. the K=512 triangle matmul under both DST modes -------------------------------
    # the trimul's own contraction: [1, C, S, S] @ [1, C, S, S], K = S = 512.
    C, S = 8, a.n          # 8 channels is enough to characterize; the arithmetic per channel
    ta = torch.randn(1, C, S, S, dtype=torch.float32) * 0.1
    tb = torch.randn(1, C, S, S, dtype=torch.float32) * 0.1
    # reference in float64 so neither device arm is being compared against its own error
    ref_mm = (ta.double() @ tb.double()).float()
    A = ttnn.from_torch(ta.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    B = ttnn.from_torch(tb.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    res["matmul_K512"] = {}
    for tag, fp32 in (("fp32_dst", True), ("bf16_dst", False)):
        o = ttnn.matmul(A, B, compute_kernel_config=ckc_for(dev, fp32),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        res["matmul_K512"][tag] = stats(ttnn.to_torch(o), ref_mm)
        ttnn.deallocate(o)
    ttnn.deallocate(A); ttnn.deallocate(B)

    # --- A. sigmoid under both DST modes, against ttnn's shipped sigmoid ------------------
    tg = torch.randn(1, 1, a.n, a.c, dtype=torch.float32) * 3.0
    G = ttnn.from_torch(tg.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref_sig = torch.sigmoid(tg.double()).float()
    arms = {}
    for tag, fp32 in (("fp32_dst", True), ("bf16_dst", False)):
        o = ttnn.sigmoid(G, compute_kernel_config=ckc_for(dev, fp32),
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        arms[tag] = ttnn.to_torch(o)
        ttnn.deallocate(o)
    res["sigmoid"] = {t: stats(v, ref_sig) for t, v in arms.items()}
    # the number that actually matters: do the two DST modes AGREE with each other?
    diff = (arms["fp32_dst"].float() != arms["bf16_dst"].float())
    res["sigmoid"]["modes_disagree_frac"] = diff.float().mean().item()
    res["sigmoid"]["modes_max_abs_diff"] = (
        arms["fp32_dst"].float() - arms["bf16_dst"].float()).abs().max().item()
    ttnn.deallocate(G)

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    m = res["matmul_K512"]
    s = res["sigmoid"]
    print(f"\nK=512 triangle matmul vs float64 ref:")
    print(f"  fp32 DST  max_abs {m['fp32_dst']['max_abs']:.6g}  mean_rel {m['fp32_dst']['mean_rel']:.6g}")
    print(f"  bf16 DST  max_abs {m['bf16_dst']['max_abs']:.6g}  mean_rel {m['bf16_dst']['mean_rel']:.6g}")
    print(f"  surrendering fp32 DST costs "
          f"{m['bf16_dst']['mean_rel'] / max(m['fp32_dst']['mean_rel'], 1e-12):.2f}x the mean rel error")
    print(f"\nsigmoid: the two DST modes disagree on {s['modes_disagree_frac'] * 100:.2f} % of "
          f"elements, max abs diff {s['modes_max_abs_diff']:.6g}")


if __name__ == "__main__":
    main()
