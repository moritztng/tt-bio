#!/usr/bin/env python3
"""Split one trunk tri-attention at N=128 into qkv-proj / SDPA / out-proj.

Tri-attention is the largest single item in a Pairformer block at the 117-aa shape
(2 per block, ~1.87 ms each against an 8.31 ms block). Knowing which third of it is
slow decides whether there is a cheap config lever or only a kernel rewrite.

All timed regions are sync-bracketed on both sides.
"""
import json
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

N = 128
C_Z = 256
H = 8
D = 32


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=5, pipe=12, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return round(med(out), 4)


def make(dev, shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype)


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    ckc_lofi = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False, packer_l1_acc=True,
    )
    torch.manual_seed(0)
    x = make(dev, (1, N, N, C_Z))
    w_qkv = make(dev, (C_Z, 3 * H * D))
    w_o = make(dev, (H * D, C_Z))
    q = make(dev, (N, H, N, D))
    k = make(dev, (N, H, N, D))
    v = make(dev, (N, H, N, D))
    res = {}

    def qkv_proj(cfg=ckc):
        ttnn.deallocate(ttnn.linear(x, w_qkv, compute_kernel_config=cfg,
                                    core_grid=CORE_GRID_MAIN,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))
    res["qkv_proj_hifi4"] = timed(dev, qkv_proj)
    res["qkv_proj_lofi"] = timed(dev, lambda: qkv_proj(ckc_lofi))

    def sdpa():
        ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False))
    res["sdpa_default"] = timed(dev, sdpa)

    for cq, ck_ in ((32, 32), (32, 128), (64, 128), (128, 128)):
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            q_chunk_size=cq, k_chunk_size=ck_, exp_approx_mode=False,
        )

        def sdpa_cfg(p=prog):
            ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, program_config=p))
        try:
            res[f"sdpa_q{cq}_k{ck_}"] = timed(dev, sdpa_cfg)
        except Exception as e:
            res[f"sdpa_q{cq}_k{ck_}"] = f"ERR {type(e).__name__}"

    o = make(dev, (1, N, N, H * D))

    def out_proj():
        ttnn.deallocate(ttnn.linear(o, w_o, compute_kernel_config=ckc,
                                    core_grid=CORE_GRID_MAIN,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))
    res["out_proj"] = timed(dev, out_proj)

    # what a plain big matmul of the same FLOPs costs, as an efficiency reference
    big_a = make(dev, (1, 1, N * N, C_Z))
    big_w = make(dev, (C_Z, 3 * H * D))

    def flat_qkv():
        ttnn.deallocate(ttnn.linear(big_a, big_w, compute_kernel_config=ckc,
                                    core_grid=CORE_GRID_MAIN,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG))
    res["qkv_proj_flat3d"] = timed(dev, flat_qkv)

    gf_qkv = N * N * C_Z * 3 * H * D * 2 / 1e9
    gf_sdpa = N * H * (N * N * D * 2) * 2 / 1e9
    gf_out = N * N * H * D * C_Z * 2 / 1e9
    print(json.dumps(res, indent=2))
    print(f"\nGFLOP: qkv={gf_qkv:.3f} sdpa={gf_sdpa:.3f} out={gf_out:.3f}")
    for k_, gf in (("qkv_proj_hifi4", gf_qkv), ("qkv_proj_flat3d", gf_qkv),
                   ("sdpa_default", gf_sdpa), ("out_proj", gf_out)):
        if isinstance(res.get(k_), float):
            print(f"{k_:20s} {res[k_]:8.4f} ms  -> {gf / (res[k_] / 1e3) / 1e3:7.2f} TFLOP/s"
                  f"  ({gf / (res[k_] / 1e3) / 1e3 / 100.6 * 100:5.1f}% of HiFi4 peak)")


if __name__ == "__main__":
    main()
