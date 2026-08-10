#!/usr/bin/env python3
"""The compute roof the trunk can actually aim at, measured at the trunk's own shapes.

A square 4096 matmul is not a roof any op in this model can reach: the trunk's contractions
are K=256 or K=1024, never 4096. This measures every matmul shape class that carries >=1% of
the 298 aa fold's arithmetic, at the padded shape it really runs (298 -> 320), trying every
backend the codebase has, and keeps the best. The FLOP-weighted result is the realistic
compute roof for this model on this card.
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=3, pipe=4, reps=5):
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
    return st.median(out)


dev = get_device()
g = dev.compute_with_storage_grid_size()
CORES = g.x * g.y
print(f"grid {g.x}x{g.y} = {CORES} cores", flush=True)
KC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)
MM = getattr(ttnn.experimental, "minimal_matmul", None)
print("minimal_matmul:", MM is not None, flush=True)

# (label, a_shape, b_shape, TFLOP carried by this class over one 298 aa fold)
# TFLOP figures come from perf/ceiling/census_fold_p298.json (logical, at 298); they are
# rescaled to the executed padded shape inside the loop.
CLASSES = [
    ("trimul.in_proj      [1,320,320,256]@[256,128]", (1, 320, 320, 256), (256, 128), 48.79),
    ("transition.up       [1,30,320,256]@[256,1024]", (1, 30, 320, 256), (256, 1024), 44.21),
    ("triatt.qkv          [320,320,256]@[256,768]", (320, 320, 256), (256, 768), 36.60),
    ("trimul.out_proj     [1,320,320,256]@[256,256]", (1, 320, 320, 256), (256, 256), 24.40),
    ("transition.down     [1,30,320,1024]@[1024,256]", (1, 30, 320, 1024), (1024, 256), 22.10),
    ("triatt.out          [320,320,256]@[256,256]", (320, 320, 256), (256, 256), 24.40),
    ("trimul.tri_matmul   [1,32,320,320]@[1,32,320,320]", (1, 32, 320, 320), (1, 32, 320, 320), 14.74),
    ("attnpairbias.qkv    [1,320,768]@[768,3072]", (1, 320, 768), (768, 3072), 6.75),
]


def flops_of(ash, bsh):
    if len(bsh) == 2:
        k, n = bsh
        m = 1
        for d in ash[:-1]:
            m *= d
        assert ash[-1] == k, (ash, bsh)
        return 2 * m * k * n
    # batched a @ b^T-free case: [.., M, K] @ [.., K, N]
    m, k = ash[-2], ash[-1]
    n = bsh[-1]
    batch = 1
    for d in ash[:-2]:
        batch *= d
    return 2 * batch * m * k * n


res = {"grid": [g.x, g.y], "cores": CORES, "classes": {}}
torch.manual_seed(0)
for label, ash, bsh, tflop_fold in CLASSES:
    a = ttnn.from_torch(torch.randn(*ash) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(*bsh) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gf = flops_of(ash, bsh) / 1e9
    variants = {}

    def add(name, fn):
        try:
            ms = timed(dev, fn)
        except Exception as e:
            print(f"    {name:26s} ERR {str(e)[:70]}", flush=True)
            return
        variants[name] = round(gf / (ms / 1e3) / 1e3, 2)
        print(f"    {name:26s} {ms:8.4f} ms  {gf/(ms/1e3)/1e3:7.2f} TFLOP/s", flush=True)

    print(f"  {label}   ({gf:.1f} GFLOP/call)", flush=True)
    add("matmul->DRAM", lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=DRAM)))
    add("matmul->L1", lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=L1)))
    add(
        "matmul+grid->DRAM",
        lambda: ttnn.deallocate(
            ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=DRAM, core_grid=ttnn.CoreGrid(x=g.x, y=g.y))
        ),
    )
    if MM is not None and len(bsh) == 2:
        add("minimal_matmul->DRAM", lambda: ttnn.deallocate(MM(a, b, memory_config=DRAM)))
        add("minimal_matmul->L1", lambda: ttnn.deallocate(MM(a, b, memory_config=L1)))
    best = max(variants.items(), key=lambda kv: kv[1]) if variants else ("none", 0.0)
    res["classes"][label] = {
        "gflop_per_call": round(gf, 2),
        "tflop_per_fold_logical": tflop_fold,
        "variants": variants,
        "best": best[0],
        "best_tflops": best[1],
    }
    print(f"    -> best {best[0]} {best[1]} TFLOP/s", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

# SDPA at the tri-attention shape
try:
    q = ttnn.from_torch(torch.randn(320, 8, 320, 32) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    k = ttnn.from_torch(torch.randn(320, 8, 320, 32) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    v = ttnn.from_torch(torch.randn(320, 8, 320, 32) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(torch.randn(1, 8, 320, 320) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    sdpa_gf = 4 * 320 * 8 * 320 * 320 * 32 / 1e9
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=g, q_chunk_size=320, k_chunk_size=320, exp_approx_mode=False
    )
    ms = timed(
        dev,
        lambda: ttnn.deallocate(
            ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, program_config=prog, compute_kernel_config=KC
            )
        ),
    )
    res["sdpa_320_8_320_32"] = {"ms": round(ms, 4), "gflop": round(sdpa_gf, 2), "tflops": round(sdpa_gf / (ms / 1e3) / 1e3, 2)}
    print(f"  SDPA [320,8,320,32] q=k=320  {ms:.4f} ms  {sdpa_gf/(ms/1e3)/1e3:.2f} TFLOP/s", flush=True)
except Exception as e:
    print("  SDPA ERR", str(e)[:200], flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
