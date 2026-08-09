#!/usr/bin/env python3
"""SDPA at the trunk tri-attention shape, on tt-bio's PRODUCTION grid.

perf/multipred_sdpa_pcc.py used dev.compute_with_storage_grid_size(); tt-bio builds its
configs on COMPUTE_GRID_MAIN. If those differ, the earlier PCC 0.281 at q128/k128 says
nothing about shipping code. This re-runs on the production grid and prints both.
"""
import time

import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device

N, H, D = 128, 8, 32


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def med(xs):
    return sorted(xs)[len(xs) // 2]


def bench(dev, fn, warm=5, pipe=12, reps=5):
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
    return med(out)


def main():
    dev = get_device()
    print(f"COMPUTE_GRID_MAIN={T.COMPUTE_GRID_MAIN}  "
          f"device_grid={dev.compute_with_storage_grid_size()}")
    prod = T._tri_att_sdpa_program_config(N, N)
    print(f"production tri-att config at seq=128: q={prod.q_chunk_size} k={prod.k_chunk_size}")

    torch.manual_seed(0)
    tq, tk, tv = (torch.randn(N, H, N, D) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(tq.float(), tk.float(), tv.float())
    q, k, v = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
               for t in (tq, tk, tv))

    cases = {"production": prod}
    for cq, ckz in ((32, 32), (32, 128), (64, 64), (64, 128), (128, 128)):
        cases[f"main_q{cq}_k{ckz}"] = T._sdpa_program_config(cq, ckz)
    cases["devgrid_q128_k128"] = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
        exp_approx_mode=False, q_chunk_size=128, k_chunk_size=128)

    base = None
    for name, cfg in cases.items():
        try:
            out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, program_config=cfg)
            t = ttnn.to_torch(out)
            ms = bench(dev, lambda c=cfg: ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, program_config=c)))
            if base is None:
                base = t
            print(f"{name:20s} q={cfg.q_chunk_size:4d} k={cfg.k_chunk_size:4d} "
                  f"{ms:8.4f} ms  pcc_vs_fp32={pcc(t, ref):.6f}  "
                  f"bitexact_vs_prod={torch.equal(t, base)}")
        except Exception as e:
            print(f"{name:20s} ERR {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    main()
