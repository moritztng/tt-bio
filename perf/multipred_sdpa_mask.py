#!/usr/bin/env python3
"""Replicate the production tri-attention SDPA call exactly (attn_mask + scale) at seq=128.

perf/multipred_sdpa_prodgrid.py called SDPA with no mask and no scale and got PCC 0.285 at
the production q128/k128 config. The real call in TriangleAttention.attend passes
attn_mask=triangle_bias (1, n_heads, S, S) and scale=self.scale**-1. If the mask restores
PCC, the earlier number is a probe artifact and shipping code is unaffected.
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
    torch.manual_seed(0)
    scale = D ** -0.5
    tq, tk, tv = (torch.randn(N, H, N, D) for _ in range(3))
    tbias = torch.randn(1, H, N, N) * 0.5

    ref = torch.nn.functional.scaled_dot_product_attention(
        tq.float(), tk.float(), tv.float(), attn_mask=tbias.float().expand(N, H, N, N),
        scale=scale)

    q, k, v = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
               for t in (tq, tk, tv))
    bias = ttnn.from_torch(tbias, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    for name, cfg in (("production q128/k128", T._tri_att_sdpa_program_config(N, N)),
                      ("q64/k128", T._sdpa_program_config(64, 128)),
                      ("q32/k32", T._sdpa_program_config(32, 32))):
        for with_mask in (True, False):
            try:
                kw = dict(attn_mask=bias) if with_mask else {}
                o = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=False, scale=scale, program_config=cfg, **kw)
                t = ttnn.to_torch(o)
                r = ref if with_mask else torch.nn.functional.scaled_dot_product_attention(
                    tq.float(), tk.float(), tv.float(), scale=scale)
                ttnn.deallocate(o)
                ms = bench(dev, lambda c=cfg, kk=kw: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, is_causal=False, scale=scale, program_config=c, **kk)))
                print(f"{name:22s} mask={str(with_mask):5s} {ms:8.4f} ms  pcc_vs_fp32={pcc(t, r):.6f}")
            except Exception as e:
                print(f"{name:22s} mask={str(with_mask):5s} ERR {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
