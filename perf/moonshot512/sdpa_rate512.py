"""The tri-attention SDPA rate at the shape the fold really runs, with the production config.

shape_roof_census.py probes SDPA at q_chunk = k_chunk = seq, which throws a circular-buffer clash
at seq >= 320. Production caps the chunk at SDPA_CHUNK_MAX = 256 (`_tri_att_sdpa_program_config`),
so the census left the single largest arithmetic class in the model unmeasured. This measures it
the production way and sweeps the chunk so the number has a context.
"""
import json
import sys
import time

import torch
import ttnn

from tt_bio import tenstorrent as T


def timed(dev, fn, iters=7):
    fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    out = sys.argv[1]
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    KC = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    res = {"grid": [g.x, g.y], "cores": g.x * g.y, "rows": []}
    for s in (320, 512):
        b, h, d = s, 8, 32
        gf = 4 * b * h * s * s * d / 1e9
        q, k, v = (
            ttnn.from_torch(torch.randn(b, h, s, d) * 0.1, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
            for _ in range(3)
        )
        bias = ttnn.from_torch(torch.randn(1, h, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
        prod = T._tri_att_sdpa_program_config(s, s)
        for label, chunk in (("production", None), ("64", 64), ("128", 128),
                             ("256", 256), ("512", 512)):
            prog = prod if chunk is None else T._sdpa_program_config(chunk, chunk)
            try:
                ms = timed(dev, lambda p=prog: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False,
                        program_config=p, compute_kernel_config=KC)))
                tf = round(gf / (ms / 1e3) / 1e3, 2)
                row = {"seq": s, "chunk": label, "ms": round(ms, 4),
                       "gflop_per_call": round(gf, 2), "tflops": tf}
                print("  seq=%d chunk=%-10s %8.4f ms %7.2f TFLOP/s" % (s, label, ms, tf),
                      flush=True)
            except Exception as e:  # noqa: BLE001
                row = {"seq": s, "chunk": label, "error": str(e)[:200]}
                print("  seq=%d chunk=%-10s ERR %s" % (s, label, str(e)[:100]), flush=True)
            res["rows"].append(row)
        for t in (q, k, v, bias):
            ttnn.deallocate(t)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out, flush=True)
    ttnn.close_device(dev)


main()
