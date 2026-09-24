"""Minimal repro for tt-metal: non-causal SDPA with attn_mask=None is wrong at q_chunk = k_chunk = 128.

    python3 perf/mgx_sdpa/repro.py [device_id]

Needs only ttnn and torch. Prints PCC against torch fp32 for the same q, k, v with no mask and
with an all-zero additive mask (mathematically identical), twice each.
"""
import sys

import torch
import ttnn

dev = ttnn.open_device(device_id=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
try:
    g = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    B, H, L, d = 1, 40, 128, 64
    q, k, v = (torch.randn(B, H, L, d) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    tt = lambda x: ttnn.from_torch(x.bfloat16(), device=dev, layout=ttnn.TILE_LAYOUT)
    zero = tt(torch.zeros(1, 1, L, L))
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g, exp_approx_mode=False,
                                 q_chunk_size=128, k_chunk_size=128)
    print(f"arch {dev.arch()} grid {g.x}x{g.y} ttnn {getattr(ttnn, '__version__', '?')}")
    for name, mask in (("attn_mask=None", None), ("zero mask", zero)):
        for run in range(2):
            o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
                tt(q), tt(k), tt(v), attn_mask=mask, is_causal=False, scale=d ** -0.5,
                program_config=cfg)).float()
            pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
            print(f"{name:15s} run {run}: PCC {pcc:.4f}")
finally:
    ttnn.close_device(dev)
