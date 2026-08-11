#!/usr/bin/env python3
"""K2 feasibility probe: what work split and what KV chains does the fold's own SDPA call get?

Two claims from reading `sdpa_program_factory.cpp` at v0.68.0 decide whether K2 is a tractable
transcription or a 470-line topology problem. This checks both against the device, at the exact
shape and program config the 512 aa fold issues, by running the op with debug logging on.

    1. Every core is given ALL 8 heads (batch_parallel_factor = min(B, num_cores) = 110 saturates
       the split before nh_parallel_factor gets anything), so a per-core mask working set is 4 MiB
       and K2 must change the work split.
    2. KV chains are grouped by (batch, head) and skipped below 2 segments. Every (batch, head) is
       owned by one core, so chains_built should be 0 and chains_skipped 0 as well -- the second
       pass never even sees a multi-segment head.

This is a diagnostic run, not a timed one, so it does not take the benchlock.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

S, C, H, D = 512, 256, 8, 32
dev = T.get_device()


def dram(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


torch.manual_seed(0)
q = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
k = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
v = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))

pc = T._sdpa_program_config(512, 256)
print("PROBE program_config:", pc, flush=True)
print("PROBE q shape:", [int(d) for d in q.shape], "bias:", [int(d) for d in bias.shape], flush=True)
o = ttnn.transformer.scaled_dot_product_attention(
    q, k, v, attn_mask=bias, is_causal=False, scale=D ** -0.5, program_config=pc)
ttnn.synchronize_device(dev)
print("PROBE ok, out shape:", [int(d) for d in o.shape], flush=True)
