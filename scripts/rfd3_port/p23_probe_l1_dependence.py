"""Is `ttnn.softmax`'s compiled program a function of how much L1 is free when it compiles?

The p23 divergence is one softmax op, one shape, one program-cache key, bit-identical input,
two different answers -- and each run compiled its own entry for that key. So the compile
itself differs. tt-metal's multi-core softmax sizes its row-reduction block from the L1 space
still available (`lowest_occupied_compute_l1_address`), which is process history, not part of
the cache key. This holds a deliberate L1 allocation alive across the first compile and looks
for a different answer.

  --l1-mb N   keep an N MB L1-interleaved tensor alive while softmax compiles
"""
from __future__ import annotations

import argparse
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=2702)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--l1-kb", type=float, default=0.0)
ap.add_argument("--tree", default=None)
args = ap.parse_args()

if args.tree:
    sys.path.insert(0, args.tree)

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()


def lowest():
    try:
        return dev.lowest_occupied_compute_l1_address()
    except Exception as e:  # noqa: BLE001
        return f"<n/a {e}>"


hold = None
if args.l1_kb:
    # interleaved L1 tensor, spread over all cores; bf16 tiles
    tiles = max(1, int(args.l1_kb * 1024 / 2048))
    hold = ttnn.from_torch(torch.zeros(1, 1, 32, 32 * tiles), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"held {tiles} L1 tiles ({tiles * 2048 / 1024:.1f} KB total)", flush=True)

print("lowest_occupied_compute_l1_address before compile:", lowest(), flush=True)

g = torch.Generator().manual_seed(7)
x = torch.randn(1, args.heads, args.target, args.target, generator=g) * 4.0
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
n0 = dev.num_program_cache_entries()
out = ttnn.softmax(t, dim=-1)
n1 = dev.num_program_cache_entries()
y = ttnn.to_torch(out).double()
print(f"L1_KB={args.l1_kb} cache {n0}->{n1} sum={y.sum().item():.12e}", flush=True)
if hold is not None:
    ttnn.deallocate(hold)
