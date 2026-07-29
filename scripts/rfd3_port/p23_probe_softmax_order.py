"""Minimal repro: is `ttnn.softmax(dim=-1)`'s answer a function of what ran before it?

p23 named `ttnn.softmax` as the first op inside RFD3AtomBlock whose output differs between
an isolated and a sequenced fold from bit-identical inputs. This takes RFD3 out of the
picture: build one deterministic [1,4,L,L] fp32 tensor, softmax it, and print the sum --
optionally after softmaxing other shapes first in the same process.

  --target L         width/height of the tensor under test (default 2702)
  --pre L [L ...]    shapes to softmax first, in order
  --pre-dtype        dtype for the warm-up shapes (default float32)
"""
from __future__ import annotations

import argparse
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=2702)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--pre", type=int, nargs="*", default=[])
ap.add_argument("--pre-dtype", default="float32")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--tree", default=None)
args = ap.parse_args()

if args.tree:
    sys.path.insert(0, args.tree)

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

DT = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16}
dev = get_device()


def make(L, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, args.heads, L, L, generator=g) * 4.0
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


for L in args.pre:
    t = make(L, DT[args.pre_dtype], 1234 + L)
    n0 = dev.num_program_cache_entries()
    out = ttnn.softmax(t, dim=-1)
    n1 = dev.num_program_cache_entries()
    print(f"pre  L={L:5d} dtype={args.pre_dtype:9s} cache {n0}->{n1}", flush=True)
    ttnn.deallocate(out)
    ttnn.deallocate(t)

t = make(args.target, DT[args.dtype], 7)
n0 = dev.num_program_cache_entries()
out = ttnn.softmax(t, dim=-1)
n1 = dev.num_program_cache_entries()
y = ttnn.to_torch(out).double()
print(f"TARGET L={args.target} dtype={args.dtype} cache {n0}->{n1} "
      f"sum={y.sum().item():.12e} rowsum_dev_max={float((y.sum(-1) - 1).abs().max()):.6e}",
      flush=True)
