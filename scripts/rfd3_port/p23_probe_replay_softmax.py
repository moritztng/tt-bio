"""Softmax the dumped real scores tensor in a clean process, from a host upload.

Gives the third data point: what `ttnn.softmax(dim=-1)` returns for this exact LOGICAL input
when the tensor is freshly uploaded (host upload zero-fills the tile padding).
"""
from __future__ import annotations

import argparse
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--tree", default=None)
ap.add_argument("--dirty", type=float, default=0.0,
                help="pre-dirty DRAM with this value before uploading")
args = ap.parse_args()
if args.tree:
    sys.path.insert(0, args.tree)

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()
x = torch.load(args.input, weights_only=True)
print("loaded", tuple(x.shape), x.dtype, "sum=%.12e" % x.double().sum().item(), flush=True)

if args.dirty:
    d = ttnn.full((1, 4, 2720, 2720), args.dirty, dtype=ttnn.float32,
                  layout=ttnn.TILE_LAYOUT, device=dev)
    print("dirtied at", hex(d.buffer_address()), flush=True)
    ttnn.deallocate(d)

t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
print("upload at", hex(t.buffer_address()), flush=True)
out = ttnn.softmax(t, dim=-1)
print("sum=%.9f" % ttnn.to_torch(out).double().sum().item(), flush=True)
