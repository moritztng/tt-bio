#!/usr/bin/env python3
"""Is `ttnn.linear(..., bias=b, activation="sigmoid")` sigmoid(x@W + b), or sigmoid(x@W) + b?

The AdaLN fold moves the gate's sigmoid into the projection matmul's epilogue. That is only
valid if the epilogue applies the activation AFTER the bias. If it applies it before, the fold
silently computes a different function, and a fold-level RMSD would read as a basin flip rather
than as the bug it is. Scored against a float64 torch reference, never against another
approximation.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, "/home/ttuser/.coworker/wt/c12-fused-eltwise-at-pin")

import torch
import ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
torch.manual_seed(0)
M, K, N = 512, 768, 768
x_t = (torch.randn(1, M, K) * 0.5)
w_t = (torch.randn(K, N) * 0.05)
b_t = (torch.randn(N) * 0.3)

f = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
x, w, b = f(x_t), f(w_t), f(b_t)

ref_after = torch.sigmoid(x_t.double() @ w_t.double() + b_t.double())   # sigmoid(xW + b)
ref_before = torch.sigmoid(x_t.double() @ w_t.double()) + b_t.double()  # sigmoid(xW) + b

y_act = ttnn.to_torch(ttnn.linear(x, w, bias=b, activation="sigmoid")).double()
y_chain = ttnn.to_torch(ttnn.sigmoid(ttnn.linear(x, w, bias=b))).double()


def err(a, r):
    return float((a - r).abs().max()), float((a - r).abs().max() / r.abs().max())


print("linear(activation='sigmoid')  vs sigmoid(xW+b) float64 : maxabs %.3e  rel %.3e" % err(y_act, ref_after))
print("linear(activation='sigmoid')  vs sigmoid(xW)+b float64 : maxabs %.3e  rel %.3e" % err(y_act, ref_before))
print("sigmoid(linear(bias=b))       vs sigmoid(xW+b) float64 : maxabs %.3e  rel %.3e" % err(y_chain, ref_after))
print("linear(activation) vs sigmoid(linear())                : maxabs %.3e" % float((y_act - y_chain).abs().max()))

# And the other half of the fold: is addcmul(bias, a, gate) == a*gate + bias?
a_t = torch.randn(1, M, N) * 0.7
g_t = torch.rand(1, M, N)
bb_t = torch.randn(1, M, N) * 0.4
a, g, bb = f(a_t), f(g_t), f(bb_t)
ref = a_t.double() * g_t.double() + bb_t.double()
y_fused = ttnn.to_torch(ttnn.addcmul(bb, a, g)).double()
y_chainm = ttnn.to_torch(ttnn.add_(ttnn.multiply(a, g), bb)).double()
print("addcmul(bias, a, gate)        vs a*gate+bias  float64   : maxabs %.3e  rel %.3e" % err(y_fused, ref))
print("multiply then add             vs a*gate+bias  float64   : maxabs %.3e  rel %.3e" % err(y_chainm, ref))
