#!/usr/bin/env python3
"""Which of upstream's two end-node functions does OUR device module compute?

`end_node_convention.json` establishes, on trained weights in float64, that upstream 0.5.0 has
two constructions of `tri_att_end` that are NOT the same function:

  A/B'  TriangleAttentionEndingNode(z)  ==  TriangleAttention(z^T, transpose_bias=False)^T
  B     TriangleAttention(z^T, transpose_bias=True)^T   <- what PairFormerBlock:388-397 calls

bit-identical inside the pair, 7.936e-01 relative between them. The trained checkpoint describes
B, because B is what the block runs.

This asks our `T.TriangleAttention(ending=True)` the same question at both settings of our own
`transpose_bias`, forward only, against both references at once. It is the discriminator the
campaign's `tb-off` arms have been reading indirectly for forty passes: those arms move our flag
and watch a gradient, which cannot say WHICH function either setting computes.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "perf/of3t_gradients")

OUT = "perf/of3t_reopen/end_node_orientation.json"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BLOCK = 2


def rel(a, b):
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main() -> int:
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttention, TriangleAttentionEndingNode)

    full = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(full, dict) and "state_dict" in full and isinstance(full["state_dict"], dict):
        full = full["state_dict"]
    pre = f"pairformer_stack.blocks.{BLOCK}.pair_stack.tri_att_end."
    sd = {k[len(pre):]: v.detach().to(torch.float64)
          for k, v in full.items() if k.startswith(pre)}
    H = sd["linear_z.weight"].shape[0]
    c_z = sd["layer_norm.weight"].shape[0]
    c_h = sd["mha.linear_q.weight"].shape[0] // H

    torch.manual_seed(7)
    N = 64
    z = (torch.randn(1, N, N, c_z) * 0.5).double()
    mask = torch.ones(1, N, N, dtype=torch.float64)

    end = TriangleAttentionEndingNode(c_z, c_h, H, inf=1e9).double().eval()
    end.load_state_dict(sd, strict=True)
    start = TriangleAttention(c_z, c_h, H, inf=1e9).double().eval()
    start.load_state_dict(sd, strict=True)
    with torch.no_grad():
        refA = end(z, mask=mask)
        refB = start(z.transpose(-2, -3), mask=mask.transpose(-1, -2),
                     transpose_bias=True).transpose(-2, -3)
    sep = rel(refA.numpy(), refB.numpy())

    flat_all = remap_pairformer_stack(full, prefix="pairformer_stack")
    L = f"layers.{BLOCK}.tri_att_end."
    n = len(L)
    ours_sd = {(k[n:][len("mha."):] if k[n:].startswith("mha.") else k[n:]): v
               for k, v in flat_all.items() if k.startswith(L)}

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)

    rep = {"instrument": "our end node against BOTH of upstream 0.5.0's end-node functions",
           "block": BLOCK, "tokens": N, "weights": CKPT,
           "refA": "TriangleAttentionEndingNode(z) == external transpose, transpose_bias=False",
           "refB": "TriangleAttention(z^T, transpose_bias=True)^T -- base_blocks.py:388-397",
           "refA_vs_refB": sep, "arms": {}}
    print(f"reference separation A vs B: {sep:.6e}")
    for arm in (True, False):
        mod = T.TriangleAttention(c_h, H, True, dict(ours_sd), ckc,
                                  scale_pair_bias=False, fp32_softmax=True,
                                  transpose_bias=arm)
        out = mod(ft(z))
        ours = ttnn.to_torch(out).to(torch.float64).numpy()
        ra, rb = rel(ours, refA.numpy()), rel(ours, refB.numpy())
        rep["arms"][f"ours_transpose_bias_{arm}"] = {"vs_refA": ra, "vs_refB": rb,
                                                     "matches": "A" if ra < rb else "B"}
        print(f"ours transpose_bias={str(arm):5s}  vs A {ra:.6e}   vs B {rb:.6e}   "
              f"-> matches {'A' if ra < rb else 'B'}")
    os.makedirs("perf/of3t_reopen", exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
