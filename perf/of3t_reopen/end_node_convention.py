#!/usr/bin/env python3
"""Which end-node bias ordering does upstream 0.5.0's PairFormerBlock actually compute?

float64 torch only, no card, no checkpoint semantics -- two of THEIR OWN constructions of the
same sub-module, on one draw:

  A  TriangleAttentionEndingNode(z)                        starting=False, transpose_bias=False
  B  TriangleAttention(z.transpose, transpose_bias=True)   what PairFormerBlock:394 calls

`base_blocks.py:284` builds `tri_att_end` as a plain `TriangleAttention` (starting=True), then
`tri_att_start_end` transposes z itself and passes `transpose_bias=True`. `bisect_grad.py`'s
alone arm builds `TriangleAttentionEndingNode` and calls it bare. If A and B are the same
function the two arms are comparable; if they differ, every ALONE measurement of `tri_att_end`
in this campaign was taken against a reference the trained checkpoint does not describe, and
D8's alone-vs-assembled asymmetry on that module is an instrument artifact before it is a
finding.

B' is the same external transpose WITHOUT their flag, which is our device's documented
meaning of `transpose_bias=True` -- "the ending variant's pair bias is transposed along with
the pair", tenstorrent.py:7388. So the three columns say whether their two constructions agree
and which one our shipped trunk reproduces.

RUN IT ON TRAINED WEIGHTS. The first version of this file built the modules from their own
initialisers and read 0.000e+00 three ways, which looked like "the constructions agree" and was
"the module output is identically zero": OF3 zero-inits `mha.linear_o`, so an untrained
TriangleAttention returns zeros whatever its bias does. The bias-permutation probe below is the
control -- it reads the two orderings directly, so a vacuous pass cannot hide behind it.
"""
from __future__ import annotations

import json
import os
import sys

OUT = "perf/of3t_reopen/end_node_convention.json"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BLOCK = 2


def rel(a, b):
    import numpy as np
    a, b = a.detach().double().numpy(), b.detach().double().numpy()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main() -> int:
    import torch
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttention, TriangleAttentionEndingNode)
    from openfold3.core.utils.tensor_utils import permute_final_dims

    full = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(full, dict) and "state_dict" in full and isinstance(full["state_dict"], dict):
        full = full["state_dict"]
    pre = f"pairformer_stack.blocks.{BLOCK}.pair_stack.tri_att_end."
    sd = {k[len(pre):]: v.detach().to(torch.float64)
          for k, v in full.items() if k.startswith(pre)}
    if not sd:
        raise SystemExit(f"no trained weights under {pre}")
    H = sd["linear_z.weight"].shape[0]
    c_z = sd["layer_norm.weight"].shape[0]
    c_h = sd["mha.linear_q.weight"].shape[0] // H

    torch.manual_seed(7)
    N = 48
    z = (torch.randn(1, N, N, c_z) * 0.5).double()
    mask = torch.ones(1, N, N, dtype=torch.float64)

    end = TriangleAttentionEndingNode(c_z, c_h, H, inf=1e9).double().eval()
    end.load_state_dict(sd, strict=True)
    start = TriangleAttention(c_z, c_h, H, inf=1e9).double().eval()
    start.load_state_dict(sd, strict=True)

    with torch.no_grad():
        # A -- the alone arm's construction
        A = end(z, mask=mask)
        # B -- PairFormerBlock's own construction, base_blocks.py:388-397
        B = start(z.transpose(-2, -3), mask=mask.transpose(-1, -2),
                  transpose_bias=True).transpose(-2, -3)
        # B' -- the same external transpose but WITHOUT their transpose_bias flag, i.e. the
        #       bias following the transposed pair. This is our device's documented meaning of
        #       transpose_bias=True, written in their code.
        Bp = start(z.transpose(-2, -3), mask=mask.transpose(-1, -2),
                   transpose_bias=False).transpose(-2, -3)

    # The control this file needed the first time: the two bias orderings, read directly. If
    # this is 0 the probe is degenerate and nothing below means anything.
    with torch.no_grad():
        xln = start.layer_norm(z.transpose(-2, -3))
        b_201 = permute_final_dims(start.linear_z(xln), (2, 0, 1))
        b_210 = permute_final_dims(start.linear_z(xln), (2, 1, 0))
    bias_sep = float((b_201 - b_210).norm() / b_210.norm())

    rep = {"instrument": "upstream 0.5.0's two end-node constructions, float64 torch, one draw",
           "shapes": {"N": N, "c_z": c_z, "c_hidden": c_h, "heads": H},
           "A_is": "TriangleAttentionEndingNode(z) -- bisect_grad.py's alone arm",
           "B_is": "TriangleAttention(z^T, transpose_bias=True)^T -- base_blocks.py:388-397, "
                   "what PairFormerBlock computes",
           "Bprime_is": "the same external transpose with transpose_bias=False -- the bias "
                        "following the transposed pair, our device's transpose_bias=True",
           "rel_A_vs_B": rel(A, B),
           "rel_A_vs_Bprime": rel(A, Bp),
           "rel_B_vs_Bprime": rel(B, Bp),
           "bias_ordering_separation": bias_sep,
           "control": "bias_ordering_separation is the two permutations against each other; a "
                      "zero there means the probe cannot see the flag at all",
           "weights": f"{CKPT} block {BLOCK} tri_att_end, trained",
           "norms": {"A": float(A.norm()), "B": float(B.norm()), "Bprime": float(Bp.norm())}}
    if min(rep["norms"].values()) == 0.0 or bias_sep == 0.0:
        rep["VACUOUS"] = ("an output norm or the bias separation is zero -- this run reads "
                          "nothing and its rel figures are not evidence")
        print("VACUOUS: " + rep["VACUOUS"])
    print(f"bias (2,0,1) vs (2,1,0)     separation             rel {bias_sep:.6e}")
    print(f"|A| {float(A.norm()):.4f}  |B| {float(B.norm()):.4f}  |B'| {float(Bp.norm()):.4f}")
    print(f"A (EndingNode bare)        vs B (PairFormerBlock)   rel {rep['rel_A_vs_B']:.6e}")
    print(f"bias (2,0,1) vs (2,1,0)     separation             rel {bias_sep:.6e}")
    print(f"|A| {float(A.norm()):.4f}  |B| {float(B.norm()):.4f}  |B'| {float(Bp.norm()):.4f}")
    print(f"A (EndingNode bare)        vs B' (bias follows z^T) rel {rep['rel_A_vs_Bprime']:.6e}")
    print(f"B (PairFormerBlock)        vs B' (bias follows z^T) rel {rep['rel_B_vs_Bprime']:.6e}")
    os.makedirs("perf/of3t_reopen", exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
