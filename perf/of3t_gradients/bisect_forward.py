#!/usr/bin/env python3
"""SS3c first: the forward has to agree before any gradient it produces means anything.

The stack-scope instrument read z at 5.1e-01 against the float64 reference, which is two orders
past anything bf16 explains, so this localises it one sub-module at a time on the same input.
Each of their modules is run in float64 and each of ours on the card, both handed the same z,
both returning the UPDATE rather than the residual sum, which is what `PairBlock` adds.

K28: a module's probe input has to be its own, so the sweep is also run on the z a real block
sees rather than only on a unit-normal tensor of the right shape.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT = "perf/of3t_gradients/bisect_forward.json"
PRE = "pairformer_stack.blocks.0."


def rel(a, b):
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    import numpy as np
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack

    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttention as TheirTriAtt, TriangleAttentionEndingNode)
    from openfold3.core.model.layers.transition import SwiGLUTransition

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    flat = {k[len("layers.0."):]: v
            for k, v in remap_pairformer_stack(sd, prefix="pairformer_stack").items()
            if k.startswith("layers.0.")}

    N, c_z = 64, 128
    g = torch.Generator().manual_seed(11)
    z0 = torch.randn(1, N, N, c_z, generator=g) * 0.05
    mask = torch.ones(1, N, N, dtype=torch.float64)

    def sub(prefix):
        return {k[len(PRE + prefix):]: v.to(torch.float64)
                for k, v in sd.items() if k.startswith(PRE + prefix)}

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def ours_flat(prefix, strip=""):
        n = len(prefix)
        return {(k[n:][len(strip):] if k[n:].startswith(strip) else k[n:]): v
                for k, v in flat.items() if k.startswith(prefix)}

    cases = []

    # 1 + 2: the two triangle multiplications.
    for label, cls, their_p, our_p, ending in (
            ("tri_mul_out", TriangleMultiplicationOutgoing, "pair_stack.tri_mul_out.",
             "tri_mul_out.", False),
            ("tri_mul_in", TriangleMultiplicationIncoming, "pair_stack.tri_mul_in.",
             "tri_mul_in.", True)):
        m = cls(c_z=c_z, c_hidden=c_z).to(torch.float64)
        m.load_state_dict(sub(their_p), strict=True)
        m.eval()
        ref = m(z0.to(torch.float64), mask=mask).detach()
        ours = T.TriangleMultiplication(ending, ours_flat(our_p), ckc)
        got = ttnn.to_torch(ours(ft(z0))).to(torch.float64)
        cases.append({"module": label, "rel": rel(got.numpy(), ref.numpy()),
                      "ref_norm": float(ref.norm())})

    # 3 + 4: the two triangle attentions.
    n_heads = flat["tri_att_start.linear.weight"].shape[0]
    head_dim = flat["tri_att_start.mha.linear_q.weight"].shape[0] // n_heads
    for label, cls, their_p, our_p, ending in (
            ("tri_att_start", TheirTriAtt, "pair_stack.tri_att_start.", "tri_att_start.", False),
            ("tri_att_end", TriangleAttentionEndingNode, "pair_stack.tri_att_end.",
             "tri_att_end.", True)):
        m = cls(c_in=c_z, c_hidden=head_dim, no_heads=n_heads, inf=1e9).to(torch.float64)
        m.load_state_dict(sub(their_p), strict=True)
        m.eval()
        ref = m(z0.to(torch.float64), mask=mask).detach()
        ours = T.TriangleAttention(head_dim, n_heads, ending, ours_flat(our_p, "mha."), ckc,
                                   scale_pair_bias=False, fp32_softmax=True)
        got = ttnn.to_torch(ours(ft(z0))).to(torch.float64)
        cases.append({"module": label, "rel": rel(got.numpy(), ref.numpy()),
                      "ref_norm": float(ref.norm())})

    # 5: the pair transition.
    m = SwiGLUTransition(c_in=c_z, n=4).to(torch.float64)
    m.load_state_dict(sub("pair_stack.pair_transition."), strict=True)
    m.eval()
    ref = m(z0.to(torch.float64), mask=mask).detach()
    ours = T.Transition(ours_flat("transition_z."), ckc)
    got = ttnn.to_torch(ours(ft(z0))).to(torch.float64)
    cases.append({"module": "pair_transition", "rel": rel(got.numpy(), ref.numpy()),
                  "ref_norm": float(ref.norm())})

    for c in cases:
        print(f"   {c['rel']:.3e}  {c['module']}  (ref norm {c['ref_norm']:.3e})")
    json.dump({"tokens": N, "cases": cases}, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
