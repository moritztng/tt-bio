#!/usr/bin/env python3
"""PROTOCOL SS3, precondition: which of their parameters have a REACHABLE gradient at all.

Before a per-parameter gradient can be compared it has to exist. SS3a established the
parameter correspondence; this establishes, at RUNTIME and on the card, which of those
parameters our stack actually produces a gradient for.

The question is not academic. `tt_bio.train.lora.weights_for` discovers the trainable set
from one census forward and refuses with "the census found no adaptable linear site ... the
forward does not route through tt_bio.ops.linear". So the trainable set is exactly the
`ops.linear` sites, and the optimised TriangleMultiplication does not obviously route
through one: it pre-fuses `g_in`/`p_in` into ttnn tensors in `__init__` and multiplies them
directly. If those weights are not reachable, 232 of our tensors -- 464 of theirs -- cannot
be compared at all, and a gradient instrument that quietly skipped them would report a clean
pass over the parameters it happened to reach.

Runtime, not static, deliberately: A2 recorded that static verb coverage misses call sites
that `tape()` cannot see, so this runs the real module on the real card and reads what the
tape produced.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

CKPT = "/home/ttuser/.boltz/of3-p2-155k.pt"
OUT = Path(__file__).resolve().parent / "grad_reach.json"


def main() -> int:
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import tenstorrent as tt
    from tt_bio import openfold3_weights as of3w

    sd = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # Block 0's incoming triangle multiplication, through the shipped remap, so the fused
    # g_in/p_in this probe is about are built exactly as production builds them.
    # The remap returns ONE flat dict with dotted keys, e.g.
    # "layers.0.tri_mul_in.g_in.weight" of shape (256, 128) -- which is already the fusion
    # visible in the shape: two of their (128, 128) projections concatenated on dim 0.
    stack = of3w.remap_pairformer_stack(sd, "pairformer_stack")
    pre = "layers.0.tri_mul_in."
    flat = {k[len(pre):]: v for k, v in stack.items()
            if k.startswith(pre) and torch.is_tensor(v)}
    if not flat:
        raise SystemExit(f"no keys under {pre!r}; got e.g. {sorted(stack)[:5]}")

    report = {"instrument": "PROTOCOL SS3 precondition -- gradient reachability",
              "module": "TriangleMultiplication (incoming), pairformer_stack.blocks.0",
              "our_param_keys": sorted(flat),
              "shapes": {k: list(v.shape) for k, v in sorted(flat.items())}}

    device = tt.get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    mod = tt.TriangleMultiplication(True, flat, ckc)

    n = 64
    c_z = flat["g_in.weight"].shape[1]
    rng = torch.Generator().manual_seed(11)
    x = torch.randn(1, n, n, c_z, generator=rng, dtype=torch.float32) * 0.05
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # --- the probe: run the shipped module under the tape and see what carries a gradient --
    leaf = ag.Tensor(xt, requires_grad=True)
    got_grad, err = {}, None
    try:
        with ag.tape():
            out = mod(leaf)
        out.backward()
        got_grad["input"] = leaf.grad is not None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    report["taped_backward_error"] = err
    report["input_received_gradient"] = got_grad.get("input")

    # Are the module's own weights autograd leaves at all? They are built by `torch_to_tt`
    # in __init__, which returns a raw ttnn handle, so this is asking whether anything in
    # the module is a thing the tape could ever hand a gradient to.
    leaves = []
    for attr in dir(mod):
        if attr.startswith("__"):
            continue
        try:
            v = getattr(mod, attr)
        except Exception:
            continue
        if isinstance(v, ag.Tensor):
            leaves.append({"attr": attr, "requires_grad": v.requires_grad,
                           "has_grad": v.grad is not None})
    report["module_autograd_leaves"] = leaves
    report["module_autograd_leaf_count"] = len(leaves)

    # --- and what the trainable-site census would find on this forward -------------------
    from tt_bio.train import lora as L
    census_err, sites = None, []
    try:
        params = L.weights_for(lambda t: mod(t), None, leaf)
        sites = sorted(params)
    except Exception as e:
        census_err = f"{type(e).__name__}: {e}"
    report["census_sites"] = sites
    report["census_site_count"] = len(sites)
    report["census_error"] = census_err

    report["conclusion"] = (
        "weights are reachable" if leaves or sites else
        "NO weight of this module is an autograd leaf and the census finds no adaptable "
        "site, so no gradient exists for it to compare")
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "shapes"}, indent=2)[:3000])
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
