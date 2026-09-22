#!/usr/bin/env python3
"""Unit gradcheck of the DiT attention tail against float64, one op at a time.

WHY HERE. The per-tensor array localises the campaign's largest gradient disagreement to ONE
AdaLN site: `attention_pair_bias.layer_norm_a`. Its sister, `conditioned_transition.layer_norm`,
is the SAME `tt_bio.tenstorrent.AdaLN` class on the SAME `s` in the SAME 24 blocks, and it is
clean in every one of them -- 0.103 at block 8 against 18.504. An op that is wrong is wrong at
both sites, so the AdaLN op is exonerated by the measurement and the suspect is what flows BACK
into `adaln_a`: the attention chain between it and the block output
(`openfold3_diffusion_transformer.py:181-228`), which the transition path does not have.

The chain's most suspicious member is `o = o[:, :, :, :HEAD_DIM]` at line 216 -- a last-axis
slice from a padded 64 to 48, which is NOT a tile multiple. Its taped backward pads 48 back to
64 with `ttnn.concat` on the last axis in TILE_LAYOUT (`taped_ttnn.py:_sliced`), and sub-tile
last-axis work on a Blackhole DRAM tensor is a known hazard class on this fleet.

So: run the tail forward and backward on the card, seeded with a random cotangent, against the
same chain in torch float64 on host. Per op, then composed. Tiny tensors, no model, no
checkpoint. Either an op's backward is wrong -- and the campaign's largest gradient defect is
located -- or every one is clean and the defect is an interaction further up, which is also a
result and narrows the list.

The forward is checked beside every backward (A18 at unit scale): a wrong forward would make
the backward comparison meaningless, and these ops are pure data movement, so the forward is
expected bit-exact and a non-zero reading there is itself the finding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

N_HEADS, HEAD_DIM, PAD_DIM = 16, 48, 64


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--act", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--out", default="perf/of3t_conditioning/DIT_TAIL_GRADCHECK.json")
    a = p.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.taped_ttnn import taped_ttnn
    from tt_bio.tenstorrent import get_device, device_dtype_override

    # `tape()` rebinds the name `ttnn` inside tt_bio modules; a script outside the package still
    # holds the real module, and the real pybind refuses an autograd.Tensor. The shim IS the
    # taped surface, so address it directly -- this is the documented way in for a verification
    # script (`taped_ttnn.taped_ttnn`'s own docstring).
    T = taped_ttnn()

    torch.manual_seed(a.seed)
    N = a.tokens
    dev = get_device()
    act = ttnn.float32 if a.act == "fp32" else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
    to_t = lambda x: ttnn.to_torch(x.value if hasattr(x, "value") else x).double()

    def rel(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        return float(torch.linalg.vector_norm(x - y)
                     / (torch.linalg.vector_norm(y) + 1e-300))

    def cosine(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        nx, ny = x.norm(), y.norm()
        return float((x * y).sum() / (nx * ny)) if nx and ny else 0.0

    def ratio(x, y):
        return float(x.reshape(-1).double().norm() / (y.reshape(-1).double().norm() + 1e-300))

    arms, results = [], {}

    # Each arm: a name, the device chain, the torch chain, and the input shape. The torch side
    # is float64 and is the reference; the device side is the shipped op through the tape.
    def arm(name, shape, dev_fn, ref_fn):
        arms.append((name, shape, dev_fn, ref_fn))

    arm("slice_64_to_48", (1, N_HEADS, N, PAD_DIM),
        lambda x: x[:, :, :, :HEAD_DIM],
        lambda x: x[:, :, :, :HEAD_DIM])
    arm("permute_0132", (1, N_HEADS, N, HEAD_DIM),
        lambda x: T.permute(x, (0, 1, 3, 2)),
        lambda x: x.permute(0, 1, 3, 2))
    arm("reshape_merge_heads", (1, N_HEADS, HEAD_DIM, N),
        lambda x: T.reshape(x, (x.shape[0], -1, x.shape[3])),
        lambda x: x.reshape(x.shape[0], -1, x.shape[3]))
    arm("permute_021", (1, N_HEADS * HEAD_DIM, N),
        lambda x: T.permute(x, (0, 2, 1)),
        lambda x: x.permute(0, 2, 1))

    def dev_tail(x):
        o = x[:, :, :, :HEAD_DIM]
        o = T.permute(o, (0, 1, 3, 2))
        o = T.reshape(o, (o.shape[0], -1, o.shape[3]))
        return T.permute(o, (0, 2, 1))

    def ref_tail(x):
        o = x[:, :, :, :HEAD_DIM]
        o = o.permute(0, 1, 3, 2)
        o = o.reshape(o.shape[0], -1, o.shape[3])
        return o.permute(0, 2, 1)

    arm("composed_tail", (1, N_HEADS, N, PAD_DIM), dev_tail, ref_tail)

    for name, shape, dev_fn, ref_fn in arms:
        xt = torch.randn(*shape)
        xr = xt.double().clone().requires_grad_(True)
        yr = ref_fn(xr)
        gt = torch.randn(*yr.shape)
        yr.backward(gt.double())
        try:
            with device_dtype_override(act), ag.tape():
                xd = ag.Tensor(ft(xt), requires_grad=True)
                yd = dev_fn(xd)
                fwd = rel(to_t(yd), yr.detach())
                ag.backward([yd], [ft(gt)])
            gd = to_t(xd.grad)
            results[name] = {
                "shape": list(shape), "out_shape": list(yr.shape),
                "forward_rel": fwd,
                "backward_rel": rel(gd, xr.grad),
                "backward_norm_ratio": ratio(gd, xr.grad),
                "backward_cos": cosine(gd, xr.grad),
                "ref_grad_norm": float(xr.grad.norm()),
                "dev_grad_norm": float(gd.norm()),
            }
        except Exception as e:
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        r = results[name]
        if "error" in r:
            print(f"[{time.perf_counter()-t0:.0f}s] {name:<22} RAISED {r['error'][:80]}",
                  flush=True)
        else:
            print(f"[{time.perf_counter()-t0:.0f}s] {name:<22} fwd {r['forward_rel']:.3e} | "
                  f"bwd {r['backward_rel']:.3e}  r {r['backward_norm_ratio']:.6f}  "
                  f"cos {r['backward_cos']:.6f}", flush=True)

    rep = {"what": "unit gradcheck of the DiT attention tail against a float64 host reference, "
                   "per op and composed. These ops are pure data movement, so the forward is "
                   "expected bit-exact and any non-zero backward is a defect, not precision.",
           "tokens": N, "n_heads": N_HEADS, "head_dim": HEAD_DIM, "padded_head_dim": PAD_DIM,
           "act": a.act, "seed": a.seed, "bar": 5.0e-02, "results": results,
           "over_bar": sorted(k for k, v in results.items()
                              if "error" not in v and v["backward_rel"] > 5.0e-02)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True, default=str)
    print("\nover the 5.0e-02 bar:", rep["over_bar"] or "none")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
