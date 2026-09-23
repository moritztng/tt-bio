#!/usr/bin/env python3
"""The shipped tt-bio trunk Pairformer over the captured boundary, with named kwarg overrides.

`of3t-trunk043ref/device_arm.py` runs the shipped configuration and the transpose_bias lever.
This runs the same stack with any Pairformer kwarg overridden, so the pair-bias scale
convention can be measured end to end without editing the construction site: the OF3 trunk
default is deliberately `scale_pair_bias=False` and `compose_verify.sh` asserts it stays that
way, so the override lives in the harness and never in the tree.

The shipped configuration is still read off `OF3Trunk`'s own construction by spying on the
Pairformer constructor, and both the shipped and the overridden kwargs are recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import time

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--permute-input", action="store_true")
    ap.add_argument("--break-seed", type=int, default=20260920)
    a = ap.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    import tt_bio.openfold3_trunk as OT

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    spy = {}

    class _Stop(Exception):
        pass

    def _spy(*ar, **kw):
        spy["n_blocks"] = ar[0]
        spy["dims"] = list(ar[1:5])
        spy["transform_s"] = ar[5]
        spy["kwargs"] = {k: (v if isinstance(v, (bool, int, float, str, type(None))) else str(v))
                         for k, v in kw.items()}
        raise _Stop()

    real_pf = OT.Pairformer
    OT.Pairformer = _spy
    try:
        OT.OF3Trunk(sd, ckc)
    except _Stop:
        pass
    finally:
        OT.Pairformer = real_pf
    if "kwargs" not in spy:
        raise SystemExit("the spy never reached Pairformer")
    shipped = dict(spy["kwargs"])
    kwargs = dict(shipped)
    for kv in a.set:
        k, _, v = kv.partition("=")
        kwargs[k] = {"true": True, "false": False}.get(v.lower(), v)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    s_in, z_in = b["s_in"], b["z_in"]
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(z_in.shape[1])
    if a.permute_input:
        gen = torch.Generator().manual_seed(a.break_seed)
        real = int(sm.sum())
        perm = torch.randperm(real, generator=gen)
        idx = torch.arange(N)
        idx[:real] = perm
        s_in = s_in[:, idx].contiguous()
        z_in = z_in[:, idx][:, :, idx].contiguous()

    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    flat = (flat_all if a.blocks == spy["n_blocks"]
            else {k: v for k, v in flat_all.items() if int(k.split(".")[1]) < a.blocks})
    mod = T.Pairformer(a.blocks, *spy["dims"], spy["transform_s"], flat, ckc, **kwargs)

    s_dtype = ttnn.float32 if kwargs.get("s_fp32_residual", False) else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
    so, zo = mod(fts(s_in), ft(z_in), ft(pm), ft(attn), ft(attn))
    s, z = ttnn.to_torch(so).to(torch.float64), ttnn.to_torch(zo).to(torch.float64)

    torch.save({"s": s, "z": z, "config": kwargs, "shipped_config": shipped,
                "overrides": a.set, "permuted": bool(a.permute_input)}, a.out)
    rep = {"blocks": a.blocks, "config": kwargs, "shipped_config": shipped,
           "overrides": a.set, "permuted": bool(a.permute_input),
           "tokens": N, "real_tokens": int(sm.sum()),
           "s_norm": float(s.norm()), "z_norm": float(z.norm()),
           "seconds": time.perf_counter() - t0}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
