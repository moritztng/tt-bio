#!/usr/bin/env python3
"""Each s-side op of one shipped PairformerLayer, fed the REFERENCE's own inputs.

`dev_block.py` feeds a whole block one state and reports every leaf, which attributes a leaf's
error to the leaf plus everything upstream of it inside the block. This closes that gap: the
AttentionPairBias is handed the reference's LayerNorm'd single track and the reference's pair
output, and the single transition is handed the reference's post-attention single track, both
rounded to the bf16 the device holds and nothing else. A leaf that is still wrong here is wrong
on its own.

It also records what crosses `AttentionPairBias._attention` -- q, k, v, the assembled bias and
the attention output -- so the op can be split into the projection, the attention and the
gated output projection.
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
    ap.add_argument("--ref", required=True, help="REF64_b<idx>.pt, the f64 reference arm")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[])
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
    shipped_kwargs = dict(spy["kwargs"])
    kwargs = dict(shipped_kwargs)
    for kv in a.set:
        k, _, v = kv.partition("=")
        kwargs[k] = {"true": True, "false": False}.get(v.lower(), v)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    ref = torch.load(a.ref, map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(pm.shape[1])

    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    pre = f"layers.{a.block}."
    flat = {"layers.0." + k[len(pre):]: v for k, v in flat_all.items() if k.startswith(pre)}
    mod = T.Pairformer(1, *spy["dims"], spy["transform_s"], flat, ckc, **kwargs)
    layer = mod.blocks[0]

    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    tt = lambda x: ttnn.to_torch(x).to(torch.float64)

    # record what crosses _attention; the method is looked up on the instance, so an instance
    # attribute is enough and the shipped __call__ runs unchanged
    apb = layer.attention_pair_bias
    sink = {}
    inner = apb._attention

    def rec_attention(q, k, v, bias):
        sink["q"], sink["k"], sink["v"], sink["bias"] = tt(q), tt(k), tt(v), tt(bias)
        o = inner(q, k, v, bias)
        sink["attn_out"] = tt(o)
        return o

    apb._attention = rec_attention

    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
    sn_ref, z5_ref, s1_ref = ref["shared"]["sn"], ref["shared"]["z5"], ref["shared"]["s1"]
    sn_t, z5_t = ft(sn_ref), ft(z5_ref)
    fed = {"sn": tt(sn_t), "z5": tt(z5_t)}
    u1 = tt(layer.attention_pair_bias(sn_t, z5_t, seq_mask=ft(attn)))

    s1_t = ft(s1_ref)
    fed["s1"] = tt(s1_t)
    u2 = tt(layer.transition_s(s1_t))

    nm = lambda x: float(x.norm())
    out = {"block": a.block, "config": kwargs, "shipped_config": shipped_kwargs,
           "overrides": a.set, "fed": fed, "u1": u1, "u2": u2, "attn": sink}
    torch.save(out, a.out)
    rep = {"block": a.block, "config": kwargs, "overrides": a.set,
           "fed_norms": {k: nm(v) for k, v in fed.items()},
           "u1_norm": nm(u1), "u2_norm": nm(u2),
           "attn_norms": {k: nm(v) for k, v in sink.items()},
           "attn_shapes": {k: list(v.shape) for k, v in sink.items()},
           "seconds": time.perf_counter() - t0}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
