#!/usr/bin/env python3
"""The inside of one shipped token-level AttentionPairBias, fed the reference's own inputs.

The token-level site does NOT go through `AttentionPairBias._attention`: `__call__` computes
q@k^T, the bias add, the softmax and probs@v inline, which is also why `fp32_softmax` and
`accurate_softmax` are dead at this site. So the internals are read by patching the three
module-level entry points that branch uses -- `nlp_create_qkv_heads`, `batched_matmul` and
`ttnn.softmax` -- for the duration of one call, and restoring them after.

Recorded: q, k, v; the raw q@k^T; the softmax's input (the biased, scaled logits) and its
output; the attention output; and the module's own return value. A float64 recomputation from
the same q/k/v/logits is done by `attn_score.py`, so every stage has an exact reference.
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
    ap.add_argument("--ref", required=True)
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
    kwargs = dict(spy["kwargs"])
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

    sink, mm_calls = {}, []
    real_heads = ttnn.experimental.nlp_create_qkv_heads
    real_bmm = T.batched_matmul
    real_soft = ttnn.softmax

    def p_heads(*ar, **kw):
        out = real_heads(*ar, **kw)
        sink["q"], sink["k"], sink["v"] = (tt(x) for x in out)
        return out

    def p_bmm(x, y, **kw):
        o = real_bmm(x, y, **kw)
        mm_calls.append((tt(x), tt(y), tt(o)))
        return o

    def p_soft(x, *ar, **kw):
        o = real_soft(x, *ar, **kw)
        sink["logits"] = tt(x)
        sink["probs"] = tt(o)
        return o

    ttnn.experimental.nlp_create_qkv_heads = p_heads
    T.batched_matmul = p_bmm
    ttnn.softmax = p_soft
    try:
        sn_t, z5_t = ft(ref["shared"]["sn"]), ft(ref["shared"]["z5"])
        fed = {"sn": tt(sn_t), "z5": tt(z5_t)}
        attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
        u1 = tt(layer.attention_pair_bias(sn_t, z5_t, seq_mask=ft(attn)))
    finally:
        ttnn.experimental.nlp_create_qkv_heads = real_heads
        T.batched_matmul = real_bmm
        ttnn.softmax = real_soft

    if len(mm_calls) >= 2:
        sink["qk_raw"] = mm_calls[0][2]
        sink["attn_out"] = mm_calls[-1][2]
    nm = lambda x: float(x.norm())
    torch.save({"block": a.block, "config": kwargs, "fed": fed, "u1": u1, "sink": sink,
                "n_matmuls": len(mm_calls)}, a.out)
    rep = {"block": a.block, "config": kwargs, "overrides": a.set, "n_matmuls": len(mm_calls),
           "u1_norm": nm(u1),
           "norms": {k: nm(v) for k, v in sink.items()},
           "shapes": {k: list(v.shape) for k, v in sink.items()},
           "absmax": {k: float(v.abs().max()) for k, v in sink.items()},
           "seconds": time.perf_counter() - t0}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
