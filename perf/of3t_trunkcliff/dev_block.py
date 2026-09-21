#!/usr/bin/env python3
"""ONE shipped tt-bio PairformerLayer on device, by block index, over a supplied state.

The decomposition is taken by WRAPPING the layer's submodules, not by reimplementing
`PairformerLayer.__call__`: the shipped call runs, and each wrapper records its own inputs and
outputs on the way past. A reimplementation would be a different function from the one being
explained, and the whole point of this row is that the difference lives inside a block.

Recorded, in the order the shipped call makes them:

  z0..z4   the pair track as each sub-module sees it (the wrapper's input)
  dz1..dz5 each sub-module's update (the wrapper's output)
  sn       the LayerNorm of the single track, as AttentionPairBias receives it
  u1       the AttentionPairBias update
  s1       the single track as the transition receives it
  u2       the transition update

`s2` and `z5` are the layer's own return values, so the decomposition cannot disagree with the
shipped output by construction; the residual chain is checked against them anyway.

The input state is cast to whatever dtype the shipped construction asks for (bfloat16 unless
s_fp32_residual is on), which is the same quantisation the `--quantise-input` reference arm
applies, so the two are fed the same bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


class Rec:
    """Transparent wrapper: forwards the call, records the tensors that crossed it."""

    def __init__(self, inner, name, sink, to_torch):
        self._inner, self._name, self._sink, self._tt = inner, name, sink, to_torch

    def __call__(self, *a, **kw):
        for i, x in enumerate(a):
            if hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "layout"):
                self._sink[f"{self._name}.in{i}"] = self._tt(x)
        out = self._inner(*a, **kw)
        if isinstance(out, tuple):
            for i, x in enumerate(out):
                self._sink[f"{self._name}.out{i}"] = self._tt(x)
        else:
            self._sink[f"{self._name}.out"] = self._tt(out)
        return out

    def __getattr__(self, k):
        return getattr(self._inner, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--permute-input", action="store_true",
                    help="break control: permute the real token positions of the fed state")
    ap.add_argument("--break-seed", type=int, default=20260920)
    ap.add_argument("--set", action="append", default=[],
                    help="override one shipped Pairformer kwarg, key=true|false|<number>; "
                         "the shipped value is still read off OF3Trunk and recorded beside it")
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

    # The SHIPPED configuration, read off OF3Trunk's own construction, never a hardcoded copy.
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
    shipped_kwargs = dict(spy["kwargs"])
    kwargs = dict(shipped_kwargs)
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in kwargs:
            raise SystemExit(f"{k} is not a shipped Pairformer kwarg: {sorted(kwargs)}")
        kwargs[k] = {"true": True, "false": False}.get(v.lower(), v)
        if isinstance(kwargs[k], str):
            kwargs[k] = float(v) if "." in v else int(v)

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    st = torch.load(a.state, map_location="cpu", weights_only=False)
    sm, pm = b["single_mask"], b["pair_mask"]
    N = int(pm.shape[1])
    s_in, z_in = st["s"].to(torch.float64), st["z"].to(torch.float64)
    if a.permute_input:
        g = torch.Generator().manual_seed(a.break_seed)
        real = int(sm.sum())
        perm = torch.randperm(real, generator=g)
        idx = torch.arange(N)
        idx[:real] = perm
        s_in = s_in[:, idx].contiguous()
        z_in = z_in[:, idx][:, :, idx].contiguous()

    # one block's weights, re-indexed to layers.0 so a 1-block Pairformer carries block `idx`
    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    pre = f"layers.{a.block}."
    flat = {"layers.0." + k[len(pre):]: v for k, v in flat_all.items() if k.startswith(pre)}
    if not flat:
        raise SystemExit(f"no weights for block {a.block}")

    mod = T.Pairformer(1, *spy["dims"], spy["transform_s"], flat, ckc, **kwargs)
    layer = mod.blocks[0]

    s_fp32 = bool(kwargs.get("s_fp32_residual", False))
    s_dtype = ttnn.float32 if s_fp32 else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
    fts = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=s_dtype)
    tt = lambda x: ttnn.to_torch(x).to(torch.float64)

    sink: dict = {}
    for attr, name in (("triangle_multiplication_start", "tri_mul_out"),
                       ("triangle_multiplication_end", "tri_mul_in"),
                       ("triangle_attention_start", "tri_att_start"),
                       ("triangle_attention_end", "tri_att_end"),
                       ("transition_z", "transition_z"),
                       ("attention_pair_bias", "attn_pair_bias"),
                       ("transition_s", "transition_s")):
        if hasattr(layer, attr):
            setattr(layer, attr, Rec(getattr(layer, attr), name, sink, tt))

    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
    s_t, z_t = fts(s_in), ft(z_in)
    s_fed, z_fed = tt(s_t), tt(z_t)          # exactly the bytes the device was handed
    so, zo = layer(s_t, ft(z_in), ft(pm), ft(attn), ft(attn))
    s_out, z_out = tt(so), tt(zo)

    # the shared ten, assembled from what crossed the wrappers
    sh = {}
    sh["z1"] = sink["tri_mul_out.in0"] + sink["tri_mul_out.out"]
    sh["z2"] = sink["tri_mul_in.in0"] + sink["tri_mul_in.out"]
    sh["z3"] = sink["tri_att_start.in0"] + sink["tri_att_start.out"]
    sh["z4"] = sink["tri_att_end.in0"] + sink["tri_att_end.out"]
    sh["z5"] = sink["transition_z.in0"] + sink["transition_z.out"]
    sh["sn"] = sink["attn_pair_bias.in0"]
    sh["u1"] = sink["attn_pair_bias.out"]
    sh["s1"] = sink["transition_s.in0"]
    sh["u2"] = sink["transition_s.out"]
    sh["s2"] = s_out

    nm = lambda x: float(x.norm())
    chain = {
        # each sub-module's input must be the previous one's residual sum, bit for bit
        "z2_in_vs_z1": nm(sink["tri_mul_in.in0"] - sh["z1"]),
        "z3_in_vs_z2": nm(sink["tri_att_start.in0"] - sh["z2"]),
        "z4_in_vs_z3": nm(sink["tri_att_end.in0"] - sh["z3"]),
        "z5_in_vs_z4": nm(sink["transition_z.in0"] - sh["z4"]),
        "z_out_vs_z5": nm(z_out - sh["z5"]) / max(nm(z_out), 1e-300),
        "s_out_vs_s1_plus_u2": nm(s_out - (sh["s1"] + sh["u2"])) / max(nm(s_out), 1e-300),
    }

    torch.save({"block": a.block, "config": kwargs, "s_fed": s_fed, "z_fed": z_fed,
                "s_out": s_out, "z_out": z_out, "shared": sh, "sink": sink,
                "chain": chain, "permuted": bool(a.permute_input),
                "shipped_config": shipped_kwargs}, a.out)
    rep = {"block": a.block, "config": kwargs, "shipped_config": shipped_kwargs,
           "overrides": a.set, "s_input_dtype": str(s_dtype),
           "permuted": bool(a.permute_input), "tokens": N, "real_tokens": int(sm.sum()),
           "state": a.state, "chain": chain,
           "shared_norms": {k: nm(v) for k, v in sh.items()},
           "s_fed_norm": nm(s_fed), "z_fed_norm": nm(z_fed),
           "s_out_norm": nm(s_out), "z_out_norm": nm(z_out),
           "seconds": time.perf_counter() - t0}
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({"block": a.block, "s_out_norm": rep["s_out_norm"],
                      "z_out_norm": rep["z_out_norm"], "chain": chain}, indent=2))
    print(json.dumps(rep["shared_norms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
