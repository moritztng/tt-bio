#!/usr/bin/env python3
"""Score the recorded AttentionPairBias internals against a float64 recomputation.

The float64 side is tt-bio's OWN formula, in the same padded-head-lane layout the device uses,
from the same fed sn and z5 and the same checkpoint weights. Comparing the device against this
answers "is the op imprecise", separately from "is the formula right", which the block-level
arms against upstream already answer.

Norm ratio and error cosine beside every relative L2. The three attention stages are reported
with the dynamic range of each one beside them, because the hypothesis under test is a
fixed-scale bf16 step whose headroom runs out.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def metrics(a, r):
    a = a.reshape(-1).double()
    r = r.reshape(-1).double()
    na, nr = a.norm().item(), r.norm().item()
    d = nr if nr > 0 else 1e-300
    return {"rel": ((a - r).norm() / d).item(), "ratio": na / d,
            "cos": ((a @ r) / ((na * nr) or 1e-300)).item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P = torch.load(a.probe, map_location="cpu", weights_only=False)
    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    sm = b["single_mask"]
    N = int(sm.shape[1])
    sn = P["fed"]["sn"].double()
    z5 = P["fed"]["z5"].double()
    S = P["sink"]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("state_dict", sd)
    pre = f"pairformer_stack.blocks.{a.block}.attn_pair_bias."
    g = lambda k: sd[pre + k].double()
    Wq, Wk, Wv = g("mha.linear_q.weight"), g("mha.linear_k.weight"), g("mha.linear_v.weight")
    bq = g("mha.linear_q.bias")
    Wz, lnz_w, lnz_b = g("linear_z.weight"), g("layer_norm_z.weight"), g("layer_norm_z.bias")
    nh, cs = Wz.shape[0], Wq.shape[0]
    hd = cs // nh
    phd = hd + (-hd % 32)

    def heads(x, bias=None):
        y = sn.reshape(-1, cs) @ x.t()
        if bias is not None:
            y = y + bias
        y = y.reshape(-1, nh, hd)
        z = torch.zeros(y.shape[0], nh, phd - hd, dtype=y.dtype)
        return torch.cat([y, z], dim=-1).permute(1, 0, 2).unsqueeze(0)

    q64, k64, v64 = heads(Wq, bq), heads(Wk), heads(Wv)
    zn = torch.nn.functional.layer_norm(z5, (z5.shape[-1],), lnz_w, lnz_b, 1e-5)
    bias64 = (zn @ Wz.t()).permute(0, 3, 1, 2)
    seq = ((1.0 - sm.double()).reshape(1, 1, 1, N)) * -1e9
    qk64 = q64 @ k64.transpose(-1, -2)
    logits64 = (qk64 + bias64 + seq) * hd ** -0.5
    probs64 = torch.softmax(logits64, dim=-1)
    out64 = probs64 @ v64

    # the gate and the output projection, in the device's padded-lane layout
    def relane_out(t):                       # proj_g: output axis is head-major
        x = t.reshape(*t.shape[:-1], nh, hd)
        return torch.nn.functional.pad(x, (0, phd - hd)).reshape(*t.shape[:-1], nh * phd)

    def relane_in(t):                        # proj_o: input axis is head-major
        x = t.reshape(nh, hd, *t.shape[1:])
        return torch.nn.functional.pad(x, (0, 0, 0, phd - hd)).reshape(nh * phd, *t.shape[1:])

    Wg = relane_out(g("mha.linear_g.weight").t())
    Wo = relane_in(g("mha.linear_o.weight").t())

    def tail(o):
        oc = o.permute(0, 2, 1, 3).reshape(1, -1, nh * phd)
        return (oc * torch.sigmoid(sn.reshape(1, -1, cs) @ Wg)) @ Wo

    u1_64 = tail(out64)
    u1_from_dev_attn = tail(S["attn_out"])   # f64 tail on the DEVICE's attention output

    rep = {"block": a.block, "head_dim": hd, "padded_head_dim": phd, "n_heads": nh,
           "stages": {}, "range": {}, "tail": {}}
    rep["tail"]["u1_device_vs_f64"] = metrics(P["u1"], u1_64)
    rep["tail"]["u1_f64tail_on_device_attn_vs_f64"] = metrics(u1_from_dev_attn, u1_64)
    rep["tail"]["u1_device_vs_f64tail_on_device_attn"] = metrics(P["u1"], u1_from_dev_attn)
    # how much cancellation the output projection does: the row-wise sum of |term| against
    # the magnitude that survives it
    oc = (out64.permute(0, 2, 1, 3).reshape(1, -1, nh * phd)
          * torch.sigmoid(sn.reshape(1, -1, cs) @ Wg))
    terms = oc.abs() @ Wo.abs()
    rep["tail"]["cancellation_median"] = float((terms / u1_64.abs().clamp(min=1e-300)).median())
    rep["tail"]["norms"] = {"attn_out": float(out64.norm()), "gated": float(oc.norm()),
                            "u1": float(u1_64.norm())}
    pairs = [("q", q64), ("k", k64), ("v", v64), ("qk_raw", qk64),
             ("logits", logits64), ("probs", probs64), ("attn_out", out64)]
    for name, r in pairs:
        if name not in S:
            continue
        rep["stages"][name] = metrics(S[name], r)

    # the dynamic range each bf16 stage has to hold, and what the softmax actually needs
    real = int(sm.sum())
    lg = logits64[..., :real, :real]
    rows_max = lg.amax(dim=-1, keepdim=True)
    spread = (rows_max - lg.amin(dim=-1, keepdim=True))
    rep["range"] = {
        "qk_raw_absmax": float(qk64[..., :real, :real].abs().max()),
        "bias_absmax": float(bias64[..., :real, :real].abs().max()),
        "logits_absmax_real": float(lg.abs().max()),
        "logits_row_spread_median": float(spread.median()),
        "logits_row_max_median": float(rows_max.median()),
        # bf16 keeps 8 explicit mantissa bits: the step at a row's own scale, against the
        # spread that decides the softmax. A ratio near or above 1 means the distribution the
        # softmax sees is quantisation noise.
        "bf16_ulp_at_row_max_median": float((2.0 ** (torch.floor(torch.log2(
            rows_max.abs().clamp(min=1e-300))) - 7)).median()),
    }
    rep["range"]["ulp_over_spread"] = (rep["range"]["bf16_ulp_at_row_max_median"]
                                       / max(rep["range"]["logits_row_spread_median"], 1e-300))
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"=== block {a.block}: device vs float64 of tt-bio's own formula ===")
    for k, v in rep["stages"].items():
        print(f"  {k:>9}  rel {v['rel']:.4e}  ratio {v['ratio']:.6f}  cos {v['cos']:.6f}")
    print("  range:", json.dumps(rep["range"], indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
