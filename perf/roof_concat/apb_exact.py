#!/usr/bin/env python3
"""Is the one-op head re-assembly the same transform, and how far is each arm from the truth?

Two AttentionPairBias modules from the SAME torch weights in one process, one built with
TT_BIO_APB_CONCAT_HEADS on and one with it off, fed the same activations. A head re-assembly
moves bytes and computes nothing, so the first question is `torch.equal`. The pad lanes the
fused arm keeps widen the gate and the output projection from n_heads*head_dim to
n_heads*padded_head_dim, which regroups the K axis of the projection matmul, so equality is
expected but not guaranteed -- hence the float64 reference underneath both arms. Never compare
one approximation against the other and call the difference an error.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T

HERE = Path(__file__).resolve().parent


def torch_ref(w, a64, z64, n_heads, head_dim, dtype):
    """The unit as written, in float64: qkv, biased softmax attention, gate, projection."""
    a = a64.to(dtype)
    z = z64.to(dtype)
    S = a.shape[-2]
    q = (a @ w["proj_q.weight"].to(dtype).t() + w["proj_q.bias"].to(dtype))
    k = a @ w["proj_k.weight"].to(dtype).t()
    v = a @ w["proj_v.weight"].to(dtype).t()
    def heads(x):
        return x.reshape(1, S, n_heads, head_dim).permute(0, 2, 1, 3)
    q, k, v = heads(q), heads(k), heads(v)
    logits = (q @ k.transpose(-1, -2) + z) * head_dim ** -0.5
    o = torch.softmax(logits, dim=-1) @ v
    o = o.permute(0, 2, 1, 3).reshape(1, S, n_heads * head_dim)
    g = torch.sigmoid(a @ w["proj_g.weight"].to(dtype).t())
    return (o * g) @ w["proj_o.weight"].to(dtype).t()


def metrics(x, ref):
    d = (x - ref).flatten()
    r = ref.flatten()
    rel = (d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()
    xc, rc = x.flatten() - x.mean(), r - r.mean()
    pcc = (xc @ rc / (xc.norm() * rc.norm())).item()
    return {"rel_rms": rel, "pcc": pcc, "max_abs": d.abs().max().item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "apb_exact.json")
    ap.add_argument("--seq", type=int, nargs="+", default=[320, 512])
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=16)
    a = ap.parse_args()

    dev = T.get_device()
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    dim, H = a.dim, a.heads
    hd = dim // H
    torch.manual_seed(0)
    w = {f"proj_{n}.weight": torch.randn(dim, dim, dtype=torch.float32) * 0.02
         for n in ("q", "k", "v", "g", "o")}
    w["proj_q.bias"] = torch.randn(dim, dtype=torch.float32) * 0.02

    rows = []
    for on in (False, True):
        T._APB_CONCAT_HEADS = on
        m = T.AttentionPairBias(head_dim=hd, n_heads=H, compute_pair_bias=False,
                                atom_level=False, state_dict=dict(w),
                                compute_kernel_config=kc)
        m.token_dit = True
        rows.append(m)
    ship, cat = rows
    assert not ship._concat_heads and cat._concat_heads, (ship._concat_heads, cat._concat_heads)

    out = {"host": platform.node(), "arch": str(dev.arch()), "dim": dim, "heads": H,
           "head_dim": hd, "padded_head_dim": cat.padded_head_dim,
           "token_dit_sdpa": T._B2_TOKEN_DIT_SDPA, "seqs": []}
    for S in a.seq:
        torch.manual_seed(S)
        a_t = torch.randn(1, S, dim, dtype=torch.float32) * 0.5
        z_t = torch.randn(1, H, S, S, dtype=torch.float32) * 0.1
        a_d = ttnn.from_torch(a_t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        z_d = ttnn.from_torch(z_t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        got = {}
        for name, mod in (("ship", ship), ("concat", cat)):
            r = mod(a_d, z_d)
            got[name] = ttnn.to_torch(r).to(torch.float32)
            ttnn.deallocate(r)
        ttnn.deallocate(a_d); ttnn.deallocate(z_d)
        # the reference reads exactly the bf16 the device read, so the only difference
        # measured is arithmetic, not the input rounding
        a64 = a_t.to(torch.bfloat16).to(torch.float64)
        z64 = z_t.to(torch.bfloat16).to(torch.float64)
        ref = torch_ref(w, a64, z64, H, hd, torch.float64)
        rec = {"S": S, "bit_exact": bool(torch.equal(got["ship"], got["concat"])),
               "arm_vs_arm": metrics(got["concat"].to(torch.float64),
                                     got["ship"].to(torch.float64))}
        for n in ("ship", "concat"):
            rec[n] = metrics(got[n].to(torch.float64), ref)
        out["seqs"].append(rec)
        print(json.dumps(rec), flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
