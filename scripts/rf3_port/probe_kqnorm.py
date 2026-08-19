#!/usr/bin/env python3
"""Verify the optional kq_norm path against the reference's own formulation.

A shared-code change that passes the other models' tests is only proven not to
break them. This proves it does the right thing for RF3: it drives the new path
with RF3's real query/key layer-norm weights and compares against

    Q = query_layer_norm(Q.reshape(-1, n_head * c)).reshape(Q.shape)
    K = key_layer_norm(K.reshape(-1, n_head * c)).reshape(K.shape)

taken verbatim from AttentionPairBiasDiffusion.forward.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/probe_kqnorm.py --ckpt ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PREFIX = ("shadow.feature_initializer.input_feature_embedder.atom_attention_encoder"
          ".atom_transformer.diffusion_transformer.blocks.0.attention_pair_bias.")
N_HEADS, HEAD_DIM = 4, 32


def pcc(a, b) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import ttnn

    from tt_bio.tenstorrent import AttentionPairBias, get_device

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    w = {k[len(PREFIX):]: v.float() for k, v in sd.items() if k.startswith(PREFIX)}
    if "query_layer_norm.weight" not in w:
        print("no kq_norm weights at that prefix")
        return 1

    width = N_HEADS * HEAD_DIM
    torch.manual_seed(args.seed)
    qkv = torch.randn(1, args.n, 3 * width)

    # reference: norm the Q and K slices over n_head * c, leave V alone
    def ref_norm(part, prefix):
        ln = torch.nn.LayerNorm((width,))
        ln.weight.data = w[f"{prefix}.weight"].clone()
        ln.bias.data = w[f"{prefix}.bias"].clone()
        ln.eval()
        with torch.no_grad():
            return ln(part)

    want = torch.cat([
        ref_norm(qkv[..., :width], "query_layer_norm"),
        ref_norm(qkv[..., width:2 * width], "key_layer_norm"),
        qkv[..., 2 * width:],
    ], dim=-1)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # Build the block just to own the weights, then drive the new path directly.
    stub = dict(w)
    stub.setdefault("proj_q.weight", torch.zeros(width, width))
    stub.setdefault("proj_q.bias", torch.zeros(width))
    for k in ("proj_k.weight", "proj_v.weight", "proj_g.weight", "proj_o.weight"):
        stub.setdefault(k, torch.zeros(width, width))
    stub.setdefault("proj_z.0.weight", torch.ones(16))
    stub.setdefault("proj_z.0.bias", torch.zeros(16))
    stub.setdefault("proj_z.1.weight", torch.zeros(N_HEADS, 16))

    attn = AttentionPairBias(HEAD_DIM, N_HEADS, True, False, stub, cfg)
    attn._load_kq_norm()
    if not attn.kq_norm:
        print("kq_norm did not activate despite weights being present")
        return 1

    got_tt = attn._apply_kq_norm(ttnn.from_torch(
        qkv, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16))
    got = torch.Tensor(ttnn.to_torch(got_tt)).float().reshape(want.shape)

    # --- atom-level branch: q and kv are separate tensors, not one fused qkv ---
    # This is the branch RF3's atom transformer actually takes, and the first cut of
    # kq_norm missed it entirely because this probe drove the helper rather than the
    # code path. Score it explicitly.
    B, K, W, ATOM_DIM = 1, 2, 32, 128
    q_t = torch.randn(B, K, W, width)
    kv_t = torch.randn(B, K, ATOM_DIM, 2 * width)
    want_q = ref_norm(q_t, "query_layer_norm")
    want_kv = torch.cat(
        [ref_norm(kv_t[..., :width], "key_layer_norm"), kv_t[..., width:]], dim=-1)

    def to_tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    got_q_tt, got_kv_tt = attn._apply_kq_norm_atom(to_tt(q_t), to_tt(kv_t))
    got_q = torch.Tensor(ttnn.to_torch(got_q_tt)).float().reshape(want_q.shape)
    got_kv = torch.Tensor(ttnn.to_torch(got_kv_tt)).float().reshape(want_kv.shape)

    diff = (got - want).abs()
    rep = {
        "tokens": args.n,
        "activated": bool(attn.kq_norm),
        "pcc": round(pcc(got, want), 6),
        "maxabs": round(float(diff.max()), 6),
        "rel_rms": round(float(diff.pow(2).mean().sqrt() / want.std()), 6),
        # the V slice must be untouched
        "v_slice_pcc": round(pcc(got[..., 2 * width:], qkv[..., 2 * width:]), 6),
        "atom_q_pcc": round(pcc(got_q, want_q), 6),
        "atom_kv_pcc": round(pcc(got_kv, want_kv), 6),
        # the V half of kv must be untouched
        "atom_v_half_pcc": round(pcc(got_kv[..., width:], kv_t[..., width:]), 6),
    }
    rep["verdict"] = "PASS" if min(
        rep["pcc"], rep["v_slice_pcc"], rep["atom_q_pcc"],
        rep["atom_kv_pcc"], rep["atom_v_half_pcc"]) > 0.999 else "GAP"
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
