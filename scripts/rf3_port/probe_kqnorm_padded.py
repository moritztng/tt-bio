#!/usr/bin/env python3
"""kq_norm with a non-tile-aligned head_dim (RF3's token DiT: 768/16 = 48).

Drives __call__'s own helper on a fused qkv laid out the way the class lays it out
(pad lanes zero), and checks three things the padded path can get wrong:
  * the norm reduces over n_heads*head_dim, not over the padded width
  * the pad lanes come back ZERO, not carrying layer-norm bias into every q.k
  * V is untouched
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias."


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    import ttnn
    from tt_bio.tenstorrent import AttentionPairBias, get_device

    H, D = 16, 48
    Dp = D + (-D % 32)
    W, Wp = H * D, H * Dp
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    w = {k[len(PREFIX):]: v.float() for k, v in sd.items() if k.startswith(PREFIX)}

    torch.manual_seed(0)
    # real (unpadded) per-head values; pad lanes are zero, as the class builds them
    real = torch.randn(1, args.n, H, D)
    q_p = torch.cat([real, torch.zeros(1, args.n, H, Dp - D)], -1).reshape(1, args.n, Wp)
    k_real = torch.randn(1, args.n, H, D)
    k_p = torch.cat([k_real, torch.zeros(1, args.n, H, Dp - D)], -1).reshape(1, args.n, Wp)
    v_p = torch.randn(1, args.n, Wp)
    qkv = torch.cat([q_p, k_p, v_p], dim=-1)

    def ref(x, pre):
        ln = torch.nn.LayerNorm((W,))
        ln.weight.data = w[f"{pre}.weight"].clone()
        ln.bias.data = w[f"{pre}.bias"].clone()
        ln.eval()
        with torch.no_grad():
            return ln(x.reshape(1, args.n, W))

    want_q = ref(real, "query_layer_norm")
    want_k = ref(k_real, "key_layer_norm")

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    stub = dict(w)
    stub.setdefault("proj_q.bias", torch.zeros(W))
    for k in ("proj_q.weight", "proj_k.weight", "proj_v.weight",
              "proj_g.weight", "proj_o.weight"):
        stub.setdefault(k, torch.zeros(W, W))
    stub.setdefault("proj_z.0.weight", torch.ones(128))
    stub.setdefault("proj_z.0.bias", torch.zeros(128))
    stub.setdefault("proj_z.1.weight", torch.zeros(H, 128))

    attn = AttentionPairBias(D, H, True, False, stub, cfg)
    attn._load_kq_norm()
    out = attn._apply_kq_norm(ttnn.from_torch(qkv, layout=ttnn.TILE_LAYOUT,
                                              device=dev, dtype=ttnn.bfloat16))
    got = torch.Tensor(ttnn.to_torch(out)).float().reshape(1, args.n, 3 * Wp)

    g_q = got[..., :Wp].reshape(1, args.n, H, Dp)
    g_k = got[..., Wp:2 * Wp].reshape(1, args.n, H, Dp)
    rep = {
        "head_dim": D, "padded_head_dim": Dp, "activated": bool(attn.kq_norm),
        "q_pcc": round(pcc(g_q[..., :D].reshape(1, args.n, W), want_q), 6),
        "k_pcc": round(pcc(g_k[..., :D].reshape(1, args.n, W), want_k), 6),
        "q_pad_lanes_zero": bool(float(g_q[..., D:].abs().max()) == 0.0),
        "k_pad_lanes_zero": bool(float(g_k[..., D:].abs().max()) == 0.0),
        "v_untouched_pcc": round(pcc(got[..., 2 * Wp:], v_p), 6),
    }
    rep["verdict"] = ("PASS" if rep["q_pcc"] > 0.999 and rep["k_pcc"] > 0.999
                      and rep["q_pad_lanes_zero"] and rep["k_pad_lanes_zero"]
                      and rep["v_untouched_pcc"] > 0.999 else "GAP")
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
