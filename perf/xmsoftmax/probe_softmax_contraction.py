#!/usr/bin/env python3
"""What survives `probs @ v`: the row-sum deficit, or the per-element error?

rf3-port-p3's root cause is that `ttnn.softmax`'s missing 2.4% of row mass is a MULTIPLICATIVE
error on every weight in a row, so it does not cancel in the contraction with v, while an
error field of the same size that is not a uniform deficit largely does (the reference's own
bf16 probs error was 0.095 and collapsed to 0.0038 at `probs @ v`).

If that is the mechanism, then renormalising the fused output -- 2 extra ops, not the manual
chain's 5 -- should recover most of the win, even though it scores badly on rel_rms(probs).
rf3 rejected renorm on rel_rms(probs) = 0.009879. This scores every arm on rel_rms(o) instead,
which is the quantity the model actually propagates.

Softmax runs on device; the contraction runs on host in fp64 against the same v for every arm,
so the only thing that differs between arms is the softmax.
"""
import argparse, json, os
import torch
import ttnn


def rel_rms(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,16,512,512")
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--spread", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    B, H, S, K = [int(x) for x in a.shape.split(",")]
    torch.manual_seed(a.seed)
    logits = (torch.rand(B, H, S, K, dtype=torch.float64) * a.spread) - a.spread / 2
    v = torch.randn(B, H, K, a.head_dim, dtype=torch.float64)
    p_ref = torch.softmax(logits, dim=-1)
    o_ref = p_ref @ v

    res = {"shape": [B, H, S, K], "head_dim": a.head_dim, "spread": a.spread, "arms": {}}

    def score(name, p):
        p = p.to(torch.float64)
        rs = p.sum(-1)
        res["arms"][name] = {
            "rowsum_mean": float(rs.mean()), "rowsum_min": float(rs.min()),
            "rel_rms_probs": rel_rms(p, p_ref),
            "rel_rms_o": rel_rms(p @ v, o_ref),
        }

    # host ceiling: a CORRECT softmax evaluated in bf16. This is the "reference bf16" arm rf3
    # used as the accuracy ceiling -- an error field of the same magnitude that is not a deficit.
    score("host_bf16_correct", torch.softmax(logits.to(torch.bfloat16).to(torch.float32),
                                             dim=-1).to(torch.float64))

    dev = ttnn.open_device(device_id=0)
    try:
        x = ttnn.from_torch(logits.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.float32)
        score("fused", ttnn.to_torch(ttnn.softmax(x, dim=-1)))

        f = ttnn.softmax(x, dim=-1)
        score("fused_renorm", ttnn.to_torch(ttnn.divide(f, ttnn.sum(f, dim=-1, keepdim=True))))

        m = ttnn.max(x, dim=-1, keepdim=True)
        d = ttnn.subtract(x, m)
        ttnn.exp(d, output_tensor=d)
        score("manual", ttnn.to_torch(ttnn.divide(d, ttnn.sum(d, dim=-1, keepdim=True))))
    finally:
        ttnn.close_device(dev)

    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
