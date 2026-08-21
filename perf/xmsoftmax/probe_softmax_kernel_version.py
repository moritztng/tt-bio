#!/usr/bin/env python3
"""Is the `ttnn.softmax` mass deficit still there in a newer ttnn?

rf3-port-p3 measured `ttnn.softmax` on [1,16,512,512] fp32 returning rows that sum to
0.9769 (min 0.9613) against a fp64 softmax, on ttnn 0.67.4. If a later ttnn fixed the
kernel, the cross-model answer is a version bump, not a 4x manual chain. Card-free of
nothing: it opens one device, runs one op per variant, and reports rowsum + rel_rms.

Standalone on purpose (no tt_bio import), so it runs unchanged under a scout venv whose
ttnn does not match the repo pin.
"""
import argparse, json, os, sys
import torch
import ttnn


def rel_rms(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,16,512,512")
    ap.add_argument("--spread", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    shape = [int(x) for x in a.shape.split(",")]
    torch.manual_seed(a.seed)
    logits = (torch.rand(shape, dtype=torch.float64) * a.spread) - a.spread / 2
    ref = torch.softmax(logits, dim=-1)
    lg32 = logits.to(torch.float32)

    dev = ttnn.open_device(device_id=0)
    try:
        x = ttnn.from_torch(lg32, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
        out = {"ttnn_version": getattr(ttnn, "__version__", "?"), "shape": shape,
               "spread": a.spread, "variants": {}}

        fused = ttnn.to_torch(ttnn.softmax(x, dim=-1)).to(torch.float64)
        rs = fused.sum(-1)
        out["variants"]["fused"] = {"rowsum_mean": float(rs.mean()),
                                    "rowsum_min": float(rs.min()),
                                    "rel_rms_vs_fp64": rel_rms(fused, ref)}

        # renormalise the fused output: 2 extra ops, not 5. rf3-port-p3 measured 0.009879 on its
        # own logits, 2.8x better than fused and 28x worse than the manual chain. It is the cheap
        # middle option, and the arm that decides a model whose needed win is small.
        f2 = ttnn.softmax(x, dim=-1)
        rsum = ttnn.sum(f2, dim=-1, keepdim=True)
        ren = ttnn.to_torch(ttnn.divide(f2, rsum)).to(torch.float64)
        rs = ren.sum(-1)
        out["variants"]["fused_renorm"] = {"rowsum_mean": float(rs.mean()),
                                           "rowsum_min": float(rs.min()),
                                           "rel_rms_vs_fp64": rel_rms(ren, ref)}

        m = ttnn.max(x, dim=-1, keepdim=True)
        d = ttnn.subtract(x, m)
        ttnn.exp(d, output_tensor=d)
        s = ttnn.sum(d, dim=-1, keepdim=True)
        man = ttnn.to_torch(ttnn.divide(d, s)).to(torch.float64)
        rs = man.sum(-1)
        out["variants"]["manual"] = {"rowsum_mean": float(rs.mean()),
                                     "rowsum_min": float(rs.min()),
                                     "rel_rms_vs_fp64": rel_rms(man, ref)}
    finally:
        ttnn.close_device(dev)

    print(json.dumps(out, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
