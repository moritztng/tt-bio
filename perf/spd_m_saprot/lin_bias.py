"""Signed bias of a SaProt projection's bf16 output against float64: mean((got - ref) * sign(ref)) / mean(|ref|).
Round-to-nearest gives ~0; truncation toward zero gives about -2^-9 (-0.00195). Arms: the output dtype (bf16, fp32) at
the model's compute config, normal (HiFi4) and --fast (bfp8 weights, HiFi3), on the qkv shape of the page batch.

    python3 perf/spd_m_saprot/lin_bias.py --out lin_bias.jsonl
"""
import argparse
import json

import torch
import ttnn


def bias(got, ref):
    return float(((got - ref) * ref.sign()).mean() / ref.abs().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true", help="also K in (32, 256, 1280, 5120) x HiFi4/HiFi2/LoFi x packer_l1_acc, fp32 out")
    a = ap.parse_args()
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3808, 1280, generator=gen).to(torch.bfloat16)
    w = (torch.randn(1280, 3840, generator=gen) / 1280 ** 0.5).to(torch.bfloat16)
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = open(a.out, "w")
    for mode, wdt, fid in (("normal", ttnn.bfloat16, ttnn.MathFidelity.HiFi4), ("fast", ttnn.bfloat8_b, ttnn.MathFidelity.HiFi3)):
        wt = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=wdt)
        w64 = ttnn.to_torch(wt).double()   # the weights the device holds (bfp8 already rounded)
        ref = x.double() @ w64
        ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                    fp32_dest_acc_en=True, packer_l1_acc=False)
        for odt in (ttnn.bfloat16, ttnn.float32):
            y = ttnn.to_torch(ttnn.linear(xt, wt, compute_kernel_config=ck, dtype=odt)).double()
            rec = {"mode": mode, "out": str(odt), "bias": bias(y, ref), "rel_l2": float((y - ref).norm() / ref.norm())}
            if odt == ttnn.float32:   # what the bf16 output would be under round-to-nearest of the same fp32 values
                rne = y.float().to(torch.bfloat16).double()
                rec["rne_bias"], rec["rne_rel_l2"] = bias(rne, ref), float((rne - ref).norm() / ref.norm())
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
    if a.sweep:   # where does the bias come from: the multiply (fidelity), the K accumulation, or the partials' spill
        for k in (32, 256, 1280, 5120):
            xk = torch.randn(3808, k, generator=gen).to(torch.bfloat16)
            wk = (torch.randn(k, 1280, generator=gen) / k ** 0.5).to(torch.bfloat16)
            ref = xk.double() @ wk.double()
            xkt = ttnn.from_torch(xk, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            wkt = ttnn.from_torch(wk, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi):
                for l1acc in (False, True):
                    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                                fp32_dest_acc_en=True, packer_l1_acc=l1acc)
                    y = ttnn.to_torch(ttnn.linear(xkt, wkt, compute_kernel_config=ck, dtype=ttnn.float32)).double()
                    rec = {"k": k, "fid": str(fid), "l1acc": l1acc, "bias": bias(y, ref),
                           "rel_l2": float((y - ref).norm() / ref.norm())}
                    print(json.dumps(rec), flush=True)
                    out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
