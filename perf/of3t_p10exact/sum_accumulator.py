"""Does `ttnn.sum` on a bf16 input keep an fp32 running sum, or a bf16 one?

`of3t-perf10` measured on CPU that a bf16 ACCUMULATOR loses a reduction as sqrt(N) while bf16
inputs into an fp32 accumulator do not (`bf16_reduction.py`). `autograd.py:2698` asserts in prose
that the device takes the bad branch -- "a bf16 result means a bf16 running sum however precise
the destination register is" -- and six token-axis reductions in that file are written as if it
did not. This turns the prose into a measurement.

The input is already bf16-exact, so nothing here measures input rounding: the float64 reference
is the sum of the SAME values the device holds, and every difference is accumulation.

    python3 sum_accumulator.py --out ACC.json
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tt_bio.tenstorrent as T
from tt_bio.autograd import precise_config, _sum_leading, _flat2d


def rel_l2(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def cos(a, ref):
    a, ref = a.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    return float(a @ ref / (np.linalg.norm(a) * np.linalg.norm(ref)))


def arms(dev, N, C, seed=0):
    g = torch.randn(N, C, generator=torch.Generator().manual_seed(seed))
    xb = g.to(torch.bfloat16)                       # the exact values the device will hold
    ref = xb.to(torch.float64).numpy().sum(0)       # float64 sum of THOSE values
    x = ttnn.from_torch(xb, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = {}

    def read(t):
        return ttnn.to_torch(t).to(torch.float64).numpy().reshape(-1)[:C]

    # `ttnn.sum` in this build takes no `dtype` kwarg (measured: TypeError, the signature is
    # dim/keepdim/memory_config/compute_kernel_config/scalar/correction/sub_core_grids). So the
    # only lever on its accumulator is the INPUT dtype, and that is what these arms separate.
    s = ttnn.sum(x, dim=0, keepdim=True, compute_kernel_config=precise_config())
    out["ttnn_sum_bf16_in_precise"] = (read(s), str(s.dtype))
    s = ttnn.sum(x, dim=0, keepdim=True)
    out["ttnn_sum_bf16_in_default"] = (read(s), str(s.dtype))
    sl = _sum_leading(x, [C])
    out["sum_leading_bf16_in"] = (read(sl), str(sl.dtype))          # production dgamma today

    xf = ttnn.typecast(x, ttnn.float32)
    s = ttnn.sum(xf, dim=0, keepdim=True, compute_kernel_config=precise_config())
    out["ttnn_sum_fp32_in_precise"] = (read(s), str(s.dtype))
    sl = _sum_leading(xf, [C])
    out["sum_leading_fp32_in_tree"] = (read(sl), str(sl.dtype))     # the fp32 add tree
    ttnn.deallocate(xf)

    return {k: {"rel_l2": rel_l2(v, ref), "cos_vs_float64": cos(v, ref), "out_dtype": d}
            for k, (v, d) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=64)
    a = ap.parse_args()

    dev = T.get_device()
    rows = {}
    for N in (64, 4096, 147456):
        t0 = time.time()
        rows[N] = arms(dev, N, a.channels)
        print(f"N={N} ({time.time() - t0:.1f}s)")
        for k, v in rows[N].items():
            print("   %-34s %-20s rel %.3e  cos %.6f"
                  % (k, v["out_dtype"], v["rel_l2"], v["cos_vs_float64"]))
    T.cleanup()

    rec = {"what": "ttnn.sum accumulator precision on a bf16 input, vs a float64 sum of the same values",
           "channels": a.channels, "rows": {str(k): v for k, v in rows.items()}}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
