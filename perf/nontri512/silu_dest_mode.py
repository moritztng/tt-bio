#!/usr/bin/env python3
"""Is the fused-silu penalty the FUSION, or is it the fp32 dest accumulator?

`ckernel_sfpu_silu.h` calls `_sfpu_sigmoid_<is_fp32_dest_acc_en>`, and that template parameter picks
the algorithm:

    fp32 dest -> _sfpu_exp_fp32_accurate_ (Cody-Waite) + _sfpu_reciprocal_<2>
    bf16 dest -> _sfpu_exp_21f_bf16_                   + _sfpu_reciprocal_<1>

If that is the mechanism then a silu with NO matmul anywhere near it must show the same ~2x purely
from the dest mode, and a cheap unary (relu) must NOT, since relu has no such branch. relu is the
control that separates "accurate transcendental" from "fp32 moves twice the bytes".

Also measures how many elements the two sigmoid algorithms actually disagree on at bf16 output,
which is the number the parity decision needs.
"""

import argparse
import collections
import json
import statistics as st
import time

import torch

import ttnn
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

L1 = ttnn.L1_MEMORY_CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = get_device()
    N, H, h = args.n, 1024, 16
    nch = N // h
    print(f"hidden chunk [1,{h},{N},{H}] x{nch}, grid {CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)

    # The production hidden: fc1's output, one chunk, L1-resident, values in the range a real
    # layer-norm'd activation reaches.
    tt_h = torch.randn(1, h, N, H) * 2.0
    hid_b = ttnn.from_torch(tt_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    hid_f = ttnn.from_torch(tt_h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    samples = collections.defaultdict(list)

    def run_phase(legs):
        order = list(legs)
        for k in order:
            for _ in range(args.warm):
                try:
                    legs[k]()
                except Exception as e:
                    print(f"  {k}: WARM ERR {str(e)[:160]}", flush=True)
                    break
            ttnn.synchronize_device(dev)
        for r in range(args.rounds):
            for k in order:
                try:
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    legs[k]()
                    ttnn.synchronize_device(dev)
                    samples[k].append((time.perf_counter() - t0) * 1e3)
                except Exception as e:
                    print(f"  {k}: ERR {str(e)[:160]}", flush=True)
                    samples[k].append(float("nan"))
            print(f"  round {r+1}: " + " ".join(f"{k}={samples[k][-1]:.3f}" for k in order), flush=True)

    def mk(op, src):
        def leg():
            for _ in range(nch):
                ttnn.deallocate(op(src, memory_config=L1))
        return leg

    legs = collections.OrderedDict([
        ("silu_bf16_x32",    mk(ttnn.silu, hid_b)),
        ("silu_fp32_x32",    mk(ttnn.silu, hid_f)),
        ("sigmoid_bf16_x32", mk(ttnn.sigmoid, hid_b)),
        ("sigmoid_fp32_x32", mk(ttnn.sigmoid, hid_f)),
        ("relu_bf16_x32",    mk(ttnn.relu, hid_b)),
        ("relu_fp32_x32",    mk(ttnn.relu, hid_f)),
    ])
    print("--- dest-mode legs, interleaved ---", flush=True)
    run_phase(legs)

    res = {}
    print(f"\n=== median of {args.rounds} interleaved rounds ===", flush=True)
    for k, v in samples.items():
        v = [s for s in v if s == s]
        if not v:
            continue
        res[k] = {"ms_median": round(st.median(v), 4), "ms_min": round(min(v), 4),
                  "ms_max": round(max(v), 4), "n": len(v)}
        print(f"  {k:20s} {st.median(v):9.3f} ms  (min {min(v):.3f} max {max(v):.3f})", flush=True)

    def ratio(a, b):
        if a in res and b in res and res[b]["ms_median"]:
            return round(res[a]["ms_median"] / res[b]["ms_median"], 4)
        return None
    rr = {"silu_fp32_over_bf16": ratio("silu_fp32_x32", "silu_bf16_x32"),
          "sigmoid_fp32_over_bf16": ratio("sigmoid_fp32_x32", "sigmoid_bf16_x32"),
          "relu_fp32_over_bf16": ratio("relu_fp32_x32", "relu_bf16_x32")}
    print("\n  fp32/bf16 dest ratio: " + " ".join(f"{k.split('_')[0]}={v}" for k, v in rr.items()), flush=True)

    # --- numerics: how far apart are the two sigmoid algorithms at bf16 output? ----------------
    small = torch.randn(1, 1, 512, 1024) * 2.0
    sb = ttnn.from_torch(small, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    sf = ttnn.from_torch(small.to(torch.bfloat16).to(torch.float32), dtype=ttnn.float32,
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    fast = ttnn.to_torch(ttnn.silu(sb, memory_config=L1)).to(torch.float32)
    acc = ttnn.to_torch(ttnn.silu(sf, memory_config=L1)).to(torch.bfloat16).to(torch.float32)
    ref = torch.nn.functional.silu(small.to(torch.bfloat16).to(torch.float32)).to(torch.bfloat16).to(torch.float32)
    n = fast.numel()
    num = {
        "elems": n,
        "fast_vs_accurate_differ_frac": round(float((fast != acc).sum()) / n, 6),
        "fast_vs_accurate_max_abs": float((fast - acc).abs().max()),
        "fast_vs_torchbf16_differ_frac": round(float((fast != ref).sum()) / n, 6),
        "accurate_vs_torchbf16_differ_frac": round(float((acc != ref).sum()) / n, 6),
    }
    print("\n  numerics (bf16-output silu, 512x1024 elems):", flush=True)
    for k, v in num.items():
        print(f"    {k:36s} {v}", flush=True)

    if args.out:
        json.dump({"N": N, "c_hid": H, "h": h, "legs": res, "ratios": rr, "numerics": num,
                   "grid": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                   "samples": {k: [round(s, 4) for s in v] for k, v in samples.items()}},
                  open(args.out, "w"), indent=2)
        print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
