#!/usr/bin/env python3
"""Does the fused bias the kernels always had, and `mm_generic` never bound, actually work?

The wheel's minimal_matmul kernels read a bias into `c_4` under `FUSE_BIAS` and add it before the
pack. `tt_bio/mm_generic.py` transcribed everything except the four bindings that turn it on. This
checks the binding on hardware, at the token-DiT qkv shape, three ways:

  correctness   against `ttnn.linear(s, w, bias=b)` and, separately, against the same generic
                matmul with no bias plus the bias added in torch -- the second is the one that
                isolates the binding from the matmul's own numerics;
  a negative control that must move the answer (bias perturbed in one channel);
  cost          the fused bias against the separate `ttnn.add` it replaces.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S, C, H, HD = 512, 768, 16, 64
BLK = (4, 8, 4, 4, 2)          # the screen's fastest at this shape, 103.18 us


def timed(ttnn, dev, fn, reps=20, blocks=5):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    w = []
    for _ in range(blocks):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        w.append(1e6 * (time.perf_counter() - t) / reps)
    return round(st.median(w), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio import mm_generic as G

    dev = T.get_device(trace_region_size=512 << 20)
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "blk": list(BLK), "shape": [S, C, 3 * H * HD],
                   "loadavg": open("/proc/loadavg").read().split()[:3]}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1))

    tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,   # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    N = 3 * H * HD
    t_s, t_w, t_b = torch.randn(1, S, C), torch.randn(C, N), torch.randn(1, N)
    s, w, b = tt(t_s), tt(t_w), tt(t_b)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, S, N // 3]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]

    def gen(bias):
        G.generic_minimal_matmul(dev, s, w, outs, (BLK, grid), G.ckc_args(ckc), bias=bias)
        return torch.cat([ttnn.to_torch(o) for o in outs], dim=-1)

    nobias = gen(None).clone()
    fused = gen(b).clone()
    ref = ttnn.to_torch(ttnn.linear(s, w, bias=b, compute_kernel_config=ckc,
                                    core_grid=T.CORE_GRID_MAIN)).clone()

    def cmp(x, y):
        d = (x.float() - y.float()).abs()
        return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                "rms": round(float(y.float().pow(2).mean().sqrt()), 4),
                "max_rel_to_rms": float(d.max()) / float(y.float().pow(2).mean().sqrt()),
                "bit_exact": bool(d.max() == 0)}

    out["binding"] = {
        # the isolating check: same matmul, bias added on the host instead of in the kernel
        "vs_same_matmul_plus_host_bias": cmp(fused, (nobias.float() + t_b.float()).bfloat16()),
        "vs_ttnn_linear_bias": cmp(fused, ref),
        "nobias_vs_ttnn_linear_bias": cmp(nobias, ref),
    }
    dump()

    t_b2 = t_b.clone()
    t_b2[0, 7] += 4.0
    b2 = tt(t_b2)
    ctrl = gen(b2).clone()
    out["binding"]["negative_control_differs"] = bool((ctrl.float() - fused.float()).abs().max() > 0)
    out["binding"]["negative_control_max_abs"] = float((ctrl.float() - fused.float()).abs().max())
    dump()
    print("  binding:", json.dumps(out["binding"], indent=1), flush=True)

    bq = tt(torch.zeros(1, 1, N // 3))
    out["cost_us"] = {
        "generic_no_bias": timed(ttnn, dev, lambda: G.generic_minimal_matmul(
            dev, s, w, outs, (BLK, grid), G.ckc_args(ckc))),
        "generic_fused_bias": timed(ttnn, dev, lambda: G.generic_minimal_matmul(
            dev, s, w, outs, (BLK, grid), G.ckc_args(ckc), bias=b)),
        "separate_add_it_replaces": timed(ttnn, dev, lambda: ttnn.deallocate(
            ttnn.add(outs[0], bq, memory_config=ttnn.DRAM_MEMORY_CONFIG))),
        "ttnn_linear_with_bias": timed(ttnn, dev, lambda: ttnn.deallocate(
            ttnn.linear(s, w, bias=b, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN))),
    }
    c = out["cost_us"]
    c["fused_bias_costs_us"] = round(c["generic_fused_bias"] - c["generic_no_bias"], 2)
    c["saved_vs_separate_add_us"] = round(c["separate_add_it_replaces"]
                                          - c["fused_bias_costs_us"], 2)
    dump()
    print("  cost:", json.dumps(c, indent=1), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
