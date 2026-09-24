#!/usr/bin/env python3
"""ttnn's auto-configured matmul against `tenstorrent._k1_program_config` (in0_block_w = 1):
does the narrowed config remove the Wormhole fault, and what does it cost?

For each [M, K] x [K, N] shape: ttnn.matmul with no program config ("auto") and with the k1
config ("w1"), both scored against float64 (fault = |err| > 0.25 x the dot product's own scale,
as linear_probe.py) and timed interleaved, AICLK sampled DURING. ttnn does not expose the config
it resolves (`create_matmul_attributes` returns program_config None, graph capture prints
`<variant>`), so the auto arm's block width is not recorded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from perf.clocksample import during  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=16384)
ap.add_argument("--shapes", default="768x3072,768x1536,1536x768,384x1536,512x128,128x512,256x64")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = T.get_device()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
           packer_l1_acc=True)


rows = []
with during() as clk:
    for shp in a.shapes.split(","):
        k, n = (int(x) for x in shp.split("x"))
        g = torch.Generator().manual_seed(4000 + k + n)
        A = torch.randn(1, 1, a.m, k, generator=g).bfloat16()
        B = torch.randn(k, n, generator=g).bfloat16()
        ref = torch.matmul(A.double(), B.double())
        scale = torch.matmul(A.double() ** 2, B.double() ** 2).sqrt()
        ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev)
        tb = ttnn.from_torch(B, layout=ttnn.TILE_LAYOUT, device=dev)
        arms = {"auto": None, "w1": T._k1_program_config(a.m // 32, n // 32)}
        # the same config at the widest K block <= 8 that fits: the price of w = 1 within one family
        base = T._k1_program_config(a.m // 32, n // 32)  # shared (cached): copy before mutating
        wk = type(base).from_json(base.to_json())
        kt = k // 32
        wk.in0_block_w = max(d for d in range(min(8, kt), 0, -1) if kt % d == 0
                             and T._matmul_cb_bytes(d, wk.out_block_h, wk.out_block_w, 2) <= T._matmul_cb_budget())
        arms["wk"] = wk
        ms = {x: [] for x in arms}
        for rep in range(a.reps + 1):
            for name, cfg in arms.items():
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                out = ttnn.matmul(ta, tb, compute_kernel_config=ckc, program_config=cfg, dtype=ttnn.bfloat16)
                ttnn.synchronize_device(dev)
                if rep:
                    ms[name].append((time.perf_counter() - t0) * 1e3)
                if rep == 1:
                    e = ttnn.to_torch(out).double().reshape(ref.shape) - ref
                    q = e.abs() / scale.clamp(min=1e-30)
                    ms[name + "_fault"] = int((q > 0.25).sum())
                    ms[name + "_maxq"] = round(float(q.max()), 4)
                ttnn.deallocate(out)
        for name, cfg in arms.items():
            x = sorted(ms[name])
            r = {"m": a.m, "k": k, "n": n, "arm": name, "config": T._bmm_cfg_fields(cfg), "fault": ms[name + "_fault"], "max_q": ms[name + "_maxq"],
                 "median_ms": round(x[len(x) // 2], 4), "elems": ref.numel()}
            rows.append(r)
            print(json.dumps(r), flush=True)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
print(clk.line(), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps({"arch": str(dev.arch()), "aiclk": clk.summary(), "rows": rows}, indent=1) + "\n")
