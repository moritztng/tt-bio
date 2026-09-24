#!/usr/bin/env python3
"""Why the triangle-attention batched matmuls run at 6-18 % of roof, and what recovers it.

One chunk of `autograd.triangle_attention` at the census arm (n=256, 4 heads x 32, chunk 128):
  QK^T  [128,4,128,32] x [128,4,256,32]^T  -> [128,4,128,256]
  PV    [128,4,128,256] x [128,4,256,32]   -> [128,4,128,32]
  dV    [128,4,128,256]^T x [128,4,128,32] -> [128,4,256,32]
Each is timed as issued today (rank 4, precise_config) and as variants that do the same
arithmetic: a rank-3 view, an explicit batched program config, and the default fidelity (priced,
never shipped here). Every variant's output is compared to the shipped one, bit for bit.

Timing: R back-to-back calls, one sync, median over reps. AICLK sampled during.
"""
import argparse
import json
import statistics
import sys
import time
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.e2e_distogram import ClockTrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--R", type=int, default=10)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio.autograd import precise_config

    dev = tt.get_device()
    grid = dev.compute_with_storage_grid_size()
    clocks = ClockTrace(period=1.0).start()
    hifi = precise_config()
    lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                            math_approx_mode=True, fp32_dest_acc_en=False,
                                            packer_l1_acc=True)
    g = torch.Generator().manual_seed(0)

    def T(*s):
        return ttnn.from_torch(torch.randn(*s, generator=g).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    B, H, nq, nk, d = 128, 4, 128, 256, 32
    q, k, v = T(B, H, nq, d), T(B, H, nk, d), T(B, H, nk, d)
    p = T(B, H, nq, nk)
    go = T(B, H, nq, d)

    def r3(t):
        s = [int(x) for x in t.shape]
        return ttnn.reshape(t, [s[0] * s[1], s[2], s[3]])

    def bmm_cfg(M, N, K, per_core_M=None, in0_block_w=None):
        Mt, Nt, Kt = M // 32, N // 32, K // 32
        pcm = per_core_M or Mt
        bw = in0_block_w or Kt
        sw = min(Nt, 4) if pcm * min(Nt, 4) <= 8 else 1
        while Nt % sw:
            sw -= 1
        sh = 1
        for h in range(pcm, 0, -1):
            if pcm % h == 0 and h * sw <= 8:
                sh = h
                break
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=sh,
            out_subblock_w=sw, per_core_M=pcm, per_core_N=Nt)

    cases = {
        "PV": dict(a=p, b=v, kw={}, M=nq, N=d, K=nk),
        "QKt": dict(a=q, b=k, kw=dict(transpose_b=True), M=nq, N=nk, K=d),
        "dV_PtdO": dict(a=p, b=go, kw=dict(transpose_a=True), M=nk, N=d, K=nq),
    }
    res = {"grid": [grid.x, grid.y], "cases": {}}
    for name, c in cases.items():
        a, b, kw = c["a"], c["b"], c["kw"]
        flops = 2.0 * B * H * c["M"] * c["N"] * c["K"]
        variants = {
            "shipped_r4_hifi": lambda: ttnn.matmul(a, b, compute_kernel_config=hifi, **kw),
            "r3_hifi": lambda: ttnn.matmul(r3(a), r3(b), compute_kernel_config=hifi, **kw),
            "r4_default": lambda: ttnn.matmul(a, b, **kw),
            "r4_lofi": lambda: ttnn.matmul(a, b, compute_kernel_config=lofi, **kw),
        }
        if not kw:
            variants["r4_hifi_reusecfg"] = lambda: ttnn.matmul(
                a, b, compute_kernel_config=hifi,
                program_config=bmm_cfg(c["M"], c["N"], c["K"]))
            variants["r3_hifi_reusecfg"] = lambda: ttnn.matmul(
                r3(a), r3(b), compute_kernel_config=hifi,
                program_config=bmm_cfg(c["M"], c["N"], c["K"]))
        ref = ttnn.to_torch(variants["shipped_r4_hifi"]())
        out = {}
        for vn, fn in variants.items():
            try:
                y = fn()
                yt = ttnn.to_torch(y).reshape(ref.shape)
                exact = bool(torch.equal(yt, ref))
                maxdiff = float((yt.float() - ref.float()).abs().max())
                ttnn.synchronize_device(dev)
                ts = []
                t_w0 = time.time()
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    for _ in range(args.R):
                        y = fn()
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / args.R)
                t_w1 = time.time()
                us = statistics.median(ts) * 1e6
                out[vn] = dict(us=us, us_min=min(ts) * 1e6, us_max=max(ts) * 1e6,
                               tflops=flops / (us * 1e-6) / 1e12, bitexact=exact,
                               maxdiff=maxdiff, clock=clocks.window(t_w0, t_w1))
                print(f"{name:8s} {vn:18s} {us:8.1f} us  {out[vn]['tflops']:6.2f} TF/s  "
                      f"exact={exact} maxdiff={maxdiff:.3g} clk={out[vn]['clock']}", flush=True)
            except Exception as e:
                out[vn] = dict(error=str(e)[:300])
                print(f"{name:8s} {vn:18s} ERROR {str(e)[:200]}", flush=True)
        res["cases"][name] = out
    clocks.stop()
    res["clock_meta"] = clocks.summary()
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
