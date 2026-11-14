"""ESMC's linears at the page shape, swept over matmul programs, against a float64 reference.

The WH census (woc2) puts ESMC-300M's linears at 83 ms of a 178 ms batch of 7, the fc1 half [3808, 960] x [960, 2560]
at 0.61 ms a call, ~31 TF/s against a ~98 TF/s HiFi3 peak, and math fidelity does not move them (wef). So the program
is the suspect. Arms per shape:
  shipped   ttnn.linear(core_grid=CORE_GRID_MAIN), ESMC's kernel config (HiFi3, fp32 dest, packer L1 acc)
  auto      ttnn.linear with no grid
  mm        ttnn.experimental.minimal_matmul, default config
  2d.*      MatmulMultiCoreReuseMultiCastProgramConfig on the full grid: in0_block_w x transpose_mcast x fp32 dest
Per arm: device ms (median of --repeat, --inner calls per synchronize), TF/s, rel_l2 and max error / peak against
float64 of the same bf16 operands, and whether the output equals the shipped arm's byte for byte.

    python3 perf/spd_gap_esmf/lin_bench.py --model esmc-300m --out lin.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

# (d_model, d_ff, rows of the page batch): fc1 runs as two halves of d_ff each (split SwiGLU).
MODELS = {"esmc-300m": (960, 2560, 7 * 544), "esmc-600m": (1152, 3072, 7 * 544), "esmc-6b": (2560, 6912, 544)}


def shapes(model):
    d, ff, m = MODELS[model]
    return {"qkv": (m, d, 3 * d), "out": (m, d, d), "fc1h": (m, d, ff), "fc2": (m, ff, d)}


def subblock(pcm, pcn, cap):
    w = max(x for x in range(1, min(cap, pcn) + 1) if pcn % x == 0)
    h = max(x for x in range(1, cap // w + 1) if pcm % x == 0)
    return h, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=sorted(MODELS))
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN

    def ck(fp32):
        return ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
            fp32_dest_acc_en=fp32, packer_l1_acc=True)

    out = open(args.out, "w")
    g = torch.Generator().manual_seed(0)
    for name, (m, k, n) in shapes(args.model).items():
        x = torch.randn(m, k, generator=g).to(torch.bfloat16)
        w = (torch.randn(k, n, generator=g) / math.sqrt(k)).to(torch.bfloat16)
        ref = x.double() @ w.double()
        xd = ttnn.from_torch(x.reshape(1, 1, m, k), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        mt, kt, nt = -(-m // 32), k // 32, n // 32
        arms = {
            "shipped": lambda: ttnn.linear(xd, wd, compute_kernel_config=ck(True), dtype=ttnn.bfloat16,
                                           core_grid=T.CORE_GRID_MAIN),
            "auto": lambda: ttnn.linear(xd, wd, compute_kernel_config=ck(True), dtype=ttnn.bfloat16),
            "mm": lambda: ttnn.experimental.minimal_matmul(xd, wd, compute_kernel_config=ck(True),
                                                           dtype=ttnn.bfloat16),
        }
        for tr in (False, True):
            ry, rx = (gx, gy) if tr else (gy, gx)        # M spread over ry cores, N over rx
            pcm, pcn = -(-mt // ry), -(-nt // rx)
            for fp32 in (True, False):
                sh, sw = subblock(pcm, pcn, 4 if fp32 else 8)
                for ibw in [b for b in (1, 2, 3, 4, 5, 6, 8) if kt % b == 0]:
                    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh,
                        out_subblock_w=sw, per_core_M=pcm, per_core_N=pcn, transpose_mcast=tr,
                        fused_activation=None, fuse_batch=True)
                    arms[f"2d.t{int(tr)}.f{int(fp32)}.b{ibw}"] = (
                        lambda cfg=cfg, fp32=fp32: ttnn.linear(xd, wd, program_config=cfg, dtype=ttnn.bfloat16,
                                                               compute_kernel_config=ck(fp32)))
        base = None
        for arm, fn in arms.items():
            rec = {"model": args.model, "shape": name, "mkn": [m, k, n], "arm": arm, "grid": [gx, gy]}
            try:
                o = fn()
                got = ttnn.to_torch(o).reshape(m, n).double()
                ttnn.deallocate(o)
                ts = []
                for _ in range(args.repeat):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(args.inner):
                        ttnn.deallocate(fn())
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / args.inner * 1e3)
                ms = sorted(ts)[len(ts) // 2]
                err = got - ref
                rec.update(ms=round(ms, 4), tflops=round(2 * m * k * n / ms / 1e9, 2),
                           finite=bool(torch.isfinite(got).all()),
                           rel_l2=float(err.norm() / ref.norm()), max_err_peak=float(err.abs().max() / ref.abs().max()))
                if arm == "shipped":
                    base = got
                rec["equal_shipped"] = bool(base is not None and torch.equal(got, base))
            except Exception as e:  # an L1 refusal or an illegal config is a result, not a crash
                rec["error"] = str(e).splitlines()[0][:200]
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()
        ttnn.deallocate(xd); ttnn.deallocate(wd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
