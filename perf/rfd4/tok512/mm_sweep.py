"""Token-side matmul sweep for RFD4 under the device profiler: ttnn.linear (the trunk's `_lin` call) against
minimal_matmul blocks, per fidelity and weight dtype, at the batched token counts each size serves.

    perf/rfd4/tok512/mm_sweep.sh <out dir> <card> [--m 512 2240 2048 4096] [--reps 8]

Each config runs `--reps` times between `B|<label>` / `E|<label>` signposts; mm_table.py reads DEVICE KERNEL DURATION
per label. Error is rel RMS against the float64 product of the same bf16 operands (row sample), with
N(0,1) activations and N(0, 1/K) weights, so it ranks arms, it is not the trunk's error.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402
from tracy import signpost  # noqa: E402

import clocksample  # noqa: E402

#: name, K, N, bias, activation (the trunk's token-side linears; recycle is the [n*n, 384] pair projection)
SITES = [
    ("fc_x", 768, 3072, False, None),
    ("fc_g", 768, 3072, False, "silu"),
    ("fc_out", 3072, 768, False, None),
    ("qkv", 768, 768, True, None),
    ("gate", 768, 768, True, "sigmoid"),
    ("ada", 256, 768, True, "sigmoid"),
    ("beta", 256, 768, False, None),
    ("factors", 768, 384, False, None),
    ("recycle", 384, 384, False, None),
]
ROWS = 256
ACT = {"silu": ttnn.UnaryOpType.SILU, "sigmoid": ttnn.UnaryOpType.SIGMOID, "relu": ttnn.UnaryOpType.RELU}


def blocks(kt, nt, mt):
    for M in (2, 4, 5, 7, 8):
        for Kb in sorted({min(kt, 8), min(kt, 4), min(kt, 6)}):
            for Nb, sh, sw in ((1, 4, 1), (2, 2, 2), (4, 1, 4), (2, 1, 2), (4, 2, 4)):
                if kt % Kb or nt % Nb or mt % M or M % sh or sh * sw > 4 or Nb % sw:
                    continue
                yield M, Kb, Nb, sh, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[512, 2240])
    ap.add_argument("--recycle-n", type=int, nargs="+", default=[512, 224])
    ap.add_argument("--sites", default=",".join(s[0] for s in SITES))
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()

    from tt_bio.rfd4.trunk import compute_kernel_config
    from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, CORE_GRID_MAIN, _mm_core_coord, get_device, mm_sigmoid, silu_ckc

    dev = get_device()
    base = compute_kernel_config()
    dt = {"bf16": ttnn.bfloat16, "b8": ttnn.bfloat8_b}
    grid = _mm_core_coord(*COMPUTE_GRID_MAIN)

    def ckc(fid):
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=base.math_approx_mode,
            fp32_dest_acc_en=base.fp32_dest_acc_en, packer_l1_acc=base.packer_l1_acc)

    up = lambda t, d: ttnn.from_torch(t.float().contiguous(), dtype=dt[d], layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                      device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = torch.Generator().manual_seed(0)
    res, want = [], a.sites.split(",")

    def run(label, fn, reps, ref):
        try:
            y = fn()
        except RuntimeError as e:
            return {"label": label, "refused": str(e).splitlines()[0][:160]}
        got = ttnn.to_torch(y).reshape(-1, ref.shape[-1])[:ROWS].double()
        ttnn.deallocate(y)
        err = float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())
        ttnn.synchronize_device(dev)
        signpost(f"B|{label}")
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        signpost(f"E|{label}")
        ttnn.ReadDeviceProfiler(dev)   # the device buffer holds a few thousand ops; drain it per label
        return {"label": label, "err": err}

    signpost("elt_prof")
    with clocksample.during() as clk:
        for name, K, N, has_b, act in SITES:
            if name not in want:
                continue
            for m in ([n * n for n in a.recycle_n] if name == "recycle" else a.m):
                x = torch.randn(m, K, generator=g)
                w = torch.randn(K, N, generator=g) / K ** 0.5
                b = torch.randn(N, generator=g) * 0.1 if has_b else None
                xb = x.bfloat16().double()
                ref = xb[:ROWS] @ w.bfloat16().double() + (0 if b is None else b.bfloat16().double())
                ref = {"silu": torch.nn.functional.silu, "sigmoid": torch.sigmoid}.get(act, lambda t: t)(ref)
                xt = up(x[None, None], "bf16")
                bt = None if b is None else up(b[None], "bf16")
                for wd in ("bf16", "b8"):
                    wt = up(w, wd)
                    for fid in ("HiFi4", "HiFi2"):
                        c = ckc(fid)
                        lc = silu_ckc(c) if act == "silu" else c
                        lact = mm_sigmoid(c) if act == "sigmoid" else act
                        tag = f"{name}|{m}|{wd}|{fid}"
                        r = run(f"{tag}|linear", lambda: ttnn.linear(
                            xt, wt, bias=bt, activation=lact, compute_kernel_config=lc, dtype=ttnn.bfloat16,
                            core_grid=CORE_GRID_MAIN), a.reps, ref)
                        res.append(r)
                        print(json.dumps(r), flush=True)
                        # block sweep on the two arms that matter and activation-free sites; the rest take minimal_matmul's own plan
                        swept = act is None and (wd, fid) in (("bf16", "HiFi4"), ("b8", "HiFi2"))
                        for blk in [None, *(blocks(K // 32, N // 32, m // 32) if swept else ())]:
                            cfg = blk and ttnn.MinimalMatmulConfig(
                                M_block_size=blk[0], K_block_size=blk[1], N_block_size=blk[2], subblock_h=blk[3],
                                subblock_w=blk[4], compute_with_storage_grid_size=grid)
                            print("try", tag, blk, flush=True)   # a hang names its config
                            r = run(f"{tag}|mm{'-'.join(map(str, blk or ['auto']))}", lambda: ttnn.experimental.minimal_matmul(
                                input_tensor=xt, weight_tensor=wt, bias_tensor=bt, compute_kernel_config=c,
                                fused_activation=ttnn.UnaryWithParam(ACT[act]) if act else None,
                                dtype=ttnn.bfloat16, config=cfg), a.reps, ref)
                            res.append(r)
                    ttnn.deallocate(wt)
                for t in (xt, bt):
                    if t is not None:
                        ttnn.deallocate(t)
    signpost("elt_end")
    Path(a.out).write_text(json.dumps(dict(rows=res, reps=a.reps, aiclk=clk.summary(),
                                           load=clk.load and max(clk.load)), indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
