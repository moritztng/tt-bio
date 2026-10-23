"""RFD4's token-track linears alone at n tokens, per program config: ms a call, rel RMS against float64, max |diff|
against production's call. Census v7 has these at 2.4-5.7x their byte floor on 64 cores, and the transition's
768 -> 3072 pair at the same ms at 256 as at 512: weight-read bound, not row bound.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/k256/tok_mm.py --out <json> [--n 256] [--rounds 4] [--reps 10]

Arms: `prod` = Module._lin's call (ops.linear, CORE_GRID_MAIN), `plain` = ttnn.linear with no core grid,
`1d:w=<in0_block_w>` = 1D multicast of x over ceil(Nt / per_core_N) cores (per_core_N the smallest that fits the
grid), `2d:w=<in0_block_w>` = 2D multicast on the full grid. No activation or bias in any arm, so arms differ only
in the program.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

# (name, K, N, calls a pass): transition fc_x / fc_g, transition out, attention q/k/v/g/o, adaLN gamma/beta
SHAPES = (("tr_in", 768, 3072, 28), ("tr_out", 3072, 768, 14), ("attn", 768, 768, 70), ("adaln", 256, 768, 84))


def pc_1d(gx, gy, mt, kt, nt, w):
    pn = math.ceil(nt / (gx * gy))
    sw = max(d for d in (4, 3, 2, 1) if pn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1, out_subblock_w=sw,
        per_core_M=mt, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)


def pc_2d(gx, gy, mt, kt, nt, w):
    pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
    sw = max(d for d in (4, 3, 2, 1) if pn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=w, out_subblock_h=1, out_subblock_w=sw,
        out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--arms", default="prod|plain|1d:w=2|1d:w=4|1d:w=8|2d:w=2|2d:w=4")
    a = ap.parse_args()

    from tt_bio import ops
    from tt_bio import tenstorrent as T
    from tt_bio.rfd4.trunk import compute_kernel_config

    dev = T.get_device()
    ckc = compute_kernel_config()
    gx, gy = T.COMPUTE_GRID_X_11, T.COMPUTE_GRID_Y
    g = torch.Generator().manual_seed(0)
    rows = []
    with clocksample.during() as clk:
        for name, K, N, per_pass in SHAPES:
            x64 = torch.randn(1, 1, a.n, K, generator=g, dtype=torch.float64)
            w64 = torch.randn(K, N, generator=g, dtype=torch.float64) / math.sqrt(K)
            x = ttnn.from_torch(x64.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            w = ttnn.from_torch(w64.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ref = x64 @ w64
            rn = ref.pow(2).mean().sqrt()
            mt, kt, nt = a.n // 32, K // 32, N // 32

            def call(arm):
                if arm == "prod":
                    return ops.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
                if arm == "plain":
                    return ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
                kind, wpart = arm.split(":")
                bw = int(wpart.split("=")[1])
                if kt % bw:
                    raise RuntimeError(f"in0_block_w {bw} does not divide Kt {kt}")
                pc = (pc_1d if kind == "1d" else pc_2d)(gx, gy, mt, kt, nt, bw)
                return ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, program_config=pc,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

            live, times, first = [], {}, None
            for arm in a.arms.split("|"):
                try:
                    r = call(arm)
                    got = ttnn.to_torch(r).double().reshape(ref.shape)
                    ttnn.deallocate(r)
                except Exception as exc:  # noqa: BLE001
                    rows.append(dict(n=a.n, op=name, arm=arm, error=str(exc).splitlines()[0][:200]))
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                if first is None:
                    first = got
                live.append((arm, dict(rel_rms_vs_f64=float((got - ref).pow(2).mean().sqrt() / rn),
                                       max_abs_vs_prod=float((got - first).abs().max()))))
                times[arm] = []
            for _ in range(a.rounds):
                for arm, _e in live:
                    for _ in range(a.reps):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        r = call(arm)
                        ttnn.synchronize_device(dev)
                        times[arm].append(time.perf_counter() - t0)
                        ttnn.deallocate(r)
            floor_ms = 1e3 * 2 * (a.n * K + K * N + a.n * N) / 436e9
            for arm, e in live:
                ts = sorted(times[arm][1:])
                ms = 1e3 * statistics.median(ts)
                rows.append(dict(n=a.n, op=name, arm=arm, ms=round(ms, 4), p10_ms=round(1e3 * ts[len(ts) // 10], 4),
                                 ms_a_pass=round(ms * per_pass, 3), x_floor=round(ms / floor_ms, 2),
                                 reps=len(ts), **e))
                print(json.dumps(rows[-1]), flush=True)
            ttnn.deallocate(x)
            ttnn.deallocate(w)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, aiclk=clk.summary(), core_grid_main=str(T.CORE_GRID_MAIN),
                                           script="perf/rfd4/k256/tok_mm.py"), indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
