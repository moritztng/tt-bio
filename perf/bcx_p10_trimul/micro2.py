#!/usr/bin/env python3
"""The trimul backward's projection VJPs at AF2's RAGGED token axis, with a float64 grade.

At n=275 the pair tensor is logically [1, 275, 275, C] and its second-last dim is not a whole
number of tiles, so autograd._via2d declines the 2-D collapse and the cotangent goes to
ttnn.matmul as a 4-D operand with transpose_b. That plan is what the census reads at 20-35
GB/s. The forward does the same contraction through ttnn.experimental.minimal_matmul on the
same ragged shape and is 8x faster, so this asks whether the backward can simply call it, and
what the two kernels cost in accuracy against a float64 reference.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_stack"))
OUT = ROOT / "perf" / "bcx_p10_trimul"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n", type=int, default=275)
    ap.add_argument("--out", default="micro2.json")
    args = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as AG
    from tt_bio import reblock_permute as RB
    from tt_bio.af2 import compute_kernel_config
    from perf.bcx_stack import stack as S

    dev = get_device()
    ckc = compute_kernel_config()
    clock = S.Clock()
    N, C, H4 = args.n, 128, 512

    def up(t):
        return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    def timeit(fn, reps):
        r = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(r)
        xs = []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(dev)
            xs.append(time.perf_counter() - t0)
            ttnn.deallocate(r)
        return S.dist(xs)

    cases, gates = {}, {}

    def add(name, fn, flop, byt, ref=None):
        try:
            d = timeit(fn, args.reps)
            rec = {**d, "tflops": flop / d["median"] / 1e12, "gbs": byt / d["median"] / 1e9,
                   "load": os.getloadavg()[0]}
            if ref is not None:
                y = ttnn.to_torch(fn()).to(torch.float64)
                rec["rel_l2_vs_f64"] = float(
                    torch.linalg.vector_norm(y - ref) / torch.linalg.vector_norm(ref))
            cases[name] = rec
            print(json.dumps({name: {
                "ms": round(d["median"] * 1e3, 4), "tflops": round(rec["tflops"], 2),
                "gbs": round(rec["gbs"], 1),
                "rel": (None if ref is None else float(f'{rec["rel_l2_vs_f64"]:.3e}'))}}),
                flush=True)
        except Exception as e:                                          # noqa: BLE001
            cases[name] = {"error": f"{type(e).__name__}: {e}"[:220]}
            print(json.dumps({name: cases[name]}), flush=True)

    # ---- A: the in-projection's dgrad.  g [1,N,N,4h] against w [c_z, 4h]
    tg4 = torch.randn(1, N, N, H4) * 0.05
    tw4 = torch.randn(C, H4) * 0.05
    g4, w4, w4t = up(tg4), up(tw4), up(tw4.t().contiguous())
    refA = (tg4.to(torch.float64) @ tw4.to(torch.float64).t())
    F_A, B_A = 2 * N * N * H4 * C, (N * N * H4 + C * H4 + N * N * C) * 2
    add("A.ttnn.matmul 4D transpose_b  (SHIPPED)", lambda: ttnn.matmul(
        g4, w4, transpose_b=True, compute_kernel_config=ckc), F_A, B_A, refA)
    add("A.ag.bmm 4D transpose_b", lambda: AG.bmm(
        g4, w4, False, True, compute_kernel_config=ckc), F_A, B_A, refA)
    add("A.minimal_matmul 4D, w transposed once", lambda: ttnn.experimental.minimal_matmul(
        g4, w4t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        compute_kernel_config=ckc), F_A, B_A, refA)

    # ---- B: an output projection's dgrad.  g [1,N,N,c_z] against w [c_z, c_z]
    tg1 = torch.randn(1, N, N, C) * 0.05
    tw1 = torch.randn(C, C) * 0.05
    g1, w1, w1t = up(tg1), up(tw1), up(tw1.t().contiguous())
    refB = (tg1.to(torch.float64) @ tw1.to(torch.float64).t())
    F_B, B_B = 2 * N * N * C * C, (N * N * C * 2 + C * C) * 2
    add("B.ttnn.matmul 4D transpose_b  (SHIPPED)", lambda: ttnn.matmul(
        g1, w1, transpose_b=True, compute_kernel_config=ckc), F_B, B_B, refB)
    add("B.minimal_matmul 4D, w transposed once", lambda: ttnn.experimental.minimal_matmul(
        g1, w1t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        compute_kernel_config=ckc), F_B, B_B, refB)

    # ---- C: the channel move and its inverse, at the ragged shape
    p_in = up(torch.randn(1, N, N, C) * 0.05)
    p_out = up(torch.randn(1, C, N, N) * 0.05)
    mc = ttnn.DRAM_MEMORY_CONFIG
    gates = {"eligible_fwd(ragged)": bool(RB.eligible(p_in, mc)),
             "eligible_back(ragged)": bool(RB.eligible_back(p_out, mc)),
             "shape_fwd": [int(d) for d in p_in.shape],
             "shape_back": [int(d) for d in p_out.shape]}
    print(json.dumps(gates), flush=True)
    B_M = (N * N * C * 2) * 2
    add("C.channel_move  ttnn.permute (0,3,1,2)  (SHIPPED under tape)",
        lambda: ttnn.permute(p_in, (0, 3, 1, 2), memory_config=mc), 0, B_M)
    if gates["eligible_fwd(ragged)"]:
        add("C.channel_move  reblock_permute kernel",
            lambda: RB.reblock_permute(p_in, mc), 0, B_M)
    add("C.channel_back  ttnn.permute (0,2,3,1)  (SHIPPED in bwd)",
        lambda: ttnn.permute(p_out, (0, 2, 3, 1), memory_config=mc), 0, B_M)
    if gates["eligible_back(ragged)"]:
        add("C.channel_back  reblock_permute_back kernel",
            lambda: RB.reblock_permute_back(p_out, mc), 0, B_M)

    clock.stop()
    blob = {"stamp": {"card": int(os.environ.get("TT_VISIBLE_DEVICES", "0")),
                      "aiclk_node": clock.path, "load": os.getloadavg()},
            "reps": args.reps, "n": N, "c_z": C, "hidden4": H4,
            "gates": gates, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print("wrote " + str(OUT / args.out), flush=True)


if __name__ == "__main__":
    main()
