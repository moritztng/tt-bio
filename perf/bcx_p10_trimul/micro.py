#!/usr/bin/env python3
"""The trimul backward's matmul and permute shapes, against every kernel that can serve them.

The census reads the taped backward's projection VJPs at 20-35 GB/s and 2-3 % of the FLOP
roof while the forward does the same arithmetic through ttnn.experimental.minimal_matmul.
This times the shapes on their own, synced, median of many warm reps, so a loud host cannot
move the ranking.
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
    ap.add_argument("--out", default="micro.json")
    args = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as AG
    from tt_bio.af2 import compute_kernel_config
    from perf.bcx_stack import stack as S

    dev = get_device()
    ckc = compute_kernel_config()
    clock = S.Clock()

    def up(shape, dtype=ttnn.bfloat16):
        t = torch.randn(*shape) * 0.05
        return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=dtype)

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

    N, R, C, H4 = 288, 275, 128, 512
    g4 = up((1, R, N, H4))            # cotangent of the fused in-projection
    w4 = up((C, H4))                  # the in-projection weight, [c_z, 4*hidden]
    w4t = up((H4, C))
    g1 = up((1, R, N, C))             # cotangent of an out-projection
    w1 = up((C, C))
    a3 = up((1, C, N, N))             # a moved channel chunk
    b3 = up((1, C, N, N))
    g2 = ttnn.reshape(g4, [R * N, H4])
    g2c = ttnn.reshape(g1, [R * N, C])

    cases = {}

    def add(name, fn, flop, byt):
        try:
            d = timeit(fn, args.reps)
        except Exception as e:                                          # noqa: BLE001
            cases[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            print(json.dumps({name: cases[name]}), flush=True)
            return
        cases[name] = {**d, "tflops": flop / d["median"] / 1e12,
                       "gbs": byt / d["median"] / 1e9, "load": os.getloadavg()[0]}
        print(json.dumps({name: {"ms": round(d["median"] * 1e3, 4),
                                 "tflops": round(cases[name]["tflops"], 2),
                                 "gbs": round(cases[name]["gbs"], 1)}}), flush=True)

    F_A = 2 * R * N * H4 * C
    B_A = (R * N * H4 + C * H4 + R * N * C) * 2
    add("A.dgrad_inproj  ttnn.matmul 4D tb", lambda: ttnn.matmul(
        g4, w4, transpose_b=True, compute_kernel_config=ckc), F_A, B_A)
    add("A.dgrad_inproj  ag.bmm 4D tb", lambda: AG.bmm(
        g4, w4, False, True, compute_kernel_config=ckc), F_A, B_A)
    add("A.dgrad_inproj  ttnn.matmul 2D tb", lambda: ttnn.matmul(
        g2, w4, transpose_b=True, compute_kernel_config=ckc), F_A, B_A)
    add("A.dgrad_inproj  ttnn.matmul 2D untransposed w", lambda: ttnn.matmul(
        g2, w4t, compute_kernel_config=ckc), F_A, B_A)
    add("A.dgrad_inproj  minimal_matmul 4D untransposed w",
        lambda: ttnn.experimental.minimal_matmul(
            g4, w4t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc), F_A, B_A)
    add("A.dgrad_inproj  minimal_matmul 2D untransposed w",
        lambda: ttnn.experimental.minimal_matmul(
            g2, w4t, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc), F_A, B_A)

    F_B = 2 * R * N * C * C
    B_B = (R * N * C + C * C + R * N * C) * 2
    add("B.dgrad_outproj ttnn.matmul 4D tb", lambda: ttnn.matmul(
        g1, w1, transpose_b=True, compute_kernel_config=ckc), F_B, B_B)
    add("B.dgrad_outproj ag.bmm 4D tb", lambda: AG.bmm(
        g1, w1, False, True, compute_kernel_config=ckc), F_B, B_B)
    add("B.dgrad_outproj ttnn.matmul 2D tb", lambda: ttnn.matmul(
        g2c, w1, transpose_b=True, compute_kernel_config=ckc), F_B, B_B)
    add("B.dgrad_outproj minimal_matmul 4D untransposed w",
        lambda: ttnn.experimental.minimal_matmul(
            g1, w1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc), F_B, B_B)
    add("B.dgrad_outproj minimal_matmul 2D untransposed w",
        lambda: ttnn.experimental.minimal_matmul(
            g2c, w1, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc), F_B, B_B)

    F_C = 2 * C * N * N * N
    B_C = (C * N * N * 3) * 2
    add("C.n3_contract   ttnn.matmul batched", lambda: ttnn.matmul(
        a3, b3, compute_kernel_config=ckc), F_C, B_C)
    add("C.n3_contract   ag.bmm batched", lambda: AG.bmm(
        a3, b3, compute_kernel_config=ckc), F_C, B_C)
    add("C.n3_contract   ttnn.matmul batched tb", lambda: ttnn.matmul(
        a3, b3, transpose_b=True, compute_kernel_config=ckc), F_C, B_C)
    add("C.n3_contract   ag.bmm batched tb", lambda: AG.bmm(
        a3, b3, False, True, compute_kernel_config=ckc), F_C, B_C)

    x = up((1, R, N, C))
    add("D.fwd_inproj    minimal_matmul (shipped fwd)",
        lambda: ttnn.experimental.minimal_matmul(
            x, w4, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc), F_A, B_A)

    p_in = up((1, R, N, C))
    p_out = up((1, C, N, N))
    add("E.channel_move  ttnn.permute (0,3,1,2)", lambda: ttnn.permute(
        p_in, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG), 0,
        (R * N * C + N * N * C) * 2)
    add("E.channel_back  ttnn.permute (0,2,3,1)", lambda: ttnn.permute(
        p_out, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG), 0,
        (N * N * C * 2) * 2)
    from tt_bio import reblock_permute as RB
    mc = ttnn.DRAM_MEMORY_CONFIG
    gates = {"reblock_eligible_fwd": bool(RB.eligible(p_in, mc)),
             "reblock_eligible_back": bool(RB.eligible_back(p_out, mc))}
    print(json.dumps(gates), flush=True)
    if gates["reblock_eligible_fwd"]:
        add("E.channel_move  reblock_permute kernel", lambda: RB.reblock_permute(p_in, mc), 0,
            (R * N * C + N * N * C) * 2)
    if gates["reblock_eligible_back"]:
        add("E.channel_back  reblock_permute_back kernel",
            lambda: RB.reblock_permute_back(p_out, mc), 0, (N * N * C * 2) * 2)

    clock.stop()
    blob = {"stamp": {"card": int(os.environ.get("TT_VISIBLE_DEVICES", "0")),
                      "aiclk_node": clock.path, "load": os.getloadavg(),
                      "aiclk": clock.window([(0.0, 9e18)]) if hasattr(clock, "window") else None},
            "reps": args.reps, "n_pad": N, "rows": R, "c_z": C, "hidden4": H4,
            "gates": gates, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print("wrote " + str(OUT / args.out), flush=True)


if __name__ == "__main__":
    main()
