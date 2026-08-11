#!/usr/bin/env python3
"""Why the `Transition` oracle floor is 13.15 ms/call: price the three GEMMs, and the shape they run at.

The Phase-2 screen (perf/nontri512/transition_screen.py) found the whole glue -- layer_norm, chunk,
concat, and every per-op dispatch in the 32x6 chain -- worth 1.588 ms of a 14.738 ms call. So the
call is its three GEMMs. This leg asks what those GEMMs are bound by, at the production shape and
at wider row chunks, plus a same-FLOP contraction with a 4x deeper K to test the skinny-K mechanism.

Every leg: real shapes, HiFi4 + fp32_dest_acc + packer_l1_acc, `core_grid=CORE_GRID_MAIN`, synced
both sides, interleaved, median of --rounds.
"""

import argparse
import collections
import json
import statistics as st
import time

import torch

import ttnn
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    N, C, H = args.n, 256, 1024
    print(f"N={N} c={C} c_hid={H} grid={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)

    # Weights, real shapes (values irrelevant to timing).
    w_up = ttnn.from_torch(torch.randn(C, H) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=DRAM)
    w_dn = ttnn.from_torch(torch.randn(H, C) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=DRAM)

    def rows(h):
        return ttnn.from_torch(torch.randn(1, h, N, C) * 0.5, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

    nch = N // 16

    def mm(a, w, omem, act=None):
        ttnn.deallocate(ttnn.linear(a, w, activation=act, compute_kernel_config=ckc,
                                    memory_config=omem, dtype=ttnn.bfloat16,
                                    core_grid=CORE_GRID_MAIN))

    per_call_flop = 3 * 2 * (N * N) * C * H
    a_sq = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    b_sq = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

    def run_phase(legs, flops, samples):
        order = list(legs)
        for k in order:
            for _ in range(args.warm):
                try:
                    legs[k]()
                except Exception as e:
                    print(f"  {k}: WARM ERR {str(e)[:120]}", flush=True)
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
                    print(f"  {k}: ERR {str(e)[:120]}", flush=True)
                    samples[k].append(float("nan"))
            print(f"  round {r+1}: " + " ".join(f"{k}={samples[k][-1]:.3f}" for k in order), flush=True)

    samples = collections.defaultdict(list)
    flops = {"square4096_x3": 3 * 2 * 4096 ** 3}

    # --- phase 1: the up-projections (K = 8 tiles), nothing large resident in L1 -------------
    x16 = rows(16)
    p1 = collections.OrderedDict()
    p1["fc1_silu_x32"] = lambda: [mm(x16, w_up, L1, "silu") for _ in range(nch)]
    p1["fc1_nosilu_x32"] = lambda: [mm(x16, w_up, L1, None) for _ in range(nch)]
    p1["fc1_toDRAM_x32"] = lambda: [mm(x16, w_up, DRAM, "silu") for _ in range(nch)]
    p1["square4096_x3"] = lambda: [ttnn.deallocate(
        ttnn.matmul(a_sq, b_sq, compute_kernel_config=ckc, memory_config=DRAM)) for _ in range(3)]
    for k in ("fc1_silu_x32", "fc1_nosilu_x32", "fc1_toDRAM_x32"):
        flops[k] = per_call_flop / 3
    for h in (32, 64):
        xh = rows(h)
        k = N // h
        p1[f"fc1_h{h}_x{k}"] = (lambda xh=xh, k=k: [mm(xh, w_up, DRAM, "silu") for _ in range(k)])
        flops[f"fc1_h{h}_x{k}"] = per_call_flop / 3
    print("--- phase 1: up-projection, K=8 tiles ---", flush=True)
    run_phase(p1, flops, samples)
    ttnn.deallocate(x16)

    # --- phase 2: the down-projection (K = 32 tiles), hidden resident in L1 as in production --
    hid16 = ttnn.from_torch(torch.randn(1, 16, N, H) * 0.5, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    p2 = collections.OrderedDict()
    p2["fc3_x32"] = lambda: [mm(hid16, w_dn, DRAM, None) for _ in range(nch)]
    p2["fc3_toL1_x32"] = lambda: [mm(hid16, w_dn, L1, None) for _ in range(nch)]
    for k in ("fc3_x32", "fc3_toL1_x32"):
        flops[k] = per_call_flop / 3
    print("--- phase 2: down-projection, K=32 tiles ---", flush=True)
    run_phase(p2, flops, samples)

    res = {}
    print(f"\n=== GEMM legs, N={N}, median of {args.rounds} interleaved rounds per phase ===", flush=True)
    for k, v in samples.items():
        v = [s for s in v if s == s]
        if not v:
            continue
        med = st.median(v)
        tf = flops.get(k, 0) / (med * 1e-3) / 1e12 if k in flops else None
        res[k] = {"ms_median": round(med, 4), "ms_min": round(min(v), 4),
                  "TFLOPs": round(tf, 2) if tf else None}
        print(f"  {k:18s} {med:9.3f} ms" + (f"   {tf:7.2f} TFLOP/s" if tf else ""), flush=True)
    if args.out:
        json.dump({"N": N, "c": C, "c_hid": H, "grid": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                   "per_call_flop": per_call_flop, "legs": res,
                   "samples": {k: [round(s, 4) for s in v] for k, v in samples.items()}},
                  open(args.out, "w"), indent=2)
        print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
