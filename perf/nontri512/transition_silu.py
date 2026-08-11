#!/usr/bin/env python3
"""The 5.08 ms: what the fused `activation="silu"` on Transition fc1 actually costs, and what the
alternatives cost, at the production 512-aa pair shape.

perf/nontri512/transition_gemm.py measured fc1 at 7.040 ms per 32 chunks with the fused silu and
1.958 ms without it. This leg prices every piece of the chain that is left, so the 14.738 ms call has
a closed ledger, and prices the two silu routes:

  silu_standalone_x32   ttnn.silu in place on the L1 hidden, the `_UNFUSED_SILU` path's SFPU pass
  multiply_x32          the SwiGLU gate
  layer_norm_x32        the pre-norm
  A_fused / A_unfused   the WHOLE production call, silu fused vs unfused, interleaved A/B

`_UNFUSED_SILU` is release-gated: it applies silu to the bf16-packed matmul output rather than to the
fp32 dest accumulator, so it is NOT bit-exact. It is measured here to price the mechanism, not to
propose shipping it as is.
"""

import argparse
import collections
import json
import statistics as st
import time

import torch

import ttnn
import tt_bio.tenstorrent as T
from tt_bio import protenix_weights as PW
from tt_bio.tenstorrent import CORE_GRID_MAIN, PairformerLayer, get_device

CKPT = "/home/ttuser/.boltz/protenix-v2.pt"
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def build_transition(ckc):
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len("pairformer_stack.blocks.0."):]: v
           for k, v in sd.items() if k.startswith("pairformer_stack.blocks.0.")}
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    layer = PairformerLayer(32, c_z // 32, 384 // 16, 16, True, remapped, ckc)
    return layer.transition_z, c_z


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
    tz, C = build_transition(ckc)
    N, H, h = args.n, 1024, 16
    nch = N // h
    print(f"N={N} c={C} c_hid={H} grid={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)

    x = ttnn.from_torch(torch.randn(1, N, N, C) * 0.5, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    samples = collections.defaultdict(list)

    def run_phase(legs):
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

    # --- phase 1: the whole call, silu fused vs unfused, interleaved ---------------------------
    def a_fused():
        T._UNFUSED_SILU = False
        ttnn.deallocate(tz(x))

    def a_unfused():
        T._UNFUSED_SILU = True
        ttnn.deallocate(tz(x))
        T._UNFUSED_SILU = False

    def layer_norm_x32():
        c = x[:, 0:h]
        for _ in range(nch):
            ttnn.deallocate(ttnn.layer_norm(c, weight=tz.norm_weight, bias=tz.norm_bias,
                                            epsilon=1e-5, compute_kernel_config=ckc,
                                            memory_config=L1))
        ttnn.deallocate(c)

    print("--- phase 1: the whole call, and the pre-norm ---", flush=True)
    run_phase(collections.OrderedDict([("A_fused", a_fused), ("A_unfused", a_unfused),
                                       ("layer_norm_x32", layer_norm_x32)]))

    # --- phase 2: the SFPU passes, hidden resident in L1 as production has it ------------------
    hid = ttnn.from_torch(torch.randn(1, h, N, H) * 0.5, dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    hid2 = ttnn.from_torch(torch.randn(1, h, N, H) * 0.5, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)

    def silu_standalone_x32():
        for _ in range(nch):
            ttnn.silu(hid, memory_config=L1, output_tensor=hid)

    def silu_outofplace_x32():
        for _ in range(nch):
            ttnn.deallocate(ttnn.silu(hid, memory_config=L1))

    def multiply_x32():
        for _ in range(nch):
            ttnn.multiply(hid, hid2, memory_config=L1, output_tensor=hid)

    print("--- phase 2: SFPU passes on the L1 hidden ---", flush=True)
    run_phase(collections.OrderedDict([("silu_standalone_x32", silu_standalone_x32),
                                       ("silu_outofplace_x32", silu_outofplace_x32),
                                       ("multiply_x32", multiply_x32)]))

    res = {}
    print(f"\n=== N={N}, median of {args.rounds} interleaved rounds per phase ===", flush=True)
    for k, v in samples.items():
        v = [s for s in v if s == s]
        if not v:
            continue
        res[k] = {"ms_median": round(st.median(v), 4), "ms_min": round(min(v), 4),
                  "ms_max": round(max(v), 4), "n": len(v)}
        print(f"  {k:22s} {st.median(v):9.3f} ms  (min {min(v):.3f} max {max(v):.3f})", flush=True)
    if args.out:
        json.dump({"N": N, "c": C, "c_hid": H, "h": h, "legs": res,
                   "grid": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                   "samples": {k: [round(s, 4) for s in v] for k, v in samples.items()}},
                  open(args.out, "w"), indent=2)
        print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
