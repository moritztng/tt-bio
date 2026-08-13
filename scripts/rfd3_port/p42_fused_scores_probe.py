"""L6b: the fused score+bias kernel, gated on torch.equal and priced against the shipped chain.

The shipped path spends five ops per atom-attention call at ``[1, 4, 3359, 3360]``: a ``-1e4``
bf16 template, ``ttnn.scatter`` of the neighbour pair bias into it, the bias' ``bf16 -> fp32``
widen, the scores' widen, and the scaled ``add`` that folds ``head_dim**-0.5`` in as a
MUL_UNARY_SFPU activation. ``state/rfd3-host-half.md`` §3 prices those at 8.5 ms/call, and L6a
(``sparse_bias_fp32``) already replaced the first three with one 0.932 ms pass. This probe gates
the fusion of all five: ``rfd3_bias.fused_scores_bias_fp32`` never materialises the dense fp32
bias at all.

Two gates, both required:

1. ``torch.equal`` on the full [1, H, L, N] fp32 result against the five-op chain, and again on
   ``ttnn.softmax`` of it, on three index distributions. The parity argument is in
   ``compute_fused_scores.cpp``: the widen is a shift, ttnn's fp32 add is the SFPU's, and the
   scaled operand survives ttnn's intermediate fp32 pack unchanged -- so this is a check of an
   argument, not a hope.
2. device time per call for the fused op against both halves of what it replaces, sync-bracketed,
   after a warm call.

Run:
    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half PYTHONPATH=$PWD \\
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/p42_fused_scores_probe.py \\
      --out perf/p42/fused_scores.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

from tt_bio import rfd3_bias
from scripts.rfd3_port.p36_bias_kernel_probe import align_tile, make_indices, timed

REPS = 5


def one_shape(device, H, L, K, kind, results, scale, slots=(None,)):
    N = align_tile(L)
    print(f"\n=== H{H}_L{L}_K{K}_{kind} ===", flush=True)

    idx = make_indices(kind, L, K)
    pb = torch.randn(1, H, L, K)
    sc = torch.randn(1, H, L, N)
    pair_bias = ttnn.from_torch(
        pb, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    scores = ttnn.from_torch(
        sc, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    idx_tiled = ttnn.from_torch(
        idx.unsqueeze(1).expand(1, H, L, K).contiguous().to(torch.int32),
        layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.uint32,
    )
    idx_rm = ttnn.from_torch(
        idx.unsqueeze(1).to(torch.int32).contiguous(),
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32,
    )
    template = ttnn.full(
        (1, H, L, N), -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    act = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)]

    def ref_bias():
        """The three ops L6a already replaces, kept as the reference bias source."""
        b = ttnn.scatter(template, 3, idx_tiled, pair_bias)
        f = ttnn.typecast(b, ttnn.float32, memory_config=b.memory_config())
        ttnn.deallocate(b)
        return f

    def ref_scores(bias_f):
        s = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
        o = ttnn.add(s, bias_f, input_tensor_a_activations=act)
        ttnn.deallocate(s)
        return o

    def ref_full():
        bias_f = ref_bias()
        o = ref_scores(bias_f)
        ttnn.deallocate(bias_f)
        return o

    def got():
        return rfd3_bias.fused_scores_bias_fp32(scores, pair_bias, idx_rm, scale)

    ms_bias, all_bias = timed(ref_bias, device)
    bias_f = ref_bias()
    ms_sc, all_sc = timed(lambda: ref_scores(bias_f), device)
    ms_l6a, all_l6a = timed(
        lambda: rfd3_bias.sparse_bias_fp32(pair_bias, idx_rm, device=device), device)

    for s in slots:
        if s is not None:
            rfd3_bias.F_BIAS_SLOTS = s
            rfd3_bias._FCACHE.clear()
        tag = f"H{H}_L{L}_K{K}_{kind}" + (f"_slots{s}" if s is not None else "")
        a, b = ref_full(), got()
        ta, tb = ttnn.to_torch(a), ttnn.to_torch(b)
        sa, sb = ttnn.softmax(a, dim=-1), ttnn.softmax(b, dim=-1)
        tsa, tsb = ttnn.to_torch(sa), ttnn.to_torch(sb)
        for t in (a, b, sa, sb):
            ttnn.deallocate(t)
        equal = bool(torch.equal(ta, tb))
        equal_sm = bool(torch.equal(tsa, tsb))
        maxabs = float((ta - tb).abs().max())
        if not equal:
            d = (ta != tb).nonzero()
            print(f"  {len(d)} of {ta.numel()} elements differ; first 5 {d[:5].tolist()}")
            for pos in d[:5].tolist():
                print(f"   at {pos}: ref={ta[tuple(pos)]:.8f} got={tb[tuple(pos)]:.8f}")
        ms_got, all_got = timed(got, device)
        replaced = ms_bias + ms_sc
        gbps = (90.3 + 180.6) * (H / 4) * (N / 3360) * (L / 3359) / ms_got
        print(f"bias_slots={rfd3_bias.F_BIAS_SLOTS:>3}  equal={equal} equal_softmax={equal_sm} "
              f"maxabs={maxabs:g}\n  shipped bias {ms_bias:.3f} + scores {ms_sc:.3f} = "
              f"{replaced:.3f} ms   L6a bias {ms_l6a:.3f}   fused {ms_got:.3f} ms   "
              f"{replaced / ms_got:.2f}x vs shipped   {gbps:.0f} GB/s of a 385 roof", flush=True)
        results[tag] = {
            "H": H, "L": L, "K": K, "N": N, "kind": kind,
            "bias_slots": rfd3_bias.F_BIAS_SLOTS, "out_slots": rfd3_bias.F_OUT_SLOTS,
            "window": rfd3_bias.F_WINDOW, "scores_slots": rfd3_bias.F_SCORES_SLOTS,
            "bit_exact": equal, "bit_exact_softmax": equal_sm, "maxabs": maxabs,
            "ms_ref_bias": ms_bias, "ms_ref_scores": ms_sc, "ms_l6a_bias": ms_l6a,
            "ms_fused": ms_got, "speedup_vs_shipped": replaced / ms_got,
            "implied_gbps": gbps,
            "ms_ref_bias_all": all_bias, "ms_ref_scores_all": all_sc,
            "ms_l6a_all": all_l6a, "ms_fused_all": all_got,
        }
    ttnn.deallocate(bias_f)
    for t in (pair_bias, scores, idx_tiled, idx_rm, template):
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/p42/fused_scores.json")
    ap.add_argument("--kinds", default="random,banded,mixed")
    ap.add_argument("--slots", default="", help="comma-separated cb_bias depth sweep")
    ap.add_argument("--head-dim", type=int, default=32)
    args = ap.parse_args()

    slots = tuple(int(s) for s in args.slots.split(",")) if args.slots else (None,)
    scale = args.head_dim ** -0.5
    rfd3_bias.set_enabled(True)
    rfd3_bias.set_fused_enabled(True)
    device = ttnn.open_device(device_id=0)
    results: dict = {}
    try:
        for kind in args.kinds.split(","):
            one_shape(device, 4, 3359, 128, kind, results, scale, slots)
    finally:
        ttnn.close_device(device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    bad = [k for k, v in results.items() if not v["bit_exact"]]
    print("NOT bit-exact: " + (", ".join(bad) if bad else "none"))


if __name__ == "__main__":
    main()
