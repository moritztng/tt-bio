#!/usr/bin/env python3
"""p71 -- where the DiT bias chain's time really is, and which fusion shape is worth building.

p70 said the six-op chain is 1.0810 ms/call and that 64 % of it is the bias side. This run splits
the bias side op by op and screens a DIFFERENT fusion shape from the one state §3 specified,
because reading the kernels made §3's cost model look incomplete:

§3 has the kernel read the compact pair bias in its PRE-permute `[1, I, J, H]` layout, so the
permute never happens. But in that layout a tile is `[32 j x 32 h]`, so one output tile
`(h, it, jt)` needs column `h` of 32 different pages, and assembling it is a per-element strided
gather -- 16 x 685 x 704 = 7.7 M scalar L1 moves per call. §3's cost model
(`r 31.7/390 + w 30.9/269.6 = 0.196 ms`) counts none of it. The sparse kernel's own measured poke
rate (3.1 M pokes in 0.819 ms of a 1.68 ms call) puts that gather at ~2.1 ms/call on its own,
i.e. worse than the whole shipped chain. So §3's kernel is priced wrong, and the arms below price
the alternative.

The alternative, in two independent halves:

  * **batched bias prep, pure ttnn, no kernel.** `_PAIRBIAS_FUSED` already makes ONE
    `[1, I, J, 18*32]` projection per recycle. Do the permute, the mask add and the key-axis pad
    ONCE on that 576-wide tensor instead of 18 times on a 16-wide slice. Same bytes either way --
    the win is entirely op efficiency, because a 16-wide last dim is half a tile and p70 measured
    the per-block permute at 77 GB/s against 334.6 for a clone of the same bytes.
  * **a streaming kernel** for the last three ops (`typecast`, `typecast`, scaled add) that reads
    the ALREADY-permuted bf16 bias at a dim-1 head offset, so no slice and no gather. That kernel
    needs one new reader; `compute_fused_scores.cpp` and `writer_fused_scores.cpp` are reusable.

Arm K is the cheap thing that would make the kernel unnecessary: ttnn's own binary add with bf16
operands and an fp32 output. Timed AND checked with `torch.equal`, because `model.py:1533`'s
comment says the folded form rounds the scaled operand to the input dtype -- if that is what
happens here it is not exact and the kernel is needed after all.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p71_bias_prep_arms.py \
          perf/p71/bias_prep_arms.json
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p71/bias_prep_arms.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5

I = 685
J = 685
H = 16
NKEY = 704
SLOT = 32              # _PAIRBIAS_SLOT
NBLK = 18              # DiT blocks
CWIDE = SLOT * NBLK    # 576
HEAD_DIM = 48
SCALE = HEAD_DIM ** -0.5
CALLS = 36             # block calls per step
RECYCLES = 2           # batched prep runs once per recycle
MB = 1024.0 * 1024.0
RES = {}


def mb(*ts):
    tot = 0.0
    for t in ts:
        n = 1
        for d in t.padded_shape:
            n *= int(d)
        tot += n * (4 if t.dtype == ttnn.float32 else 2)
    return tot / MB


def timeit(fn, n=N, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(DEV)
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out), [round(v, 4) for v in out]


def arm(name, fn, bytes_mb, per_step, note=""):
    ms, all_ms = timeit(fn)
    RES[name] = {"ms": round(ms, 4), "all": all_ms, "mb": round(bytes_mb, 1),
                 "gb_s": round(bytes_mb / 1024.0 / (ms / 1e3), 1),
                 "calls_per_step": per_step,
                 "ms_per_step": round(ms * per_step, 3), "note": note}
    print("[p71] %-28s %8.4f ms  %7.1f MB  %6.1f GB/s   x%-3d = %7.3f ms/step  %s"
          % (name, ms, bytes_mb, bytes_mb / 1024.0 / (ms / 1e3), per_step,
             ms * per_step, note), flush=True)
    return ms


def main():
    global DEV
    DEV = get_device()
    torch.manual_seed(0)
    tt = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(          # noqa: E731
        x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    mask_t = torch.where(torch.rand(1, 1, I, J) < 0.05, -1e4, 0.0)
    mask = tt(mask_t)
    scores_t = torch.randn(1, H, I, NKEY) * 0.3
    scores = tt(scores_t)

    # --- the shipped per-block ops, each alone -------------------------------------------------
    pb16 = tt(torch.randn(1, I, J, H) * 0.5)
    arm("I_permute_16wide", lambda: ttnn.deallocate(ttnn.permute(pb16, (0, 3, 1, 2))),
        2 * mb(pb16), CALLS, "the shipped per-block permute")

    pbp = ttnn.permute(pb16, (0, 3, 1, 2))               # [1,H,I,J]
    arm("J_add_mask_16wide", lambda: ttnn.deallocate(ttnn.add(pbp, mask)),
        2 * mb(pbp) + mb(mask), CALLS, "the shipped per-block mask add")
    biasm = ttnn.add(pbp, mask)
    padspec = [(0, 0)] * 4
    padspec[3] = (0, NKEY - J)
    arm("J2_pad_16wide", lambda: ttnn.deallocate(ttnn.pad(biasm, padspec, -1e4)),
        2 * mb(biasm), CALLS, "the shipped per-block key-axis pad")
    bias_bf = ttnn.pad(biasm, padspec, -1e4)             # [1,H,I,704] bf16, the ref operand
    ttnn.deallocate(pbp)
    ttnn.deallocate(biasm)
    ttnn.deallocate(pb16)

    # --- the batched-prep arms, once per recycle on the 576-wide projection --------------------
    wide = tt(torch.randn(1, I, J, CWIDE) * 0.5)
    print("[p71] wide projection tensor %.1f MB" % mb(wide), flush=True)
    arm("E_permute_576wide", lambda: ttnn.deallocate(ttnn.permute(wide, (0, 3, 1, 2))),
        2 * mb(wide), RECYCLES, "one permute for all 18 blocks")
    widep = ttnn.permute(wide, (0, 3, 1, 2))            # [1,576,I,J]
    ttnn.deallocate(wide)
    arm("F_add_mask_576wide", lambda: ttnn.deallocate(ttnn.add(widep, mask)),
        2 * mb(widep) + mb(mask), RECYCLES, "one mask add, broadcast over dim 1")
    widem = ttnn.add(widep, mask)
    ttnn.deallocate(widep)
    arm("G_pad_576wide", lambda: ttnn.deallocate(ttnn.pad(widem, padspec, -1e4)),
        2 * mb(widem), RECYCLES, "one key-axis pad")
    widepad = ttnn.pad(widem, padspec, -1e4)            # [1,576,I,704]
    ttnn.deallocate(widem)

    def slice1():
        o = ttnn.slice(widepad, [0, 0, 0, 0], [1, H, I, NKEY])
        ttnn.deallocate(o)

    arm("H_slice_dim1_16", slice1, 2 * mb(bias_bf), CALLS,
        "per-block dim-1 slice, needed ONLY if no kernel reads at an offset")

    # --- the three ops a streaming kernel would replace, and ttnn's own folded form -----------
    def two_casts_and_add():
        sf = ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config())
        bf = ttnn.typecast(bias_bf, ttnn.float32, memory_config=bias_bf.memory_config())
        o = ttnn.add(sf, bf, input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
        for t in (sf, bf, o):
            ttnn.deallocate(t)

    arm("L_shipped_tail_3op", two_casts_and_add,
        mb(scores) + mb(bias_bf) + 4 * (mb(scores) + mb(bias_bf)) / 2 * 3, CALLS,
        "typecast + typecast + scaled add: what the kernel replaces")

    def bf16_add_fp32_out():
        o = ttnn.add(scores, bias_bf, dtype=ttnn.float32,
                     input_tensor_a_activations=[
                         ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
        ttnn.deallocate(o)

    try:
        k_ms = arm("K_bf16_add_fp32_out", bf16_add_fp32_out,
                   mb(scores) + mb(bias_bf) + 2 * mb(scores) * 2, CALLS,
                   "ttnn's folded form -- the zero-kernel candidate")
    except Exception as e:                                              # noqa: BLE001
        k_ms = None
        RES["K_bf16_add_fp32_out"] = {"error": str(e)[:300]}
        print("[p71] K rejected: %s" % str(e)[:200], flush=True)

    # --- exactness: is arm K bit-identical to the shipped tail? -------------------------------
    ref = ttnn.add(
        ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config()),
        ttnn.typecast(bias_bf, ttnn.float32, memory_config=bias_bf.memory_config()),
        input_tensor_a_activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
    ref_h = ttnn.to_torch(ref).float()
    RES["exactness"] = {}
    if k_ms is not None:
        got = ttnn.add(scores, bias_bf, dtype=ttnn.float32,
                       input_tensor_a_activations=[
                           ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE)])
        got_h = ttnn.to_torch(got).float()
        eq = bool(torch.equal(ref_h, got_h))
        mx = float((ref_h - got_h).abs().max())
        RES["exactness"]["K_vs_shipped_tail"] = {"torch_equal": eq, "maxabs": mx}
        print("[p71] arm K vs shipped tail: torch.equal=%s  maxabs=%.6g" % (eq, mx), flush=True)

    # batched prep vs per-block prep, on block 0's slice
    sl = ttnn.slice(widepad, [0, 0, 0, 0], [1, H, I, NKEY])
    RES["exactness"]["batched_prep_shape"] = [int(d) for d in sl.shape]
    ttnn.deallocate(sl)

    # --- the arithmetic -----------------------------------------------------------------------
    shipped_step = 1.0810 * CALLS       # p70 arm A, the whole six-op chain
    per_block_prep = (RES["I_permute_16wide"]["ms"] + RES["J_add_mask_16wide"]["ms"]
                      + RES["J2_pad_16wide"]["ms"]) * CALLS
    batched_prep = (RES["E_permute_576wide"]["ms"] + RES["F_add_mask_576wide"]["ms"]
                    + RES["G_pad_576wide"]["ms"]) * RECYCLES
    tail = RES["L_shipped_tail_3op"]["ms"] * CALLS
    slices = RES["H_slice_dim1_16"]["ms"] * CALLS
    RES["arithmetic"] = {
        "p70_chain_ms_per_step": round(shipped_step, 3),
        "shipped_prep_ms_per_step": round(per_block_prep, 3),
        "shipped_tail_ms_per_step": round(tail, 3),
        "batched_prep_ms_per_step": round(batched_prep, 3),
        "dim1_slices_ms_per_step": round(slices, 3),
        "A_batched_prep_plus_slices_no_kernel": round(
            per_block_prep - (batched_prep + slices), 3),
        "B_batched_prep_no_slices_kernel_reads_at_offset": round(
            per_block_prep - batched_prep, 3),
    }
    if k_ms is not None:
        RES["arithmetic"]["C_tail_replaced_by_arm_K"] = round((tail - k_ms * CALLS), 3)
    print("\n[p71] per-block prep     %7.3f ms/step" % per_block_prep)
    print("[p71] batched prep      %7.3f ms/step  (+ %7.3f if dim-1 slices are needed)"
          % (batched_prep, slices))
    print("[p71] shipped tail      %7.3f ms/step" % tail)
    for k, v in RES["arithmetic"].items():
        if k.startswith(("A_", "B_", "C_")):
            print("[p71] saving %-45s %+8.3f ms/step  %+.3f s/design"
                  % (k, v, v * 200 / 1e3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"shape": {"I": I, "J": J, "H": H, "n_key": NKEY,
                                         "c_wide": CWIDE, "head_dim": HEAD_DIM},
                               "n": N, "arms": RES}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
