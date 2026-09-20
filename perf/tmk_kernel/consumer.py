#!/usr/bin/env python3
"""Pass 2. Price the CONSUMER route against the composite it would replace, and test the one
sub-question that decides whether it has a prize at all.

The fused consumer kernel replaces three programs -- gated move for a, gated move for b, and the
contraction -- so K-B s denominator is that composite, not the matmul alone. Byte accounting, all
at 512 aa with C = 128 and the fused projection `blk` = [1,512,512,512] bf16:

  shipped composite  read 2 x 134.2 (blk quarters) + 134.2 (a,b) + write 2 x 67.1 + 67.1 = 603.9 MB
  fused consumer     read 268.4 (blk once) + write 67.1                                  = 335.5 MB

1.80x fewer bytes, which is over K-B s 1.30x bar -- but only if the fused kernel s gather runs at
the rate the composite runs at. The shipped gated move does 134.2 MB in ~1.73 ms, 77.4 GB/s. If the
gather is stuck there, reading 268.4 MB costs 3.47 ms and the fused kernel loses to the thing it
replaces before it computes anything. `tmk-assumptions` measured a tile-granular control at 364.7
GB/s but used `transpose(-2,-1)`, which is the hardware s native tile transpose and NOT the d-axis
rotation, so it bounds the memory system and not this permutation. The repo already contains an
honest tile-granular version of THIS permutation -- `reblock_permute`, the ungated forward move,
reported at 221 GB/s at N=1024 and gated off below N=384. This measures it at N=512 on this card.

Roof control is taken IN THIS SESSION ON THIS CARD: `tmk-assumptions` measured 123.65 on one chip
and 113.44 on another of the same board class, 8.3 % apart, so a quoted roof from another card is
not a roof here.
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
import torch, ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE.parent))
from clocksample import during  # noqa: E402
import tt_bio  # noqa: E402
assert str(REPO) in tt_bio.__file__
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import reblock_permute as RB  # noqa: E402

S, C, N_REP = 512, 128, 7
TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
PIN_NODE = int(os.environ.get("TMK_PIN_NODE", "1"))
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
SHIPPED_PC = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(11, 10), in0_block_w=16, out_subblock_h=1, out_subblock_w=1,
    out_block_h=2, out_block_w=2, per_core_M=2, per_core_N=2, transpose_mcast=False,
    fused_activation=None, fuse_batch=False)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def pin():
    p = subprocess.Popen([sys.executable, str(REPO / "perf/pvxcust/pin_aiclk.py"),
                          str(PIN_NODE), "1350"], stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        print("  pin:", line.rstrip(), flush=True)
        if line.strip() == "READY":
            return p
    raise RuntimeError("no READY")


def timed(fn, dev, rep=N_REP, free=True):
    ts = []
    for _ in range(rep + 2):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if free and out is not None:
            ttnn.deallocate(out)
    ts = sorted(ts[2:])
    return ts[len(ts) // 2]


def main():
    proc = pin(); dev = tt.get_device(); torch.manual_seed(0)
    res = {"host": TAG, "board": "p150a", "card_umd": 0, "S": S, "C": C, "n_rep": N_REP}
    f = lambda x, mc=DRAM: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=dev, memory_config=mc)

    with during(period=2.0) as clk:
        # ---- roof control, this card, this session ------------------------------------------
        q = f(torch.randn(1, 1, 4096, 4096) * 0.05)
        ms = timed(lambda: ttnn.matmul(q, q, compute_kernel_config=KC, memory_config=DRAM,
                                       dtype=ttnn.bfloat16), dev)
        res["cube_tflops"] = round(2 * 4096**3 / (ms * 1e-3) / 1e12, 2)
        res["cube_ms"] = round(ms, 4)
        print("  ROOF cube 4096^3 this card   %8.4f ms  %7.2f TFLOP/s" % (ms, res["cube_tflops"]), flush=True)
        ttnn.deallocate(q)

        # ---- the move census, same 134.2 MB moved every way we have ------------------------
        blk = f(torch.randn(1, S, S, 4 * C) * 0.05)              # fused projection, 268.4 MB
        chan = f(torch.randn(1, C, S, S) * 0.05)                 # already channel-major
        pair = f(torch.randn(1, S, S, C) * 0.05)                 # one role, pair-major
        moved = C * S * S * 2                                    # 67.1 MB produced
        mv = {}

        def note(k, ms_, rd_wr_mb):
            mv[k] = {"ms": round(ms_, 4), "gbs": round(rd_wr_mb * 1e6 / (ms_ * 1e-3) / 1e9, 1)}
            print("  MOVE %-38s %8.4f ms  %6.1f GB/s" % (k, ms_, mv[k]["gbs"]), flush=True)

        a_out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, S, S]), ttnn.bfloat16,
                                               ttnn.TILE_LAYOUT, dev, DRAM)
        p_off, g_off = tt.gp_off("p_a", C), tt.gp_off("g_a", C)
        note("gated move (SHIPPED) blk->a DRAM",
             timed(lambda: RB.reblock_permute_gated(blk, p_off, g_off, C, out=a_out) and None,
                   dev, free=False), (2 * moved + moved) / 1e6)
        note("reblock_permute ungated (tile-granular)",
             timed(lambda: RB.reblock_permute(pair, memory_config=DRAM), dev),
             2 * moved / 1e6)
        note("ttnn.permute (0,3,1,2) fallback",
             timed(lambda: ttnn.permute(pair, (0, 3, 1, 2), memory_config=DRAM), dev),
             2 * moved / 1e6)
        note("clone control, same bytes (copy speed)",
             timed(lambda: ttnn.clone(chan, memory_config=DRAM), dev), 2 * moved / 1e6)
        note("transpose(-2,-1) native tile-transpose",
             timed(lambda: ttnn.transpose(chan, -2, -1, memory_config=DRAM), dev),
             2 * moved / 1e6)
        res["moves"] = mv

        # ---- the composite K-B must be scored against --------------------------------------
        b_out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, C, S, S]), ttnn.bfloat16,
                                               ttnn.TILE_LAYOUT, dev, DRAM)

        def composite():
            RB.reblock_permute_gated(blk, p_off, g_off, C, out=a_out)
            RB.reblock_permute_gated(blk, tt.gp_off("p_b", C), tt.gp_off("g_b", C), C, out=b_out)
            return ttnn.matmul(a_out, b_out, compute_kernel_config=KC, memory_config=DRAM,
                               program_config=SHIPPED_PC, dtype=ttnn.bfloat16, transpose_b=True)
        cms = timed(composite, dev)
        mms = timed(lambda: ttnn.matmul(a_out, b_out, compute_kernel_config=KC, memory_config=DRAM,
                                        program_config=SHIPPED_PC, dtype=ttnn.bfloat16,
                                        transpose_b=True), dev)
        res["composite_ms"] = round(cms, 4)
        res["matmul_ms"] = round(mms, 4)
        res["matmul_tflops"] = round(C * 2 * S**3 / (mms * 1e-3) / 1e12, 3)
        res["composite_mb"] = 603.9
        res["composite_gbs"] = round(603.9e6 / (cms * 1e-3) / 1e9, 1)
        res["fused_mb"] = 335.5
        print("  COMPOSITE 2x gated move + matmul %8.4f ms  %6.1f GB/s   (matmul alone %.4f ms, %.2f TFLOP/s)" % (cms, res["composite_gbs"], mms, res["matmul_tflops"]), flush=True)

        # what the fused consumer would cost if its gather ran at each measured rate
        proj = {}
        for k, v in mv.items():
            proj[k] = round(335.5e6 / (v["gbs"] * 1e9) * 1e3, 4)
        res["fused_ms_if_gather_at"] = proj
        res["fused_x_if_gather_at"] = {k: round(cms / v, 3) for k, v in proj.items()}
        for k in proj:
            print("  PROJ fused at %-34s %8.4f ms  %6.3f x composite" % (k[:34], proj[k], cms/proj[k]), flush=True)

    res["clock"] = clk.summary()
    (HERE / f"consumer_{TAG}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["clock"]), flush=True)
    proc.terminate(); proc.wait(timeout=30)


if __name__ == "__main__":
    main()
