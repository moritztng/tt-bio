#!/usr/bin/env python3
"""De-risk pass 3: is the a/b DRAM round trip actually worth deleting?

The fused consumer kernel's entire saving over the shipped composite is that the moved operands
never reach DRAM -- the gather writes them into the matmul's CBs instead. Its gather transaction
count is unchanged (`trimul-fused-kernel-final`: L1->L1 is only 1.20x over DRAM->DRAM because the
writer RISC's transaction count binds, not where the bytes live). So the fusion's prize is exactly
the DRAM write of a and b plus the matmul's DRAM read of them, and that is measurable today by
pointing the SHIPPED gated move at an L1 destination and letting the SHIPPED matmul read it there.

No kernel needed. If this reads ~1.0x, the fused consumer cannot be worth building and K-B fires
on the route; if it reads near the 1.88x byte projection, pass 3 has its mandate.

Chunked over channels because a + b at C=128 is 134.2 MB against ~134.11 MB of usable L1 -- that
exact residency is already on the DO-NOT-REPEAT list, so this runs the ladder below it. Arms
interleaved in one process, warm, median of 7, AICLK pinned and sampled DURING.
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

S, N_REP = 512, 7
TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
PIN_NODE = int(os.environ.get("TMK_PIN_NODE", "1"))
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def pin():
    p = subprocess.Popen([sys.executable, str(REPO / "perf/pvxcust/pin_aiclk.py"),
                          str(PIN_NODE), "1350"], stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        print("  pin: " + line.rstrip(), flush=True)
        if line.strip() == "READY":
            return p
    raise RuntimeError("no READY")


def pcfg(st):
    pm = max(1, -(-st // 10)); pn = max(1, -(-st // 11))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(11, 10), in0_block_w=st, out_subblock_h=1,
        out_subblock_w=1, out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn,
        transpose_mcast=False, fused_activation=None, fuse_batch=False)


def timed(fn, dev):
    ts = []
    for _ in range(N_REP + 2):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if out is not None:
            ttnn.deallocate(out)
    ts = sorted(ts[2:])
    return ts[len(ts) // 2]


def main():
    proc = pin(); dev = tt.get_device(); torch.manual_seed(0)
    rows = []
    PC = pcfg(S // 32)
    with during(period=2.0) as clk:
        for G in (32, 64):
            blk = ttnn.from_torch(torch.randn(1, S, S, 4 * G) * 0.05, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            ent = {"chunk_channels": G}
            for tag, mc in (("dram", DRAM), ("l1", L1)):
                try:
                    a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, S, S]), ttnn.bfloat16,
                                                       ttnn.TILE_LAYOUT, dev, mc)
                    b = ttnn.allocate_tensor_on_device(ttnn.Shape([1, G, S, S]), ttnn.bfloat16,
                                                       ttnn.TILE_LAYOUT, dev, mc)

                    def comp(a=a, b=b, mc=mc):
                        RB.reblock_permute_gated(blk, tt.gp_off("p_a", G), tt.gp_off("g_a", G),
                                                 G, out=a)
                        RB.reblock_permute_gated(blk, tt.gp_off("p_b", G), tt.gp_off("g_b", G),
                                                 G, out=b)
                        return ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=DRAM,
                                           program_config=PC, dtype=ttnn.bfloat16,
                                           transpose_b=True)
                    ms = timed(comp, dev)
                    ent[tag + "_ms"] = round(ms, 4)
                    ttnn.deallocate(a); ttnn.deallocate(b)
                except Exception as exc:                                   # noqa: BLE001
                    ent[tag + "_err"] = type(exc).__name__ + ": " + str(exc)[:120]
            if "dram_ms" in ent and "l1_ms" in ent:
                ent["l1_speedup"] = round(ent["dram_ms"] / ent["l1_ms"], 4)
            rows.append(ent)
            print("  C=%3d  composite DRAM operands %8.4f ms  |  L1 operands %8.4f ms  -> %.4fx  %s"
                  % (G, ent.get("dram_ms", -1), ent.get("l1_ms", -1), ent.get("l1_speedup", 0),
                     ent.get("l1_err", "")[:70]), flush=True)
            ttnn.deallocate(blk)
    out = {"host": TAG, "board": "p150a", "card_umd": 0, "S": S, "n_rep": N_REP,
           "clock": clk.summary(), "ladder": rows}
    (HERE / ("l1_round_trip_" + TAG + ".json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["clock"]), flush=True)
    proc.terminate(); proc.wait(timeout=30)


if __name__ == "__main__":
    main()
