#!/usr/bin/env python3
"""Does the per-core rate of a big output block survive FULL grid occupancy?

`unit_rate.py` measured 870 GF/s/core on 16 cores and 1057 on 4, against the shipped config s
590 on 64. Both fast arms ran on a nearly empty chip, so the rate could be an artefact of having
the DRAM and the NOC to themselves. This asks the same question with every core busy.

Two ways to fill the grid at a 128x128-per-core block, neither of which the shipped config can
reach, because `fuse_batch=False` tiles ONE batch item over the grid and 16x16 output tiles at
4x4 per core is 16 cores:

  REUSE   `MatmulMultiCoreReuseProgramConfig` -- the non-multicast batched variant. It spreads
          batch x M-blocks x N-blocks over the grid, so the batch axis fills the cores. This is
          the closest stock analogue of the DESIGN section 4 kernel and it costs nothing to run.
  MCAST-b the multicast config at a batch sized so the 4x4 grid is busy for a long time.

Also measures the DRAM roof on this board rather than asserting it, because the no-multicast
route s cost is bytes and a byte roof that is quoted rather than measured has retracted a number
in this campaign family before.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from clocksample import during  # noqa: E402

import tt_bio  # noqa: E402
assert str(REPO) in tt_bio.__file__, f"wrong tt_bio: {tt_bio.__file__}"
from tt_bio import tenstorrent as tt  # noqa: E402

S, CZ, N_REP = 512, 128, 7
TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
PIN_NODE = int(os.environ.get("TMK_PIN_NODE", "1"))
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def pin():
    p = subprocess.Popen([sys.executable, str(REPO / "perf/pvxcust/pin_aiclk.py"),
                          str(PIN_NODE), "1350"], stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        print("  pin:", line.rstrip(), flush=True)
        if line.strip() == "READY":
            return p
    raise RuntimeError("no READY")


def timed(fn, dev):
    ts = []
    for _ in range(N_REP + 2):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(out)
    ts = sorted(ts[2:])
    return ts[len(ts) // 2]


def main():
    proc = pin()
    dev = tt.get_device()
    torch.manual_seed(0)
    f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rows = []

    # ---- DRAM roof, measured on this board, not quoted -------------------------------------
    big = f(torch.randn(1, 1, 8192, 8192))
    with during(period=2.0) as clkd:
        t = timed(lambda: ttnn.clone(big, memory_config=ttnn.DRAM_MEMORY_CONFIG), dev)
    by = 2 * 8192 * 8192 * 2
    dram_gbs = by / (t * 1e-3) / 1e9
    print(f"  DRAM clone 8192^2 bf16 {t:.4f} ms -> {dram_gbs:.1f} GB/s (r+w)", flush=True)
    ttnn.deallocate(big)

    # ---- occupancy arms ---------------------------------------------------------------------
    # (label, batch, kind, grid, per_core_M, per_core_N, sub_h, sub_w, in0_block_w)
    ARMS = [
        ("MCAST  shipped 11x10 pc2x2 sub1x1", CZ, "mc", (11, 10), 2, 2, 1, 1, 16),
        ("MCAST  4x4 pc4x4 sub1x4", CZ, "mc", (4, 4), 4, 4, 1, 4, 16),
        ("REUSE  11x10 pc4x4 sub1x4", CZ, "re", (11, 10), 4, 4, 1, 4, 16),
        ("REUSE  11x10 pc4x4 sub2x2", CZ, "re", (11, 10), 4, 4, 2, 2, 16),
        ("REUSE  11x10 pc8x8 sub2x2", CZ, "re", (11, 10), 8, 8, 2, 2, 8),
        ("REUSE  11x10 pc16x16 sub2x2", CZ, "re", (11, 10), 16, 16, 2, 2, 2),
        ("REUSE  11x10 pc2x2 sub1x1", CZ, "re", (11, 10), 2, 2, 1, 1, 16),
        ("REUSE  8x8 pc4x4 sub1x4", CZ, "re", (8, 8), 4, 4, 1, 4, 16),
        ("REUSE  11x10 pc4x16 sub1x4", CZ, "re", (11, 10), 4, 16, 1, 4, 8),
        ("REUSE  11x10 pc16x4 sub4x1", CZ, "re", (11, 10), 16, 4, 4, 1, 8),
    ]
    a = f(torch.randn(1, CZ, S, S) * 0.05)
    b = f(torch.randn(1, CZ, S, S) * 0.05)
    ref_ms = None
    with during(period=2.0) as clk:
        for label, bat, kind, g, pm, pn, sh, sw, bw in ARMS:
            if kind == "mc":
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=g, in0_block_w=bw,
                    out_subblock_h=sh, out_subblock_w=sw, out_block_h=pm, out_block_w=pn,
                    per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                    fused_activation=None, fuse_batch=False)
            else:
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=g, in0_block_w=bw,
                    out_subblock_h=sh, out_subblock_w=sw, per_core_M=pm, per_core_N=pn)
            try:
                ms = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=KC,
                                               memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                               program_config=pc, dtype=ttnn.bfloat16), dev)
            except Exception as exc:                                       # noqa: BLE001
                rows.append({"arm": label, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                print(f"  {label:34s} REFUSED {str(exc)[:90]}", flush=True)
                continue
            fl = bat * 2 * S * S * S
            tfs = fl / (ms * 1e-3) / 1e12
            if ref_ms is None:
                ref_ms = ms
            rows.append({"arm": label, "kind": kind, "grid": list(g), "pc": [pm, pn, sh, sw, bw],
                         "ms_median": round(ms, 4), "tflops": round(tfs, 3),
                         "vs_shipped": round(ref_ms / ms, 4)})
            print(f"  {label:34s} {ms:8.4f} ms  {tfs:7.3f} TF/s  {ref_ms/ms:6.3f}x shipped",
                  flush=True)

    out = {"host": TAG, "board": "p150a", "card_umd": 0, "S": S, "cz": CZ, "n_rep": N_REP,
           "dram_gbs_measured": round(dram_gbs, 1), "clock_dram": clkd.summary(),
           "clock": clk.summary(), "arms": rows}
    (HERE / f"occupancy_{TAG}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["clock"]), flush=True)
    proc.terminate(); proc.wait(timeout=30)


if __name__ == "__main__":
    main()
