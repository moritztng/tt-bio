#!/usr/bin/env python3
"""Screen DESIGN.md section 4 before writing a line of kernel C++: is the per-core rate of the
"one core owns a (channel, j-block) unit" geometry actually higher than the shipped call site
gets per core?

The shipped contraction is ttnn.matmul([1,128,512,512] @ [1,128,512,512]) under
`_triangle_mul_program_config(16)`: an 11x10 grid, per_core 2x2 tiles, out_subblock 1x1,
fuse_batch=False. 16x16 output tiles at 2x2 per core engages ceil(16/2)^2 = 64 cores and leaves
46 idle -- a closed finding, not re-opened here, only priced.

DESIGN section 4 wants one core to own a 128x128 output block with the whole 512-long K, an
operand pair of 2*128*512*2 = 262144 B (17.9 % of a core L1). That geometry IS expressible as a
stock program config: grid 4x4, per_core 4x4, which tiles one 512x512 channel across exactly 16
cores with no core idle. It cannot fill 110 cores -- fuse_batch=False tiles one batch item at a
time -- and filling them from the batch axis is what a hand-written kernel would add. So the
screen measures PER-CORE rate, and the projection to 110 cores is stated separately as a
projection, never as a measurement.

Every arm: same operands, same compute kernel config, same dtype, warm, interleaved, median of
N_REP, AICLK pinned and sampled DURING.
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

S = 512
K_T = S // 32                      # 16 tiles
CZ = 128                           # channels per contraction call at 512 aa (sites_512_qb1c0)
N_REP = 7
TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
PIN_NODE = int(os.environ.get("TMK_PIN_NODE", "1"))   # UMD 0 == bus 01:00.0 == sysfs node 1
PIN_TARGET = 1350

KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def pin():
    p = subprocess.Popen([sys.executable, str(REPO / "perf/pvxcust/pin_aiclk.py"),
                          str(PIN_NODE), str(PIN_TARGET)],
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        print("  pin:", line.rstrip(), flush=True)
        if line.strip() == "READY":
            return p
    raise RuntimeError("pin_aiclk never reported READY")


def cfg(gx, gy, pm, pn, osh, osw, bw):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
        out_subblock_h=osh, out_subblock_w=osw, out_block_h=pm, out_block_w=pn,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None,
        fuse_batch=False)


def cores_engaged(gx, gy, pm, pn):
    """Cores a MultiCastProgramConfig actually lights up for a K_T x K_T tile output."""
    return min(gy, -(-K_T // pm)) * min(gx, -(-K_T // pn))


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
    return ts[len(ts) // 2], ts[0], ts[-1]


def main():
    proc = pin()
    dev = tt.get_device()
    torch.manual_seed(0)
    f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    rows = []
    # Arms. Each is (label, batch, (gx,gy,per_core_M,per_core_N,out_sub_h,out_sub_w,in0_block_w)).
    # B is set per arm so every arm does ~1 ms of work: a 0.1 ms arm reads the 0.05142 ms host
    # bracket floor `fixterm-decompose` measured, not the kernel.
    ARMS = [
        ("SHIPPED  11x10 pc2x2 sub1x1", CZ, (11, 10, 2, 2, 1, 1, 16)),
        ("shipped-grid-only 8x8 pc2x2 sub1x1", CZ, (8, 8, 2, 2, 1, 1, 16)),
        ("sub2x2   11x10 pc2x2 sub2x2", CZ, (11, 10, 2, 2, 2, 2, 16)),
        ("UNIT     4x4 pc4x4 sub1x1", 32, (4, 4, 4, 4, 1, 1, 16)),
        ("UNIT     4x4 pc4x4 sub2x2", 32, (4, 4, 4, 4, 2, 2, 16)),
        ("UNIT     4x4 pc4x4 sub4x1", 32, (4, 4, 4, 4, 4, 1, 16)),
        ("UNIT     4x4 pc4x4 sub1x4", 32, (4, 4, 4, 4, 1, 4, 16)),
        ("UNIT2    2x2 pc8x8 sub2x2", 8, (2, 2, 8, 8, 2, 2, 16)),
        ("SOLO     1x1 pc16x16 sub2x2", 2, (1, 1, 16, 16, 2, 2, 16)),
        ("wide     8x8 pc2x2 sub2x2", CZ, (8, 8, 2, 2, 2, 2, 16)),
    ]
    ops = {}
    for _, b, _c in ARMS:
        if b not in ops:
            ops[b] = (f(torch.randn(1, b, S, S) * 0.05), f(torch.randn(1, b, S, S) * 0.05))

    with during(period=2.0) as clk:
        for label, b, c in ARMS:
            a, bb = ops[b]
            pc = cfg(*c)
            try:
                run = lambda: ttnn.matmul(a, bb, compute_kernel_config=KC,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                          program_config=pc, dtype=ttnn.bfloat16,
                                          transpose_b=True)
                med, lo, hi = timed(run, dev)
            except Exception as exc:                                      # noqa: BLE001
                rows.append({"arm": label, "batch": b, "cfg": list(c),
                             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                print(f"  {label:34s} REFUSED {type(exc).__name__}", flush=True)
                continue
            gx, gy, pm, pn = c[0], c[1], c[2], c[3]
            ce = cores_engaged(gx, gy, pm, pn)
            fl = b * 2 * S * S * S
            tfs = fl / (med * 1e-3) / 1e12
            rows.append({"arm": label, "batch": b, "cfg": list(c), "cores_engaged": ce,
                         "ms_median": round(med, 4), "ms_min": round(lo, 4),
                         "ms_max": round(hi, 4), "gflop": round(fl / 1e9, 3),
                         "tflops_aggregate": round(tfs, 3),
                         "gflops_per_core": round(tfs * 1e3 / ce, 1)})
            print(f"  {label:34s} {med:8.4f} ms  {tfs:7.3f} TF/s  "
                  f"{ce:3d} cores  {tfs*1e3/ce:7.1f} GF/s/core", flush=True)

    out = {"host": TAG, "board": "p150a", "card_umd": 0, "pin_node": PIN_NODE,
           "S": S, "cz": CZ, "n_rep": N_REP, "clock": clk.summary(), "arms": rows}
    (HERE / f"unit_rate_{TAG}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["clock"]), flush=True)
    proc.terminate()
    proc.wait(timeout=30)


if __name__ == "__main__":
    main()
