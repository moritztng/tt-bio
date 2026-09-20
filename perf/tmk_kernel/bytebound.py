#!/usr/bin/env python3
"""Is the trimul contraction rate-bound or byte-bound? The campaign s ground truth calls 3.491 ms
of the module a "kernel-rate deficit", which presumes rate. `occupancy.py` measured this board s
DRAM roof at 381.3 GB/s and the shipped contraction moving 201.3 MB in 0.9673 ms, which is 208
GB/s, 55 % of that roof -- close enough that the premise needs testing rather than assuming.

The discriminator: run the SAME matmul with the same program config, same dtype, same compute
kernel config, at a channel count small enough that all three tensors fit in L1, and compare the
L1 arm against the DRAM arm. If the op is compute-bound the two read the same; if it is
byte-bound the L1 arm pulls away. Interleaved, warm, median of 7, clock pinned and sampled.

Also runs the n-ladder in channels, because `fixterm-decompose` showed a sync;call;sync bracket
charges a 0.05142 ms host floor and an intercept read off a single point is that floor.
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

S, N_REP = 512, 7
TAG = sys.argv[1] if len(sys.argv) > 1 else "qb1c0"
PIN_NODE = int(os.environ.get("TMK_PIN_NODE", "1"))
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
SHIPPED = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(11, 10), in0_block_w=16, out_subblock_h=1, out_subblock_w=1,
    out_block_h=2, out_block_w=2, per_core_M=2, per_core_N=2, transpose_mcast=False,
    fused_activation=None, fuse_batch=False)


def pin():
    p = subprocess.Popen([sys.executable, str(REPO / "perf/pvxcust/pin_aiclk.py"),
                          str(PIN_NODE), "1350"], stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in p.stdout:
        print("  pin:", line.rstrip(), flush=True)
        if line.strip() == "READY":
            return p
    raise RuntimeError("no READY")


def timed(fn, dev, rep=N_REP):
    ts = []
    for _ in range(rep + 2):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(out)
    ts = sorted(ts[2:])
    return ts[len(ts) // 2]


def main():
    proc = pin(); dev = tt.get_device(); torch.manual_seed(0)
    rows = []
    with during(period=2.0) as clk:
        # n-ladder in channels, DRAM, so the intercept is read off a line and not one point.
        for cz in (2, 4, 8, 16, 32, 64, 128):
            mk = lambda mc: ttnn.from_torch(torch.randn(1, cz, S, S) * 0.05,
                                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                            device=dev, memory_config=mc)
            ent = {"cz": cz, "gflop": round(cz * 2 * S**3 / 1e9, 3)}
            for tagmc, mc in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
                try:
                    a, b = mk(mc), mk(mc)
                    ms = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=KC,
                                                   memory_config=mc, program_config=SHIPPED,
                                                   dtype=ttnn.bfloat16), dev)
                    ent[tagmc + "_ms"] = round(ms, 4)
                    ent[tagmc + "_tflops"] = round(cz * 2 * S**3 / (ms * 1e-3) / 1e12, 3)
                    # bytes: operand pair read once (multicast) + output written
                    by = cz * 3 * S * S * 2
                    ent[tagmc + "_gbs"] = round(by / (ms * 1e-3) / 1e9, 1)
                    ttnn.deallocate(a); ttnn.deallocate(b)
                except Exception as exc:                                    # noqa: BLE001
                    ent[tagmc + "_err"] = f"{type(exc).__name__}: {str(exc)[:110]}"
                    for t in ("a", "b"):
                        pass
            rows.append(ent)
            g = lambda k, d=0.0: ent.get(k, d)
            print("  cz=%4d  DRAM %8.4f ms %6.2f TF/s %6.1f GB/s  |  L1 %8.4f ms %6.2f TF/s  %s"
                  % (cz, g("dram_ms"), g("dram_tflops"), g("dram_gbs"),
                     g("l1_ms", -1.0), g("l1_tflops"), ent.get("l1_err", "")[:60]), flush=True)
    out = {"host": TAG, "board": "p150a", "card_umd": 0, "S": S, "n_rep": N_REP,
           "clock": clk.summary(), "ladder": rows}
    (HERE / f"bytebound_{TAG}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["clock"]), flush=True)
    proc.terminate(); proc.wait(timeout=30)


if __name__ == "__main__":
    main()
