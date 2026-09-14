#!/usr/bin/env python3
"""Wormhole envelopes for three Blackhole-fitted constants in `tt_bio/tenstorrent.py`.

One process, one device, one session. Every arm set runs whole once per block in a fixed order,
so a JIT warm-up or a clock ramp cannot bias one arm against another; the reported time is the
minimum over blocks. The shipped arm runs twice under two labels, so the table carries its own
A/A floor. The dense bf16 HiFi4 4096-cube runs in the same session, so the roof every ratio is
quoted against was measured here and not asserted.

The constants are patched on the module object rather than in the source tree: the whglx
checkout under `/home/agent/tt-bio-wt` is shared with every other worker on the box.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG


def kernel_config(dev):
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    return kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, reps, blocks, dev):
    """Minimum wall time per call over `blocks` blocks of `reps` enqueued calls."""
    best = float("inf")
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
            ttnn.deallocate(out)
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e3


def part_facts(dev):
    cc = dev.compute_with_storage_grid_size()
    f = {"host": platform.node(), "arch": str(dev.arch()),
         "compute_grid_x": cc.x, "compute_grid_y": cc.y, "cores": cc.x * cc.y}
    for name, call in (("l1_size_per_core", "l1_size_per_core"),
                       ("dram_size_per_channel", "dram_size_per_channel"),
                       ("num_dram_channels", "num_dram_channels")):
        try:
            f[name] = int(getattr(dev, call)())
        except Exception as e:                                          # noqa: BLE001
            f[name] = f"n/a ({type(e).__name__})"
    return f


# ---- roof -------------------------------------------------------------------------------
def roof(dev, kc, blocks):
    a = ttnn.from_torch(torch.randn(4096, 4096, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    b = ttnn.from_torch(torch.randn(4096, 4096, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=DRAM)
    ms = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=DRAM), 3,
               blocks, dev)
    ms_aa = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=DRAM), 3,
                  blocks, dev)
    ttnn.deallocate(a); ttnn.deallocate(b)
    flops = 2 * 4096 ** 3
    return {"ms": ms, "TFLOPs": flops / (ms * 1e-3) / 1e12, "aa_ms": ms_aa,
            "aa_pct": 100 * abs(ms_aa - ms) / ms}


# ---- arm 1: _FP32_SOFTMAX_L1_GRID -------------------------------------------------------
#: AF2-IG's triangle attention, the model this constant actually serves: 4 heads, head_dim 32,
#: bias added raw. `_fp32_softmax_attention` is on that path unconditionally whenever a pair
#: bias exists (af2.py:486), so unlike Boltz-2 it is not behind BOLTZ2_FP32_SOFTMAX.
def fp32_softmax_case(dev, kc, S, heads=4, hd=32):
    q = ttnn.from_torch(torch.randn(S, heads, S, hd, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    k = ttnn.from_torch(torch.randn(S, heads, S, hd, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    v = ttnn.from_torch(torch.randn(S, heads, S, hd, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    bias = ttnn.from_torch(torch.randn(1, heads, S, S, dtype=torch.bfloat16),
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    return q, k, v, bias


def fp32_softmax_call(q, k, v, bias, kc, hd):
    return T._fp32_softmax_attention(q, k, v, bias, scale_inv=float(hd) ** -0.5,
                                     compute_kernel_config=kc, out_dtype=ttnn.bfloat16,
                                     bias_scale_inv=1.0)


def clear_fp32_state():
    T._FP32_SOFTMAX_L1_ROW_CAP.clear()
    T._FP32_SOFTMAX_L1_FREE_ROW_CAP.clear()
    T._FP32_SOFTMAX_DRAM_ROW_CAP.clear()
    # `_fp32_softmax_l1_plan` is lru_cached on the SHAPE only, so without this every arm after
    # the first reads the first arm's grid. In production the constant never moves and the cache
    # is correct; in an A/B it silently collapses the sweep to one arm.
    T._fp32_softmax_l1_plan.cache_clear()
    T._fp32_softmax_core_grid.cache_clear()
    for key in T.FP32_SOFTMAX_STATS:
        T.FP32_SOFTMAX_STATS[key] = 0


def plan_for(grid, per_row, height_per_row, k_len, float_cores):
    """What the planner returns for this grid, with no device work: (rows, cores)."""
    old_g, old_f = T._FP32_SOFTMAX_L1_GRID, T._FP32_SOFTMAX_L1_FLOAT_CORES
    T._FP32_SOFTMAX_L1_GRID, T._FP32_SOFTMAX_L1_FLOAT_CORES = grid, float_cores
    clear_fp32_state()
    try:
        return T._fp32_softmax_l1_plan(per_row, height_per_row, k_len, None, None, None)
    finally:
        T._FP32_SOFTMAX_L1_GRID, T._FP32_SOFTMAX_L1_FLOAT_CORES = old_g, old_f
        clear_fp32_state()


def sweep_fp32_softmax(dev, kc, S, grids, blocks, reps, float_cores, want_equal):
    heads, hd = 4, 32
    q, k, v, bias = fp32_softmax_case(dev, kc, S, heads, hd)
    height_per_row = heads * S
    per_row = height_per_row * S * 4
    rows = []
    ref = None
    for label, grid in grids:
        old_g, old_f = T._FP32_SOFTMAX_L1_GRID, T._FP32_SOFTMAX_L1_FLOAT_CORES
        T._FP32_SOFTMAX_L1_GRID, T._FP32_SOFTMAX_L1_FLOAT_CORES = grid, float_cores
        clear_fp32_state()
        rec = {"arm": label, "grid": list(grid), "tuned_cores": grid[0] * grid[1],
               "float_cores": float_cores, "S": S}
        try:
            plan = T._fp32_softmax_l1_plan(per_row, height_per_row, S, None, None, None)
            rec["plan_rows"], rec["plan_cores"] = int(plan[0]), int(plan[1])
            rec["ms"] = timed(lambda: fp32_softmax_call(q, k, v, bias, kc, hd), reps, blocks, dev)
            rec["stats"] = dict(T.FP32_SOFTMAX_STATS)
            if want_equal:
                out = fp32_softmax_call(q, k, v, bias, kc, hd)
                host = ttnn.to_torch(out)
                ttnn.deallocate(out)
                if ref is None:
                    ref = host
                    rec["equal_to_shipped"] = True
                else:
                    rec["equal_to_shipped"] = bool(torch.equal(host, ref))
                    rec["max_abs"] = float((host.float() - ref.float()).abs().max())
        except Exception as e:                                          # noqa: BLE001
            rec["refused"] = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            T._FP32_SOFTMAX_L1_GRID, T._FP32_SOFTMAX_L1_FLOAT_CORES = old_g, old_f
            clear_fp32_state()
        rows.append(rec)
        print("  %-18s plan=%s  %s" % (label, (rec.get("plan_rows"), rec.get("plan_cores")),
                                       rec.get("refused") or "%.4f ms" % rec["ms"]), flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fp32_softmax_grid.json")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--no-equal", action="store_true")
    a = ap.parse_args()

    # `get_device()` takes the card lease itself, over the whole visible set.
    try:
        dev = T.get_device()
        facts = part_facts(dev)
        print(json.dumps(facts), flush=True)
        kc = kernel_config(dev)
        r = roof(dev, kc, a.blocks)
        print("cube4096  %.4f ms  %.4f TFLOP/s   A/A %.3f %%"
              % (r["ms"], r["TFLOPs"], r["aa_pct"]), flush=True)

        cc = dev.compute_with_storage_grid_size()
        # (y, x) as the constant is written. Shipped first so it is the A/A reference, then the
        # whole-grid rectangle, then rungs below and one deliberately off the end of the part.
        grids = [("shipped_8x8", (8, 8)), ("aa_8x8", (8, 8)),
                 ("full_%dx%d" % (cc.y, cc.x), (cc.y, cc.x)),
                 ("y7x8", (7, 8)), ("y6x8", (6, 8)), ("y4x8", (4, 8)),
                 ("over_%dx%d" % (cc.y + 1, cc.x), (cc.y + 1, cc.x)),
                 ("over_y8x%d" % (cc.x + 2), (8, cc.x + 2))]
        out = {"facts": facts, "roof": r, "grids": [list(g) for _, g in grids], "sweeps": []}
        for S in [int(s) for s in a.sizes.split(",")]:
            for fc in (False, True):
                print("S=%d float_cores=%s" % (S, fc), flush=True)
                out["sweeps"].append({
                    "S": S, "float_cores": fc,
                    "rows": sweep_fp32_softmax(dev, kc, S, grids, a.blocks, a.reps, fc,
                                               not a.no_equal)})
    finally:
        T.cleanup()

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
