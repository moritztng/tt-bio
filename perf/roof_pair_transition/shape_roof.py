#!/usr/bin/env python3
"""The shape-honest roof for `Transition|1x512x512x128`, and where its time actually goes.

One process, one device, one session. Every arm is timed the same way -- R enqueues per sync so
host dispatch is amortised, B blocks, minimum over blocks -- and the blocks INTERLEAVE the arms
(arm order is fixed, the whole arm set runs once per block) so a JIT-warm-up or a clock ramp
cannot bias one arm against another.

The dense cube runs in the same session as everything else, so the ratio `arm / cube` is
internally consistent even when the card is not the card the budget was taken on.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio.device_lease import CardSetLease                                   # noqa: E402

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

# The shipped unit, from tt_bio/tenstorrent.py:7763 and the budget's own capture.
H = W = 512          # pair tensor is [1, 512, 512, 128]
C = 128              # channel
HID = 512            # fc1/fc2 hidden
SHIP_H = 16          # TRANSITION_H_CHUNK_SIZE at W=512 > TRANSITION_H_CHUNK_BIG_MAX_W=384
FULL_FLOP = 2 * H * W * C * HID * 3


def _grid(dev):
    cc = dev.compute_with_storage_grid_size()
    return cc.x, cc.y


def build(dev, gx, gy):
    """Everything the arms need, allocated once. Returns (arms, tensors_to_free)."""
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    # The fold runs the Transition's linears on an 11x10 grid (CORE_GRID_MAIN); clamp to the part.
    cg = ttnn.CoreGrid(y=min(10, gy), x=min(11, gx))

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    # weights, exactly the shipped shapes (ttnn.linear takes [in, out])
    w1 = t((C, HID)); w2 = t((C, HID)); w3 = t((HID, C))
    nw = ttnn.from_torch(torch.ones(C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    nb = ttnn.from_torch(torch.zeros(C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)

    z = t((1, H, W, C))                       # the pair tensor the unit is called on
    blk = t((1, SHIP_H, W, C))                # one shipped row block, DRAM (what chunk() leaves)
    blk_l1 = t((1, SHIP_H, W, C), L1)         # one row block already in L1
    wide_l1 = t((1, SHIP_H, W, HID), L1)      # a hidden-width block in L1 (fc3's input)
    cube4 = t((4096, 4096)); cube4b = t((4096, 4096))
    cube2 = t((2048, 2048)); cube2b = t((2048, 2048))
    flat = t((1, H * W, C))                   # the whole unit as ONE matmul, no row blocking

    keep = [w1, w2, w3, nw, nb, z, blk, blk_l1, wide_l1, cube4, cube4b, cube2, cube2b, flat]

    def lin(x, w, out_mc, act=None):
        return ttnn.linear(x, w, activation=act, compute_kernel_config=kc,
                           memory_config=out_mc, dtype=ttnn.bfloat16, core_grid=cg)

    BF = 2 * SHIP_H * W * C * HID           # FLOP of ONE fc1-shaped matmul on one row block

    def swiglu(x, out_mc=DRAM, act="silu"):
        """The shipped block, byte for byte: LN(L1) -> fc1+silu(L1) -> fc2(L1) -> mul -> fc3."""
        xn = ttnn.layer_norm(x, weight=nw, bias=nb, epsilon=1e-5,
                             compute_kernel_config=kc, memory_config=L1)
        a = lin(xn, w1, L1, act)
        if act is None:
            a = ttnn.silu(a, memory_config=L1, output_tensor=a)
        b = lin(xn, w2, L1)
        ttnn.deallocate(xn)
        a = ttnn.multiply_(a, b)
        ttnn.deallocate(b)
        o = lin(a, w3, out_mc)
        ttnn.deallocate(a)
        return o

    def full(h, concat=True, out_mc=DRAM):
        chunks = ttnn.chunk(z, -(-H // h), dim=1)
        parts = [swiglu(c, out_mc) for c in chunks]
        for c in chunks:
            ttnn.deallocate(c)
        if not concat:
            for p in parts[1:]:
                ttnn.deallocate(p)
            return parts[0]
        o = ttnn.concat(parts, dim=1)
        for p in parts:
            ttnn.deallocate(p)
        return o

    def chunk_only(h):
        cs = ttnn.chunk(z, -(-H // h), dim=1)
        for c in cs[1:]:
            ttnn.deallocate(c)
        return cs[0]

    A = {}
    A["cube4096"] = (lambda: ttnn.matmul(cube4, cube4b, compute_kernel_config=kc,
                                         memory_config=DRAM), 2 * 4096 ** 3)
    A["cube2048"] = (lambda: ttnn.matmul(cube2, cube2b, compute_kernel_config=kc,
                                         memory_config=DRAM), 2 * 2048 ** 3)
    # the three real-shape matmuls, one row block, isolated
    A["fc1_dram_dram"] = (lambda: lin(blk, w1, DRAM), BF)
    A["fc1_l1_l1"] = (lambda: lin(blk_l1, w1, L1), BF)
    A["fc1_l1_l1_silu"] = (lambda: lin(blk_l1, w1, L1, "silu"), BF)
    A["fc3_l1_dram"] = (lambda: lin(wide_l1, w3, DRAM), BF)
    A["fc3_l1_l1"] = (lambda: lin(wide_l1, w3, L1), BF)
    # the block's arithmetic only: three matmuls, L1 resident, no LN / no silu / no multiply
    def mm3():
        a = lin(blk_l1, w1, L1)
        b = lin(blk_l1, w2, L1)
        ttnn.deallocate(b)
        o = lin(wide_l1, w3, DRAM)
        ttnn.deallocate(a)
        return o
    A["mm3_block"] = (mm3, 3 * BF)
    # the non-matmul stages, priced on their own
    A["layer_norm_block"] = (lambda: ttnn.layer_norm(blk_l1, weight=nw, bias=nb, epsilon=1e-5,
                                                     compute_kernel_config=kc, memory_config=L1), 0)
    A["multiply_block"] = (lambda: ttnn.mul(wide_l1, wide_l1, memory_config=L1), 0)
    A["silu_block"] = (lambda: ttnn.silu(wide_l1, memory_config=L1), 0)
    # the whole shipped block
    A["block_ship"] = (lambda: swiglu(blk, DRAM), 3 * BF)
    A["block_l1in"] = (lambda: swiglu(blk_l1, DRAM), 3 * BF)
    A["block_unfused_silu"] = (lambda: swiglu(blk, DRAM, None), 3 * BF)
    # the row blocking itself
    A["chunk_only_h16"] = (lambda: chunk_only(16), 0)
    # the whole unit
    A["full_h16_ship"] = (lambda: full(16), FULL_FLOP)
    A["full_h32"] = (lambda: full(32), FULL_FLOP)
    A["full_h64"] = (lambda: full(64), FULL_FLOP)
    A["full_h128"] = (lambda: full(128), FULL_FLOP)
    A["full_h8"] = (lambda: full(8), FULL_FLOP)
    A["full_h4"] = (lambda: full(4), FULL_FLOP)
    A["full_h16_noconcat"] = (lambda: full(16, concat=False), FULL_FLOP)
    A["full_h64_noconcat"] = (lambda: full(64, concat=False), FULL_FLOP)
    # the unit with no row blocking at all: one flat [262144,128] chain, intermediates in DRAM
    def flat_chain():
        xn = ttnn.layer_norm(flat, weight=nw, bias=nb, epsilon=1e-5,
                             compute_kernel_config=kc, memory_config=DRAM)
        a = lin(xn, w1, DRAM, "silu")
        b = lin(xn, w2, DRAM)
        ttnn.deallocate(xn)
        a = ttnn.multiply_(a, b)
        ttnn.deallocate(b)
        o = lin(a, w3, DRAM)
        ttnn.deallocate(a)
        return o
    A["full_flat_dram"] = (flat_chain, FULL_FLOP)
    return A, keep, kc, cg


def time_arms(arms, order, reps, blocks, dev, warm=2):
    """Warm every arm first, drop the ones this part refuses, then interleave the timed blocks."""
    err, live = {}, []
    for n in order:
        fn, _ = arms[n]
        try:
            for _ in range(warm):
                ttnn.deallocate(fn())
            live.append(n)
        except Exception as e:                                                # noqa: BLE001
            err[n] = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
            print("SKIP %-22s %s" % (n, err[n]), flush=True)
    order = live
    ttnn.synchronize_device(dev)
    best = {n: None for n in order}
    for _ in range(blocks):
        for n in order:
            fn, _ = arms[n]
            outs = []
            t0 = time.perf_counter()
            for _ in range(reps):
                outs.append(fn())
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / reps
            for o in outs:
                ttnn.deallocate(o)
            best[n] = dt if best[n] is None else min(best[n], dt)
    return best, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "shape_roof.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    # The bench opens the device itself, so it takes the same flock the model's get_device()
    # takes -- otherwise a second job on this card collides at the fd level with no error.
    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        gx, gy = _grid(dev)
        arms, keep, kc, cg = build(dev, gx, gy)
        order = [n for n in arms if not a.only or n in a.only.split(",")]
        best, err = time_arms(arms, order, a.reps, a.blocks, dev)
        order = [n for n in order if n in best]
        rows = []
        for n in order:
            s = best[n]
            f = arms[n][1]
            rows.append({"arm": n, "ms": s * 1e3,
                         "TFLOPs": (f / s / 1e12) if f else None})
        cube = next((r["TFLOPs"] for r in rows if r["arm"] == "cube4096"), None)
        for r in rows:
            r["pct_of_cube"] = (100 * r["TFLOPs"] / cube) if (cube and r["TFLOPs"]) else None
        out = {
            "host": platform.node(),
            "arch": str(dev.arch()),
            "grid": [gx, gy],
            "core_grid_used": [cg.x, cg.y],
            "ttnn": getattr(ttnn, "__version__", "?"),
            "reps": a.reps, "blocks": a.blocks,
            "loadavg": open("/proc/loadavg").read().split()[:3],
            "cube4096_TFLOPs": cube,
            "refused": err,
            "rows": rows,
        }
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            print("%-*s  %9.4f ms  %s  %s" % (
                w, r["arm"], r["ms"],
                ("%7.2f TFLOP/s" % r["TFLOPs"]) if r["TFLOPs"] else "        -      ",
                ("%5.1f %% of cube" % r["pct_of_cube"]) if r["pct_of_cube"] else ""), flush=True)
        for t_ in keep:
            ttnn.deallocate(t_)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
