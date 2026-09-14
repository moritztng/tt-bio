#!/usr/bin/env python3
"""What pins the six triangle matmul classes at 13-15 % of the dense cube.

`perf/roof_shape/` measured the fraction each class reaches. It did not measure WHY, and its one
sentence of mechanism ("the MAC array failing to fill at a 4-tile K") is refused by its own table:
the triangle product is a 16-tile-K matmul and sits BELOW the 4-tile-K in-projection.

The discriminator here is the math-fidelity ladder. HiFi4 is four MAC passes per tile, HiFi2 two,
LoFi one. A shape whose time the MAC array sets speeds up toward 4x from HiFi4 to LoFi; a shape
limited by anything else -- DRAM, NOC, the packer, dispatch -- barely moves. The dense cube runs at
all three fidelities in the same session as the control, so the ladder is internally calibrated.

Method is `perf/roof_shape/shape_roofs.py`'s and unchanged: one process, one device, one session,
`reps` enqueues per synchronize so host dispatch is amortised, the whole arm set run in a fixed
order once per block so no arm can be biased by warm-up or a clock ramp, minimum over blocks.
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

S = 512
CZ = 128
HD = 32
NH = 4
FIDS = ("LoFi", "HiFi2", "HiFi4")


def _f(*d):
    n = 2
    for x in d:
        n *= x
    return n


def build(dev, want_k, want_n):
    arch = dev.arch()
    kcls = (ttnn.types.WormholeComputeKernelConfig if arch == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid, acc=True):
        return kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                    fp32_dest_acc_en=acc, packer_l1_acc=True)

    cc = dev.compute_with_storage_grid_size()
    gx, gy = cc.x, cc.y
    full = ttnn.CoreGrid(y=min(10, gy), x=min(11, gx))
    half = ttnn.CoreGrid(y=max(1, min(10, gy) // 2), x=min(11, gx))

    def t(shape, mc=DRAM):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    def lin(x, w, fid, out_mc=DRAM, grid=None, acc=True):
        kw = {"core_grid": grid} if grid is not None else {}
        return ttnn.linear(x, w, compute_kernel_config=kc(fid, acc), memory_config=out_mc,
                           dtype=ttnn.bfloat16, **kw)

    zflat = t((S * S, CZ))
    w640 = t((CZ, 5 * CZ))
    w544 = t((CZ, 3 * NH * HD + CZ + 32))
    w128 = t((CZ, CZ))
    ta = t((1, CZ, S, S)); tb = t((1, CZ, S, S))
    q = t((S, NH, S, HD)); k = t((S, NH, S, HD)); v = t((S, NH, S, HD))
    bias = t((1, NH, S, S))
    cube4 = t((4096, 4096)); cube4b = t((4096, 4096))
    big = t((8192, 8192)); bigb = t((8192, 8192))
    keep = [zflat, w640, w544, w128, ta, tb, q, k, v, bias, cube4, cube4b, big, bigb]

    # K ladder: 262144 x K @ K x 128. in0 at K=512 is 268.4 MB, the largest thing here.
    kla = {}
    for kk in want_k:
        if kk == CZ:
            kla[kk] = (zflat, w128)
        else:
            a = t((S * S, kk)); b = t((kk, CZ))
            kla[kk] = (a, b); keep += [a, b]
    # N ladder: 262144 x 128 @ 128 x N.
    nla = {}
    for nn in want_n:
        w = {CZ: w128, 5 * CZ: w640}.get(nn)
        if w is None:
            w = t((CZ, nn)); keep.append(w)
        nla[nn] = w

    A = {}
    A["bw_add8192"] = (lambda: ttnn.add(big, bigb, memory_config=DRAM), 3 * 8192 * 8192 * 2, 3)

    for fid in FIDS:
        A["cube4096_%s" % fid] = (lambda f=fid: ttnn.matmul(
            cube4, cube4b, compute_kernel_config=kc(f), memory_config=DRAM),
            _f(4096, 4096, 4096), 3)
        A["trimul_in_%s" % fid] = (lambda f=fid: lin(zflat, w640, f),
                                   _f(1, S * S, CZ, 5 * CZ), 2)
        A["triatt_in_%s" % fid] = (lambda f=fid: lin(zflat, w544, f),
                                   _f(1, S * S, CZ, 17 * 32), 2)
        A["pair_out128_%s" % fid] = (lambda f=fid: lin(zflat, w128, f),
                                     _f(1, S * S, CZ, CZ), 3)
        A["trimul_einsum_%s" % fid] = (lambda f=fid: ttnn.matmul(
            ta, tb, compute_kernel_config=kc(f), memory_config=DRAM), _f(CZ, S, S, S), 3)
        pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=cc, exp_approx_mode=False,
                                    q_chunk_size=256, k_chunk_size=256)
        A["triatt_sdpa_%s" % fid] = (
            lambda f=fid, pc=pc: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                program_config=pc, compute_kernel_config=kc(f)),
            2 * _f(S * NH, S, S, HD), 2)

    # The second dimension: fp32 destination accumulation. With fp32 dest the DST register holds
    # half as many tiles and every pack is a format conversion, so a matmul that the MAC array does
    # not bind can be bound by DST capacity instead. The fold sets it True almost everywhere.
    for fid in ("LoFi", "HiFi4"):
        A["cube4096_%s_noacc" % fid] = (lambda f=fid: ttnn.matmul(
            cube4, cube4b, compute_kernel_config=kc(f, False), memory_config=DRAM),
            _f(4096, 4096, 4096), 3)
        A["trimul_in_%s_noacc" % fid] = (lambda f=fid: lin(zflat, w640, f, acc=False),
                                         _f(1, S * S, CZ, 5 * CZ), 2)
        A["pair_out128_%s_noacc" % fid] = (lambda f=fid: lin(zflat, w128, f, acc=False),
                                           _f(1, S * S, CZ, CZ), 3)
        A["trimul_einsum_%s_noacc" % fid] = (lambda f=fid: ttnn.matmul(
            ta, tb, compute_kernel_config=kc(f, False), memory_config=DRAM),
            _f(CZ, S, S, S), 3)
        pcn = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=cc, exp_approx_mode=False,
                                     q_chunk_size=256, k_chunk_size=256)
        A["triatt_sdpa_%s_noacc" % fid] = (
            lambda f=fid, pc=pcn: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                program_config=pc, compute_kernel_config=kc(f, False)),
            2 * _f(S * NH, S, S, HD), 2)

    for kk, (a, b) in kla.items():
        A["kladder_k%d" % kk] = (lambda a=a, b=b: lin(a, b, "HiFi4"),
                                 _f(1, S * S, kk, CZ), 3)
    for nn, w in nla.items():
        A["nladder_n%d" % nn] = (lambda w=w, nn=nn: lin(zflat, w, "HiFi4"),
                                 _f(1, S * S, CZ, nn), 2)

    A["trimul_in_HiFi4_gfull"] = (lambda: lin(zflat, w640, "HiFi4", grid=full),
                                  _f(1, S * S, CZ, 5 * CZ), 2)
    A["trimul_in_HiFi4_ghalf"] = (lambda: lin(zflat, w640, "HiFi4", grid=half),
                                  _f(1, S * S, CZ, 5 * CZ), 2)
    # A/A: the same callable entered twice under two names bounds session drift.
    A["cube4096_HiFi4_AA"] = A["cube4096_HiFi4"]
    A["trimul_in_HiFi4_AA"] = A["trimul_in_HiFi4"]
    A["pair_out128_HiFi4_AA"] = A["pair_out128_HiFi4"]
    return A, keep, (gx, gy), (full.x, full.y), (half.x, half.y)


def time_arms(arms, order, blocks, dev, warm=2):
    err, live = {}, []
    for n in order:
        fn = arms[n][0]
        try:
            tw = time.perf_counter()
            for _ in range(warm):
                ttnn.deallocate(fn())
            live.append(n)
            print("warm %-26s %7.2f s" % (n, time.perf_counter() - tw), flush=True)
        except Exception as e:                                                # noqa: BLE001
            err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
            print("SKIP %-26s %s" % (n, err[n]), flush=True)
    order = live
    ttnn.synchronize_device(dev)
    best = {n: None for n in order}
    for _b in range(blocks):
        tb0 = time.perf_counter()
        for n in list(order):
            if n not in best:
                continue
            fn, _flop, reps = arms[n]
            outs = []
            try:
                t0 = time.perf_counter()
                for _ in range(reps):
                    outs.append(fn())
                    if len(outs) > 4:
                        ttnn.deallocate(outs.pop(0))
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) / reps
            except Exception as e:                                            # noqa: BLE001
                err[n] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160])
                print("DROP %-26s %s" % (n, err[n]), flush=True)
                best.pop(n, None)
                for o in outs:
                    ttnn.deallocate(o)
                ttnn.synchronize_device(dev)
                continue
            for o in outs:
                ttnn.deallocate(o)
            best[n] = dt if best[n] is None else min(best[n], dt)
        print("block %d  %7.2f s" % (_b, time.perf_counter() - tb0), flush=True)
    return {n: v for n, v in best.items() if v is not None}, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "tri_mech.json")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--kladder", default="128,256,512")
    ap.add_argument("--nladder", default="128,256,640")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    lease = CardSetLease().acquire()
    dev = ttnn.open_device(device_id=int(os.environ.get("TT_BIO_ROOF_DEV", "0")))
    try:
        arms, keep, grid, gfull, ghalf = build(
            dev, [int(x) for x in a.kladder.split(",") if x],
            [int(x) for x in a.nladder.split(",") if x])
        order = [n for n in arms if not a.only or n in a.only.split(",")]
        best, err = time_arms(arms, order, a.blocks, dev)
        order = [n for n in order if n in best]
        rows = [{"arm": n, "ms": best[n] * 1e3, "TFLOPs": arms[n][1] / best[n] / 1e12,
                 "reps": arms[n][2]} for n in order]
        cube = next((r["TFLOPs"] for r in rows if r["arm"] == "cube4096_HiFi4"), None)
        for r in rows:
            r["pct_of_cube_hifi4"] = 100 * r["TFLOPs"] / cube if cube else None
        out = {"host": platform.node(), "arch": str(dev.arch()), "grid": list(grid),
               "grid_full": list(gfull), "grid_half": list(ghalf), "blocks": a.blocks,
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "cube4096_HiFi4_TFLOPs": cube, "refused": err, "rows": rows}
        a.out.write_text(json.dumps(out, indent=1))
        w = max(len(r["arm"]) for r in rows)
        for r in rows:
            print("%-*s  %10.4f ms  %8.2f TFLOP/s  %6.1f %% of HiFi4 cube"
                  % (w, r["arm"], r["ms"], r["TFLOPs"], r["pct_of_cube_hifi4"] or 0), flush=True)
        for x in keep:
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
