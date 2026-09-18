#!/usr/bin/env python3
"""Decompose the thin-K matmul's fixed term. Not reduce it -- find out what it IS.

c13-matmul-rate measured, on qb2 node 3 (p300c), at fixed b=16, M=512, N=512 and K scaled
128 -> 2048:  t(kt) = 0.0723 + 0.00554*kt ms.  At the fold's own K=128 (kt=4) that makes 73.9 %
of the call a cost which does not scale with K. It survived bandwidth (an arm with no DRAM
traffic at all still capped at 23.57 TFLOP/s), the FPU (a 4x cut in passes moved it 1.02x) and
the batch loop (flat2d moved it 1.008x). This row asks what the 0.0723 ms is.

THE FIRST THING TO RULE OUT IS THE INSTRUMENT, and no earlier pass did. c13's per-call number
comes from `synchronize_device; call; synchronize_device`, which is max(host, device) plus a
full drain and a fresh program launch per call. `roof-launch-floor-per-device-program-not-python
-call` measured exactly this overstatement on a cheap op: ttnn.add reads 9.72 us synced against
6.22 us traced, 56 % high. The fold issues these 17,920 calls back to back, so if the synced
bracket carries a per-bracket floor, part of the 0.0723 ms is a cost the fold never pays -- and
that is consistent with c10-trace-lever finding nothing to recover at the fold level.

So the ladders here are run on three instruments over the same arms:
  synced    synchronize; one call; synchronize          (what c13 used)
  stream    synchronize; n identical calls; synchronize (fit L + n*c, c is the marginal call)
  trace     capture n calls, replay, device only        (host removed by construction)
`n`-ladder first, then every decomposition ladder on the marginal instrument.

The components, each with an experiment that isolates it and nothing else:
  launch/drain   n-ladder per bracket                   -> L, the instrument's own floor
  per-op fixed   N-ladder and batch-ladder intercepts   -> paid once per call, whatever the size
  per-out-tile   batch-ladder slope at fixed grid       -> paid per output tile at kt=4
  dest drain     out_subblock sweep at identical work   -> the packer/dest granularity term
  operand mcast  grid SHAPE at constant core count      -> mcast width, not core count
  DRAM read      in_l1 delta at both intercept and slope
  DRAM write     out_l1 delta at both intercept and slope
  idle cores     core-count sweep, to say cause or consequence

The batch ladder is the load-bearing one and it is why this row can attribute what a K ladder
alone cannot (`matmul-fixed-cost-needs-n-ladder-to-attribute-per-op-vs-per-tile`): scaling b at
fixed M, K, N scales the output tile count with the core-grid mapping held EXACTLY fixed, because
ttnn loops the batch on each core. An N ladder moves the tile count and the grid width at once.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))

import clk                                                                     # noqa: E402
import ttnn                                                                    # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# The fold's own shape for key A: b=16, M=512, K=128, N=512, 17,920 calls, 19.241 TFLOP.
B, M, K, N = 16, 512, 128, 512


def flops(b, m, k, n):
    return 2.0 * b * m * k * n


def tile_flops(b, m, k, n):
    assert m % 32 == 0 and k % 32 == 0 and n % 32 == 0
    return float(b) * (m // 32) * (k // 32) * (n // 32) * 2 * 32 * 32 * 32


def min_bytes(b, m, k, n, elem=2):
    """Compulsory DRAM traffic: in0 once, the shared weight once, the result once."""
    return float(elem) * (b * m * k + k * n + b * m * n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--stages", default="ctl,n,k,batch,nlad,grid,sub")
    ap.add_argument("--only", default="both", choices=("both", "dram", "l1"),
                    help="which residency's arms to build. An L1 session must not also "
                         "allocate the DRAM session's L1 tensors: s1 lost 40 arms to "
                         "'circular buffers clash with L1 buffers' purely from co-residency.")
    ap.add_argument("--nlo", type=int, default=4)
    ap.add_argument("--nhi", type=int, default=16)
    ap.add_argument("--nlo-l1", type=int, default=2)
    ap.add_argument("--nhi-l1", type=int, default=8)
    ap.add_argument("--kmults", default="1,2,4,8,16",
                    help="K multipliers for the K ladder. An L1 K ladder cannot hold all five "
                         "at once: the kt=64 L1 input alone is 33.6 MB of 165 MB.")
    a = ap.parse_args()
    stages = set(a.stages.split(","))

    import torch
    from tt_bio import tenstorrent as T

    device = T.get_device(trace_region_size=200_000_000)
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(a.clock, nodes)
    t_ramp = time.time()
    while time.time() - t_ramp < 10.0:
        if all(clk.aiclk(n) >= a.clock - 5 for n in held):
            break
        time.sleep(0.01)
    reached = {n: clk.aiclk(n) for n in held}
    if any(v < a.clock - 5 for v in reached.values()):
        print("REFUSING to measure: %r after %.0f ms" % (reached, 1e3 * (time.time() - t_ramp)))
        return 2
    print("nodes %r forced to %d MHz in %.0f ms" % (held, a.clock, 1e3 * (time.time() - t_ramp)),
          flush=True)
    sampler = clk.Sampler(held[0])
    t_measure_start = time.time()

    g = device.compute_with_storage_grid_size()
    cores, GRID = g.x * g.y, T.CORE_GRID_MAIN
    print("grid %dx%d = %d cores, CORE_GRID_MAIN=%r, arch=%r"
          % (g.x, g.y, cores, GRID, device.arch()), flush=True)

    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)

    def kc(fid=ttnn.MathFidelity.HiFi4, fp32=True, pl1=True):
        return kcls(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32,
                    packer_l1_acc=pl1)

    KC_SHIP = kc()            # the fold's own config for this class, HiFi4/fp32acc/packerl1acc
    KC_ROOF = kc(fp32=False, pl1=False)

    torch.manual_seed(0)

    def dev(t, mc=DRAM):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=mc)

    # ---- instruments -------------------------------------------------------------------
    def synced(fn, reps, warm, n=1):
        """synchronize; n calls; synchronize -- returns ms per BRACKET, not per call."""
        out = []
        for i in range(reps + warm):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            rs = [fn() for _ in range(n)]
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            for r in rs:
                if isinstance(r, ttnn.Tensor):
                    ttnn.deallocate(r)
            if i >= warm:
                out.append((t1 - t0) * 1e3)
        return out

    def traced(fn, reps, warm, n):
        """Capture n calls into a trace, replay it. Device time, host removed by construction."""
        for _ in range(2):
            rs = [fn() for _ in range(n)]
            ttnn.synchronize_device(device)
            for r in rs:
                if isinstance(r, ttnn.Tensor):
                    ttnn.deallocate(r)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        keep = [fn() for _ in range(n)]
        ttnn.end_trace_capture(device, tid, cq_id=0)
        out = []
        for i in range(reps + warm):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            if i >= warm:
                out.append((t1 - t0) * 1e3)
        ttnn.release_trace(device, tid)
        for r in keep:
            if isinstance(r, ttnn.Tensor):
                try:
                    ttnn.deallocate(r)
                except Exception:                                          # noqa: BLE001
                    pass
        return out

    arms = []      # (name, callable, nper, instrument, meta)

    def add(name, fn, nper=1, inst="synced", **meta):
        arms.append((name, fn, nper, inst, meta))

    # Every decomposition arm is a PAIR of bracket sizes, never a single one. The synced
    # bracket charges a launch/drain floor L once per bracket (measured below at 0.05 ms,
    # comparable to the whole cost of the call), so a per-call number from any single bracket
    # is L/n too high. The difference between two bracket sizes cancels L exactly, with no
    # assumption about its size: marginal = (t_hi - t_lo) / (hi - lo).
    NLO, NHI = a.nlo, a.nhi
    NLO_L1, NHI_L1 = a.nlo_l1, a.nhi_l1   # 16 live 8.39 MB L1 outputs do not fit
    WANT_DRAM, WANT_L1 = a.only in ("both", "dram"), a.only in ("both", "l1")

    def addpair(name, fn, l1=False, **meta):
        if (l1 and not WANT_L1) or (not l1 and not WANT_DRAM):
            meta.pop("amount_per_call")
            return
        lo, hi = (NLO_L1, NHI_L1) if l1 else (NLO, NHI)
        amt = meta.pop("amount_per_call")
        add("%s_x%02d" % (name, lo), fn, lo, amount=amt * lo, pair=name, **meta)
        add("%s_x%02d" % (name, hi), fn, hi, amount=amt * hi, pair=name, **meta)

    def lin(xa, wa, mc, kcfg=KC_SHIP, grid=GRID, pcfg=None):
        def f():
            kw = dict(compute_kernel_config=kcfg, memory_config=mc, dtype=ttnn.bfloat16)
            if pcfg is not None:
                kw["program_config"] = pcfg
            elif grid is not None:
                kw["core_grid"] = grid
            return ttnn.linear(xa, wa, **kw)
        return f

    hold = []

    # ---- stage ctl: known-answer controls, this part, this session ---------------------
    if "ctl" in stages:
        c8, c8b = dev(torch.randn(8192, 8192) * 0.05), dev(torch.randn(8192, 8192) * 0.05)
        ad, bd = dev(torch.randn(8192, 8192) * 0.05), dev(torch.randn(8192, 8192) * 0.05)
        hold += [c8, c8b, ad, bd]
        F8 = flops(1, 8192, 8192, 8192)
        assert F8 == 1099511627776.0 and tile_flops(1, 8192, 8192, 8192) == F8
        BW = 3 * 8192 * 8192 * 2
        assert BW == 402653184
        for nm in ("ctl_cube_ship", "ctl_cube_ship_aa"):
            add(nm, lambda: ttnn.matmul(c8, c8b, compute_kernel_config=KC_SHIP,
                                        memory_config=DRAM), kind="flops", amount=F8)
        add("ctl_cube_roof", lambda: ttnn.matmul(c8, c8b, compute_kernel_config=KC_ROOF,
                                                 memory_config=DRAM), kind="flops", amount=F8)
        for nm in ("ctl_bwadd", "ctl_bwadd_aa"):
            add(nm, lambda: ttnn.add(ad, bd, memory_config=DRAM), kind="bytes",
                amount=float(BW))

    # the key-A operands, in every residency the ablation needs
    x_d, w_d = dev(torch.randn(1, B, M, K) * 0.05), dev(torch.randn(K, N) * 0.05)
    x_l, w_l = dev(torch.randn(1, B, M, K) * 0.05, L1), dev(torch.randn(K, N) * 0.05, L1)
    hold += [x_d, w_d, x_l, w_l]
    FA, BA = flops(B, M, K, N), min_bytes(B, M, K, N)
    assert FA == tile_flops(B, M, K, N)

    # ---- stage n: the n-ladder. Is the 0.0723 ms device time at all? ------------------
    # Same op, 1..16 calls inside ONE bracket. t(n) = L + n*c: L is the bracket's own
    # launch/drain floor, c is the marginal cost of a call issued back to back the way the
    # fold issues it. Run on both the DRAM arm and the no-DRAM arm, and TRACED at n=16 as
    # the independent check that L is host and not device.
    if "n" in stages:
        for nper in (1, 2, 4, 8, 16):
            add("n%02d_ship" % nper, lin(x_d, w_d, DRAM), nper, kind="flops",
                amount=FA * nper)
            add("n%02d_both_l1" % nper, lin(x_l, w_l, L1), nper, kind="flops",
                amount=FA * nper)
        add("n16_ship_aa", lin(x_d, w_d, DRAM), 16, kind="flops", amount=FA * 16)
        for nper in (1, 8, 16):
            add("tr%02d_ship" % nper, lin(x_d, w_d, DRAM), nper, "traced", kind="flops",
                amount=FA * nper)
            add("tr%02d_both_l1" % nper, lin(x_l, w_l, L1), nper, "traced", kind="flops",
                amount=FA * nper)

    # ---- stage k: REPRODUCED. c13's own ladder, on the synced instrument it was taken on,
    # and again streamed and traced so the fit can be read on all three.
    if "k" in stages:
        for mult in [int(x) for x in a.kmults.split(",")]:
            kk = K * mult
            xk = dev(torch.randn(1, B, M, kk) * 0.05)
            wk = dev(torch.randn(kk, N) * 0.05)
            hold += [xk, wk]
            xkl = wkl = None
            if WANT_L1:
                xkl = dev(torch.randn(1, B, M, kk) * 0.05, L1)
                wkl = dev(torch.randn(kk, N) * 0.05, L1)
                hold += [xkl, wkl]
            Fk = flops(B, M, kk, N)
            add("k%02d_ship" % (kk // 32), lin(xk, wk, DRAM), 1, kind="flops", amount=Fk,
                kt=kk // 32)
            if WANT_DRAM:
                add("k%02d_ship_tr" % (kk // 32), lin(xk, wk, DRAM), 8, "traced",
                    kind="flops", amount=Fk * 8, kt=kk // 32)
            addpair("k%02d_ship" % (kk // 32), lin(xk, wk, DRAM), kind="flops",
                    amount_per_call=Fk, kt=kk // 32)
            if WANT_L1:
                addpair("k%02d_both_l1" % (kk // 32), lin(xkl, wkl, L1), l1=True,
                        kind="flops", amount_per_call=Fk, kt=kk // 32)
                addpair("k%02d_out_l1" % (kk // 32), lin(xk, wk, L1), l1=True,
                        kind="flops", amount_per_call=Fk, kt=kk // 32)
                addpair("k%02d_in_l1_p" % (kk // 32), lin(xkl, wkl, DRAM), l1=True,
                        kind="flops", amount_per_call=Fk, kt=kk // 32)
            addpair("k%02d_in_l1" % (kk // 32), lin(xkl, wkl, DRAM), kind="flops",
                    amount_per_call=Fk, kt=kk // 32)

    # ---- stage batch: per-op fixed vs per-output-tile, with the grid mapping HELD FIXED --
    if "batch" in stages:
        for b in (1, 2, 4, 8, 16, 32):
            xb = dev(torch.randn(1, b, M, K) * 0.05)
            hold += [xb]
            xbl = dev(torch.randn(1, b, M, K) * 0.05, L1) if WANT_L1 else None
            if xbl is not None:
                hold += [xbl]
            Fb = flops(b, M, K, N)
            addpair("b%02d_ship" % b, lin(xb, w_d, DRAM), kind="flops", amount_per_call=Fb,
                    bt=b, ot=b * (M // 32) * (N // 32))
            if WANT_L1:
                addpair("b%02d_both_l1" % b, lin(xbl, w_l, L1), l1=True, kind="flops",
                        amount_per_call=Fb, bt=b, ot=b * (M // 32) * (N // 32))
            add("b%02d_ship_x01" % b, lin(xb, w_d, DRAM), 1, kind="flops", amount=Fb,
                bt=b, ot=b * (M // 32) * (N // 32))

    # ---- stage nlad: the N ladder, as the cross-check that moves grid width too ---------
    if "nlad" in stages:
        for nn in (128, 256, 512, 1024, 2048):
            wn = dev(torch.randn(K, nn) * 0.05)
            hold += [wn]
            wnl = dev(torch.randn(K, nn) * 0.05, L1) if WANT_L1 else None
            if wnl is not None:
                hold += [wnl]
            Fn = flops(B, M, K, nn)
            addpair("N%04d_ship" % nn, lin(x_d, wn, DRAM), kind="flops", amount_per_call=Fn,
                    nt=nn // 32, ot=B * (M // 32) * (nn // 32))
            if WANT_L1:
                addpair("N%04d_both_l1" % nn, lin(x_l, wnl, L1), l1=True, kind="flops",
                        amount_per_call=Fn, nt=nn // 32, ot=B * (M // 32) * (nn // 32))

    # ---- stage grid: core COUNT, and separately core SHAPE at constant count ------------
    # The shape sweep is the mcast experiment: same shape, same number of cores, different
    # mcast width and height. The count sweep says whether idle cores are cause or effect.
    if "grid" in stages:
        shapes = [(g.x, g.y), (8, 8), (6, 6), (4, 4), (2, 2), (1, 1),
                  (10, 6), (6, 10), (10, 4), (4, 10), (8, 5), (5, 8), (8, 6), (6, 8)]
        for gx, gy in shapes:
            if gx > g.x or gy > g.y:
                continue
            cg = ttnn.CoreGrid(x=gx, y=gy)
            addpair("g%02dx%02d_ship" % (gx, gy), lin(x_d, w_d, DRAM, grid=cg),
                    kind="flops", amount_per_call=FA, gx=gx, gy=gy, gc=gx * gy)
            addpair("g%02dx%02d_both_l1" % (gx, gy), lin(x_l, w_l, L1, grid=cg), l1=True,
                    kind="flops", amount_per_call=FA, gx=gx, gy=gy, gc=gx * gy)

    # ---- stage sub: the pipeline the fixed term is supposed to be living in ------------
    # flat2d, because c13 measured the leading batch dim free (1.008x) and an explicit program
    # config is 2-D. An 8x8 grid, because c13 measured 8x8 to match the full 11x10 within the
    # A/A floor on this shape and it makes per_core_M an exact 32 so the block ladder divides.
    # Three sweeps, all at IDENTICAL shape, cores, FLOPs and bytes:
    #   out_block_h   how many blocks the SAME output is cut into  -> pipeline fill/drain
    #   out_subblock  the dest-register drain granularity          -> packer/dest term
    #   in0_block_w   K-blocks per output block (needs a fat K)    -> CB round trips
    if "sub" in stages:
        x_f = dev(torch.randn(B * M, K) * 0.05)
        x_fl = dev(torch.randn(B * M, K) * 0.05, L1)
        hold += [x_f, x_fl]
        mt, nt, kt = (B * M) // 32, N // 32, K // 32
        DEST_TILES = 4              # BH dest with fp32_dest_acc_en; 8 is the bf16-dest figure
        gx, gy = 8, 8
        per_m, per_n = -(-mt // gy), -(-nt // gx)          # 32, 2
        cc = ttnn.CoreCoord(gx, gy)
        addpair("flat2d_ship", lin(x_f, w_d, DRAM), kind="flops", amount_per_call=FA)
        addpair("flat2d_g8x8_ship", lin(x_f, w_d, DRAM, grid=ttnn.CoreGrid(x=gx, y=gy)),
                kind="flops", amount_per_call=FA)

        def pcfg(ibw, sh, sw, obh, obw):
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=cc, in0_block_w=ibw,
                out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
                per_core_M=per_m, per_core_N=per_n, transpose_mcast=False,
                fused_activation=None)

        for obh in (1, 2, 4, 8, 16, 32):
            sh = min(obh, DEST_TILES // per_n)
            try:
                pc = pcfg(kt, sh, per_n, obh, per_n)
            except Exception as e:                                         # noqa: BLE001
                print("  obh %d refused at build: %r" % (obh, repr(e)[:120]), flush=True)
                continue
            addpair("obh%02d_ship" % obh, lin(x_f, w_d, DRAM, pcfg=pc), kind="flops",
                    amount_per_call=FA, obh=obh, sh=sh, blocks=per_m // obh)
            addpair("obh%02d_both_l1" % obh, lin(x_fl, w_l, L1, pcfg=pc), l1=True,
                    kind="flops", amount_per_call=FA, obh=obh, sh=sh, blocks=per_m // obh)

        for sh, sw in ((1, 1), (2, 1), (4, 1), (1, 2), (2, 2)):
            if per_m % sh or per_n % sw or sh * sw > DEST_TILES:
                continue
            try:
                pc = pcfg(kt, sh, sw, per_m, per_n)
            except Exception as e:                                         # noqa: BLE001
                print("  sub %dx%d refused at build: %r" % (sh, sw, repr(e)[:120]), flush=True)
                continue
            addpair("sub%dx%d_ship" % (sh, sw), lin(x_f, w_d, DRAM, pcfg=pc),
                    kind="flops", amount_per_call=FA, sh=sh, sw=sw)
            addpair("sub%dx%d_both_l1" % (sh, sw), lin(x_fl, w_l, L1, pcfg=pc), l1=True,
                    kind="flops", amount_per_call=FA, sh=sh, sw=sw)

        xk8 = dev(torch.randn(B * M, K * 8) * 0.05)
        wk8 = dev(torch.randn(K * 8, N) * 0.05)
        hold += [xk8, wk8]
        Fk8 = flops(B, M, K * 8, N)
        for ibw in (1, 2, 4, 8, 16, 32):
            try:
                pc = pcfg(ibw, min(2, DEST_TILES // per_n), per_n, 2, per_n)
            except Exception as e:                                         # noqa: BLE001
                print("  ibw %d refused at build: %r" % (ibw, repr(e)[:120]), flush=True)
                continue
            addpair("ibw%02d_ship" % ibw, lin(xk8, wk8, DRAM, pcfg=pc), kind="flops",
                    amount_per_call=Fk8, ibw=ibw)

    # ---- warm, dropping what the part refuses, then run interleaved --------------------
    print("arms=%d reps=%d warm=%d" % (len(arms), a.reps, a.warm), flush=True)
    refused, live = {}, []
    t_warm = time.time()
    for nm, fn, nper, inst, meta in arms:
        try:
            if inst == "synced":
                synced(fn, 0, 1, nper)
            live.append((nm, fn, nper, inst, meta))
        except Exception as e:                                             # noqa: BLE001
            refused[nm] = repr(e)[:400]
            print("  REFUSED %s: %s" % (nm, repr(e)[:200]), flush=True)
    print("warm+compile %.1f s, live=%d refused=%d" % (time.time() - t_warm, len(live),
                                                       len(refused)), flush=True)

    # traced arms cannot interleave (one capture each), so take them first, whole
    res = {}
    for nm, fn, nper, inst, meta in live:
        if inst != "traced":
            continue
        try:
            res[nm] = traced(fn, a.reps, a.warm, nper)
        except Exception as e:                                             # noqa: BLE001
            refused[nm] = "traced: " + repr(e)[:400]
            print("  REFUSED(trace) %s: %s" % (nm, repr(e)[:200]), flush=True)
    sync_arms = [t for t in live if t[3] == "synced"]
    for nm, *_ in sync_arms:
        res[nm] = []
    for rep in range(a.reps):
        order = sync_arms if rep % 2 == 0 else list(reversed(sync_arms))
        for nm, fn, nper, inst, meta in order:
            res[nm] += synced(fn, 1, 0, nper)
        if (rep + 1) % 5 == 0:
            print("  rep %d/%d" % (rep + 1, a.reps), flush=True)

    t_measure_end = time.time()
    clock = sampler.stop()
    meta_by = {nm: (nper, inst, m) for nm, fn, nper, inst, m in live}
    out = {"host": os.uname().nodename, "tag": a.tag, "cores": cores,
           "grid": [g.x, g.y], "arch": repr(device.arch()), "nodes": held,
           "clock_mhz": a.clock, "clock": clock, "reps": a.reps, "warm": a.warm,
           "t_measure_start": t_measure_start, "t_measure_end": t_measure_end,
           "key": {"b": B, "m": M, "k": K, "n": N, "flops": FA, "min_bytes": BA},
           "refused": refused, "arms": {}}
    for nm, ms in res.items():
        if not ms:
            continue
        nper, inst, m = meta_by[nm]
        lo = min(ms)
        row = {"inst": inst, "nper": nper, "ms_min": lo, "ms_med": st.median(ms),
               "ms_all": ms, "n": len(ms), "spread_pct": 100.0 * (max(ms) - lo) / lo,
               "per_call_ms": lo / nper}
        row.update(m)
        if m.get("kind") == "flops":
            row["tflops"] = m["amount"] / (lo * 1e-3) / 1e12
        elif m.get("kind") == "bytes":
            row["gbs"] = m["amount"] / (lo * 1e-3) / 1e9
        out["arms"][nm] = row

    dest = HERE / ("fixterm_%s.json" % a.tag)
    dest.write_text(json.dumps(out, indent=1))
    print("\n%-22s %5s %7s %10s %10s %9s" % ("arm", "n", "inst", "ms_min", "per_call", "TFLOP/s"))
    for nm in sorted(out["arms"]):
        r = out["arms"][nm]
        print("%-22s %5d %7s %10.5f %10.5f %9s"
              % (nm, r["nper"], r["inst"], r["ms_min"], r["per_call_ms"],
                 "%.2f" % r["tflops"] if r.get("tflops") else
                 ("%.1f GB/s" % r["gbs"] if r.get("gbs") else "-")))
    print("\nclock during: %r" % (clock,))
    print("span %.1f s, wrote %s" % (t_measure_end - t_measure_start, dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
