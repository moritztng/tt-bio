#!/usr/bin/env python3
"""p2-matmul-ceiling — the Phase-2 experiment set, qb1 card 0, ttnn 0.67.4.

Arms:
  roofs  the unary DRAM read/write/copy roofs and the matmul-writer write roof in both operand
         placements, re-measured on THIS card. Nothing is inherited.
  h1     M sweep at K=256, nt=8, DRAM output -> does time scale with M, and does in0's buffer type
         change the per-element cost?
  h2     52.43 MB of output written two ways (nt=8 M=102400, nt=64 M=12800) -> is overlap set by
         per-core block width or by total bytes?
  h3     in0_block_w 1/2/4/8 at K=256 with an L1 output, everything else pinned.
  q12    achieved write rate as a function of the op's own read:write byte ratio, DRAM operands
         against a matched L1-operand arm at the same shape.
  sites  the two narrow-output production shapes at the fold's own [1,298,298,256], baseline
         `core_grid=CORE_GRID_MAIN` against the tuned 1D config, with a core ladder and parity.

Every timed region synchronises immediately before the clock starts and immediately before it
stops (`ttnn-sync-before-every-timed-region`).
"""
import argparse, json, math, statistics as st, sys, time

import torch
import ttnn

from tt_bio.tenstorrent import (get_device, CORE_GRID_MAIN, COMPUTE_GRID_MAIN,
                                _pair_proj_program_config)

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = 32
dev = get_device()
dg = dev.compute_with_storage_grid_size()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
GX, GY = COMPUTE_GRID_MAIN
NUM_CORES = GX * GY
try:
    from tt_bio.tenstorrent import _l1_bank_bytes
    L1_BANK = _l1_bank_bytes()
except Exception:                                                             # noqa: BLE001
    L1_BANK = 1_461_760
print(f"L1 bank budget {L1_BANK} B", flush=True)
print(f"compute_grid={dg.x}x{dg.y}  CORE_GRID_MAIN={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}  "
      f"COMPUTE_GRID_MAIN={GX}x{GY} ({NUM_CORES} cores)", flush=True)


def timed(fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def T(shape, mc, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ttnn.from_torch(torch.randn(*shape, generator=g), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


def pc(m_tiles, k_tiles, n_tiles, bw, cores=None, obh=5):
    """A 1D in1-mcast config (M sharded over the grid), the class every production site here is.

    Same construction as `_pair_proj_program_config` but with `out_block_h` and the core count
    exposed, because H2 is about `out_block_h` and the ladder needs the core count.
    """
    cores = cores or NUM_CORES
    if k_tiles % bw:
        return None
    per_core_M = -(-(-(-m_tiles // cores)) // obh) * obh
    if per_core_M > m_tiles or -(-m_tiles // per_core_M) > cores:
        return None
    obh = min(obh, per_core_M)
    while per_core_M % obh:
        obh -= 1
    obw = n_tiles
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0)
    # L1 bank budget, `_pair_proj_program_config`'s own arithmetic: in0 and in1 double-buffered per
    # K block, plus the output block's bf16 tile and the fp32 partial the packer accumulates into.
    # The output term is NOT dropped here -- `perfwar-programconfig-gate-output-not-subtracted`.
    need = (2 * bw * (obh + obw) * 2048 + obh * obw * (2048 + 4096) + 128 * 1024)
    if need > L1_BANK:
        return None
    try:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(GX, GY),
            in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
            out_block_h=obh, out_block_w=obw, per_core_M=per_core_M, per_core_N=n_tiles,
            fuse_batch=True, fused_activation=None, mcast_in0=False)
    except Exception as e:                                                    # noqa: BLE001
        print("  pc build failed:", str(e)[:90], flush=True)
        return None


def pc_cores(m_tiles, per_core_M):
    return -(-m_tiles // per_core_M)


def run_mm(a, b, omem, **kw):
    return timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                     memory_config=omem, **kw)))


# ---------------------------------------------------------------------------------------------
def arm_roofs():
    r = {}
    print("=== unary roofs on THIS card (card 0) ===", flush=True)
    n = 1024 * 4096                                     # 8.39 MB per clone, x8 below
    big = (1, 1, 8192, 4096)                            # 67.1 MB
    src_d = T(big, DRAM)
    src_l = None
    try:
        src_l = T(big, L1)
    except Exception as e:                                                    # noqa: BLE001
        print("  L1 source alloc failed:", str(e)[:80], flush=True)
    nb = big[2] * big[3] * 2
    t = timed(lambda: ttnn.deallocate(ttnn.clone(src_d, memory_config=L1)))
    r["read_GBs"] = round(nb / t / 1e9, 1)
    print(f"  DRAM->L1 read roof   {nb/t/1e9:7.1f} GB/s  ({t*1e6:.1f} us)", flush=True)
    if src_l is not None:
        t = timed(lambda: ttnn.deallocate(ttnn.clone(src_l, memory_config=DRAM)))
        r["write_GBs"] = round(nb / t / 1e9, 1)
        print(f"  L1->DRAM write roof  {nb/t/1e9:7.1f} GB/s  ({t*1e6:.1f} us)", flush=True)
        ttnn.deallocate(src_l)
    t = timed(lambda: ttnn.deallocate(ttnn.clone(src_d, memory_config=DRAM)))
    r["copy_GBs"] = round(2 * nb / t / 1e9, 1)
    print(f"  DRAM->DRAM copy roof {2*nb/t/1e9:7.1f} GB/s  ({t*1e6:.1f} us)", flush=True)
    ttnn.deallocate(src_d)

    print("=== compute roof: square bf16 HiFi4, L1 output ===", flush=True)
    for M in (4096,):
        a, b = T((1, 1, M, M), DRAM), T((1, 1, M, M), DRAM)
        t = run_mm(a, b, DRAM, core_grid=ttnn.CoreGrid(y=dg.y, x=dg.x))
        r["compute_square_TFLOPs"] = round(2 * M ** 3 / t / 1e12, 2)
        print(f"  {M}^3 DRAM out  {2*M**3/t/1e12:7.2f} TFLOP/s ({t*1e6:.1f} us)", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)

    print("=== matmul-writer write roof, K=256 nt=64 M=12800, both operand placements ===",
          flush=True)
    M, K, N = 12800, 256, 2048
    outB = M * N * 2
    for lbl, mc in (("L1 operands", L1), ("DRAM operands", DRAM)):
        try:
            a, b = T((1, 1, M, K), mc), T((1, 1, K, N), mc)
        except Exception as e:                                                # noqa: BLE001
            print(f"  {lbl}: alloc {str(e)[:70]}", flush=True); continue
        best, bestcfg = None, None
        cfgs = [("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)}),
                ("cg_11x10", {"core_grid": CORE_GRID_MAIN})]
        p = pc(M // TILE, K // TILE, N // TILE, 8)
        if p is not None:
            cfgs.append(("pc_bw8_obh5", {"program_config": p}))
        for cl, kw in cfgs:
            try:
                t = run_mm(a, b, DRAM, **kw)
            except Exception as e:                                            # noqa: BLE001
                print(f"  {lbl:14s} {cl:12s} ERR {str(e)[:60]}", flush=True); continue
            g = outB / t / 1e9
            print(f"  {lbl:14s} {cl:12s} {t*1e6:8.2f} us  write {g:6.1f} GB/s  "
                  f"{2*M*K*N/t/1e12:6.2f} TFLOP/s", flush=True)
            if best is None or g > best:
                best, bestcfg = g, cl
        r[f"mmwrite_roof_{'L1' if mc is L1 else 'DRAM'}_operands_GBs"] = round(best, 1)
        r[f"mmwrite_roof_{'L1' if mc is L1 else 'DRAM'}_operands_cfg"] = bestcfg
        ttnn.deallocate(a); ttnn.deallocate(b)
    return r


# ---------------------------------------------------------------------------------------------
def arm_h1():
    K, N = 256, 256
    rows = []
    print("=== H1: M sweep, K=256 nt=8, DRAM output, L1 operands ===", flush=True)
    for M in (4096, 8192, 16384, 32768):
        try:
            a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
        except Exception as e:                                                # noqa: BLE001
            print(f"  M={M}: L1 alloc {str(e)[:70]}", flush=True); continue
        gflop = 2 * M * K * N / 1e9
        for cl, kw in (("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                       ("pc_bw8", {"program_config": pc(M // TILE, K // TILE, N // TILE, 8)})):
            if kw.get("program_config", 1) is None:
                continue
            t = run_mm(a, b, DRAM, **kw)
            rows.append({"arm": "L1_in", "M": M, "cfg": cl, "us": round(t * 1e6, 2),
                         "tflops": round(gflop / t / 1e3, 2),
                         "ns_per_row": round(t * 1e9 / M, 3)})
            print(f"  M={M:6d} {cl:9s} {t*1e6:8.2f} us  {gflop/t/1e3:6.2f} TFLOP/s  "
                  f"{t*1e9/M:7.3f} ns/row", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)

    print("--- second falsifier: in0 in DRAM instead of L1, identical shape ---", flush=True)
    for M in (16384,):
        a, b = T((1, 1, M, K), DRAM), T((1, 1, K, N), L1)
        gflop = 2 * M * K * N / 1e9
        for cl, kw in (("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                       ("pc_bw8", {"program_config": pc(M // TILE, K // TILE, N // TILE, 8)})):
            if kw.get("program_config", 1) is None:
                continue
            t = run_mm(a, b, DRAM, **kw)
            rows.append({"arm": "DRAM_in", "M": M, "cfg": cl, "us": round(t * 1e6, 2),
                         "tflops": round(gflop / t / 1e3, 2),
                         "ns_per_row": round(t * 1e9 / M, 3)})
            print(f"  M={M:6d} {cl:9s} DRAM in0 {t*1e6:8.2f} us  {gflop/t/1e3:6.2f} TFLOP/s  "
                  f"{t*1e9/M:7.3f} ns/row", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    return rows


# ---------------------------------------------------------------------------------------------
def arm_h2():
    """52.43 MB of output, written two ways. Compute basis and write basis measured separately."""
    out_rows = []
    K = 256
    write_bytes = 12800 * 2048 * 2
    print(f"=== H2: {write_bytes/1e6:.2f} MB of DRAM output, nt=8 vs nt=64 ===", flush=True)

    print("--- write basis: unary L1->DRAM clone of exactly those bytes ---", flush=True)
    wb = None
    try:
        s = T((1, 1, 12800, 2048), L1)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(s, memory_config=DRAM)))
        wb = t
        print(f"  {t*1e6:8.2f} us  {write_bytes/t/1e9:6.1f} GB/s", flush=True)
        ttnn.deallocate(s)
    except Exception as e:                                                    # noqa: BLE001
        print("  write basis alloc:", str(e)[:70], flush=True)

    for nt, M in ((8, 102400), (64, 12800)):
        N = nt * TILE
        print(f"--- nt={nt} M={M} (out {M*N*2/1e6:.2f} MB) ---", flush=True)
        try:
            a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
        except Exception as e:                                                # noqa: BLE001
            print(f"  L1 alloc {str(e)[:70]}", flush=True); continue
        gflop = 2 * M * K * N / 1e9
        cfgs = [("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)})]
        p = pc(M // TILE, K // TILE, nt, 8)
        if p is not None:
            cfgs.append(("pc_bw8_obh5", {"program_config": p}))
        best_dram = None
        for cl, kw in cfgs:
            try:
                t = run_mm(a, b, DRAM, **kw)
            except Exception as e:                                            # noqa: BLE001
                print(f"  out=DRAM {cl:12s} ERR {str(e)[:60]}", flush=True); continue
            print(f"  out=DRAM {cl:12s} {t*1e6:9.2f} us  {gflop/t/1e3:6.2f} TFLOP/s  "
                  f"write {M*N*2/t/1e9:6.1f} GB/s", flush=True)
            out_rows.append({"nt": nt, "M": M, "out": "DRAM", "cfg": cl,
                             "us": round(t * 1e6, 2), "tflops": round(gflop / t / 1e3, 2),
                             "write_GBs": round(M * N * 2 / t / 1e9, 1)})
            if best_dram is None or t < best_dram:
                best_dram = t
        # compute basis: the same matmul with an L1 output, at whatever M fits, scaled by FLOPs
        cb = None
        for Mc in (M, M // 2, M // 4, M // 8, M // 16):
            try:
                ac = a if Mc == M else T((1, 1, Mc, K), L1)
                tc = run_mm(ac, b, L1, core_grid=ttnn.CoreGrid(y=dg.y, x=dg.x))
            except Exception:                                                 # noqa: BLE001
                continue
            rate = 2 * Mc * K * N / tc / 1e12
            cb = gflop / 1e3 / rate
            print(f"  compute basis at M={Mc}: {rate:6.2f} TFLOP/s -> {cb*1e6:9.2f} us at M={M}",
                  flush=True)
            if ac is not a:
                ttnn.deallocate(ac)
            break
        if best_dram and cb and wb:
            mx, sm = max(cb, wb), cb + wb
            out_rows.append({"nt": nt, "M": M, "summary": True,
                             "measured_us": round(best_dram * 1e6, 2),
                             "compute_us": round(cb * 1e6, 2), "write_us": round(wb * 1e6, 2),
                             "max_us": round(mx * 1e6, 2), "sum_us": round(sm * 1e6, 2),
                             "frac_of_sum": round(best_dram / sm, 3),
                             "ratio_to_max": round(best_dram / mx, 3)})
            print(f"  ==> measured {best_dram*1e6:9.2f} us | max() {mx*1e6:9.2f} | "
                  f"sum() {sm*1e6:9.2f} | measured/sum {best_dram/sm:.3f} | "
                  f"measured/max {best_dram/mx:.3f}", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    return out_rows


# ---------------------------------------------------------------------------------------------
def arm_h3():
    """`in0_block_w` swept 1/2/4/8 with everything else pinned, at three K and three output widths.

    Output width is held at the production narrow value (nt=8) for the K ladder, so the K=256 arm
    and the K=4096 arm differ in nothing but the contraction length: if H3 holds, bw=8 at K=256
    climbs toward the K=4096 rate.
    """
    rows = []
    print("=== H3: in0_block_w sweep at nt=8, K = 256 / 1024 / 4096, L1 in and L1 out ===",
          flush=True)
    for K, M in ((256, 16384), (1024, 16384), (4096, 4096)):
        nt, N = 8, 256
        try:
            a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
        except Exception as e:                                                # noqa: BLE001
            print(f"  K={K} L1 alloc {str(e)[:70]}", flush=True); continue
        gflop = 2 * M * K * N / 1e9
        for bw in (1, 2, 4, 8, 16):
            p = pc(M // TILE, K // TILE, nt, bw)
            if p is None:
                continue
            for omem, ol in ((L1, "L1"), (DRAM, "DRAM")):
                try:
                    t = run_mm(a, b, omem, program_config=p)
                except Exception as e:                                        # noqa: BLE001
                    print(f"  K={K} bw={bw} out={ol:4s} ERR {str(e)[:70]}", flush=True); continue
                rows.append({"K": K, "nt": nt, "M": M, "bw": bw, "out": ol,
                             "us": round(t * 1e6, 2), "tflops": round(gflop / t / 1e3, 2)})
                print(f"  K={K:5d} bw={bw:2d} out={ol:4s} {t*1e6:9.2f} us  "
                      f"{gflop/t/1e3:7.2f} TFLOP/s", flush=True)
        # the config-free reference at the same shape
        for cl, kw in (("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                       ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)})):
            for omem, ol in ((L1, "L1"), (DRAM, "DRAM")):
                try:
                    t = run_mm(a, b, omem, **kw)
                except Exception as e:                                        # noqa: BLE001
                    print(f"  K={K} {cl} out={ol} ERR {str(e)[:60]}", flush=True); continue
                rows.append({"K": K, "nt": nt, "M": M, "bw": None, "cfg": cl, "out": ol,
                             "us": round(t * 1e6, 2), "tflops": round(gflop / t / 1e3, 2)})
                print(f"  K={K:5d} {cl:9s} out={ol:4s} {t*1e6:9.2f} us  "
                      f"{gflop/t/1e3:7.2f} TFLOP/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)

    print("=== H3: does the same bw sweep move a WIDE output? (nt=32, K=256) ===", flush=True)
    K, M, nt, N = 256, 16384, 32, 1024
    try:
        a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
        gflop = 2 * M * K * N / 1e9
        for bw in (1, 2, 4, 8):
            p = pc(M // TILE, K // TILE, nt, bw)
            if p is None:
                print(f"  nt=32 bw={bw}: config exceeds the L1 bank budget", flush=True); continue
            try:
                t = run_mm(a, b, L1, program_config=p)
            except Exception as e:                                            # noqa: BLE001
                print(f"  nt=32 bw={bw} ERR {str(e)[:60]}", flush=True); continue
            rows.append({"K": K, "nt": nt, "M": M, "bw": bw, "out": "L1",
                         "us": round(t * 1e6, 2), "tflops": round(gflop / t / 1e3, 2)})
            print(f"  nt=32 bw={bw} {t*1e6:9.2f} us  {gflop/t/1e3:7.2f} TFLOP/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    except Exception as e:                                                    # noqa: BLE001
        print("  nt=32 alloc", str(e)[:70], flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
def arm_q12():
    """Achieved matmul write rate vs the op's own read:write byte ratio.

    Output held at 48.82 MB and nt=8 throughout (the fold's own pair-track output), K varied so
    that read:write moves through 1:4 .. 3:1. Each K runs twice: operands in DRAM (the real op,
    which pays read/write contention) and operands in L1 (contention-free, the writer alone).
    """
    rows = []
    M = 95360                     # 298 x 320, the fold's own row count
    N, nt = 256, 8
    outB = M * N * 2
    print(f"=== Q12: write rate vs read:write, out {outB/1e6:.2f} MB at nt=8 ===", flush=True)
    for K in (64, 128, 256, 512, 768):
        readB = M * K * 2 + K * N * 2
        ratio = readB / outB
        p = pc(M // TILE, max(1, K // TILE), nt, min(8, max(1, K // TILE)))
        for lbl, mc in (("DRAM", DRAM), ("L1", L1)):
            try:
                a, b = T((1, 1, M, K), mc), T((1, 1, K, N), mc)
            except Exception as e:                                            # noqa: BLE001
                print(f"  K={K} {lbl} operands: alloc {str(e)[:60]}", flush=True); continue
            for cl, kw in (("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                           ("pc_bw", {"program_config": p})):
                if kw.get("program_config", 1) is None:
                    continue
                try:
                    t = run_mm(a, b, DRAM, **kw)
                except Exception as e:                                        # noqa: BLE001
                    print(f"  K={K} {lbl} {cl} ERR {str(e)[:60]}", flush=True); continue
                rows.append({"K": K, "operands": lbl, "cfg": cl, "read_MB": round(readB / 1e6, 2),
                             "write_MB": round(outB / 1e6, 2), "read_over_write": round(ratio, 3),
                             "us": round(t * 1e6, 2), "write_GBs": round(outB / t / 1e9, 1),
                             "tflops": round(2 * M * K * N / t / 1e12, 2)})
                print(f"  K={K:4d} r:w={ratio:5.2f} {lbl:4s} {cl:9s} {t*1e6:9.2f} us  "
                      f"write {outB/t/1e9:6.1f} GB/s", flush=True)
            ttnn.deallocate(a); ttnn.deallocate(b)

    print("=== Q12: T1's two rows at their own shapes ===", flush=True)
    for name, K, N2 in (("qkv@1428", 256, 768), ("gate@1434", 256, 256)):
        Mq = 95360
        oB = Mq * N2 * 2
        rB = Mq * K * 2 + K * N2 * 2
        for lbl, mc in (("DRAM", DRAM), ("L1", L1)):
            try:
                a, b = T((1, 1, Mq, K), mc), T((1, 1, K, N2), mc)
            except Exception as e:                                            # noqa: BLE001
                print(f"  {name} {lbl}: alloc {str(e)[:60]}", flush=True); continue
            p = pc(Mq // TILE, K // TILE, N2 // TILE, 8)
            for cl, kw in (("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                           ("pc_bw8", {"program_config": p})):
                if kw.get("program_config", 1) is None:
                    continue
                try:
                    t = run_mm(a, b, DRAM, **kw)
                except Exception as e:                                        # noqa: BLE001
                    print(f"  {name} {lbl} {cl} ERR {str(e)[:60]}", flush=True); continue
                rows.append({"site": name, "K": K, "N": N2, "operands": lbl, "cfg": cl,
                             "read_MB": round(rB / 1e6, 2), "write_MB": round(oB / 1e6, 2),
                             "read_over_write": round(rB / oB, 3), "us": round(t * 1e6, 2),
                             "write_GBs": round(oB / t / 1e9, 1)})
                print(f"  {name:10s} r:w={rB/oB:5.2f} {lbl:4s} {cl:9s} {t*1e6:9.2f} us  "
                      f"write {oB/t/1e9:6.1f} GB/s", flush=True)
            ttnn.deallocate(a); ttnn.deallocate(b)
    return rows


# ---------------------------------------------------------------------------------------------
SITES = [("PWA z->bias @tenstorrent.py:2832", 1, 240), ("template z proj @protenix.py:306", 64, 40)]


def arm_sites():
    rows = []
    tok, cz = 298, 256
    for name, cout, calls in SITES:
        n_tiles = -(-cout // TILE)
        print(f"=== {name}: [1,{tok},{tok},{cz}] x [{cz},{cout}] (nt={n_tiles}), "
              f"{calls} calls/fold ===", flush=True)
        xt = torch.randn(1, tok, tok, cz, generator=torch.Generator().manual_seed(1))
        wt = torch.randn(cz, cout, generator=torch.Generator().manual_seed(2))
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=DRAM)
        w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=DRAM)
        m_tiles = tok * -(-tok // TILE)
        k_tiles = cz // TILE
        readB = tok * (-(-tok // TILE) * TILE) * cz * 2
        writeB = tok * (-(-tok // TILE) * TILE) * (n_tiles * TILE) * 2
        print(f"  m_tiles={m_tiles} k_tiles={k_tiles} n_tiles={n_tiles}  "
              f"read {readB/1e6:.2f} MB  write {writeB/1e6:.2f} MB", flush=True)

        def call(**kw):
            return ttnn.linear(x, w, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                               memory_config=DRAM, **kw)

        base_t = timed(lambda: ttnn.deallocate(call(core_grid=CORE_GRID_MAIN)))
        base_out = ttnn.to_torch(call(core_grid=CORE_GRID_MAIN))
        print(f"  baseline core_grid=11x10 : {base_t*1e6:9.2f} us   "
              f"{base_t*calls*1e6/1e3:7.1f} ms/fold  read {readB/base_t/1e9:6.1f} GB/s",
              flush=True)
        rows.append({"site": name, "cfg": "baseline core_grid 11x10", "us": round(base_t * 1e6, 2),
                     "ms_fold": round(base_t * calls * 1e3, 2), "calls": calls,
                     "read_GBs": round(readB / base_t / 1e9, 1),
                     "write_GBs": round(writeB / base_t / 1e9, 1)})

        print("  -- baseline core ladder (cores engaged) --", flush=True)
        for gx, gy in ((2, 2), (4, 4), (6, 6), (8, 8), (11, 10), (13, 10)):
            if gx > dg.x or gy > dg.y:
                continue
            try:
                t = timed(lambda: ttnn.deallocate(call(core_grid=ttnn.CoreGrid(y=gy, x=gx))))
            except Exception as e:                                            # noqa: BLE001
                print(f"    {gx}x{gy}: ERR {str(e)[:60]}", flush=True); continue
            rows.append({"site": name, "cfg": f"ladder core_grid {gx}x{gy}",
                         "cores": gx * gy, "us": round(t * 1e6, 2)})
            print(f"    {gx}x{gy} ({gx*gy:3d} cores) {t*1e6:9.2f} us   "
                  f"x vs 11x10 = {t/base_t:.3f}", flush=True)

        print("  -- tuned 1D config: in0_block_w = k_tiles, per_core_M ~ 30 --", flush=True)
        best = None
        for bw in (1, 2, 4, 8):
            for obh in (5, 10, 15, 30):
                p = pc(m_tiles, k_tiles, n_tiles, bw, obh=obh)
                if p is None:
                    continue
                try:
                    t = timed(lambda: ttnn.deallocate(call(program_config=p)))
                except Exception as e:                                        # noqa: BLE001
                    print(f"    bw={bw} obh={obh}: ERR {str(e)[:60]}", flush=True); continue
                cores = pc_cores(m_tiles, p.per_core_M)
                rows.append({"site": name, "cfg": f"pc bw={bw} obh={p.out_block_h} "
                                                  f"per_core_M={p.per_core_M}",
                             "bw": bw, "cores": cores, "us": round(t * 1e6, 2),
                             "ms_fold": round(t * calls * 1e3, 2),
                             "speedup": round(base_t / t, 3),
                             "read_GBs": round(readB / t / 1e9, 1)})
                print(f"    bw={bw} obh={p.out_block_h:2d} per_core_M={p.per_core_M:3d} "
                      f"cores={cores:3d}/{NUM_CORES} {t*1e6:9.2f} us  {base_t/t:5.3f}x  "
                      f"{t*calls*1e3:6.1f} ms/fold  read {readB/t/1e9:6.1f} GB/s", flush=True)
                if best is None or t < best[0]:
                    best = (t, p, bw)
        if best is not None:
            t, p, bw = best
            tuned_out = ttnn.to_torch(call(program_config=p))
            eq = bool(torch.equal(base_out, tuned_out))
            d = (base_out.float() - tuned_out.float())
            rmsd = float(torch.sqrt((d ** 2).mean()))
            denom = float(base_out.float().std())
            pcc = float(torch.corrcoef(torch.stack([base_out.float().flatten(),
                                                    tuned_out.float().flatten()]))[0, 1])
            rows.append({"site": name, "parity": True, "best_bw": bw,
                         "best_us": round(t * 1e6, 2), "torch_equal": eq,
                         "rmsd": rmsd, "rel_rmsd": rmsd / denom if denom else None, "pcc": pcc})
            print(f"  PARITY best (bw={bw}): torch.equal={eq}  RMSD={rmsd:.3e}  "
                  f"rel={rmsd/denom:.3e}  PCC={pcc:.8f}", flush=True)
            # bw=1 alone (drain schedule only) must be bit-exact
            p1 = pc(m_tiles, k_tiles, n_tiles, 1, obh=5)
            if p1 is not None:
                o1 = ttnn.to_torch(call(program_config=p1))
                print(f"  PARITY bw=1 obh=5 (drain schedule only): "
                      f"torch.equal={bool(torch.equal(base_out, o1))}", flush=True)
                rows.append({"site": name, "parity": True, "cfg": "pc bw=1 obh=5",
                             "torch_equal": bool(torch.equal(base_out, o1))})
        ttnn.deallocate(x); ttnn.deallocate(w)
    return rows




# ---------------------------------------------------------------------------------------------
def arm_h3b():
    """H3 again, with `per_core_M` as the free variable.

    The nt=8 M=16384 arm above put per_core_M at 5 and the rate FELL with `in0_block_w`; the two
    production sites put it at 30 and the rate ROSE ~2x. So the K-loop reading of H3 cannot be
    right on its own: what a wider `in0_block_w` buys is one in1 mcast barrier / DEST clear /
    packer pass per output block instead of Kt of them, and a core only has blocks to amortise
    over when it owns many tile rows. This arm sweeps per_core_M directly.
    """
    rows = []
    K, N, nt = 256, 256, 8
    print("=== H3b: in0_block_w x per_core_M at K=256, nt=8 ===", flush=True)
    for M, omems in ((16384, ("L1", "DRAM")), (47680, ("L1", "DRAM")), (95360, ("DRAM",))):
        m_tiles = M // TILE
        try:
            a, b = T((1, 1, M, K), DRAM), T((1, 1, K, N), DRAM)
        except Exception as e:                                                # noqa: BLE001
            print(f"  M={M} alloc {str(e)[:70]}", flush=True); continue
        gflop = 2 * M * K * N / 1e9
        base = {}
        for ol in omems:
            omem = L1 if ol == "L1" else DRAM
            try:
                tb = run_mm(a, b, omem, core_grid=CORE_GRID_MAIN)
            except Exception as e:                                            # noqa: BLE001
                print(f"  M={M} out={ol} baseline ERR {str(e)[:60]}", flush=True); continue
            base[ol] = tb
            rows.append({"M": M, "m_tiles": m_tiles, "out": ol, "cfg": "cg_11x10",
                         "us": round(tb * 1e6, 2), "tflops": round(gflop / tb / 1e3, 2)})
            print(f"  M={M:6d} out={ol:4s} cg_11x10 {tb*1e6:9.2f} us  "
                  f"{gflop/tb/1e3:7.2f} TFLOP/s", flush=True)
        for bw in (1, 2, 4, 8):
            p = pc(m_tiles, K // TILE, nt, bw)
            if p is None:
                continue
            for ol in omems:
                omem = L1 if ol == "L1" else DRAM
                try:
                    t = run_mm(a, b, omem, program_config=p)
                except Exception as e:                                        # noqa: BLE001
                    print(f"  M={M} bw={bw} out={ol} ERR {str(e)[:60]}", flush=True); continue
                rows.append({"M": M, "m_tiles": m_tiles, "per_core_M": p.per_core_M,
                             "cores": pc_cores(m_tiles, p.per_core_M), "bw": bw, "out": ol,
                             "us": round(t * 1e6, 2), "tflops": round(gflop / t / 1e3, 2),
                             "vs_bw1": None})
                print(f"  M={M:6d} pcM={p.per_core_M:3d} cores={pc_cores(m_tiles,p.per_core_M):3d} "
                      f"bw={bw} out={ol:4s} {t*1e6:9.2f} us  {gflop/t/1e3:7.2f} TFLOP/s"
                      + (f"  {base[ol]/t:5.3f}x vs cg" if ol in base else ""), flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    return rows


def _pc_any(m_tiles, k_tiles, n_tiles, bw, cores=None):
    """Best legal config over a range of out_block_h, for wide outputs where obh=5 will not fit."""
    out = []
    for obh in (1, 2, 3, 5, 10, 15, 30):
        p = pc(m_tiles, k_tiles, n_tiles, bw, cores=cores, obh=obh)
        if p is not None:
            out.append(p)
    return out


def arm_q12b():
    """The write roof a real trunk matmul reaches, and the decomposition that explains Q12."""
    rows = []
    mm = getattr(ttnn.experimental, "minimal_matmul", None)
    print(f"=== Q12b: minimal_matmul available: {mm is not None} ===", flush=True)

    print("--- the highest write rate ANY matmul reaches on this card (K=256, L1 operands) ---",
          flush=True)
    K = 256
    for nt in (8, 16, 32, 64, 128):
        N = nt * TILE
        M = max(TILE, int(52.4e6 / (2 * N)) // TILE * TILE)
        outB = M * N * 2
        try:
            a, b = T((1, 1, M, K), L1), T((1, 1, K, N), L1)
        except Exception as e:                                                # noqa: BLE001
            print(f"  nt={nt}: L1 alloc {str(e)[:60]}", flush=True); continue
        best = None
        cands = [("cg_11x10", {"core_grid": CORE_GRID_MAIN}),
                 ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)})]
        for bw in (1, 2, 4, 8):
            for p in _pc_any(M // TILE, K // TILE, nt, bw):
                cands.append((f"pc bw={bw} obh={p.out_block_h}", {"program_config": p}))
        for cl, kw in cands:
            try:
                t = run_mm(a, b, DRAM, **kw)
            except Exception:                                                 # noqa: BLE001
                continue
            g = outB / t / 1e9
            if best is None or g > best[0]:
                best = (g, cl, t)
        if mm is not None:
            try:
                t = timed(lambda: ttnn.deallocate(mm(a, b, memory_config=DRAM,
                                                     dtype=ttnn.bfloat16,
                                                     compute_kernel_config=CKC)))
                g = outB / t / 1e9
                print(f"  nt={nt:3d} minimal_matmul {t*1e6:9.2f} us  write {g:6.1f} GB/s",
                      flush=True)
                rows.append({"probe": "write_roof_search", "nt": nt, "M": M, "cfg":
                             "minimal_matmul", "us": round(t * 1e6, 2), "write_GBs": round(g, 1)})
                if best is None or g > best[0]:
                    best = (g, "minimal_matmul", t)
            except Exception as e:                                            # noqa: BLE001
                print(f"  nt={nt} minimal_matmul ERR {str(e)[:60]}", flush=True)
        if best:
            print(f"  nt={nt:3d} M={M:7d} out {outB/1e6:6.2f} MB  BEST {best[0]:6.1f} GB/s "
                  f"({best[1]}, {best[2]*1e6:.2f} us)", flush=True)
            rows.append({"probe": "write_roof_search", "nt": nt, "M": M,
                         "out_MB": round(outB / 1e6, 2), "best_write_GBs": round(best[0], 1),
                         "best_cfg": best[1], "us": round(best[2] * 1e6, 2)})
        ttnn.deallocate(a); ttnn.deallocate(b)

    print("--- Q12 decomposition: DRAM-out time minus L1-out time = the exposed write ---",
          flush=True)
    M, N, nt = 95360, 256, 8
    outB = M * N * 2
    for K in (64, 128, 256, 512, 768):
        kt = max(1, K // TILE)
        try:
            a, b = T((1, 1, M, K), DRAM), T((1, 1, K, N), DRAM)
        except Exception as e:                                                # noqa: BLE001
            print(f"  K={K} alloc {str(e)[:60]}", flush=True); continue
        cands = [("cg_11x10", {"core_grid": CORE_GRID_MAIN})]
        for bw in (1, 2, 4, 8):
            if kt % bw:
                continue
            for p in _pc_any(M // TILE, kt, nt, bw):
                cands.append((f"pc bw={bw} obh={p.out_block_h}", {"program_config": p}))
        res = {}
        for ol, omem in (("L1", L1), ("DRAM", DRAM)):
            best = None
            for cl, kw in cands:
                try:
                    t = run_mm(a, b, omem, **kw)
                except Exception:                                             # noqa: BLE001
                    continue
                if best is None or t < best[0]:
                    best = (t, cl)
            res[ol] = best
        if res.get("L1") and res.get("DRAM"):
            tl, cl_l = res["L1"]
            td, cl_d = res["DRAM"]
            exposed = td - tl
            print(f"  K={K:4d} r:w={((M*K*2)+(K*N*2))/outB:5.2f}  L1-out {tl*1e6:8.2f} us "
                  f"({cl_l})  DRAM-out {td*1e6:8.2f} us ({cl_d})  exposed write "
                  f"{exposed*1e6:8.2f} us = {outB/max(exposed,1e-9)/1e9:6.1f} GB/s  "
                  f"achieved write {outB/td/1e9:6.1f} GB/s", flush=True)
            rows.append({"probe": "decomp", "K": K, "read_over_write":
                         round(((M * K * 2) + (K * N * 2)) / outB, 3),
                         "l1_out_us": round(tl * 1e6, 2), "l1_cfg": cl_l,
                         "dram_out_us": round(td * 1e6, 2), "dram_cfg": cl_d,
                         "exposed_write_us": round(exposed * 1e6, 2),
                         "exposed_write_GBs": round(outB / max(exposed, 1e-9) / 1e9, 1),
                         "achieved_write_GBs": round(outB / td / 1e9, 1)})
        ttnn.deallocate(a); ttnn.deallocate(b)

    print("--- T1's two rows, production op (`minimal_matmul`) vs a tuned ttnn.linear ---",
          flush=True)
    tok = 298
    for name, cout, calls in (("qkv@1428", 768, 1048), ("gate@1434", 256, 1048)):
        xt = torch.randn(1, tok, tok, 256, generator=torch.Generator().manual_seed(3))
        wt = torch.randn(256, cout, generator=torch.Generator().manual_seed(4))
        Mt = tok * -(-tok // TILE)
        readB = tok * (-(-tok // TILE) * TILE) * 256 * 2
        writeB = tok * (-(-tok // TILE) * TILE) * cout * 2
        for ol, mc in (("DRAM", DRAM), ("L1", L1)):
            try:
                x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=mc)
                w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=mc)
            except Exception as e:                                            # noqa: BLE001
                print(f"  {name} operands {ol}: alloc {str(e)[:60]}", flush=True); continue
            cands = [("cg_11x10", {"core_grid": CORE_GRID_MAIN})]
            for bw in (1, 2, 4, 8):
                for p in _pc_any(Mt, 8, -(-cout // TILE), bw):
                    cands.append((f"pc bw={bw} obh={p.out_block_h}", {"program_config": p}))
            best = None
            for cl, kw in cands:
                try:
                    t = timed(lambda: ttnn.deallocate(
                        ttnn.linear(x, w, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                                    memory_config=DRAM, **kw)))
                except Exception:                                             # noqa: BLE001
                    continue
                if best is None or t < best[0]:
                    best = (t, cl)
            if mm is not None:
                try:
                    t = timed(lambda: ttnn.deallocate(mm(input_tensor=x, weight_tensor=w,
                                                         compute_kernel_config=CKC,
                                                         dtype=ttnn.bfloat16)))
                    print(f"  {name:10s} operands={ol:4s} minimal_matmul {t*1e6:9.2f} us  "
                          f"write {writeB/t/1e9:6.1f} GB/s  {t*calls*1e3:7.1f} ms/fold",
                          flush=True)
                    rows.append({"site": name, "operands": ol, "cfg": "minimal_matmul",
                                 "us": round(t * 1e6, 2),
                                 "write_GBs": round(writeB / t / 1e9, 1),
                                 "read_GBs": round(readB / t / 1e9, 1),
                                 "ms_fold": round(t * calls * 1e3, 2)})
                    if best is None or t < best[0]:
                        best = (t, "minimal_matmul")
                except Exception as e:                                        # noqa: BLE001
                    print(f"  {name} {ol} minimal_matmul ERR {str(e)[:70]}", flush=True)
            if best:
                t, cl = best
                print(f"  {name:10s} operands={ol:4s} BEST {cl:20s} {t*1e6:9.2f} us  "
                      f"write {writeB/t/1e9:6.1f} GB/s  read {readB/t/1e9:6.1f} GB/s  "
                      f"{t*calls*1e3:7.1f} ms/fold", flush=True)
                rows.append({"site": name, "operands": ol, "cfg": "BEST:" + cl,
                             "us": round(t * 1e6, 2), "write_GBs": round(writeB / t / 1e9, 1),
                             "read_GBs": round(readB / t / 1e9, 1),
                             "ms_fold": round(t * calls * 1e3, 2)})
            ttnn.deallocate(x); ttnn.deallocate(w)
    return rows


ARMS = {"roofs": arm_roofs, "h1": arm_h1, "h2": arm_h2, "h3": arm_h3, "h3b": arm_h3b,
        "q12": arm_q12, "q12b": arm_q12b, "sites": arm_sites}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS) + ["all"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for name in (sorted(ARMS) if a.arm == "all" else [a.arm]):
        try:
            res[name] = ARMS[name]()
        except Exception as e:                                                # noqa: BLE001
            import traceback; traceback.print_exc()
            res[name] = {"error": str(e)[:300]}
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote " + a.out, flush=True)


