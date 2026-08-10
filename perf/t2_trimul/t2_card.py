#!/usr/bin/env python3
"""T2 (trimul + pair-track projections) — roofs and mechanism probes on THIS card.

Everything in one process on one device, so a roof and the op it scores share a card, a ttnn
build and an allocator state. Every timed region synchronises on both sides.

Sections
  A  compute roof, HiFi4 dense bf16, fp32_dest_acc + packer_l1_acc (the config the model runs)
  B  the K-corrected compute rate at K=256, which is the only K the pair-track projections have
  C  DRAM read roof, swept
  D  DRAM WRITE roof by writer path  -- Q4. Three write-dominated paths with identical output
     bytes: unary clone (L1->DRAM), minimal_matmul, ttnn.linear 1D-mcast.
  E  the pair-track projection 1144/1154/1358, three configs + a measured core-utilisation A/B
  F  the L1-resident trimul rows: bytes, achieved GB/s, and a tile-count staircase that measures
     how many cores the op actually engages (period of the staircase = cores engaged)
  G  the channel-move permute: GB/s vs chunk width C at constant bytes (transaction-rate test)

    PYTHONPATH=<wt> python3 perf/t2_trimul/t2_card.py --out perf/t2_trimul/t2_card_qb2c0.json
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN, COMPUTE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = 32


def timed(dev, fn, warm=4, pipe=6, reps=7):
    """Median seconds per call, dispatch amortised over `pipe`, synced on both sides."""
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(pipe)]
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
        del keep
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()

    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    gx, gy = COMPUTE_GRID_MAIN
    ncore = gx * gy
    try:
        l1_per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception:
        l1_per_core = -1
    res = {"device": {"compute_grid": f"{dg.x}x{dg.y}", "core_grid_main": f"{gx}x{gy}",
                      "cores": ncore, "l1_unreserved_per_core": l1_per_core}}
    print(f"grid={dg.x}x{dg.y} main={gx}x{gy} cores={ncore} l1/core={l1_per_core}", flush=True)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # ---- A. compute roof --------------------------------------------------------------------
    if "A" not in skip:
        print("=== A compute roof (HiFi4, fp32_dest_acc+packer_l1_acc) ===", flush=True)
        A = {}
        for n in (2048, 4096, 6144):
            a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            gf = 2 * n ** 3 / 1e9
            try:
                s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                   memory_config=DRAM), warm=3, pipe=3, reps=5)
                A[str(n)] = {"ms": round(s * 1e3, 4), "tflops": round(gf / s / 1e3, 2)}
                print(f"  N={n:<5} {s*1e3:8.3f} ms  {gf/s/1e3:7.2f} TFLOP/s", flush=True)
            except Exception as e:
                A[str(n)] = {"err": str(e)[:90]}
                print(f"  N={n} ERR {str(e)[:90]}", flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        res["A_compute_roof"] = {"runs": A,
                                 "peak_tflops": max((v.get("tflops", 0) for v in A.values()),
                                                    default=0.0)}

    # ---- B. K-corrected compute rate at K=256 -----------------------------------------------
    # The pair-track projection is 102400x256 @ 256x256. K=256 is 8 tiles, so a core can never
    # amortise the pipeline over a long inner product. Measure what K=256 can actually reach.
    if "B" not in skip:
        print("=== B compute rate at K=256 (square-roof correction) ===", flush=True)
        B = {}
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        for m in (102400, 204800):
            x = ttnn.from_torch(torch.randn(1, 1, m, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            gf = 2 * m * 256 * 256 / 1e9
            for lbl, cfg in (("core_grid", None), ("bw8_obh5", "cfg")):
                try:
                    if cfg is None:
                        s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                           memory_config=DRAM,
                                                           core_grid=CORE_GRID_MAIN),
                                  warm=3, pipe=4, reps=5)
                    else:
                        mt, kt, nt = m // TILE, 8, 8
                        pcm = -(-(-(-mt // ncore)) // 5) * 5
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy), in0_block_w=8,
                            out_subblock_h=1, out_subblock_w=4, out_block_h=5, out_block_w=nt,
                            per_core_M=pcm, per_core_N=nt, fuse_batch=True,
                            fused_activation=None, mcast_in0=False)
                        s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                           memory_config=DRAM, program_config=pc),
                                  warm=3, pipe=4, reps=5)
                    B[f"{m}_{lbl}"] = {"ms": round(s * 1e3, 4), "tflops": round(gf / s / 1e3, 2)}
                    print(f"  M={m:<7} {lbl:10s} {s*1e3:8.3f} ms {gf/s/1e3:7.2f} TFLOP/s",
                          flush=True)
                except Exception as e:
                    B[f"{m}_{lbl}"] = {"err": str(e)[:90]}
                    print(f"  M={m} {lbl} ERR {str(e)[:90]}", flush=True)
            ttnn.deallocate(x)
        ttnn.deallocate(w)
        res["B_k256"] = B

    # ---- C. DRAM read roof, swept ------------------------------------------------------------
    if "C" not in skip:
        print("=== C DRAM read roof (DRAM -> L1 clone; DRAM sees reads only) ===", flush=True)
        C = []
        for mb in (4, 8, 16, 32, 48, 64, 96, 128):
            rows = int(mb * 1e6 / 2) // 4096
            nb = rows * 4096 * 2
            rec = {"MB": round(nb / 1e6, 2)}
            try:
                x = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
                s = timed(dev, lambda: ttnn.clone(x, memory_config=L1), warm=2, pipe=3, reps=5)
                rec["gbps"] = round(nb / s / 1e9, 1)
                rec["ms"] = round(s * 1e3, 4)
                ttnn.deallocate(x)
            except Exception as e:
                rec["err"] = str(e)[:80]
            C.append(rec)
            print("  " + json.dumps(rec), flush=True)
        res["C_read_roof"] = C

    # ---- D. Q4: the write roof by writer path -------------------------------------------------
    # Three paths, same 52.4 MB of DRAM output, arithmetic negligible in all three.
    if "D" not in skip:
        print("=== D write roof by writer path (Q4) ===", flush=True)
        D = {}
        # D1: unary clone L1 -> DRAM, swept
        sweep = []
        for mb in (4, 8, 16, 24, 32):
            rows = int(mb * 1e6 / 2) // 4096
            nb = rows * 4096 * 2
            rec = {"MB": round(nb / 1e6, 2)}
            try:
                x = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
                s = timed(dev, lambda: ttnn.clone(x, memory_config=DRAM), warm=2, pipe=3, reps=5)
                rec["write_gbps"] = round(nb / s / 1e9, 1)
                rec["ms"] = round(s * 1e3, 4)
                ttnn.deallocate(x)
            except Exception as e:
                rec["err"] = str(e)[:80]
            sweep.append(rec)
            print("  clone " + json.dumps(rec), flush=True)
        D["clone_L1_to_DRAM"] = sweep

        # D2/D3: write-dominated matmuls. M=102400, K=32 (1 tile), N=256.
        m, k, n = 102400, 32, 256
        xw = ttnn.from_torch(torch.randn(1, 1, m, k), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        ww = ttnn.from_torch(torch.randn(k, n), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        out_b = m * n * 2
        in_b = m * k * 2 + k * n * 2
        gf = 2 * m * k * n / 1e9
        mt, nt = m // TILE, n // TILE
        pcm = -(-(-(-mt // ncore)) // 5) * 5
        legs = {
            "minimal_matmul": lambda: ttnn.experimental.minimal_matmul(
                xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc),
            "linear_core_grid": lambda: ttnn.linear(
                xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                core_grid=CORE_GRID_MAIN),
            "linear_1dmcast_pc": lambda: ttnn.linear(
                xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=1, out_subblock_h=1,
                    out_subblock_w=4, out_block_h=5, out_block_w=nt, per_core_M=pcm,
                    per_core_N=nt, fuse_batch=True, fused_activation=None, mcast_in0=False)),
        }
        for lbl, fn in legs.items():
            try:
                s = timed(dev, fn, warm=3, pipe=4, reps=5)
                D[lbl] = {"ms": round(s * 1e3, 4),
                          "write_gbps": round(out_b / s / 1e9, 1),
                          "rw_gbps": round((out_b + in_b) / s / 1e9, 1),
                          "gflops": round(gf / s, 1),
                          "out_MB": round(out_b / 1e6, 2), "in_MB": round(in_b / 1e6, 2)}
                print(f"  {lbl:20s} {s*1e3:8.3f} ms  write {out_b/s/1e9:6.1f} GB/s  "
                      f"r+w {(out_b+in_b)/s/1e9:6.1f} GB/s", flush=True)
            except Exception as e:
                D[lbl] = {"err": str(e)[:90]}
                print(f"  {lbl} ERR {str(e)[:90]}", flush=True)
        ttnn.deallocate(xw)
        ttnn.deallocate(ww)
        res["D_write_roof"] = D

    # ---- E. the pair-track projection ---------------------------------------------------------
    if "E" not in skip:
        print("=== E pair-track projection 102400x256 @ 256x256 ===", flush=True)
        E = {}
        m, k, n = 102400, 256, 256
        x = ttnn.from_torch(torch.randn(1, 320, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        gf = 2 * m * k * n / 1e9
        rd = (m * k + k * n) * 2
        wr = m * n * 2

        def pc_of(bw, obh, grid):
            ggx, ggy = grid
            nc = ggx * ggy
            mt, nt = m // TILE, n // TILE
            pcm = -(-(-(-mt // nc)) // obh) * obh
            if -(-mt // pcm) > nc:
                return None, None
            sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
            sw = max(ww_ for ww_ in range(min(4 // sh, nt), 0, -1) if nt % ww_ == 0)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(ggx, ggy), in0_block_w=bw, out_subblock_h=sh,
                out_subblock_w=sw, out_block_h=obh, out_block_w=nt, per_core_M=pcm,
                per_core_N=nt, fuse_batch=True, fused_activation=None, mcast_in0=False), \
                -(-mt // pcm)

        cfgs = [("core_grid_legacy", None, None, (gx, gy)),
                ("prod_bw1_obh5", 1, 5, (gx, gy)),
                ("bw8_obh5", 8, 5, (gx, gy)),
                ("bw1_obhPCM", 1, None, (gx, gy))]
        for lbl, bw, obh, grid in cfgs:
            try:
                if bw is None:
                    fn = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM,
                                             core_grid=CORE_GRID_MAIN)
                    cores = None
                elif obh is None:
                    mt = m // TILE
                    pcm = -(-mt // (gx * gy))
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                        out_subblock_h=1, out_subblock_w=4, out_block_h=pcm, out_block_w=8,
                        per_core_M=pcm, per_core_N=8, fuse_batch=True, fused_activation=None,
                        mcast_in0=False)
                    fn = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM,
                                             program_config=pc)
                    cores = -(-mt // pcm)
                else:
                    pc, cores = pc_of(bw, obh, grid)
                    fn = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM,
                                             program_config=pc)
                s = timed(dev, fn, warm=3, pipe=4, reps=5)
                E[lbl] = {"ms": round(s * 1e3, 4), "tflops": round(gf / s / 1e3, 2),
                          "write_gbps": round(wr / s / 1e9, 1),
                          "read_gbps": round(rd / s / 1e9, 1),
                          "cores_from_cfg": cores}
                print(f"  {lbl:18s} {s*1e3:8.4f} ms {gf/s/1e3:6.2f} TFLOP/s  "
                      f"w {wr/s/1e9:6.1f} GB/s cores~{cores}", flush=True)
            except Exception as e:
                E[lbl] = {"err": str(e)[:90]}
                print(f"  {lbl} ERR {str(e)[:90]}", flush=True)

        # measured core utilisation: shrink the grid the program config may use.
        occ = []
        for ggx in range(gx, 2, -2):
            pc, cores = pc_of(1, 5, (ggx, gy))
            if pc is None:
                continue
            try:
                s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                   memory_config=DRAM, program_config=pc),
                          warm=2, pipe=3, reps=5)
                occ.append({"grid": f"{ggx}x{gy}", "grid_cores": ggx * gy, "cfg_cores": cores,
                            "ms": round(s * 1e3, 4)})
                print(f"  occ grid={ggx}x{gy} cfg_cores={cores} {s*1e3:8.4f} ms", flush=True)
            except Exception as e:
                occ.append({"grid": f"{ggx}x{gy}", "err": str(e)[:70]})
        E["occupancy_ab"] = occ
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        res["E_pair_proj"] = E

    # ---- F. the L1-resident trimul rows -------------------------------------------------------
    if "F" not in skip:
        print("=== F L1-resident trimul rows ===", flush=True)
        F = {}
        N, C = 320, 32
        chan_last = lambda: ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT,
                                            device=dev, dtype=ttnn.bfloat16, memory_config=L1)
        chan_batch = lambda: ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT,
                                             device=dev, dtype=ttnn.bfloat16, memory_config=L1)
        b_chunk = N * N * C * 2

        a = chan_last()
        b = chan_last()
        try:
            s = timed(dev, lambda: ttnn.permute(a, (0, 3, 1, 2), memory_config=L1),
                      warm=3, pipe=5, reps=7)
            F["permute_in_0312"] = {"ms": round(s * 1e3, 4), "us": round(s * 1e6, 1),
                                    "bytes_MB": round(2 * b_chunk / 1e6, 2),
                                    "gbps": round(2 * b_chunk / s / 1e9, 1)}
            print(f"  permute (0,3,1,2) {s*1e6:8.1f} us  {2*b_chunk/s/1e9:6.1f} GB/s", flush=True)
        except Exception as e:
            F["permute_in_0312"] = {"err": str(e)[:90]}
        try:
            s = timed(dev, lambda: ttnn.multiply(a, b, memory_config=L1), warm=3, pipe=5, reps=7)
            F["multiply_eltwise"] = {"us": round(s * 1e6, 1),
                                     "bytes_MB": round(3 * b_chunk / 1e6, 2),
                                     "gbps": round(3 * b_chunk / s / 1e9, 1)}
            print(f"  multiply          {s*1e6:8.1f} us  {3*b_chunk/s/1e9:6.1f} GB/s", flush=True)
        except Exception as e:
            F["multiply_eltwise"] = {"err": str(e)[:90]}
        try:
            s = timed(dev, lambda: ttnn.multiply(
                a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID], memory_config=L1),
                warm=3, pipe=5, reps=7)
            F["multiply_sigmoid"] = {"us": round(s * 1e6, 1),
                                     "gbps": round(3 * b_chunk / s / 1e9, 1)}
            print(f"  multiply+sigmoid  {s*1e6:8.1f} us  {3*b_chunk/s/1e9:6.1f} GB/s", flush=True)
        except Exception as e:
            F["multiply_sigmoid"] = {"err": str(e)[:90]}
        try:
            s = timed(dev, lambda: ttnn.clone(a, memory_config=L1), warm=3, pipe=5, reps=7)
            F["clone_L1_L1"] = {"us": round(s * 1e6, 1),
                                "gbps": round(2 * b_chunk / s / 1e9, 1)}
            print(f"  clone L1->L1      {s*1e6:8.1f} us  {2*b_chunk/s/1e9:6.1f} GB/s", flush=True)
        except Exception as e:
            F["clone_L1_L1"] = {"err": str(e)[:90]}
        ttnn.deallocate(a)
        ttnn.deallocate(b)

        ab = chan_batch()
        bb = chan_batch()
        from tt_bio.tenstorrent import _triangle_mul_program_config
        pcm = _triangle_mul_program_config((N + 31) // 32)
        try:
            s = timed(dev, lambda: ttnn.matmul(ab, bb, compute_kernel_config=ckc,
                                               memory_config=L1, program_config=pcm,
                                               dtype=ttnn.bfloat16), warm=3, pipe=5, reps=7)
            gfl = C * 2 * N ** 3 / 1e9
            F["tri_matmul"] = {"us": round(s * 1e6, 1), "tflops": round(gfl / s / 1e3, 2),
                               "bytes_MB": round(3 * b_chunk / 1e6, 2),
                               "gbps": round(3 * b_chunk / s / 1e9, 1),
                               "ai_flop_per_byte": round(gfl * 1e9 / (3 * b_chunk), 1)}
            print(f"  tri matmul        {s*1e6:8.1f} us  {gfl/s/1e3:6.2f} TFLOP/s  "
                  f"{3*b_chunk/s/1e9:6.1f} GB/s", flush=True)
        except Exception as e:
            F["tri_matmul"] = {"err": str(e)[:90]}
        try:
            s = timed(dev, lambda: ttnn.permute(ab, (0, 2, 3, 1), memory_config=L1),
                      warm=3, pipe=5, reps=7)
            F["permute_out_0231"] = {"us": round(s * 1e6, 1),
                                     "gbps": round(2 * b_chunk / s / 1e9, 1)}
            print(f"  permute (0,2,3,1) {s*1e6:8.1f} us  {2*b_chunk/s/1e9:6.1f} GB/s", flush=True)
        except Exception as e:
            F["permute_out_0231"] = {"err": str(e)[:90]}
        ttnn.deallocate(ab)
        ttnn.deallocate(bb)

        # staircase: time vs output tile count for an L1 eltwise. If the op distributes tiles
        # over K cores, time is flat within a step of K tiles and jumps at each multiple.
        stair = []
        for t in list(range(1, 20)) + [32, 64, 110, 111, 128, 220, 221, 320, 640, 1280, 3200]:
            rows = t * 32
            try:
                p = ttnn.from_torch(torch.randn(1, 1, rows, 32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=ttnn.bfloat16, memory_config=L1)
                q = ttnn.from_torch(torch.randn(1, 1, rows, 32), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=ttnn.bfloat16, memory_config=L1)
                s = timed(dev, lambda: ttnn.multiply(p, q, memory_config=L1),
                          warm=3, pipe=8, reps=7)
                stair.append({"tiles": t, "us": round(s * 1e6, 2)})
                ttnn.deallocate(p)
                ttnn.deallocate(q)
            except Exception as e:
                stair.append({"tiles": t, "err": str(e)[:60]})
        F["eltwise_tile_staircase"] = stair
        print("  staircase " + json.dumps(stair), flush=True)
        res["F_l1_rows"] = F

    # ---- G. channel-move permute: GB/s vs chunk width, constant bytes -------------------------
    if "G" not in skip:
        print("=== G permute GB/s vs chunk width C (bytes held constant) ===", flush=True)
        G = []
        for C in (8, 16, 32, 64, 128, 256):
            try:
                x = ttnn.from_torch(torch.randn(1, 320, 320, C), layout=ttnn.TILE_LAYOUT,
                                    device=dev, dtype=ttnn.bfloat16, memory_config=L1)
                nb = 2 * 320 * 320 * C * 2
                s = timed(dev, lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1),
                          warm=3, pipe=4, reps=5)
                G.append({"C": C, "us": round(s * 1e6, 1), "MB": round(nb / 1e6, 2),
                          "gbps": round(nb / s / 1e9, 1),
                          "us_per_MB": round(s * 1e6 / (nb / 1e6), 2)})
                print("  " + json.dumps(G[-1]), flush=True)
                ttnn.deallocate(x)
            except Exception as e:
                G.append({"C": C, "err": str(e)[:80]})
                print(f"  C={C} ERR {str(e)[:80]}", flush=True)
        res["G_permute_width"] = G

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
