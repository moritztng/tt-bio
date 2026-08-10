#!/usr/bin/env python3
"""Q9 rescore, probe 2 — the nt curve with a fillable grid, the overlap decomposition, and stability.

Probe 1 (`r_card.py`) left three things open. This closes them, on the same card, same build.

  I  the best rate the card reaches at output width nt with the output in L1, with the grid
     actually filled. Probe 1's nt=32/64 L1 legs either under-filled the grid (mt < 3 per core) or
     failed to allocate, so its L1 curve is not the card's ceiling. Output bytes AND FLOPs are held
     constant across nt by fixing mt x nt, so only the width changes.
  J  where the DRAM-output penalty comes from. For the pair-track projection at the TRUE fold
     shape: time with an L1 output, time with a DRAM output, and the time of a separate unary clone
     of the same output bytes L1 -> DRAM. If the DRAM leg is near max() of the two the write hides;
     if it is near the sum it is serialised. Run at in0_block_w 8 (today's main) and 1 (T2's).
  K  roof stability: probe 1's load-bearing rows again, >= 12 min later, on the same card.

    PYTHONPATH=<wt> python3 perf/rescore/r_probe2.py --out perf/rescore/r_probe2_qb2c0.json
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN, COMPUTE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = 32


def timed(dev, fn, warm=3, pipe=4, reps=5):
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


def pc_1d(gx, gy, bw, obh, nt, per_core_M):
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, nt), 0, -1) if nt % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
        out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=nt,
        per_core_M=per_core_M, per_core_N=nt, fuse_batch=True,
        fused_activation=None, mcast_in0=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    run = lambda s: only is None or s in only                              # noqa: E731

    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    ncore = gx * gy
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"t_start": time.time(), "cores": ncore}
    print(f"cores={ncore} grid={gx}x{gy}", flush=True)

    # ---- I. the nt curve, grid filled, FLOPs and output bytes held constant -------------------
    if run("I"):
        print("=== I best rate vs output width nt, K=256, mt x nt = 19200 (39.3 MB out) ===",
              flush=True)
        I = []
        for nt in (8, 10, 16, 32, 64):
            mt = 19200 // nt
            m, n = mt * TILE, nt * TILE
            gf = 2 * m * 256 * n / 1e9
            out_b, in0_b = m * n * 2, m * 256 * 2
            x = ttnn.from_torch(torch.randn(1, 1, m, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            w = ttnn.from_torch(torch.randn(256, n), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            base = -(-mt // ncore)
            for bl, mc in (("l1", L1), ("dram", DRAM)):
                legs = {}
                cands = [("core_grid", None, ncore)]
                for obh in (5, 2, 1):
                    pcm = -(-base // obh) * obh
                    if -(-mt // pcm) > ncore:
                        continue
                    for bw in (8, 1):
                        cands.append((f"bw{bw}_obh{obh}_pcm{pcm}", pc_1d(gx, gy, bw, obh, nt, pcm),
                                      min(ncore, -(-mt // pcm))))
                for lbl, cfg, nc in cands:
                    try:
                        kw = {"core_grid": CORE_GRID_MAIN} if cfg is None else {"program_config": cfg}
                        s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                           memory_config=mc, **kw))
                        legs[lbl] = {"us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2),
                                     "out_gbps": round(out_b / s / 1e9, 1), "cores": nc}
                    except Exception as e:                                  # noqa: BLE001
                        legs[lbl] = {"err": str(e)[:80]}
                ok = {k: v for k, v in legs.items() if "tflops" in v}
                best = max(ok, key=lambda k: ok[k]["tflops"]) if ok else None
                rec = {"nt": nt, "N": n, "mt": mt, "M": m, "buf": bl,
                       "out_MB": round(out_b / 1e6, 2), "in0_MB": round(in0_b / 1e6, 2),
                       "AI": round(gf * 1e9 / (out_b + in0_b + 256 * n * 2), 1),
                       "best": best, "best_tflops": ok[best]["tflops"] if best else None,
                       "best_cores": ok[best]["cores"] if best else None, "legs": legs}
                I.append(rec)
                print(f"  nt={nt:<3} {bl:4s} out={out_b/1e6:.1f}MB best={best} "
                      f"{rec['best_tflops']} TFLOP/s cores={rec['best_cores']}", flush=True)
            ttnn.deallocate(x)
            ttnn.deallocate(w)
        # T3's own shape, on my card: M=10240, K=256, N=2048, L1 out.
        print("  -- T3's 95.42 shape on this card: 10240x256 @ 256x2048 --", flush=True)
        t3 = {}
        x = ttnn.from_torch(torch.randn(1, 1, 10240, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(256, 2048), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        gf = 2 * 10240 * 256 * 2048 / 1e9
        out_b = 10240 * 2048 * 2
        for bl, mc in (("l1", L1), ("dram", DRAM)):
            for lbl, kw in (("core_grid", {"core_grid": CORE_GRID_MAIN}),
                            ("bw8_obh1_pcm3", {"program_config": pc_1d(gx, gy, 8, 1, 64, 3)})):
                try:
                    s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                       memory_config=mc, **kw))
                    t3[f"{bl}_{lbl}"] = {"us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2),
                                         "out_MB": round(out_b / 1e6, 2)}
                    print(f"    {bl:4s} {lbl:14s} {s*1e6:8.2f} us {gf/s/1e3:7.2f} TFLOP/s",
                          flush=True)
                except Exception as e:                                      # noqa: BLE001
                    t3[f"{bl}_{lbl}"] = {"err": str(e)[:80]}
                    print(f"    {bl} {lbl} ERR {str(e)[:80]}", flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        res["I_nt_curve"] = {"curve": I, "t3_shape": t3}

    # ---- J. where the DRAM-output penalty comes from -----------------------------------------
    if run("J"):
        print("=== J projection at the TRUE fold shape: L1 out vs DRAM out vs a separate clone ===",
              flush=True)
        J = {}
        z = ttnn.from_torch(torch.randn(1, 298, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        mt = 2980
        m = mt * TILE
        gf = 2 * m * 256 * 256 / 1e9
        out_b = m * 256 * 2
        pcm = -(-(-(-mt // ncore)) // 5) * 5
        for bw in (8, 4, 2, 1):
            cfg = pc_1d(gx, gy, bw, 5, 8, pcm)
            row = {"in0_block_w": bw, "per_core_M": pcm, "cores": min(ncore, -(-mt // pcm))}
            for bl, mc in (("dram", DRAM), ("l1", L1)):
                try:
                    s = timed(dev, lambda: ttnn.linear(z, w, compute_kernel_config=ckc,
                                                       memory_config=mc, program_config=cfg))
                    row[bl] = {"us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2),
                               "out_gbps": round(out_b / s / 1e9, 1)}
                except Exception as e:                                       # noqa: BLE001
                    row[bl] = {"err": str(e)[:80]}
            # in1 resident in L1: if the DRAM-output rate is limited by the writer sharing BRISC
            # with the in1 multicast sender, taking in1 out of DRAM should move it.
            try:
                wl1 = ttnn.clone(w, memory_config=L1)
                s = timed(dev, lambda: ttnn.linear(z, wl1, compute_kernel_config=ckc,
                                                   memory_config=DRAM, program_config=cfg))
                row["dram_in1_l1"] = {"us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2),
                                      "out_gbps": round(out_b / s / 1e9, 1)}
                ttnn.deallocate(wl1)
            except Exception as e:                                          # noqa: BLE001
                row["dram_in1_l1"] = {"err": str(e)[:80]}
            J[f"bw{bw}"] = row
            print(f"  bw={bw}: dram {row['dram'].get('us')} us / l1 {row['l1'].get('us')} us / "
                  f"dram+in1L1 {row['dram_in1_l1'].get('us')} us", flush=True)
        # the write leg on its own: a unary clone of exactly the projection's output bytes.
        y = ttnn.from_torch(torch.randn(1, 298, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        s = timed(dev, lambda: ttnn.clone(y, memory_config=DRAM), warm=2, pipe=3)
        J["clone_out_bytes_L1_to_DRAM"] = {"us": round(s * 1e6, 2), "MB": round(out_b / 1e6, 2),
                                           "write_gbps": round(out_b / s / 1e9, 1)}
        print(f"  clone {out_b/1e6:.2f} MB L1->DRAM: {s*1e6:.2f} us  {out_b/s/1e9:.1f} GB/s",
              flush=True)
        ttnn.deallocate(y)
        ttnn.deallocate(z)
        ttnn.deallocate(w)
        res["J_overlap"] = J

    # ---- K. stability ------------------------------------------------------------------------
    if run("K"):
        el = time.time() - res["t_start"]
        print(f"=== K stability, {el/60:.1f} min into this process ===", flush=True)
        K = {"elapsed_min": round(el / 60, 1)}
        # T2's exact 34.37 configuration.
        x = ttnn.from_torch(torch.randn(1, 1, 204800, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        mt = 6400
        pcm = -(-(-(-mt // ncore)) // 5) * 5
        gf = 2 * 204800 * 256 * 256 / 1e9
        s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM,
                                           program_config=pc_1d(gx, gy, 8, 5, 8, pcm)))
        K["t2_exact_204800_bw8_obh5_dram"] = {"ms": round(s * 1e3, 4),
                                              "tflops": round(gf / s / 1e3, 2),
                                              "out_gbps": round(204800 * 256 * 2 / s / 1e9, 1)}
        print(f"  T2's config: {s*1e3:.4f} ms  {gf/s/1e3:.2f} TFLOP/s", flush=True)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        # the contraction, and the unary/matmul write roofs.
        a = ttnn.from_torch(torch.randn(1, 32, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        b = ttnn.from_torch(torch.randn(1, 32, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        pc = T._triangle_mul_program_config(10)
        gf = 2 * 32 * 320 ** 3 / 1e9
        s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                                           program_config=pc, dtype=ttnn.bfloat16), pipe=6, reps=7)
        K["contraction_b32"] = {"us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2)}
        print(f"  contraction: {s*1e6:.2f} us  {gf/s/1e3:.2f} TFLOP/s", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        rows = int(32e6 / 2) // 4096
        nb = rows * 4096 * 2
        xc = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=L1)
        s = timed(dev, lambda: ttnn.clone(xc, memory_config=DRAM), warm=2, pipe=3)
        K["unary_write_roof_32MB"] = {"gbps": round(nb / s / 1e9, 1)}
        ttnn.deallocate(xc)
        xw = ttnn.from_torch(torch.randn(1, 1, 102400, 32), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        ww = ttnn.from_torch(torch.randn(32, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        s = timed(dev, lambda: ttnn.experimental.minimal_matmul(
            xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc),
            warm=2, pipe=3)
        K["matmul_write_roof"] = {"gbps": round(102400 * 256 * 2 / s / 1e9, 1)}
        print(f"  write roofs: unary {K['unary_write_roof_32MB']['gbps']} / "
              f"matmul {K['matmul_write_roof']['gbps']} GB/s", flush=True)
        ttnn.deallocate(xw)
        ttnn.deallocate(ww)
        res["K_stability"] = K

    # ---- L. bound the DROPPED bucket ----------------------------------------------------------
    if run("L"):
        print("=== L the five dropped classes, timed standalone (a bound, not an in-block time) ===",
              flush=True)
        L_ = {}
        z = ttnn.from_torch(torch.randn(1, 298, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w128 = ttnn.from_torch(torch.randn(256, 128), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16, memory_config=DRAM)
        s = timed(dev, lambda: ttnn.experimental.minimal_matmul(
            z, w128, memory_config=L1, dtype=ttnn.bfloat16, compute_kernel_config=ckc),
            warm=2, pipe=3, reps=5)
        L_["minimal_matmul_1329"] = {"us": round(s * 1e6, 2), "calls_per_block": 16}
        ttnn.deallocate(w128)
        g = ttnn.experimental.minimal_matmul(z, ttnn.from_torch(
            torch.randn(256, 128), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
            memory_config=DRAM), memory_config=L1, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc)
        s = timed(dev, lambda: ttnn.chunk(g, chunks=4, dim=-1), warm=2, pipe=3, reps=5)
        L_["chunk_1336"] = {"us": round(s * 1e6, 2), "calls_per_block": 16}
        ttnn.deallocate(g)
        ttnn.deallocate(z)
        t = ttnn.from_torch(torch.randn(1, 30, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        gm = ttnn.from_torch(torch.randn(1, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        bm = ttnn.from_torch(torch.randn(1, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        try:
            s = timed(dev, lambda: ttnn.layer_norm(t, weight=gm, bias=bm, epsilon=1e-5,
                                                   memory_config=L1,
                                                   compute_kernel_config=ckc), warm=2, pipe=3)
            L_["layer_norm_2063"] = {"us": round(s * 1e6, 2), "calls_per_block": 10}
        except Exception as e:                                              # noqa: BLE001
            L_["layer_norm_2063"] = {"err": str(e)[:80]}
        tl1 = ttnn.clone(t, memory_config=L1)
        w1024 = ttnn.from_torch(torch.randn(256, 1024), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
        s = timed(dev, lambda: ttnn.linear(tl1, w1024, memory_config=L1, dtype=ttnn.bfloat16,
                                           compute_kernel_config=ckc,
                                           core_grid=CORE_GRID_MAIN), warm=2, pipe=3)
        L_["linear_2071_2080"] = {"us": round(s * 1e6, 2), "calls_per_block": 20}
        ttnn.deallocate(tl1)
        ttnn.deallocate(t)
        ttnn.deallocate(w1024)
        ttnn.deallocate(gm)
        ttnn.deallocate(bm)
        tot = sum(v["us"] * v["calls_per_block"] for v in L_.values() if "us" in v)
        L_["dropped_block_ms_estimate"] = round(tot / 1e3, 3)
        for k, v in L_.items():
            print(f"  {k}: {v}", flush=True)
        res["L_dropped_bound"] = L_

    res["t_end"] = time.time()
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
