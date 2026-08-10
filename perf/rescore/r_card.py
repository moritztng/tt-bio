#!/usr/bin/env python3
"""Q9 rescore — every roof measured on qb2 card 0 this pass, plus T2's five compute rows re-timed.

One process, one device, so every roof and every op it scores share a card, a ttnn build and an
allocator state. Every timed region synchronises on both sides (an unsynced drain has inverted
rankings in this codebase before).

Sections
  A  square compute roof, DRAM out and L1 out -- only for the caveat, nothing is scored against it
  B  THE Q9 SWEEP. K=256, output buffer type {DRAM, L1} x output width nt {1, 8, 32, 64}, FLOPs
     held constant across nt (same arithmetic, different output width). Plus a straight repro of
     T2's exact 34.37 TFLOP/s configuration, and the same shape with the output in L1.
  C  the triangle contraction's own class: K=320, nt=10, L1 in and L1 out. Unbatched (the roof)
     and batched b=32 with the production program config (the op).
  D  write roofs by writer structure: unary clone L1->DRAM swept, and a write-dominated matmul.
  E  DRAM read roof swept, and the DRAM->DRAM combined ceiling.
  F  the four rows standalone at the TRUE fold shape [1, 298, 320, 256]: the contraction, and the
     pair-track projection at both _PAIR_PROJ_BW settings (today's main is 16 -> in0_block_w=8;
     T2 measured 1).
  G  core utilisation by program-config grid A/B for both op classes.
  H  section B's key rows again, >= 12 min after A, for roof stability.

    PYTHONPATH=<wt> python3 perf/rescore/r_card.py --out perf/rescore/r_card_qb2c0.json
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


def pc_1d(gx, gy, bw, obh, nt, per_core_M):
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    sw = max(w for w in range(min(4 // sh, nt), 0, -1) if nt % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
        out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=nt,
        per_core_M=per_core_M, per_core_N=nt, fuse_batch=True,
        fused_activation=None, mcast_in0=False)


def k256_leg(dev, ckc, gx, gy, ncore, out_elems, nt, mc, ckc_label, kt=8):
    """Best rate the card reaches for a K=kt*32 matmul with output width nt into `mc`.

    FLOPs are held constant across nt by fixing out_elems = M*N, so the only thing that changes
    across the sweep is how wide each core's output block is (and, with it, how much in0 has to be
    streamed for the same arithmetic).
    """
    k = kt * TILE
    n = nt * TILE
    m = out_elems // n
    m = (m // TILE) * TILE
    if m < TILE:
        return None
    mt = m // TILE
    gf = 2 * m * k * n / 1e9
    out_b, in0_b, in1_b = m * n * 2, m * k * 2, k * n * 2
    x = ttnn.from_torch(torch.randn(1, 1, m, k), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    w = ttnn.from_torch(torch.randn(k, n), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    legs = {}
    cands = [("core_grid", None)]
    cores = {}
    if mt >= ncore:
        base = -(-mt // ncore)
        for obh, bw in ((5, kt), (2, kt), (1, kt), (5, 1)):
            pcm = -(-base // obh) * obh
            if -(-mt // pcm) > ncore:
                continue
            lbl = f"bw{bw}_obh{obh}_pcm{pcm}"
            cands.append((lbl, pc_1d(gx, gy, bw, obh, nt, pcm)))
            cores[lbl] = min(ncore, -(-mt // pcm))
    for lbl, cfg in cands:
        try:
            if cfg is None:
                s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                   memory_config=mc, core_grid=CORE_GRID_MAIN))
            else:
                s = timed(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                   memory_config=mc, program_config=cfg))
            legs[lbl] = {"ms": round(s * 1e3, 4), "tflops": round(gf / s / 1e3, 2),
                         "write_gbps": round(out_b / s / 1e9, 1),
                         "read_gbps": round((in0_b + in1_b) / s / 1e9, 1),
                         "cores": cores.get(lbl, ncore)}
        except Exception as e:                                             # noqa: BLE001
            legs[lbl] = {"err": str(e)[:100]}
    ttnn.deallocate(x)
    ttnn.deallocate(w)
    ok = {k2: v for k2, v in legs.items() if "tflops" in v}
    best = max(ok, key=lambda k2: ok[k2]["tflops"]) if ok else None
    rec = {"M": m, "K": k, "N": n, "nt": nt, "mt": mt, "buf": ckc_label,
           "gflop": round(gf, 3), "out_MB": round(out_b / 1e6, 2),
           "in0_MB": round(in0_b / 1e6, 2),
           "AI_flop_per_byte": round(gf * 1e9 / (out_b + in0_b + in1_b), 1),
           "legs": legs, "best": best,
           "best_tflops": ok[best]["tflops"] if best else None,
           "best_write_gbps": ok[best]["write_gbps"] if best else None,
           "best_cores": ok[best].get("cores") if best else None}
    print(f"  K={k} nt={nt:<3} {ckc_label:4s} out={out_b/1e6:6.2f}MB M={m:<8} "
          f"best={best} {rec['best_tflops']} TFLOP/s  write={rec['best_write_gbps']} GB/s "
          f"cores={rec['best_cores']}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def run(sec):
        return only is None or sec in only

    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    gx, gy = COMPUTE_GRID_MAIN
    ncore = gx * gy
    try:
        l1_bank = int(ttnn.get_memory_view(dev, ttnn.BufferType.L1).total_bytes_per_bank)
    except Exception:                                                      # noqa: BLE001
        l1_bank = -1
    res = {"device": {"compute_grid": f"{dg.x}x{dg.y}", "core_grid_main": f"{gx}x{gy}",
                      "cores": ncore, "l1_bytes_per_bank": l1_bank,
                      "pair_proj_bw_main": T._PAIR_PROJ_BW},
           "t_start": time.time()}
    print(f"grid={dg.x}x{dg.y} main={gx}x{gy} cores={ncore} l1/bank={l1_bank} "
          f"_PAIR_PROJ_BW={T._PAIR_PROJ_BW}", flush=True)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # ---- A. square compute roof, both output buffer types ------------------------------------
    if run("A"):
        print("=== A square compute roof (HiFi4, fp32_dest_acc+packer_l1_acc) ===", flush=True)
        A = {}
        for n in (2048, 4096, 6144):
            a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            gf = 2 * n ** 3 / 1e9
            for lbl, mc in (("dram", DRAM), ("l1", L1)):
                try:
                    s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                       memory_config=mc), warm=2, pipe=3, reps=5)
                    A[f"{n}_{lbl}"] = {"ms": round(s * 1e3, 4), "tflops": round(gf / s / 1e3, 2)}
                    print(f"  N={n:<5} {lbl:4s} {s*1e3:8.3f} ms  {gf/s/1e3:7.2f} TFLOP/s",
                          flush=True)
                except Exception as e:                                     # noqa: BLE001
                    A[f"{n}_{lbl}"] = {"err": str(e)[:90]}
                    print(f"  N={n} {lbl} ERR {str(e)[:90]}", flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        res["A_square_roof"] = A

    # ---- B. the Q9 sweep --------------------------------------------------------------------
    if run("B"):
        print("=== B Q9: K=256, output buffer type x output width nt, FLOPs held constant ===",
              flush=True)
        B = {"sweep": [], "t2_repro": {}}
        for out_mb in (8, 24):
            out_elems = int(out_mb * 1e6 / 2)
            for nt in (1, 8, 32, 64):
                for lbl, mc in (("dram", DRAM), ("l1", L1)):
                    r = k256_leg(dev, ckc, gx, gy, ncore, out_elems, nt, mc, lbl)
                    if r:
                        r["target_out_MB"] = out_mb
                        B["sweep"].append(r)
        # T2's exact configuration: 204800x256 @ 256x256, in0_block_w=8, out_block_h=5, and the
        # same shape with the output in L1. This is the number the org's five rows ride on.
        print("  -- T2's exact 34.37 configuration, DRAM out then L1 out --", flush=True)
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        for m in (102400, 204800):
            x = ttnn.from_torch(torch.randn(1, 1, m, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            mt = m // TILE
            gf = 2 * m * 256 * 256 / 1e9
            out_b = m * 256 * 2
            pcm = -(-(-(-mt // ncore)) // 5) * 5
            for cl, cfg in (("core_grid", None),
                            ("bw8_obh5", pc_1d(gx, gy, 8, 5, 8, pcm)),
                            ("bw1_obh5", pc_1d(gx, gy, 1, 5, 8, pcm))):
                for bl, mc in (("dram", DRAM), ("l1", L1)):
                    key = f"{m}_{cl}_{bl}"
                    try:
                        if cfg is None:
                            s = timed(dev, lambda: ttnn.linear(
                                x, w, compute_kernel_config=ckc, memory_config=mc,
                                core_grid=CORE_GRID_MAIN))
                        else:
                            s = timed(dev, lambda: ttnn.linear(
                                x, w, compute_kernel_config=ckc, memory_config=mc,
                                program_config=cfg))
                        B["t2_repro"][key] = {
                            "ms": round(s * 1e3, 4), "tflops": round(gf / s / 1e3, 2),
                            "out_gbps": round(out_b / s / 1e9, 1),
                            "cores": min(ncore, -(-mt // pcm)) if cfg is not None else None}
                        print(f"    M={m:<7} {cl:10s} {bl:4s} {s*1e3:8.3f} ms "
                              f"{gf/s/1e3:7.2f} TFLOP/s  out={out_b/s/1e9:6.1f} GB/s", flush=True)
                    except Exception as e:                                 # noqa: BLE001
                        B["t2_repro"][key] = {"err": str(e)[:100]}
                        print(f"    M={m} {cl} {bl} ERR {str(e)[:100]}", flush=True)
            ttnn.deallocate(x)
        ttnn.deallocate(w)
        res["B_q9"] = B

    # ---- C. the contraction's own class: K=320, nt=10, L1 in / L1 out ------------------------
    if run("C"):
        print("=== C K=320 nt=10 L1-in/L1-out: the contraction's own roof class ===", flush=True)
        C = {"unbatched": [], "batched": {}, "swept": []}
        # (i) the roof, with the same config search the Q9 sweep uses: K=320, nt=10, both buffers.
        for out_mb in (8, 24):
            for lbl, mc in (("l1", L1), ("dram", DRAM)):
                r = k256_leg(dev, ckc, gx, gy, ncore, int(out_mb * 1e6 / 2), 10, mc, lbl, kt=10)
                if r:
                    r["target_out_MB"] = out_mb
                    C["swept"].append(r)
        # (ii) the same shape with the default entry point, as a cross-check on the search.
        for mt in (320, 1280):
            m = mt * TILE
            gf = 2 * m * 320 * 320 / 1e9
            out_b = m * 320 * 2
            a = ttnn.from_torch(torch.randn(1, 1, m, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=L1)
            b = ttnn.from_torch(torch.randn(1, 1, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=L1)
            for lbl, mc in (("l1", L1), ("dram", DRAM)):
                try:
                    s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                       memory_config=mc))
                    C["unbatched"].append({"mt": mt, "buf": lbl, "ms": round(s * 1e3, 4),
                                           "tflops": round(gf / s / 1e3, 2),
                                           "out_MB": round(out_b / 1e6, 2),
                                           "write_gbps": round(out_b / s / 1e9, 1)})
                    print(f"  mt={mt:<5} {lbl:4s} {s*1e3:8.3f} ms {gf/s/1e3:7.2f} TFLOP/s",
                          flush=True)
                except Exception as e:                                     # noqa: BLE001
                    C["unbatched"].append({"mt": mt, "buf": lbl, "err": str(e)[:90]})
                    print(f"  mt={mt} {lbl} ERR {str(e)[:90]}", flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        # (iii) the op: batch=32, 320x320 @ 320x320, production program config, L1 everywhere.
        pc = T._triangle_mul_program_config(10)
        res.setdefault("configs", {})["triangle_mul_pc_10"] = {
            "in0_block_w": pc.in0_block_w, "out_block_h": pc.out_block_h,
            "out_block_w": pc.out_block_w, "per_core_M": pc.per_core_M,
            "per_core_N": pc.per_core_N, "grid": str(pc.compute_with_storage_grid_size)}
        for bsz in (1, 8, 32):
            a = ttnn.from_torch(torch.randn(1, bsz, 320, 320), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16, memory_config=L1)
            b = ttnn.from_torch(torch.randn(1, bsz, 320, 320), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16, memory_config=L1)
            gf = 2 * bsz * 320 * 320 * 320 / 1e9
            byt = 3 * bsz * 320 * 320 * 2
            for lbl, kw in (("prod_pc", {"program_config": pc}), ("default", {})):
                try:
                    s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                       memory_config=L1, dtype=ttnn.bfloat16,
                                                       **kw))
                    C["batched"][f"b{bsz}_{lbl}"] = {
                        "us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2),
                        "bytes_MB": round(byt / 1e6, 3),
                        "l1_gbps": round(byt / s / 1e9, 1),
                        "AI": round(gf * 1e9 / byt, 1)}
                    print(f"  batch={bsz:<3} {lbl:8s} {s*1e6:8.2f} us {gf/s/1e3:7.2f} TFLOP/s",
                          flush=True)
                except Exception as e:                                     # noqa: BLE001
                    C["batched"][f"b{bsz}_{lbl}"] = {"err": str(e)[:90]}
                    print(f"  batch={bsz} {lbl} ERR {str(e)[:90]}", flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        res["C_k320_nt10"] = C

    # ---- D. write roofs by writer structure --------------------------------------------------
    if run("D"):
        print("=== D write roofs by writer structure (mine, this card, this pass) ===", flush=True)
        D = {"unary_clone_L1_to_DRAM": [], "matmul_writer": {}}
        for mb in (4, 8, 16, 24, 32):
            rows = int(mb * 1e6 / 2) // 4096
            nb = rows * 4096 * 2
            rec = {"MB": round(nb / 1e6, 2)}
            try:
                x = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
                s = timed(dev, lambda: ttnn.clone(x, memory_config=DRAM), warm=2, pipe=3)
                rec["write_gbps"] = round(nb / s / 1e9, 1)
                rec["ms"] = round(s * 1e3, 4)
                ttnn.deallocate(x)
            except Exception as e:                                         # noqa: BLE001
                rec["err"] = str(e)[:80]
            D["unary_clone_L1_to_DRAM"].append(rec)
            print("  clone " + json.dumps(rec), flush=True)
        m, k, n = 102400, 32, 256
        xw = ttnn.from_torch(torch.randn(1, 1, m, k), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        ww = ttnn.from_torch(torch.randn(k, n), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16, memory_config=DRAM)
        out_b, in_b = m * n * 2, m * k * 2 + k * n * 2
        mt = m // TILE
        pcm = -(-(-(-mt // ncore)) // 5) * 5
        legs = {
            "minimal_matmul": lambda: ttnn.experimental.minimal_matmul(
                xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc),
            "linear_core_grid": lambda: ttnn.linear(
                xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                core_grid=CORE_GRID_MAIN),
            "linear_1dmcast_pc": lambda: ttnn.linear(
                xw, ww, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                program_config=pc_1d(gx, gy, 1, 5, n // TILE, pcm)),
        }
        for lbl, fn in legs.items():
            try:
                s = timed(dev, fn, warm=2, pipe=3)
                D["matmul_writer"][lbl] = {
                    "ms": round(s * 1e3, 4), "write_gbps": round(out_b / s / 1e9, 1),
                    "rw_gbps": round((out_b + in_b) / s / 1e9, 1)}
                print(f"  {lbl:18s} {s*1e3:8.4f} ms  write {out_b/s/1e9:6.1f} GB/s", flush=True)
            except Exception as e:                                         # noqa: BLE001
                D["matmul_writer"][lbl] = {"err": str(e)[:90]}
        ttnn.deallocate(xw)
        ttnn.deallocate(ww)
        res["D_write_roofs"] = D

    # ---- E. read roof and the combined ceiling ------------------------------------------------
    if run("E"):
        print("=== E DRAM read roof and combined read+write ceiling ===", flush=True)
        E = {"read": [], "combined": []}
        for mb in (8, 16, 32, 48):
            rows = int(mb * 1e6 / 2) // 4096
            nb = rows * 4096 * 2
            rec = {"MB": round(nb / 1e6, 2)}
            try:
                x = ttnn.from_torch(torch.randn(rows, 4096), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
                s = timed(dev, lambda: ttnn.clone(x, memory_config=L1), warm=2, pipe=3)
                rec["read_gbps"] = round(nb / s / 1e9, 1)
                rec["ms"] = round(s * 1e3, 4)
                s2 = timed(dev, lambda: ttnn.clone(x, memory_config=DRAM), warm=2, pipe=3)
                E["combined"].append({"MB": rec["MB"], "ms": round(s2 * 1e3, 4),
                                      "rw_gbps": round(2 * nb / s2 / 1e9, 1)})
                ttnn.deallocate(x)
            except Exception as e:                                         # noqa: BLE001
                rec["err"] = str(e)[:80]
            E["read"].append(rec)
            print("  " + json.dumps(rec), flush=True)
        for r in E["combined"]:
            print("  dram->dram " + json.dumps(r), flush=True)
        res["E_read_combined"] = E

    # ---- F. the four rows standalone at the TRUE fold shape -----------------------------------
    if run("F"):
        print("=== F the rows at the TRUE fold shape [1, 298, 320, 256] ===", flush=True)
        F = {}
        # (i) the contraction, exactly as the fold runs it: batch=32, 320x320@320x320, all L1.
        pc = T._triangle_mul_program_config(10)
        a = ttnn.from_torch(torch.randn(1, 32, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        b = ttnn.from_torch(torch.randn(1, 32, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        gf = 2 * 32 * 320 ** 3 / 1e9
        byt = 3 * 32 * 320 * 320 * 2
        s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                                           program_config=pc, dtype=ttnn.bfloat16), pipe=6, reps=7)
        F["contraction_1043"] = {"us": round(s * 1e6, 2), "tflops": round(gf / s / 1e3, 2),
                                 "gflop": round(gf, 3), "bytes_MB": round(byt / 1e6, 3),
                                 "l1_gbps": round(byt / s / 1e9, 1),
                                 "AI": round(gf * 1e9 / byt, 1),
                                 "cores": pc.per_core_M and 10 * 10}
        print(f"  contraction b=32 320^3: {s*1e6:.2f} us  {gf/s/1e3:.2f} TFLOP/s", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        # (ii) the pair-track projection at the true shape and at the harness's square shape,
        # at both _PAIR_PROJ_BW settings. bw=8 is what main runs today; bw=1 is what T2 measured.
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        saved_bw = T._PAIR_PROJ_BW
        for rows, tag in ((298, "true_fold"), (320, "harness_square")):
            z = ttnn.from_torch(torch.randn(1, rows, 320, 256), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
            mt = rows * 10
            m = mt * TILE
            gf = 2 * m * 256 * 256 / 1e9
            out_b = m * 256 * 2
            in_b = m * 256 * 2 + 256 * 256 * 2
            for bw in (16, 1):
                T._PAIR_PROJ_BW = bw
                cfg = T._pair_proj_config(z, w)
                try:
                    s = timed(dev, lambda: T._pair_proj_linear(z, w, ckc, ttnn.bfloat16),
                              pipe=4, reps=5)
                    pcm = cfg.per_core_M if cfg is not None else None
                    F[f"proj_{tag}_bw{bw}"] = {
                        "rows": rows, "m_tiles": mt, "M": m, "us": round(s * 1e6, 2),
                        "tflops": round(gf / s / 1e3, 2), "gflop": round(gf, 3),
                        "out_MB": round(out_b / 1e6, 2), "in_MB": round(in_b / 1e6, 2),
                        "AI": round(gf * 1e9 / (out_b + in_b), 1),
                        "write_gbps": round(out_b / s / 1e9, 1),
                        "read_gbps": round(in_b / s / 1e9, 1),
                        "in0_block_w": cfg.in0_block_w if cfg is not None else None,
                        "per_core_M": pcm,
                        "cores": min(ncore, -(-mt // pcm)) if pcm else None}
                    print(f"  proj {tag:14s} bw={bw:<2} {s*1e6:8.2f} us {gf/s/1e3:7.2f} TFLOP/s "
                          f"write={out_b/s/1e9:6.1f} GB/s cores={F[f'proj_{tag}_bw{bw}']['cores']}",
                          flush=True)
                except Exception as e:                                     # noqa: BLE001
                    F[f"proj_{tag}_bw{bw}"] = {"err": str(e)[:100]}
                    print(f"  proj {tag} bw={bw} ERR {str(e)[:100]}", flush=True)
            ttnn.deallocate(z)
        T._PAIR_PROJ_BW = saved_bw
        ttnn.deallocate(w)
        res["F_rows_true_shape"] = F

    # ---- G. core utilisation by program-config grid A/B --------------------------------------
    if run("G"):
        print("=== G core utilisation: shrink the program config's grid, hold the size ===",
              flush=True)
        G = {"contraction": [], "projection": []}
        a = ttnn.from_torch(torch.randn(1, 32, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        b = ttnn.from_torch(torch.randn(1, 32, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=L1)
        gf = 2 * 32 * 320 ** 3 / 1e9
        for ggx, ggy in ((11, 10), (10, 10), (5, 10), (10, 5), (5, 5), (2, 2)):
            pcm, pcn = -(-10 // ggy), -(-10 // ggx)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(ggx, ggy), in0_block_w=10,
                out_subblock_h=1, out_subblock_w=1, out_block_h=pcm, out_block_w=pcn,
                per_core_M=pcm, per_core_N=pcn, transpose_mcast=False,
                fused_activation=None, fuse_batch=False)
            try:
                s = timed(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                   memory_config=L1, program_config=cfg,
                                                   dtype=ttnn.bfloat16), pipe=4, reps=5)
                G["contraction"].append({"grid": f"{ggx}x{ggy}", "cores": ggx * ggy,
                                         "per_core_M": pcm, "per_core_N": pcn,
                                         "us": round(s * 1e6, 2),
                                         "tflops": round(gf / s / 1e3, 2)})
                print(f"  contraction {ggx}x{ggy} pcM={pcm} pcN={pcn} {s*1e6:8.2f} us", flush=True)
            except Exception as e:                                         # noqa: BLE001
                G["contraction"].append({"grid": f"{ggx}x{ggy}", "err": str(e)[:80]})
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        z = ttnn.from_torch(torch.randn(1, 298, 320, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        w = ttnn.from_torch(torch.randn(256, 256), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
        mt, m = 2980, 2980 * TILE
        gf = 2 * m * 256 * 256 / 1e9
        for ggx, ggy in ((11, 10), (10, 10), (11, 5), (5, 10), (5, 5)):
            nc = ggx * ggy
            if mt < nc:
                continue
            pcm = -(-(-(-mt // nc)) // 5) * 5
            if -(-mt // pcm) > nc:
                continue
            try:
                cfg = pc_1d(ggx, ggy, 8, 5, 8, pcm)
                s = timed(dev, lambda: ttnn.linear(z, w, compute_kernel_config=ckc,
                                                   memory_config=DRAM, program_config=cfg),
                          pipe=4, reps=5)
                G["projection"].append({"grid": f"{ggx}x{ggy}", "cores": nc,
                                        "cores_used": min(nc, -(-mt // pcm)),
                                        "per_core_M": pcm, "us": round(s * 1e6, 2),
                                        "tflops": round(gf / s / 1e3, 2)})
                print(f"  projection {ggx}x{ggy} pcM={pcm} used={min(nc, -(-mt//pcm))} "
                      f"{s*1e6:8.2f} us", flush=True)
            except Exception as e:                                         # noqa: BLE001
                G["projection"].append({"grid": f"{ggx}x{ggy}", "err": str(e)[:80]})
        ttnn.deallocate(z)
        ttnn.deallocate(w)
        res["G_core_util"] = G

    # ---- H. roof stability: the load-bearing rows again ---------------------------------------
    if run("H"):
        el = time.time() - res["t_start"]
        print(f"=== H roof stability re-check, {el/60:.1f} min after the first ===", flush=True)
        H = {"elapsed_min": round(el / 60, 1), "sweep": []}
        for nt in (8, 32):
            for lbl, mc in (("dram", DRAM), ("l1", L1)):
                r = k256_leg(dev, ckc, gx, gy, ncore, int(24e6 / 2), nt, mc, lbl)
                if r:
                    H["sweep"].append(r)
        res["H_stability"] = H

    res["t_end"] = time.time()
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
