#!/usr/bin/env python3
"""z-transition-chunk §7.3 -- roofs on qb2 card 2, measured in this process, plus the Transition
fc1 placed against them at both chunk heights.

Charter §4.1: roofs are per-card and this org has produced four wrong conclusions from an inherited
one. Nothing here is carried in from card 0's `perf/size512/roofs_permute_qb2c0.py` or from T3's
95.42 TFLOP/s on card 1. Every number below comes out of this device context.

What is measured, in order:

  square compute roof   bf16 HiFi4, DRAM output, only to fix this card's machine balance.
  DRAM read roof        DRAM->L1 clone, >=64 MB.
  L1 copy roof          L1->L1 clone, >=64 MB. Prices the chunk-loop data movement.
  K/nt matmul sweep     K=256 N=1024 (the fc1's own contraction and output width) with an L1
                        output, swept over per_core_M via M; plus K=1024 and nt=64 arms so the
                        K term and the nt term can be told apart (charter §4.6 asks for both
                        columns and says the nt one usually binds).
  the fc1 itself        [1,h,512,256] x [256,1024] -> L1, at h=16 (production at W=512) and
                        h=32 (the lever; isolated only -- it does not allocate in a block),
                        scored fused-silu as production runs it AND bare.

Both sides of every timed region synchronise (`ttnn-sync-before-every-timed-region`: an unsynced
`to_torch` drain has inverted a ranking in this codebase).

qb2 / ttnn 0.68.0 -- every absolute here is a RATIO input owing a qb1/0.67.4 re-take.
NOTHING IN tt_bio/ IS CHANGED. Phase 2 forbids production edits.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ttnn  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def free(*ts):
    """Deallocate unconditionally. A tensor freed only on the success path leaks its L1 on the
    OOM/clash path, and a leaked L1 buffer makes every later L1-output op clash -- which is a
    harness artefact indistinguishable from the real in-block clash this leg is measuring."""
    for t in ts:
        if t is not None:
            try:
                ttnn.deallocate(t)
            except Exception:                                                   # noqa: BLE001
                pass


def timed(dev, fn, warm=3, pipe=4, reps=5):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import importlib.metadata as im
    from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN, COMPUTE_GRID_MAIN

    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    gx, gy = COMPUTE_GRID_MAIN
    ncores = gx * gy
    res = {"host": "qb2", "card": "physical 2", "ttnn": im.version("ttnn"),
           "note": "qb2 / ttnn 0.68.0 -- RATIO inputs, owe a qb1/0.67.4 re-take",
           "compute_with_storage_grid_size": f"{dg.x}x{dg.y}",
           "core_grid_main": f"{gx}x{gy}", "cores_main": ncores,
           "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size())}
    print(json.dumps(res), flush=True)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    # --- square compute roof, only to fix the machine balance ------------------------------
    comp = {}
    for n in (2048, 4096):
        x = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        y = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        s = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(x, y, compute_kernel_config=ckc, memory_config=DRAM)),
                  warm=2, pipe=3, reps=3)
        comp[f"{n}_square_oDRAM"] = {"ms": round(s * 1e3, 4), "tflops": round(2 * n ** 3 / s / 1e12, 2)}
        print(f"  compute N={n} {comp[f'{n}_square_oDRAM']}", flush=True)
        ttnn.deallocate(x); ttnn.deallocate(y)
    res["compute_roof_square"] = {"runs": comp, "peak_TFLOPs": max(v["tflops"] for v in comp.values())}

    # --- DRAM read roof and L1 copy roof ---------------------------------------------------
    mv = []
    for mb in (32, 64, 96):
        nrow = int(mb * 1e6 / 2) // 4096
        nb = nrow * 4096 * 2
        r = {"MB": round(nb / 1e6, 2)}
        xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev, memory_config=DRAM)
        try:
            t = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)))
            r["dram_read_GBs"] = round(nb / t / 1e9, 1)
        except Exception as e:                                                  # noqa: BLE001
            r["dram_read_err"] = str(e)[:110]
        free(xd)
        xl = None
        try:
            xl = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=L1)
            t = timed(dev, lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=L1)))
            r["l1_copy_rw_GBs"] = round(2 * nb / t / 1e9, 1)
            r["l1_copy_write_GBs"] = round(nb / t / 1e9, 1)
        except Exception as e:                                                  # noqa: BLE001
            r["l1_copy_err"] = str(e)[:110]
        finally:
            free(xl)
        mv.append(r)
        print("  mv " + json.dumps(r), flush=True)
    read_peak = max((r.get("dram_read_GBs", 0) for r in mv), default=0.0)
    l1copy_peak = max((r.get("l1_copy_rw_GBs", 0) for r in mv), default=0.0)
    l1write_peak = max((r.get("l1_copy_write_GBs", 0) for r in mv), default=0.0)
    res["movement_roofs"] = {"runs": mv, "dram_read_peak_GBs": read_peak,
                             "l1_copy_rw_peak_GBs": l1copy_peak,
                             "l1_copy_write_peak_GBs": l1write_peak}
    res["machine_balance_FLOP_per_byte_read"] = round(
        res["compute_roof_square"]["peak_TFLOPs"] * 1e12 / (read_peak * 1e9), 1) if read_peak else None
    print(f"  MEASURED dram_read {read_peak} GB/s  l1_copy(rw) {l1copy_peak} GB/s  "
          f"balance {res['machine_balance_FLOP_per_byte_read']} FLOP/byte", flush=True)

    # --- matmul roofs at the fc1's own contraction and output width ------------------------
    # charter §4.6 wants two columns: the K-corrected roof at the op's own output buffer type,
    # and the best rate the card reaches at the op's own output width nt. Sweeping K and nt
    # separately is what tells them apart.
    mm = []
    for K, N, M in [(256, 1024, 2560), (256, 1024, 8192), (256, 1024, 16384), (256, 1024, 24576),
                    (1024, 1024, 8192), (256, 2048, 8192), (256, 256, 8192)]:
        Mt, nt = M // 32, N // 32
        rec = {"K": K, "N": N, "M": M, "Mt": Mt, "nt": nt,
               "per_core_M": -(-Mt // gy), "per_core_N": -(-nt // gx),
               "cores_used": min(gy, Mt) * min(gx, nt),
               "out_MB": round(M * N * 2 / 1e6, 2)}
        x = w = None
        try:
            x = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=L1)
            w = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=DRAM)
            t = timed(dev, lambda: ttnn.deallocate(ttnn.linear(
                x, w, compute_kernel_config=ckc, memory_config=L1, dtype=ttnn.bfloat16,
                core_grid=CORE_GRID_MAIN)), warm=3, pipe=4, reps=5)
            rec["ms"] = round(t * 1e3, 4)
            rec["tflops"] = round(2 * M * K * N / t / 1e12, 2)
            # charter §4.6 identity, exact for a bf16 output: TFLOP/s = write_GB/s x K
            rec["implied_l1_write_GBs"] = round(M * N * 2 / t / 1e9, 1)
        except Exception as e:                                                  # noqa: BLE001
            rec["err"] = str(e)[:140]
        finally:
            free(x, w)
        mm.append(rec)
        print("  mm " + json.dumps(rec), flush=True)
    at_op = [r for r in mm if r.get("K") == 256 and r.get("nt") == 32 and "tflops" in r]
    res["matmul_roofs"] = {
        "runs": mm,
        "K256_nt32_L1out_peak_TFLOPs": max((r["tflops"] for r in at_op), default=0.0),
        "K256_nt32_L1out_peak_write_GBs": max((r["implied_l1_write_GBs"] for r in at_op), default=0.0),
        "best_any_TFLOPs": max((r["tflops"] for r in mm if "tflops" in r), default=0.0),
    }
    print("  ROOF(K=256,nt=32,L1 out) " + json.dumps(res["matmul_roofs"]), flush=True)

    # --- L1 contamination canary ------------------------------------------------------------
    # The first run of this harness leaked a 96 MB L1 tensor on an OOM path and every later
    # L1-output matmul clashed, h=16 included -- and h=16 IS production at W=512, so it
    # demonstrably allocates in a real fold. A leaked L1 buffer is indistinguishable from the
    # real in-block clash. Prove L1 is clean before scoring anything.
    canary = {}
    for mb in (64, 96):
        c = None
        try:
            c = ttnn.from_torch(torch.zeros(int(mb * 1e6 / 2) // 4096, 4096), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
            canary[f"{mb}MB_L1_alloc"] = "ok"
        except Exception as e:                                                  # noqa: BLE001
            canary[f"{mb}MB_L1_alloc"] = "FAIL: " + str(e)[:80]
        finally:
            free(c)
    res["l1_canary_before_fc1"] = canary
    print("  canary " + json.dumps(canary), flush=True)

    # --- the fc1 itself, both heights, fused and bare --------------------------------------
    fc1 = []
    for h in (16, 32):
        M = h * 512
        Mt, nt = M // 32, 1024 // 32
        rec = {"h": h, "x_shape": [1, h, 512, 256], "M": M, "Mt": Mt, "nt": nt,
               "per_core_M": -(-Mt // gy), "per_core_N": -(-nt // gx),
               "cores_used": min(gy, Mt) * min(gx, nt), "cores_grid": ncores,
               "gflop": round(2 * M * 256 * 1024 / 1e9, 4),
               "in_MB": round(M * 256 * 2 / 1e6, 3), "w_MB": round(256 * 1024 * 2 / 1e6, 3),
               "out_MB": round(M * 1024 * 2 / 1e6, 3)}
        rec["core_util_pct"] = round(100.0 * rec["cores_used"] / ncores, 1)
        rec["M_pad_waste_pct"] = round(100.0 * (rec["per_core_M"] * gy - Mt) / Mt, 2)
        rec["N_pad_waste_pct"] = round(100.0 * (rec["per_core_N"] * gx - nt) / nt, 2)
        tot_b = (rec["in_MB"] + rec["w_MB"] + rec["out_MB"]) * 1e6
        rec["arith_intensity_FLOP_per_byte"] = round(rec["gflop"] * 1e9 / tot_b, 1)
        rec["out_share_of_traffic_pct"] = round(100.0 * rec["out_MB"] * 1e6 / tot_b, 1)
        x = w = None
        try:
            x = ttnn.from_torch(torch.randn(1, h, 512, 256), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
            w = ttnn.from_torch(torch.randn(256, 1024), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
            for lbl, act in (("fused_silu", "silu"), ("bare", None)):
                try:
                    t = timed(dev, lambda act=act: ttnn.deallocate(ttnn.linear(
                        x, w, activation=act, compute_kernel_config=ckc, memory_config=L1,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)), warm=3, pipe=4, reps=5)
                except Exception as e:                                          # noqa: BLE001
                    rec[f"{lbl}_err"] = str(e)[:220]
                    continue
                rec[f"{lbl}_ms"] = round(t * 1e3, 4)
                rec[f"{lbl}_tflops"] = round(rec["gflop"] * 1e9 / t / 1e12, 2)
                rec[f"{lbl}_l1_write_GBs"] = round(rec["out_MB"] * 1e6 / t / 1e9, 1)
        except Exception as e:                                                  # noqa: BLE001
            rec["err"] = str(e)[:220]
        finally:
            free(x, w)
        fc1.append(rec)
        print("  fc1 " + json.dumps(rec), flush=True)

    roof_tf = res["matmul_roofs"]["K256_nt32_L1out_peak_TFLOPs"]
    roof_w = res["matmul_roofs"]["K256_nt32_L1out_peak_write_GBs"]
    for rec in fc1:
        for lbl in ("fused_silu", "bare"):
            if f"{lbl}_tflops" in rec and roof_tf:
                rec[f"{lbl}_pct_of_matmul_roof_K256_nt32_L1out"] = round(
                    100.0 * rec[f"{lbl}_tflops"] / roof_tf, 1)
            if f"{lbl}_l1_write_GBs" in rec and roof_w:
                rec[f"{lbl}_pct_of_l1_write_roof_matmul"] = round(
                    100.0 * rec[f"{lbl}_l1_write_GBs"] / roof_w, 1)
            if f"{lbl}_l1_write_GBs" in rec and l1write_peak:
                rec[f"{lbl}_pct_of_l1_copy_write_roof"] = round(
                    100.0 * rec[f"{lbl}_l1_write_GBs"] / l1write_peak, 1)
        if "fused_silu_ms" in rec and "bare_ms" in rec:
            rec["silu_cost_ms"] = round(rec["fused_silu_ms"] - rec["bare_ms"], 4)
            rec["silu_cost_pct_of_fused"] = round(
                100.0 * (rec["fused_silu_ms"] - rec["bare_ms"]) / rec["fused_silu_ms"], 1)
    res["fc1"] = fc1

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote " + a.out, flush=True)
    print(json.dumps({"fc1": fc1}, indent=1), flush=True)


if __name__ == "__main__":
    main()
