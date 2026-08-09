#!/usr/bin/env python3
"""T1 triangle-attention Phase-1 probe: roofs + every op T1 owns, one process, one card.

Runs on qb1 card 2 (TT_VISIBLE_DEVICES=2). Characterisation only -- nothing here touches
production code; the ops are re-issued standalone at exactly the shapes/dtypes/buffer types the
298 aa (N=320, c_z=256) protenix-v2 Pairformer block issues them at, read off
perf/ledger_298/ops_protenix-v2_320.json.

Every timed region synchronises on both sides. Results stream to --out as they are produced so a
timeout still leaves the finished stages on disk.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN, COMPUTE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
N, CZ, NH, HD = 320, 256, 8, 32
RES = {}


def save(path):
    json.dump(RES, open(path, "w"), indent=1)


def timed(fn, warm=3, pipe=4, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(DEV)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def us(x):
    return round(x * 1e6, 1)


def T(shape, mc=DRAM, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=DEV,
                           dtype=dt, memory_config=mc)


def stage(name, fn, out):
    print(f"\n=== {name} ===", flush=True)
    try:
        RES[name] = fn()
    except Exception as e:                                             # noqa: BLE001
        RES[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print("  ERR", RES[name]["error"], flush=True)
    save(out)


# ---------------------------------------------------------------------------------- roofs

def roofs():
    r = {}
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    # compute roof: the config the model actually runs (fp32_dest_acc + packer_l1_acc, HiFi4)
    sq = {}
    for n in (2048, 4096, 6144):
        a, b = T((1, 1, n, n)), T((1, 1, n, n))
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                      memory_config=DRAM)), warm=3, pipe=3, reps=5)
        sq[n] = {"ms": round(s * 1e3, 4), "tflops": round(2 * n ** 3 / s / 1e12, 2)}
        print(f"  square N={n}: {sq[n]['tflops']} TFLOP/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    r["compute_square"] = sq
    r["compute_peak_TFLOPs"] = max(v["tflops"] for v in sq.values())

    # K-corrected compute rate: the contraction shape T1's matmuls actually run (K=256).
    kc = {}
    for (m, k, n) in ((N * N, CZ, 768), (N * N, CZ, CZ), (N * N, CZ, 32), (6144, 6144, 6144)):
        a, b = T((m, k)), T((k, n))
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                      memory_config=DRAM)), warm=3, pipe=3, reps=5)
        kc[f"{m}x{k}x{n}"] = {"ms": round(s * 1e3, 4), "tflops": round(2 * m * k * n / s / 1e12, 2)}
        print(f"  K-corrected {m}x{k}x{n}: {kc[f'{m}x{k}x{n}']['tflops']} TFLOP/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)
    r["compute_shape_specific"] = kc

    # DRAM read swept out to 128 MB (the 8-64 MB ladder is still climbing), write, and copy.
    rows = []
    for mb in (16, 32, 64, 96, 128):
        nrow = int(mb * 1e6 / 2) // 4096
        nb = nrow * 4096 * 2
        row = {"MB": round(nb / 1e6, 2)}
        xd = T((nrow, 4096), DRAM)
        row["read_GBs"] = round(nb / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1))) / 1e9, 1)
        row["copy_rw_GBs"] = round(2 * nb / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM))) / 1e9, 1)
        ttnn.deallocate(xd)
        try:
            xl = T((nrow, 4096), L1)
            row["write_GBs"] = round(nb / timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM))) / 1e9, 1)
            ttnn.deallocate(xl)
        except Exception as e:                                         # noqa: BLE001
            row["write_err"] = str(e)[:80]
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
    r["dram"] = rows
    r["read_peak_GBs"] = max(x.get("read_GBs", 0) for x in rows)
    r["write_peak_GBs"] = max(x.get("write_GBs", 0) for x in rows)
    r["copy_peak_GBs"] = max(x.get("copy_rw_GBs", 0) for x in rows)
    r["machine_balance_FLOP_per_byte"] = round(r["compute_peak_TFLOPs"] * 1e12 / (r["read_peak_GBs"] * 1e9), 1)
    print(f"  ROOFS compute {r['compute_peak_TFLOPs']} TFLOP/s  read {r['read_peak_GBs']} "
          f"write {r['write_peak_GBs']} copy {r['copy_peak_GBs']} GB/s  bal "
          f"{r['machine_balance_FLOP_per_byte']} FLOP/B", flush=True)
    return r


# ------------------------------------------------------------------------------------ SDPA

def sdpa_cfg(q, k, grid=None):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid or COMPUTE_GRID_MAIN,
                                  exp_approx_mode=False, q_chunk_size=q, k_chunk_size=k)


def sdpa_run(q, k, v, mask, cfg):
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False, scale=HD ** -0.5, program_config=cfg)


def sdpa():
    r = {}
    prod = sdpa_cfg(64, 64)
    q, k, v = (T((N, NH, N, HD)) for _ in range(3))
    mb = T((1, NH, N, N))                       # the broadcast bias the model passes

    # Q2 leg 1 -- mask present vs absent, production shape and production program config.
    t_mask = timed(lambda: ttnn.deallocate(sdpa_run(q, k, v, mb, prod)), warm=2, pipe=3, reps=5)
    t_none = timed(lambda: ttnn.deallocate(sdpa_run(q, k, v, None, prod)), warm=2, pipe=3, reps=5)
    r["prod_mask_us"], r["prod_nomask_us"] = us(t_mask), us(t_none)
    print(f"  prod cfg 64/64: mask {us(t_mask)} us, no-mask {us(t_none)} us", flush=True)

    # Q2 leg 2 -- the decisive one. A mask MATERIALISED over the batch is b times the bytes of the
    # broadcast one. If the kernel already re-reads the broadcast mask once per (batch, head) the
    # two cost the same; if it reads it once, the materialised one is far slower.
    try:
        mm = T((N, NH, N, N))
        t_full = timed(lambda: ttnn.deallocate(sdpa_run(q, k, v, mm, prod)), warm=2, pipe=2, reps=5)
        r["prod_materialised_mask_us"] = us(t_full)
        print(f"  materialised mask [{N},{NH},{N},{N}] = {N*NH*N*N*2/1e6:.1f} MB: {us(t_full)} us", flush=True)
        ttnn.deallocate(mm)
    except Exception as e:                                             # noqa: BLE001
        r["materialised_err"] = str(e)[:150]
        print("  materialised mask ERR", r["materialised_err"], flush=True)

    # Q2 leg 3 -- batch-alone sweep. q/k/v scale with b, the broadcast mask does not.
    sweep = []
    for b in (40, 80, 160, 320):
        qb, kb, vb = (T((b, NH, N, HD)) for _ in range(3))
        tm = timed(lambda: ttnn.deallocate(sdpa_run(qb, kb, vb, mb, prod)), warm=2, pipe=3, reps=5)
        tn = timed(lambda: ttnn.deallocate(sdpa_run(qb, kb, vb, None, prod)), warm=2, pipe=3, reps=5)
        sweep.append({"b": b, "mask_us": us(tm), "nomask_us": us(tn)})
        print("  " + json.dumps(sweep[-1]), flush=True)
        for t in (qb, kb, vb):
            ttnn.deallocate(t)
    r["batch_sweep"] = sweep
    if len(sweep) >= 2:
        db = sweep[-1]["b"] - sweep[0]["b"]
        r["slope_mask_us_per_b"] = round((sweep[-1]["mask_us"] - sweep[0]["mask_us"]) / db, 4)
        r["slope_nomask_us_per_b"] = round((sweep[-1]["nomask_us"] - sweep[0]["nomask_us"]) / db, 4)
        r["slope_bias_us_per_b"] = round(r["slope_mask_us_per_b"] - r["slope_nomask_us_per_b"], 4)
        bias_bytes_per_b = NH * N * N * 2
        r["implied_bias_GBs_if_per_batch_head"] = round(
            bias_bytes_per_b / (r["slope_bias_us_per_b"] * 1e-6) / 1e9, 1) if r["slope_bias_us_per_b"] > 0 else None
        print(f"  slopes: mask {r['slope_mask_us_per_b']} us/b, nomask {r['slope_nomask_us_per_b']} "
              f"us/b, bias {r['slope_bias_us_per_b']} us/b -> "
              f"{r['implied_bias_GBs_if_per_batch_head']} GB/s of bias", flush=True)

    # occupancy A/B: the only grid knob SDPA exposes is its program config's grid.
    occ = []
    for gx, gy in ((11, 10), (10, 10), (8, 8), (6, 6), (4, 4), (2, 2), (1, 1)):
        try:
            t = timed(lambda: ttnn.deallocate(sdpa_run(q, k, v, mb, sdpa_cfg(64, 64, (gx, gy)))),
                      warm=1, pipe=2, reps=3)
            occ.append({"grid": f"{gx}x{gy}", "cores": gx * gy, "us": us(t)})
        except Exception as e:                                         # noqa: BLE001
            occ.append({"grid": f"{gx}x{gy}", "cores": gx * gy, "err": str(e)[:80]})
        print("  " + json.dumps(occ[-1]), flush=True)
    r["grid_sweep"] = occ

    # chunk-config sweep, so the production 64/64 choice is scored not assumed
    ch = []
    for c in (32, 64, 128, 256, 320):
        try:
            t = timed(lambda: ttnn.deallocate(sdpa_run(q, k, v, mb, sdpa_cfg(c, c))),
                      warm=1, pipe=2, reps=3)
            ch.append({"chunk": c, "us": us(t)})
        except Exception as e:                                         # noqa: BLE001
            ch.append({"chunk": c, "err": str(e)[:80]})
        print("  " + json.dumps(ch[-1]), flush=True)
    r["chunk_sweep"] = ch

    for t in (q, k, v, mb):
        ttnn.deallocate(t)
    return r


# ------------------------------------------------------------- head split / permutes / matmuls

def heads():
    r = {}
    for n in (128, 192, 256, 320):
        x = T((n, 1, n, 3 * NH * HD))
        t = timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)],
            warm=2, pipe=3, reps=5)
        by = n * n * 3 * NH * HD * 2
        r[f"dram_n{n}"] = {"us": us(t), "in_MB": round(by / 1e6, 2),
                           "rw_GBs": round(2 * by / t / 1e9, 1)}
        print(f"  DRAM n={n}: {us(t)} us, {r[f'dram_n{n}']['rw_GBs']} GB/s r+w", flush=True)
        ttnn.deallocate(x)
    # the N=128 production regime is L1-in/L1-out (the qkv projection kept its result in L1).
    for n in (128, 192, 256, 320):
        try:
            x = T((n, 1, n, 3 * NH * HD), L1)
            t = timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
                x, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=L1)],
                warm=2, pipe=3, reps=5)
            by = n * n * 3 * NH * HD * 2
            r[f"l1_n{n}"] = {"us": us(t), "rw_GBs": round(2 * by / t / 1e9, 1)}
            print(f"  L1   n={n}: {us(t)} us, {r[f'l1_n{n}']['rw_GBs']} GB/s r+w", flush=True)
            ttnn.deallocate(x)
        except Exception as e:                                         # noqa: BLE001
            r[f"l1_n{n}"] = {"err": str(e)[:90]}
            print(f"  L1   n={n}: ERR {r[f'l1_n{n}']['err']}", flush=True)
    return r


def permutes():
    r = {}
    for n in (128, 320):
        x = T((n, n, CZ))
        by = n * n * CZ * 2
        t = timed(lambda: ttnn.deallocate(ttnn.permute(x, (1, 0, 2))), warm=2, pipe=3, reps=5)
        r[f"dram2dram_n{n}"] = {"us": us(t), "MB": round(by / 1e6, 2),
                                "rw_GBs": round(2 * by / t / 1e9, 1)}
        # the same bytes moved as whole tiles, not transposed: the piece-size control
        tc = timed(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=DRAM)), warm=2, pipe=3, reps=5)
        r[f"clone_dram_n{n}"] = {"us": us(tc), "rw_GBs": round(2 * by / tc / 1e9, 1)}
        # C2FIX: the same permute with an L1 destination
        try:
            tl = timed(lambda: ttnn.deallocate(ttnn.permute(x, (1, 0, 2), memory_config=L1)),
                       warm=2, pipe=3, reps=5)
            r[f"dram2l1_n{n}"] = {"us": us(tl), "rw_GBs": round(2 * by / tl / 1e9, 1),
                                  "speedup_vs_dram": round(t / tl, 3)}
        except Exception as e:                                         # noqa: BLE001
            r[f"dram2l1_n{n}"] = {"err": str(e)[:90]}
        try:
            xl = T((n, n, CZ), L1)
            tll = timed(lambda: ttnn.deallocate(ttnn.permute(xl, (1, 0, 2), memory_config=L1)),
                        warm=2, pipe=3, reps=5)
            r[f"l12l1_n{n}"] = {"us": us(tll), "rw_GBs": round(2 * by / tll / 1e9, 1)}
            ttnn.deallocate(xl)
        except Exception as e:                                         # noqa: BLE001
            r[f"l12l1_n{n}"] = {"err": str(e)[:90]}
        for kk in (f"dram2dram_n{n}", f"clone_dram_n{n}", f"dram2l1_n{n}", f"l12l1_n{n}"):
            print(f"  {kk}: {json.dumps(r[kk])}", flush=True)
        ttnn.deallocate(x)
    return r


def matmuls():
    r = {}
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    from tt_bio.tenstorrent import _pair_proj_linear, _pair_proj_config  # noqa: PLC0415

    x = T((N, N, CZ))
    for lbl, nout in (("qkv", 768), ("gate", CZ)):
        w = T((CZ, nout))
        t = timed(lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16)),
            warm=2, pipe=3, reps=5)
        rd = N * N * CZ * 2 + CZ * nout * 2
        wr = N * N * nout * 2
        fl = 2 * N * N * CZ * nout
        r[lbl] = {"us": us(t), "read_MB": round(rd / 1e6, 2), "write_MB": round(wr / 1e6, 2),
                  "read_GBs": round(rd / t / 1e9, 1), "write_GBs": round(wr / t / 1e9, 1),
                  "rw_GBs": round((rd + wr) / t / 1e9, 1), "tflops": round(fl / t / 1e12, 2),
                  "AI_FLOP_per_byte": round(fl / (rd + wr), 2)}
        print(f"  {lbl}: {json.dumps(r[lbl])}", flush=True)
        # occupancy A/B for the matmul class: ttnn.matmul with an explicit grid, same shape
        occ = []
        for gx, gy in ((11, 10), (8, 8), (6, 6), (4, 4), (2, 2)):
            try:
                tt_ = timed(lambda: ttnn.deallocate(ttnn.matmul(
                    x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                    core_grid=ttnn.CoreGrid(x=gx, y=gy), memory_config=DRAM)),
                    warm=1, pipe=2, reps=3)
                occ.append({"grid": f"{gx}x{gy}", "cores": gx * gy, "us": us(tt_)})
            except Exception as e:                                     # noqa: BLE001
                occ.append({"grid": f"{gx}x{gy}", "err": str(e)[:70]})
            print("    " + json.dumps(occ[-1]), flush=True)
        r[lbl + "_grid_sweep"] = occ
        ttnn.deallocate(w)

    # triangle_bias: _pair_proj_linear, the model's own helper and its own program config
    wb = T((CZ, 32))
    cfg = _pair_proj_config(x, wb)
    t = timed(lambda: ttnn.deallocate(_pair_proj_linear(x, wb, ckc, ttnn.bfloat16)),
              warm=2, pipe=3, reps=5)
    rd, wr = N * N * CZ * 2 + CZ * 32 * 2, N * N * 32 * 2
    fl = 2 * N * N * CZ * 32
    r["triangle_bias"] = {"us": us(t), "cfg": str(cfg)[:60], "read_MB": round(rd / 1e6, 2),
                          "write_MB": round(wr / 1e6, 2), "read_GBs": round(rd / t / 1e9, 1),
                          "rw_GBs": round((rd + wr) / t / 1e9, 1),
                          "tflops": round(fl / t / 1e12, 3),
                          "AI_FLOP_per_byte": round(fl / (rd + wr), 2)}
    print(f"  triangle_bias: {json.dumps(r['triangle_bias'])}", flush=True)
    occ = []
    for gx, gy in ((11, 10), (8, 8), (6, 6), (4, 4)):
        try:
            tt_ = timed(lambda: ttnn.deallocate(ttnn.linear(
                x, wb, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(x=gx, y=gy), memory_config=DRAM)),
                warm=1, pipe=2, reps=3)
            occ.append({"grid": f"{gx}x{gy}", "cores": gx * gy, "us": us(tt_)})
        except Exception as e:                                         # noqa: BLE001
            occ.append({"grid": f"{gx}x{gy}", "err": str(e)[:70]})
        print("    " + json.dumps(occ[-1]), flush=True)
    r["triangle_bias_grid_sweep"] = occ
    ttnn.deallocate(wb)

    # the sigmoid gate multiply_, in place on a DRAM pair tensor
    a, b = T((N, N, CZ)), T((N, N, CZ))
    t = timed(lambda: ttnn.multiply_(a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
              warm=2, pipe=3, reps=5)
    by = N * N * CZ * 2
    r["gate_multiply_"] = {"us": us(t), "read_MB": round(2 * by / 1e6, 2),
                           "write_MB": round(by / 1e6, 2),
                           "read_GBs": round(2 * by / t / 1e9, 1),
                           "rw_GBs": round(3 * by / t / 1e9, 1)}
    print(f"  gate_multiply_: {json.dumps(r['gate_multiply_'])}", flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b); ttnn.deallocate(x)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    global DEV
    DEV = get_device()
    dg = DEV.compute_with_storage_grid_size()
    RES["device"] = {"TT_VISIBLE_DEVICES": os.environ.get("TT_VISIBLE_DEVICES"),
                     "compute_grid": f"{dg.x}x{dg.y}", "arch": str(DEV.arch()),
                     "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                     "l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
                     "id": DEV.id()}
    print(json.dumps(RES["device"]), flush=True)
    want = set(a.only.split(",")) if a.only else None
    for nm, fn in (("roofs", roofs), ("sdpa", sdpa), ("heads", heads),
                   ("permutes", permutes), ("matmuls", matmuls)):
        if want and nm not in want:
            continue
        stage(nm, fn, a.out)
    save(a.out)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
