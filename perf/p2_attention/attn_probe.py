#!/usr/bin/env python3
"""P1 p2-attention Phase-2 probe: the three levers, on qb1 card 2 at ttnn 0.67.4.

EXPERIMENT ONLY. Nothing here touches production code; every op is re-issued standalone at the
shapes a live 298 aa protenix-v2 fold issues it at (pair tensor [298, 320, 256], q/k/v
[298, 8, 320, 32], bias [1, 8, 320, 320]).

Every timed region synchronises the device immediately before the clock starts and immediately
before it stops. Results stream to --out so a timeout still leaves the finished stages on disk.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
M, N, CZ, NH, HD = 298, 320, 256, 8, 32
RES = {}
DEV = None


def save(path):
    json.dump(RES, open(path, "w"), indent=1)


def l1_unreserved(dev):
    """`get_max_worker_l1_unreserved_size` lives on the single Device, not on the MeshDevice."""
    for obj in [dev] + list(getattr(dev, "get_devices", lambda: [])()):
        f = getattr(obj, "get_max_worker_l1_unreserved_size", None)
        if f is not None:
            return f()
    raise AttributeError("no get_max_worker_l1_unreserved_size on device or its sub-devices")


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
    t0 = time.perf_counter()
    try:
        RES[name] = fn()
    except Exception as e:                                             # noqa: BLE001
        RES[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
        print("  ERR", RES[name]["error"], flush=True)
    print(f"  [{time.perf_counter() - t0:.1f}s]", flush=True)
    save(out)


# ------------------------------------------------------------------ R1/R2/R3: roofs, this card
def roofs():
    r = {"card": "qb1 TT_VISIBLE_DEVICES=2", "note": "measured this pass, not inherited"}
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    n = 4096
    a, b = T((1, 1, n, n)), T((1, 1, n, n))
    s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                  memory_config=DRAM)), warm=2, pipe=3, reps=5)
    r["compute_square_4096_TFLOPs"] = round(2 * n ** 3 / s / 1e12, 2)
    ttnn.deallocate(a); ttnn.deallocate(b)
    print(f"  square 4096: {r['compute_square_4096_TFLOPs']} TFLOP/s", flush=True)

    rows = []
    for mb in (32, 64, 128):
        nrow = int(mb * 1e6 / 2) // 4096
        nb = nrow * 4096 * 2
        row = {"MB": round(nb / 1e6, 2)}
        xd = T((nrow, 4096), DRAM)
        row["read_GBs"] = round(nb / timed(lambda: ttnn.deallocate(
            ttnn.clone(xd, memory_config=L1)), warm=2, pipe=3, reps=5) / 1e9, 1)
        row["copy_rw_GBs"] = round(2 * nb / timed(lambda: ttnn.deallocate(
            ttnn.clone(xd, memory_config=DRAM)), warm=2, pipe=3, reps=5) / 1e9, 1)
        ttnn.deallocate(xd)
        try:
            xl = T((nrow, 4096), L1)
            row["write_GBs"] = round(nb / timed(lambda: ttnn.deallocate(
                ttnn.clone(xl, memory_config=DRAM)), warm=2, pipe=3, reps=5) / 1e9, 1)
            ttnn.deallocate(xl)
        except Exception as e:                                         # noqa: BLE001
            row["write_err"] = str(e)[:80]
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
    r["dram"] = rows
    r["read_peak_GBs"] = max(x.get("read_GBs", 0) for x in rows)
    r["copy_peak_GBs"] = max(x.get("copy_rw_GBs", 0) for x in rows)
    r["clone_write_peak_GBs"] = max(x.get("write_GBs", 0) for x in rows)

    # R3: the MATMUL WRITER's write roof, operands in L1 (B1's rule), output DRAM. K=32 keeps the
    # op write-dominated so the figure is the writer's, not the contraction's.
    best = 0.0
    for (m, k, nn) in ((M * N, 32, 2048), (M * N, 32, 4096)):
        try:
            a2, b2 = T((m, k), L1), T((k, nn), L1)
            s2 = timed(lambda: ttnn.deallocate(ttnn.matmul(a2, b2, compute_kernel_config=ckc,
                                                           memory_config=DRAM)), warm=2, pipe=3, reps=5)
            gbs = round(m * nn * 2 / s2 / 1e9, 1)
            r[f"mm_writer_{m}x{k}x{nn}_GBs"] = gbs
            best = max(best, gbs)
            ttnn.deallocate(a2); ttnn.deallocate(b2)
            print(f"  matmul-writer {m}x{k}x{nn}: {gbs} GB/s", flush=True)
        except Exception as e:                                         # noqa: BLE001
            r[f"mm_writer_{m}x{k}x{nn}_err"] = str(e)[:120]
    r["mm_writer_write_peak_GBs"] = best
    r["machine_balance_FLOP_per_byte"] = round(
        r["compute_square_4096_TFLOPs"] * 1e12 / (r["read_peak_GBs"] * 1e9), 1)
    print(f"  ROOFS read {r['read_peak_GBs']} copy {r['copy_peak_GBs']} "
          f"mm-write {best} GB/s  balance {r['machine_balance_FLOP_per_byte']} FLOP/B", flush=True)
    return r


# ---------------------------------------------------- B1/B2/B3 + L1b: the SDPA matrix, bias buffer
def cfg(chunk, grid=(11, 10)):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                  q_chunk_size=chunk, k_chunk_size=chunk)


def sdpa_matrix():
    r = {"shape": "q/k/v [298, 8, 320, 32], bias [1, 8, 320, 320]"}
    q, k, v = (T((M, NH, N, HD)) for _ in range(3))
    bias_d = T((1, NH, N, N), DRAM)
    bias_l = ttnn.to_memory_config(bias_d, L1)
    print(f"  bias DRAM buf={bias_d.memory_config().buffer_type} "
          f"bias L1 buf={bias_l.memory_config().buffer_type}", flush=True)
    r["bias_bytes"] = int(NH * N * N * 2)
    r["bias_l1_buffer_type"] = str(bias_l.memory_config().buffer_type)

    def run(mask, c):
        return timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=HD ** -0.5, program_config=cfg(c))),
            warm=2, pipe=3, reps=5)

    for c in (64, 320):
        for lbl, mask in (("bias_dram", bias_d), ("bias_l1", bias_l), ("nobias", None)):
            try:
                r[f"chunk{c}_{lbl}_us"] = us(run(mask, c))
            except Exception as e:                                     # noqa: BLE001
                r[f"chunk{c}_{lbl}_err"] = str(e)[:200]
            print(f"  chunk{c} {lbl}: {r.get(f'chunk{c}_{lbl}_us')}", flush=True)
    for c in (64, 320):
        for src in ("dram", "l1"):
            a, b = r.get(f"chunk{c}_bias_{src}_us"), r.get(f"chunk{c}_nobias_us")
            if a and b:
                r[f"chunk{c}_biasleg_{src}_us"] = round(a - b, 1)
                r[f"chunk{c}_biasleg_{src}_GBs"] = round(
                    M * NH * N * N * 2 / ((a - b) * 1e-6) / 1e9, 1)
    ttnn.deallocate(bias_l)

    # HD: halve the head count. If the L1-mask leg is bandwidth-bound it halves; if it sits at a
    # fixed reader-occupancy floor it does not.
    try:
        q4, k4, v4 = (T((M, 4, N, HD)) for _ in range(3))
        b4d = T((1, 4, N, N), DRAM)
        b4l = ttnn.to_memory_config(b4d, L1)

        def run4(mask, c):
            return timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=mask, is_causal=False, scale=HD ** -0.5,
                program_config=cfg(c))), warm=2, pipe=3, reps=5)
        for lbl, mask in (("bias_dram", b4d), ("bias_l1", b4l), ("nobias", None)):
            r[f"h4_chunk64_{lbl}_us"] = us(run4(mask, 64))
            print(f"  h4 chunk64 {lbl}: {r[f'h4_chunk64_{lbl}_us']}", flush=True)
        for src in ("dram", "l1"):
            r[f"h4_biasleg_{src}_us"] = round(
                r[f"h4_chunk64_bias_{src}_us"] - r["h4_chunk64_nobias_us"], 1)
        for t in (q4, k4, v4, b4d, b4l):
            ttnn.deallocate(t)
    except Exception as e:                                             # noqa: BLE001
        r["h4_err"] = str(e)[:200]

    # U1: the grid ladder -> core-equivalents, production chunk, production bias.
    lad = {}
    for g in ((1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (11, 10)):
        try:
            s = timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias_d, is_causal=False, scale=HD ** -0.5,
                program_config=cfg(64, g))), warm=1, pipe=2, reps=3)
            lad[f"{g[0]}x{g[1]}"] = us(s)
            print(f"  grid {g}: {us(s)} us", flush=True)
        except Exception as e:                                         # noqa: BLE001
            lad[f"{g[0]}x{g[1]}"] = str(e)[:80]
    r["grid_ladder_us"] = lad
    if isinstance(lad.get("1x1"), float) and isinstance(lad.get("11x10"), float):
        r["core_equivalents_of_110"] = round(lad["1x1"] / lad["11x10"], 1)
    for t in (q, k, v, bias_d):
        ttnn.deallocate(t)
    return r


# ------------------------------------------------------------- L3: the head split and the qkv proj
def head_split():
    r = {}
    ckc = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    xt = torch.randn(M, N, CZ)
    wt = torch.randn(CZ, 3 * NH * HD)
    x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)

    # L3d: the projection's own baseline on this card, exactly as production issues it.
    r["qkv_proj_us"] = us(timed(lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16))))
    r["qkv_proj_write_GBs"] = round(M * N * 3 * NH * HD * 2 / (r["qkv_proj_us"] * 1e-6) / 1e9, 1)
    r["qkv_proj_read_GBs"] = round(M * N * CZ * 2 / (r["qkv_proj_us"] * 1e-6) / 1e9, 1)
    r["qkv_proj_TFLOPs"] = round(2 * M * N * CZ * 3 * NH * HD / (r["qkv_proj_us"] * 1e-6) / 1e12, 2)
    print(f"  qkv proj: {r['qkv_proj_us']} us  {r['qkv_proj_TFLOPs']} TFLOP/s "
          f"write {r['qkv_proj_write_GBs']} GB/s", flush=True)

    qkv = ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w,
                                           compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    qkv_in = ttnn.unsqueeze(qkv, 1)
    r["split_us"] = us(timed(lambda: [ttnn.deallocate(o) for o in
                                      ttnn.experimental.nlp_create_qkv_heads(
                                          qkv_in, num_heads=NH, num_kv_heads=NH,
                                          transpose_k_heads=False, memory_config=DRAM)]))
    r["split_copy_GBs"] = round(2 * M * N * 3 * NH * HD * 2 / (r["split_us"] * 1e-6) / 1e9, 1)
    print(f"  split: {r['split_us']} us  {r['split_copy_GBs']} GB/s copy", flush=True)

    qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
        qkv_in, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)
    q_ref = ttnn.to_torch(qh)

    # L3a: permute the weight's OUTPUT COLUMNS head-major and check the projection is bit-exact
    # against the same permutation applied to the unpermuted output. A permutation of output
    # columns reorders nothing inside a dot product.
    perm = torch.arange(3 * NH * HD).reshape(3, NH, HD)[:, torch.arange(NH), :].reshape(-1)
    perm = torch.randperm(3 * NH * HD, generator=torch.Generator().manual_seed(0))
    wp = ttnn.from_torch(wt[:, perm].contiguous(), layout=ttnn.TILE_LAYOUT, device=DEV,
                         dtype=ttnn.bfloat16)
    qkv_p = ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=wp,
                                             compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    a_t = ttnn.to_torch(qkv)[..., perm]
    b_t = ttnn.to_torch(qkv_p)
    r["L3a_torch_equal"] = bool(torch.equal(a_t, b_t))
    r["L3a_n_differing"] = int((a_t != b_t).sum().item())
    r["L3a_max_abs_diff"] = float((a_t - b_t).abs().max().item())
    print(f"  L3a torch.equal={r['L3a_torch_equal']} differing={r['L3a_n_differing']}", flush=True)
    ttnn.deallocate(qkv_p); ttnn.deallocate(wp)

    # L3b: is the head split a reshape? Take the q third of the projection output and reshape it
    # to [M, NH, N, HD]; compare against nlp_create_qkv_heads' q.
    try:
        q_third = ttnn.to_torch(qkv)[:, :, :NH * HD]                  # [M, N, 256]
        cand = q_third.reshape(M, N, NH, HD).permute(0, 2, 1, 3).contiguous()  # the real transform
        naive = q_third.reshape(M, NH, N, HD)                          # what a reshape alone gives
        r["L3b_reshape_matches"] = bool(torch.equal(naive, q_ref))
        r["L3b_reshape_max_abs_diff"] = float((naive - q_ref).abs().max().item())
        r["L3b_permute_matches"] = bool(torch.equal(cand, q_ref))
        r["L3b_permute_max_abs_diff"] = float((cand - q_ref).abs().max().item())
        print(f"  L3b reshape_matches={r['L3b_reshape_matches']} "
              f"(max abs {r['L3b_reshape_max_abs_diff']:.4g}), "
              f"permute_matches={r['L3b_permute_matches']}", flush=True)
    except Exception as e:                                             # noqa: BLE001
        r["L3b_err"] = str(e)[:200]

    # L3c-i: the batched matmul that WOULD emit [M, NH, N, HD] directly -- in0 broadcast over the
    # head axis, in1 a per-head weight slice. If ttnn refuses the broadcast, the refusal is the
    # call signature that forbids the fusion.
    try:
        x4 = ttnn.reshape(x, (M, 1, N, CZ))
        w4 = ttnn.from_torch(wt[:, :NH * HD].t().reshape(NH, HD, CZ).permute(0, 2, 1)
                             .reshape(1, NH, CZ, HD).contiguous(),
                             layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(x4, w4, compute_kernel_config=ckc,
                                                      memory_config=DRAM)), warm=1, pipe=2, reps=3)
        r["L3c_batched_q_us"] = us(s)
        r["L3c_batched_qkv_us"] = round(us(s) * 3, 1)
        print(f"  L3c batched q: {r['L3c_batched_q_us']} us -> qkv {r['L3c_batched_qkv_us']}",
              flush=True)
        ttnn.deallocate(w4)
    except Exception as e:                                             # noqa: BLE001
        r["L3c_batched_err"] = f"{type(e).__name__}: {e}"[:300]
        print("  L3c batched refused:", r["L3c_batched_err"], flush=True)

    # L3c-ii: the fallback that always runs -- 24 narrow matmuls at output width nt=1, one per
    # (tensor, head). Priced from one, times 24.
    w1 = T((CZ, HD))
    s1 = timed(lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w1, compute_kernel_config=ckc, dtype=ttnn.bfloat16)))
    r["L3c_narrow_one_us"] = us(s1)
    r["L3c_narrow_24_us"] = round(us(s1) * 24, 1)
    r["L3c_narrow_TFLOPs"] = round(2 * M * N * CZ * HD / s1 / 1e12, 2)
    print(f"  L3c narrow nt=1: {r['L3c_narrow_one_us']} us each, x24 = "
          f"{r['L3c_narrow_24_us']} us, {r['L3c_narrow_TFLOPs']} TFLOP/s", flush=True)
    ttnn.deallocate(w1)

    for t in (qh, kh, vh, qkv_in, x, w):
        try:
            ttnn.deallocate(t)
        except Exception:                                              # noqa: BLE001
            pass
    return r


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "perf/p2_attention/attn_probe_c2.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    DEV = get_device()
    RES["grid"] = {"core_grid_main": str(CORE_GRID_MAIN)}
    for key, get in (("compute_with_storage", lambda: str(DEV.compute_with_storage_grid_size())),
                     ("l1_unreserved_per_core_B", lambda: int(l1_unreserved(DEV))),
                     ("l1_size_per_core_B", lambda: int(DEV.l1_size_per_core()))):
        try:
            RES["grid"][key] = get()
        except Exception as e:                                         # noqa: BLE001
            RES["grid"][key] = f"{type(e).__name__}: {e}"[:120]
    print(json.dumps(RES["grid"]), flush=True)
    todo = [("roofs", roofs), ("sdpa_matrix", sdpa_matrix), ("head_split", head_split)]
    for name, fn in todo:
        if a.only and name not in a.only.split(","):
            continue
        stage(name, fn, a.out)
    save(a.out)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
