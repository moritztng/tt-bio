#!/usr/bin/env python3
"""p3-sdpa probes: roofs on THIS card, the SDPA chunk A/B, the block wall, the alignment
overlap, and the L1-resident qkv split (deliverable 3).

Shapes are the ones a live 298 aa protenix-v2 fold issues: pair [298, 320, 256] padded from
logical 298, q/k/v [298, 8, 320, 32], bias [1, 8, 320, 320]. Device synchronised on BOTH sides
of every timed region. Results stream to --out so a timeout leaves finished stages on disk.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-sdpa \
      python3 perf/p3_sdpa/p3_probe.py --out perf/p3_sdpa/probe_c0.json
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
M, N, CZ, NH, HD = 298, 320, 256, 8, 32
CALLS = 1048          # counted in my own live fold, 2 x 524
RES = {}
DEV = None
CKC = None


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


def msfold(us_per_call, calls=CALLS):
    return round(us_per_call * calls / 1000.0, 1)


def T(shape, mc=DRAM, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=DEV,
                           dtype=dt, memory_config=mc)


def cfg(chunk, grid=(11, 10)):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                  q_chunk_size=chunk, k_chunk_size=chunk)


def stage(name, fn, out):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    try:
        RES[name] = fn()
    except Exception as e:                                             # noqa: BLE001
        RES[name] = {"error": f"{type(e).__name__}: {e}"[:500]}
        print("  ERR", RES[name]["error"], flush=True)
    print(f"  [{time.perf_counter() - t0:.1f}s]", flush=True)
    save(out)


# ---------------------------------------------------------------------------- roofs, on card 0
def roofs():
    r = {"card": f"qb1 TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}",
         "note": "measured this pass on this card, not inherited"}
    n = 4096
    a, b = T((1, 1, n, n)), T((1, 1, n, n))
    s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
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
            row["l1_to_l1_GBs"] = round(2 * nb / timed(lambda: ttnn.deallocate(
                ttnn.clone(xl, memory_config=L1)), warm=2, pipe=3, reps=5) / 1e9, 1)
            ttnn.deallocate(xl)
        except Exception as e:                                         # noqa: BLE001
            row["l1_err"] = str(e)[:80]
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
    r["dram"] = rows
    r["read_peak_GBs"] = max(x.get("read_GBs", 0) for x in rows)
    r["copy_peak_GBs"] = max(x.get("copy_rw_GBs", 0) for x in rows)
    r["clone_write_peak_GBs"] = max(x.get("write_GBs", 0) for x in rows)
    r["l1_to_l1_peak_GBs"] = max(x.get("l1_to_l1_GBs", 0) for x in rows)

    best = 0.0
    for (m, k, nn) in ((M * N, 32, 2048), (M * N, 32, 4096)):
        try:
            a2, b2 = T((m, k), L1), T((k, nn), L1)
            s2 = timed(lambda: ttnn.deallocate(ttnn.matmul(a2, b2, compute_kernel_config=CKC,
                                                           memory_config=DRAM)),
                       warm=2, pipe=3, reps=5)
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
    print(f"  ROOFS read {r['read_peak_GBs']} copy {r['copy_peak_GBs']} l1l1 "
          f"{r['l1_to_l1_peak_GBs']} mm-write {best} GB/s "
          f"balance {r['machine_balance_FLOP_per_byte']} FLOP/B", flush=True)
    return r


# ------------------------------------------- the SDPA chunk A/B, the legs, the grid, the alignment
def sdpa_ab():
    r = {"shape": "q/k/v [298, 8, 320, 32], bias [1, 8, 320, 320] DRAM", "calls_per_fold": CALLS}
    q, k, v = (T((M, NH, N, HD)) for _ in range(3))
    bias = T((1, NH, N, N), DRAM)

    def run(mask, c, grid=(11, 10), qq=q, kk=k, vv=v):
        return timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=mask, is_causal=False, scale=HD ** -0.5,
            program_config=cfg(c, grid))), warm=2, pipe=3, reps=5)

    for c in (64, 320):
        for lbl, mask in (("bias", bias), ("nobias", None)):
            r[f"chunk{c}_{lbl}_us"] = us(run(mask, c))
            print(f"  chunk{c} {lbl}: {r[f'chunk{c}_{lbl}_us']} us", flush=True)
        r[f"chunk{c}_biasleg_us"] = round(r[f"chunk{c}_bias_us"] - r[f"chunk{c}_nobias_us"], 1)
        r[f"chunk{c}_biasleg_GBs"] = round(
            M * NH * N * N * 2 / (r[f"chunk{c}_biasleg_us"] * 1e-6) / 1e9, 1)
    r["delta_us_per_call"] = round(r["chunk64_bias_us"] - r["chunk320_bias_us"], 1)
    r["probe_ms_per_fold"] = msfold(r["delta_us_per_call"])
    r["core_leg_speedup"] = round(r["chunk64_nobias_us"] / r["chunk320_nobias_us"], 2)
    print(f"  DELTA {r['delta_us_per_call']} us/call -> {r['probe_ms_per_fold']} ms/fold",
          flush=True)

    # U1: grid ladder at BOTH chunk sizes -> core-equivalents of the 110-core grid.
    for c in (64, 320):
        lad = {}
        for g in ((1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (11, 10)):
            try:
                lad[f"{g[0]}x{g[1]}"] = us(timed(
                    lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                        program_config=cfg(c, g))), warm=1, pipe=2, reps=3))
            except Exception as e:                                     # noqa: BLE001
                lad[f"{g[0]}x{g[1]}"] = str(e)[:80]
        r[f"grid_ladder_chunk{c}_us"] = lad
        if isinstance(lad.get("1x1"), float) and isinstance(lad.get("11x10"), float):
            r[f"core_equivalents_chunk{c}_of_110"] = round(lad["1x1"] / lad["11x10"], 1)
        print(f"  chunk{c} core-equivalents: {r.get(f'core_equivalents_chunk{c}_of_110')}",
              flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    return r


def alignment():
    """p3-align-widen's 42.3 ms/fold: the key axis is logically 298 inside a 320-padded buffer.

    Same A/B they ran (logical 298 vs logical 320 at a fixed padded 320), but at BOTH chunk
    sizes, so the CTO can see which of the two figures survives if chunk 320 lands.
    """
    r = {"method": "logical key length 298 vs 320, both padded to 320 tiles, same program config"}
    for c in (64, 320):
        for klog in (298, 320):
            q = T((M, NH, N, HD))
            kk = T((M, NH, klog, HD))
            vv = T((M, NH, klog, HD))
            bias = T((1, NH, N, klog), DRAM)
            try:
                s = timed(lambda: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, kk, vv, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                        program_config=cfg(c))), warm=2, pipe=3, reps=5)
                r[f"chunk{c}_klogical{klog}_us"] = us(s)
            except Exception as e:                                     # noqa: BLE001
                r[f"chunk{c}_klogical{klog}_err"] = str(e)[:200]
            for t in (q, kk, vv, bias):
                ttnn.deallocate(t)
        a, b = r.get(f"chunk{c}_klogical298_us"), r.get(f"chunk{c}_klogical320_us")
        if a and b:
            r[f"chunk{c}_penalty_us"] = round(a - b, 2)
            r[f"chunk{c}_penalty_ms_per_fold"] = msfold(a - b)
            r[f"chunk{c}_penalty_ratio"] = round(a / b, 3)
            print(f"  chunk{c}: 298 {a} vs 320 {b} us -> penalty "
                  f"{r[f'chunk{c}_penalty_ms_per_fold']} ms/fold", flush=True)
    return r


# ------------------------------------------------------- deliverable 3: the L1-resident qkv split
def l1_split():
    """Projection + head split, measured as a PAIR, DRAM-output baseline vs L1-output arm.

    The full [298, 320, 768] qkv output is 146.5 MB and cannot be L1-resident, so the L1 arm
    chunks the ROW axis: each chunk projects into L1 and splits L1->L1, then the SDPA-shaped
    outputs go back to DRAM (which is where the SDPA reads them from either way).
    """
    r = {"shape": f"x [{M}, {N}, {CZ}] -> qkv [{M}, {N}, {3 * NH * HD}] -> 3 x [{M}, {NH}, {N}, {HD}]"}
    xt = torch.randn(M, N, CZ)
    wt = torch.randn(CZ, 3 * NH * HD)
    x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.bfloat16)
    r["qkv_bytes"] = M * N * 3 * NH * HD * 2

    # --- baseline: production, both ops DRAM->DRAM, whole tensor -----------------------------
    def base():
        qkv = ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w,
                                               compute_kernel_config=CKC, dtype=ttnn.bfloat16)
        qi = ttnn.unsqueeze(qkv, 1)
        outs = ttnn.experimental.nlp_create_qkv_heads(
            qi, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)
        for o in outs:
            ttnn.deallocate(o)

    r["pair_dram_us"] = us(timed(base, warm=2, pipe=3, reps=5))
    r["pair_dram_ms_per_fold"] = msfold(r["pair_dram_us"])
    print(f"  baseline pair (proj+split, DRAM): {r['pair_dram_us']} us "
          f"= {r['pair_dram_ms_per_fold']} ms/fold", flush=True)

    # the two halves separately, for the ledger
    r["proj_dram_us"] = us(timed(lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, compute_kernel_config=CKC, dtype=ttnn.bfloat16))))
    qkv_full = ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w,
                                                compute_kernel_config=CKC, dtype=ttnn.bfloat16)
    qi_full = ttnn.unsqueeze(qkv_full, 1)
    r["split_dram_us"] = us(timed(lambda: [ttnn.deallocate(o) for o in
                                           ttnn.experimental.nlp_create_qkv_heads(
                                               qi_full, num_heads=NH, num_kv_heads=NH,
                                               transpose_k_heads=False, memory_config=DRAM)]))
    r["proj_dram_ms_per_fold"] = msfold(r["proj_dram_us"])
    r["split_dram_ms_per_fold"] = msfold(r["split_dram_us"])
    print(f"  proj {r['proj_dram_us']} us / split {r['split_dram_us']} us", flush=True)
    ref = [ttnn.to_torch(o) for o in ttnn.experimental.nlp_create_qkv_heads(
        qi_full, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)]

    # --- L1 arm: chunk the row axis, project into L1, split L1->L1 ---------------------------
    for cw in (32, 64, 96):
        key = f"l1_chunk{cw}"
        try:
            def arm(cw=cw):
                outs = []
                for s0 in range(0, M, cw):
                    xs = x[s0:min(s0 + cw, M)]
                    xl = ttnn.to_memory_config(xs, L1)
                    ttnn.deallocate(xs)
                    qkv_c = ttnn.experimental.minimal_matmul(
                        input_tensor=xl, weight_tensor=w, compute_kernel_config=CKC,
                        dtype=ttnn.bfloat16, memory_config=L1)
                    ttnn.deallocate(xl)
                    qi = ttnn.unsqueeze(qkv_c, 1)
                    three = ttnn.experimental.nlp_create_qkv_heads(
                        qi, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False,
                        memory_config=L1)
                    ttnn.deallocate(qi)
                    outs.append([ttnn.to_memory_config(t, DRAM) for t in three])
                    for t in three:
                        ttnn.deallocate(t)
                for grp in outs:
                    for t in grp:
                        ttnn.deallocate(t)

            r[f"{key}_us"] = us(timed(arm, warm=2, pipe=2, reps=5))
            r[f"{key}_ms_per_fold"] = msfold(r[f"{key}_us"])
            r[f"{key}_vs_baseline"] = round(r["pair_dram_us"] / r[f"{key}_us"], 3)
            print(f"  L1 arm chunk {cw}: {r[f'{key}_us']} us = {r[f'{key}_ms_per_fold']} "
                  f"ms/fold, {r[f'{key}_vs_baseline']}x vs baseline", flush=True)
        except Exception as e:                                         # noqa: BLE001
            r[f"{key}_err"] = f"{type(e).__name__}: {e}"[:300]
            print(f"  L1 arm chunk {cw} FAILED: {r[f'{key}_err']}", flush=True)

    # --- D3a: the split alone, L1->L1 vs DRAM->DRAM, at a 32-row chunk -----------------------
    try:
        xs = x[0:32]
        xl = ttnn.to_memory_config(xs, L1)
        qkv_l = ttnn.experimental.minimal_matmul(input_tensor=xl, weight_tensor=w,
                                                 compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                                                 memory_config=L1)
        qi_l = ttnn.unsqueeze(qkv_l, 1)
        qkv_d = ttnn.to_memory_config(qkv_l, DRAM)
        qi_d = ttnn.unsqueeze(qkv_d, 1)
        s_l1 = timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
            qi_l, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=L1)])
        s_dr = timed(lambda: [ttnn.deallocate(o) for o in ttnn.experimental.nlp_create_qkv_heads(
            qi_d, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)])
        nb = 2 * 32 * N * 3 * NH * HD * 2
        r["split32_l1_us"], r["split32_dram_us"] = us(s_l1), us(s_dr)
        r["split32_l1_GBs"] = round(nb / s_l1 / 1e9, 1)
        r["split32_dram_GBs"] = round(nb / s_dr / 1e9, 1)
        r["split32_speedup"] = round(s_dr / s_l1, 2)
        print(f"  D3a split32 L1 {r['split32_l1_us']} us ({r['split32_l1_GBs']} GB/s) vs DRAM "
              f"{r['split32_dram_us']} us ({r['split32_dram_GBs']} GB/s) = "
              f"{r['split32_speedup']}x", flush=True)

        # D3b: an L1 output must be bit-exact against a DRAM output -- the split is an index move.
        a_l1 = [ttnn.to_torch(o) for o in ttnn.experimental.nlp_create_qkv_heads(
            qi_l, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=L1)]
        a_dr = [ttnn.to_torch(o) for o in ttnn.experimental.nlp_create_qkv_heads(
            qi_d, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=DRAM)]
        r["split32_torch_equal"] = all(bool(torch.equal(a, b)) for a, b in zip(a_l1, a_dr))
        r["split32_n_differing"] = int(sum(int((a != b).sum()) for a, b in zip(a_l1, a_dr)))
        r["split32_n_elements"] = int(sum(a.numel() for a in a_l1))
        r["split32_max_abs_diff"] = float(max((a - b).abs().max() for a, b in zip(a_l1, a_dr)))
        print(f"  D3b torch.equal={r['split32_torch_equal']} differing="
              f"{r['split32_n_differing']}/{r['split32_n_elements']}", flush=True)
        for t in (qi_l, qi_d, qkv_l, qkv_d, xl):
            try:
                ttnn.deallocate(t)
            except Exception:                                          # noqa: BLE001
                pass
    except Exception as e:                                             # noqa: BLE001
        r["split32_err"] = f"{type(e).__name__}: {e}"[:300]
        print("  D3a/D3b failed:", r["split32_err"], flush=True)

    # --- the whole L1 chunked pair, checked against the production output --------------------
    try:
        cw = 32
        got = [[], [], []]
        for s0 in range(0, M, cw):
            xs = x[s0:min(s0 + cw, M)]
            xl = ttnn.to_memory_config(xs, L1)
            qkv_c = ttnn.experimental.minimal_matmul(
                input_tensor=xl, weight_tensor=w, compute_kernel_config=CKC,
                dtype=ttnn.bfloat16, memory_config=L1)
            qi = ttnn.unsqueeze(qkv_c, 1)
            three = ttnn.experimental.nlp_create_qkv_heads(
                qi, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False, memory_config=L1)
            for i, t in enumerate(three):
                got[i].append(ttnn.to_torch(t))
            for t in list(three) + [qi, xl, xs]:
                try:
                    ttnn.deallocate(t)
                except Exception:                                      # noqa: BLE001
                    pass
        cat = [torch.cat(g, 0) for g in got]
        r["chunked_pair_torch_equal"] = all(bool(torch.equal(a, b)) for a, b in zip(cat, ref))
        r["chunked_pair_max_abs_diff"] = float(max((a - b).abs().max() for a, b in zip(cat, ref)))
        r["chunked_pair_n_differing"] = int(sum(int((a != b).sum()) for a, b in zip(cat, ref)))
        print(f"  chunked pair vs production: torch.equal="
              f"{r['chunked_pair_torch_equal']} max abs {r['chunked_pair_max_abs_diff']}",
              flush=True)
    except Exception as e:                                             # noqa: BLE001
        r["chunked_pair_err"] = f"{type(e).__name__}: {e}"[:300]

    for t in (qi_full, qkv_full, x, w):
        try:
            ttnn.deallocate(t)
        except Exception:                                              # noqa: BLE001
            pass
    return r


def main():
    global DEV, CKC
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "perf/p3_sdpa/probe_c0.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    DEV = get_device()
    CKC = ttnn.init_device_compute_kernel_config(
        DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    RES["grid"] = {"core_grid_main": str(CORE_GRID_MAIN),
                   "loadavg": os.getloadavg(),
                   "compute_with_storage": str(DEV.compute_with_storage_grid_size())}
    print(json.dumps(RES["grid"]), flush=True)
    for name, fn in (("roofs", roofs), ("sdpa_ab", sdpa_ab), ("alignment", alignment),
                     ("l1_split", l1_split)):
        if a.only and name not in a.only.split(","):
            continue
        stage(name, fn, a.out)
    save(a.out)
    print("\nwrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
