#!/usr/bin/env python3
"""Screens for opendde-to-4x, at OpenDDE's own 512 aa shapes on this card.

Kill gates WRITTEN HERE BEFORE THE RUN. A screen that cannot stop the build is not a screen.

S1. THE PARITY QUESTION `opendde-512aa-deep-perf` LEFT OPEN.
    Its section 4 swept `K_block` over the DIVISORS of 12 (1, 2, 3, 4, 6, 12) and concluded no
    `_MM_BLOCK` entry at kt=12 can be byte-identical to the unconfigured op. It never tried
    `K_block = 8`, which is what the unconfigured op ITSELF uses.
    `determine_default_block_sizes` in tt-metal's `minimal_matmul_program_factory.cpp` (read at tag
    v0.68.0, the wheel's version, from /home/ttuser/tt-metal) returns (M, K, N) = (8, 8, 8)
    unconditionally, subblocks (2, 2) when `fp32_dest_acc_en` (tt-bio's ckc sets it True) and (4, 2)
    otherwise; and `padded_K_tiles = round_up(K_tiles, K_block_tiles)`. So at kt=12 the DEFAULT runs
    TWO K blocks, 8 real tiles then 4 real + 4 padded. A non-divisor K_block is not illegal, it is
    the default. tt-bio's `_qkv_mm_config` refuses it only via its own `kt % blk[1]` guard.
    GATE: `config=(8,8,8,2,2)` must be `torch.equal` to `config=None` at BOTH (12,36) and (12,12).
    If it is not, the byte-identical K1 route is dead and only the release-gated `mm12` remains.

S2. DOES K1 FIRE ON A DEFAULT-EQUIVALENT ENTRY?
    K1's generic-op transcription was written for "the shipped block entry only", and every shipped
    entry has K_block == kt (one K block, no padding) and N_block == 1. (8, 8, 8) is two K blocks
    with K padding and a partial last N block.
    GATE: `qkv_heads` returns non-None, its (q, k, v) are `torch.equal` to
    `nlp_create_qkv_heads(minimal_matmul(config=default))`, and the fused op beats the two stock ops
    by >= 1.10x. Anything else and the entry buys K1 nothing.

S3. `Transition[4d, c=384]` -- 15.363 s of a 91.788 s fold, 16.7 %, never decomposed before today.
    `perf/odde4x/decomp_512.json`: 528 calls at 29.096 ms, synced both sides, two folds 0.013 s apart.
    Three matmuls are 927.6 GFLOP/call, so 29.096 ms is 31.9 TFLOP/s -- 31 % of card 3's MEASURED
    102.55 TFLOP/s HiFi4 roof and HALF the 61.7 TFLOP/s the trimul in-projection measures. Bytes are
    0.587 GB/call, 1.48 ms at 397 GB/s, so this is not bandwidth.
    PRIOR NEGATIVE, cited before proposing: `perfwar-chunked-transition-cb` tuned the transition
    matmuls' 2D PROGRAM CONFIG, measured 1.15x-1.79x op-isolated and NOTHING at the fold
    (opendde 298 aa 49.149 -> 49.282 s), and its own section 6.1 says the honest verdict is "no
    measurable gain, not exactly zero" with an upper bound of ~0.7 s/fold at 298 aa. That lever is
    also NOT bit-exact (in0_block_w moves the K-fold). THIS screen is a different lever: the ROW
    CHUNK boundary, which is row-local and therefore bit-exact by construction, and it attacks the
    per-chunk fixed cost rather than the matmul rate. `_ref / (w_eff * c)` = 131072 / (512 * 384) =
    0.667 shrinks `TRANSITION_H_CHUNK_SIZE` 16 -> 10, so 52 chunks per call; OpenDDE is the only
    model on Blackhole that factor shrinks at all.
    GATE: some h must take >= 3.0 ms/call off 29.1 (>= 1.58 s/fold over 528 calls, 121x the 0.013 s
    A/A floor) AND be `torch.equal` to h=10. Below that a shared default's blast radius is not
    worth it.

S4. The inherited 0.82 s/fold `channel_move_back` leg (predecessor section 9.7, never screened).
    GATE: three 128-channel move-backs plus the concat must beat one 384-channel move-back by
    > 0.3 ms/call.

S5. Where the 29.096 ms actually goes, per op, at h=10 -- so the residual has a mechanism and the
    S3 prediction is not a guess. No gate; this one is attribution.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WARM, REPS = 2, 7
DEV = [None]


def _free(o):
    import ttnn
    if o is None:
        return
    for t in (o if isinstance(o, (list, tuple)) else (o,)):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass


def bench(fn, free=True):
    """Median ms over REPS after WARM, synced both sides, freeing every produced tensor."""
    import ttnn
    for _ in range(WARM):
        o = fn()
        if free:
            _free(o)
    ttnn.synchronize_device(DEV[0])
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(DEV[0])
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(DEV[0])
        ts.append((time.perf_counter() - t0) * 1e3)
        if free:
            _free(o)
    return round(st.median(ts), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="s1,s2,s3,s4,s5")
    a = ap.parse_args()
    only = set(a.only.split(","))

    import importlib.metadata as im
    import torch, ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
    dev = DEV[0] = get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [gx, gy], "ttnn": im.version("ttnn"),
           "loadavg": open("/proc/loadavg").read().split()[:3]}
    torch.manual_seed(0)

    S, CZ = 512, 384

    def cfg(M, K, N, sh, sw):
        return ttnn.MinimalMatmulConfig(
            M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))

    # ---- S1: is the op's own default expressible as a _MM_BLOCK entry, byte-identically? -----
    if "s1" in only:
        rows = []
        x = ttnn.from_torch(torch.randn(1, S * S, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        for nt, N in ((36, 1152), (12, 384)):
            w = ttnn.from_torch(torch.randn(CZ, N), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
            mm = lambda c: ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=c)
            base = mm(None)
            base_t = ttnn.to_torch(base)
            _free(base)
            t_base = bench(lambda: mm(None))
            for c4 in ((8, 8, 8, 2, 2), (8, 8, 8, 4, 2), (4, 8, 8, 2, 2), (8, 8, 1, 2, 2),
                       (8, 12, 8, 2, 2), (8, 8, 8, 2, 4)):
                row = {"kt": 12, "nt": nt, "cfg": list(c4), "base_ms": t_base}
                try:
                    c = cfg(*c4)
                    o = mm(c)
                    ot = ttnn.to_torch(o)
                    _free(o)
                    row["torch_equal"] = bool(torch.equal(ot, base_t))
                    row["max_abs"] = float((ot - base_t).abs().max())
                    row["ms"] = bench(lambda c=c: mm(c))
                    row["ratio"] = round(t_base / row["ms"], 4)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:200]
                rows.append(row)
                print(" S1", row, flush=True)
            _free(w)
        _free(x)
        res["s1_default_equivalence"] = rows

    # ---- S2: K1 head-major qkv on the default-equivalent entry ------------------------------
    if "s2" in only:
        import tt_bio.triatt_qkv as K1
        rows = []
        w_qkv = ttnn.from_torch(torch.randn(CZ, 1152), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
        x3 = ttnn.from_torch(torch.randn(1, S * S, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        saved = dict(T._MM_BLOCK)
        for entry in ((8, 8, 8, 2, 2), (12, 12, 1, 2, 2)):
            T._MM_BLOCK[(12, 36)] = entry
            K1.STATS[:] = [0, 0]
            K1.REJECTS.clear()
            mm_cfg = T._qkv_mm_config(x3, w_qkv)
            row = {"entry": list(entry), "qkv_mm_config_is_none": mm_cfg is None}
            if mm_cfg is not None:
                def stock_pair():
                    s = ttnn.experimental.minimal_matmul(
                        input_tensor=x3, weight_tensor=w_qkv, compute_kernel_config=ckc,
                        dtype=ttnn.bfloat16, config=mm_cfg)
                    o = ttnn.experimental.nlp_create_qkv_heads(
                        ttnn.unsqueeze(s, 1), num_heads=12, num_kv_heads=12,
                        transpose_k_heads=False)
                    _free(s)
                    return o
                sref = stock_pair()
                ref = [ttnn.to_torch(t) for t in sref]
                _free(sref)
                out = K1.qkv_heads(x3, w_qkv, ckc, 12, 32, ttnn.bfloat16, mm_cfg)
                row["k1_served"] = out is not None
                row["rejects"] = {str(k): v for k, v in K1.REJECTS.items()}
                if out is not None:
                    got = [ttnn.to_torch(t) for t in out]
                    _free(out)
                    row["torch_equal"] = [bool(torch.equal(g, r)) for g, r in zip(got, ref)]
                    row["max_abs"] = [float((g - r).abs().max()) for g, r in zip(got, ref)]
                    row["k1_ms"] = bench(lambda: K1.qkv_heads(
                        x3, w_qkv, ckc, 12, 32, ttnn.bfloat16, mm_cfg))
                    row["stock_ms"] = bench(stock_pair)
                    row["ratio"] = round(row["stock_ms"] / row["k1_ms"], 4)
            rows.append(row)
            print(" S2", row, flush=True)
        T._MM_BLOCK.clear()
        T._MM_BLOCK.update(saved)
        _free(w_qkv)
        _free(x3)
        res["s2_k1_on_default_entry"] = rows

    # ---- S3 / S5: the Transition row chunk, and where its 29.096 ms goes ---------------------
    if "s3" in only or "s5" in only:
        H4 = 4 * CZ
        sd = {"norm.weight": torch.randn(CZ), "norm.bias": torch.randn(CZ),
              "fc1.weight": torch.randn(H4, CZ), "fc2.weight": torch.randn(H4, CZ),
              "fc3.weight": torch.randn(CZ, H4)}
        tr = T.Transition(sd, ckc)
        z = ttnn.from_torch(torch.randn(1, S, S, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        gflop = 3 * 2 * (S * S) * CZ * H4 / 1e9
        if "s3" in only:
            rows, ref = [], None
            for h in (10, 12, 16, 20, 26, 32, 44, 64):
                T.TRANSITION_H_CHUNK_SIZE = h
                eff = max(1, int(h * min(1.0, (1024 * 128) / (S * CZ))))
                row = {"h_const": h, "h_effective": eff, "n_chunks": -(-S // eff)}
                try:
                    o = tr(z)
                    ot = ttnn.to_torch(o)
                    _free(o)
                    if ref is None:
                        ref = ot
                    row["torch_equal_vs_h10"] = bool(torch.equal(ot, ref))
                    row["max_abs"] = float((ot - ref).abs().max())
                    row["ms"] = bench(lambda: tr(z))
                    row["tflops"] = round(gflop / (row["ms"] * 1e-3) / 1e3, 2)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:200]
                rows.append(row)
                print(" S3", row, flush=True)
            T.TRANSITION_H_CHUNK_SIZE = 16
            res["s3_transition_row_chunk"] = {"gflop_per_call": round(gflop, 1), "rows": rows}

        if "s5" in only:
            # One row chunk, op by op, at the shipped h=10 and at h=32.
            legs = {}
            for h in (10, 32):
                c = z[:, 0:h]
                xn = ttnn.layer_norm(c, weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5,
                                     compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
                lin = lambda w, mc, act=None: ttnn.linear(
                    xn, w, activation=act, compute_kernel_config=ckc, memory_config=mc,
                    dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)
                x1 = lin(tr.fc1_weight, ttnn.L1_MEMORY_CONFIG, "silu")
                x2 = lin(tr.fc2_weight, ttnn.L1_MEMORY_CONFIG)
                prod = ttnn.multiply(x1, x2, memory_config=ttnn.L1_MEMORY_CONFIG)
                legs[f"h{h}"] = {
                    "slice_ms": bench(lambda h=h: z[:, 0:h]),
                    "layer_norm_ms": bench(lambda: ttnn.layer_norm(
                        c, weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5,
                        compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)),
                    "fc1_silu_ms": bench(lambda: lin(tr.fc1_weight, ttnn.L1_MEMORY_CONFIG, "silu")),
                    "fc2_ms": bench(lambda: lin(tr.fc2_weight, ttnn.L1_MEMORY_CONFIG)),
                    "multiply_ms": bench(lambda: ttnn.multiply(
                        x1, x2, memory_config=ttnn.L1_MEMORY_CONFIG)),
                    "fc3_dram_ms": bench(lambda: ttnn.linear(
                        prod, tr.fc3_weight, compute_kernel_config=ckc,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                        core_grid=T.CORE_GRID_MAIN)),
                    "n_chunks_at_this_h": -(-S // h),
                }
                legs[f"h{h}"]["sum_ms"] = round(sum(
                    v for k, v in legs[f"h{h}"].items() if k.endswith("_ms")), 4)
                legs[f"h{h}"]["projected_call_ms"] = round(
                    legs[f"h{h}"]["sum_ms"] * legs[f"h{h}"]["n_chunks_at_this_h"], 3)
                _free([xn, x1, x2, prod, c])
                print(" S5", h, legs[f"h{h}"], flush=True)
            res["s5_per_op"] = legs
        _free(z)

    # ---- S4: channel_move_back at width 384 vs three 128-wide passes ------------------------
    if "s4" in only:
        c = ttnn.from_torch(torch.randn(1, CZ, S, S), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        mc = ttnn.DRAM_MEMORY_CONFIG
        row = {}
        try:
            o = T._channel_move_back(c, mc)
            ref = ttnn.to_torch(o)
            _free(o)
            row["wide_384_ms"] = bench(lambda: T._channel_move_back(c, mc))

            def sliced():
                parts = [T._channel_move_back(c[:, w:w + 128], mc) for w in range(0, CZ, 128)]
                out = ttnn.concat(parts, dim=-1)
                _free(parts)
                return out
            o2 = sliced()
            got = ttnn.to_torch(o2)
            _free(o2)
            row["sliced_3x128_ms"] = bench(sliced)
            row["delta_ms"] = round(row["wide_384_ms"] - row["sliced_3x128_ms"], 4)
            row["shapes_match"] = list(got.shape) == list(ref.shape)
            row["torch_equal"] = bool(row["shapes_match"] and torch.equal(got, ref))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        _free(c)
        res["s4_channel_move_back"] = row
        print(" S4", row, flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
