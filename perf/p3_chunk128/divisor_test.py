#!/usr/bin/env python3
"""p3-chunk128: does the SDPA bias leg pay for a RAGGED chunk tail?

The chunk ladder at the fold's padded 320 axis shows the bias leg flat at chunks that DIVIDE 320
(64, 160, 320) and 1.67x more expensive at chunks that do not (96, 128). The mechanism hypothesis
is that the mask is fetched per (q_chunk, k_chunk) pair at the FULL chunk footprint, so a ragged
tail re-reads padding: bytes ~ ceil(N/c)^2 * c^2 instead of N^2.

The test: move the padded key axis to 384, where 96 and 128 divide exactly and 160 does not. If the
hypothesis holds the expensive rungs swap over. Same band (256 < len <= 384), same op, same card.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-chunk128 \
      python3 perf/p3_chunk128/divisor_test.py --out perf/p3_chunk128/divisor_c0.json
"""
import argparse, json, math, os, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
M, NH, HD = 298, 8, 32
DEV = None


def timed(fn, warm=2, pipe=3, reps=5):
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


def cfg(c):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False,
                                  q_chunk_size=c, k_chunk_size=c)


def T(shape):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=DEV,
                           dtype=ttnn.bfloat16, memory_config=DRAM)


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    DEV = get_device()
    res = {"meta": dict(host=os.uname().nodename, visible=os.environ.get("TT_VISIBLE_DEVICES"),
                        loadavg=os.getloadavg(),
                        model="bias bytes ~ ceil(N/c)^2 * c^2 * NH * M * 2")}
    for N in (320, 384):
        q, k, v = (T((M, NH, N, HD)) for _ in range(3))
        bias = T((1, NH, N, N))
        rows = {}
        for c in (64, 96, 128, 160, 192, 320, 384):
            if c > N:
                continue
            try:
                b = timed(lambda: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5,
                        program_config=cfg(c))))
                nb = timed(lambda: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=False, scale=HD ** -0.5,
                        program_config=cfg(c))))
                nchunk = math.ceil(N / c)
                padded = nchunk * nchunk * c * c
                rows[c] = dict(bias_us=round(b * 1e6, 1), nobias_us=round(nb * 1e6, 1),
                               biasleg_us=round((b - nb) * 1e6, 1),
                               divides=(N % c == 0), n_chunks=nchunk,
                               model_pad_ratio=round(padded / (N * N), 3),
                               nominal_biasleg_GBs=round(
                                   M * NH * N * N * 2 / (b - nb) / 1e9, 1),
                               model_biasleg_GBs=round(
                                   M * NH * padded * 2 / (b - nb) / 1e9, 1))
                print(f"  N={N} c={c:3d} divides={N % c == 0!s:5s} bias {rows[c]['bias_us']:8.1f} "
                      f"nobias {rows[c]['nobias_us']:8.1f} leg {rows[c]['biasleg_us']:8.1f} us  "
                      f"model_ratio {rows[c]['model_pad_ratio']}  "
                      f"model GB/s {rows[c]['model_biasleg_GBs']}", flush=True)
            except Exception as e:                                     # noqa: BLE001
                rows[c] = {"error": f"{type(e).__name__}: {e}"[:300]}
                print(f"  N={N} c={c}: ERR {rows[c]['error'][:120]}", flush=True)
        res[f"N{N}"] = rows
        for t in (q, k, v, bias):
            ttnn.deallocate(t)
        json.dump(res, open(args.out, "w"), indent=1, default=str)
    json.dump(res, open(args.out, "w"), indent=1, default=str)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
