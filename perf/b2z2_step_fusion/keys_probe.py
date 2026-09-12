#!/usr/bin/env python3
"""The atom key window is a contiguous slice, so it does not need a matmul to build it.

`AttentionPairBias.__call__`'s atom branch builds `s_kv`, the 128-atom key window for each
32-atom query window, with a one-hot matmul and four layout ops around it:

    reshape (1,140,32,128)->(1,280,16,128)   65.65 us   sub-tile split, the W/2 half windows
    permute            ->(1,16,128,280)      69.13 us
    matmul  x keys_indexing                  98.42 us
    permute            ->(1,1120,16,128)    192.93 us
    reshape            ->(1,140,128,128)    206.66 us

632.79 us and five programs, six times a step: 3.80 ms of the 41.482 ms WH step, 9.2 %, to move
bytes. `boltz2.get_indexing_matrix` says what the matmul computes:

    index[k,j] = clamp(j - 2k + h/2, 0, h+1), classes 1..h kept
    => s_kv[k, c*(W/2) + i, d] = single[2k + c + 1 - h/2, i, d]
    => s_kv[k, r, d] = flat[k*W + (W/2 - H/2) + r, d]      r = 0 .. H-1

which is a CONTIGUOUS window of H atoms starting W/2 - H/2 before the query window, zero outside
the sequence. So it is four tile-aligned slices of a once-shifted copy, concatenated -- no matmul,
no half-window split, no permute.

This screens that construction at the production shape against the shipped one for bit-exactness
and for time, before any model code is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

B, K, W, H, D = 1, 140, 32, 128, 128


def timed(ttnn, dev, fn, reps=20, blocks=5):
    for _ in range(4):
        fn()
    ttnn.synchronize_device(dev)
    meds = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        meds.append((time.perf_counter() - t0) / reps)
    return round(1e6 * st.median(meds), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    sys.path.insert(0, str(ROOT))
    from tt_bio.boltz2 import get_indexing_matrix

    dev = T.get_device()
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "B K W H D": [B, K, W, H, D]}}

    src = torch.randn(B, K, W, D)
    s = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    ki = get_indexing_matrix(K, W, H, torch.device("cpu"))
    out["keys_indexing_shape"] = list(ki.shape)

    ckc = T.get_compute_kernel_config() if hasattr(T, "get_compute_kernel_config") else None

    def stock(ki_tt):
        x = ttnn.reshape(s, (B, 2 * K, W // 2, -1))
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.matmul(x, ki_tt) if ckc is None else ttnn.matmul(
            x, ki_tt, compute_kernel_config=ckc)
        x = ttnn.permute(x, (0, 3, 1, 2))
        return ttnn.reshape(x, (B, K, -1, D))

    def window(pad_mode):
        nchunk = H // W
        front = H // 2 - W // 2
        nblk = K + nchunk - 1
        back = nblk * W - K * W - front
        flat = ttnn.reshape(s, (B, 1, K * W, D))
        if pad_mode == "tile":
            p = ttnn.pad(flat, [[0, 0], [0, 0], [front, back], [0, 0]], 0.0)
        else:
            rm = ttnn.to_layout(flat, ttnn.ROW_MAJOR_LAYOUT)
            rm = ttnn.pad(rm, [[0, 0], [0, 0], [front, back], [0, 0]], 0.0)
            p = ttnn.to_layout(rm, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        P = ttnn.reshape(p, (B, nblk, W, D))
        return ttnn.concat([P[:, c:c + K] for c in range(nchunk)], dim=2)

    for dt in ("bfloat4_b", "bfloat16"):
        ki_tt = ttnn.from_torch(ki, layout=ttnn.TILE_LAYOUT, dtype=getattr(ttnn, dt), device=dev)
        r = stock(ki_tt)
        out[f"stock_{dt}"] = {"shape": list(r.shape), "us": timed(ttnn, dev, lambda: stock(ki_tt))}
        out[f"stock_{dt}"]["torch"] = None
        globals()[f"_ref_{dt}"] = ttnn.to_torch(r)
        print(f"  stock[{dt}] {list(r.shape)}  {out[f'stock_{dt}']['us']:.2f} us", flush=True)

    ref4 = globals()["_ref_bfloat4_b"]
    ref16 = globals()["_ref_bfloat16"]
    out["stock_bf4_vs_bf16_bit_exact"] = bool(torch.equal(ref4, ref16))
    out["stock_bf4_vs_bf16_max_abs"] = float((ref4.float() - ref16.float()).abs().max())
    # what the gather SHOULD be, in torch, from the definition
    flat_t = src.reshape(B, K * W, D)
    front = H // 2 - W // 2
    padded = torch.zeros(B, K * W + 2 * front, D)
    padded[:, front:front + K * W] = flat_t
    exact = torch.stack([padded[:, k * W:k * W + H] for k in range(K)], dim=1)
    exact_bf = exact.to(torch.bfloat16).float()
    out["stock_bf4_vs_exact_max_abs"] = float((ref4.float() - exact_bf).abs().max())
    out["stock_bf4_vs_exact_bit_exact"] = bool(torch.equal(ref4.float(), exact_bf))

    out["window"] = {}
    for mode in ("tile", "row_major"):
        rec = {}
        try:
            g = window(mode)
            rec["shape"] = list(g.shape)
            gt = ttnn.to_torch(g)
            rec["bit_exact_vs_bf4"] = bool(torch.equal(gt, ref4))
            rec["max_abs_vs_bf4"] = float((gt.float() - ref4.float()).abs().max())
            rec["bit_exact_vs_exact"] = bool(torch.equal(gt.float(), exact_bf))
            rec["us"] = timed(ttnn, dev, lambda: window(mode))
        except Exception as e:                                            # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out["window"][mode] = rec
        print(f"  window[{mode}]: {json.dumps(rec)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
