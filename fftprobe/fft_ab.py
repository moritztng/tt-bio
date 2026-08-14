#!/usr/bin/env python3
"""The A/B the gate asks for, at the altitude a user sees, plus the A/A pair that licenses it.

Arm A is the DFT-by-matmul path from the feasibility pass: the same complex 2D transform built out
of stock ttnn ops, eight batched ttnn.matmul calls through DRAM. Arm B is the fused kernel. Same
host, same card, same data, same batch, back to back.

The A/A pair runs arm B twice with nothing changed. Any A/B delta smaller than the A/A spread is not
a result -- co-tenancy on this host has inverted rankings before.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

sys_path = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path))
from tt_bio.fft2d import DTYPES, build, dft_blocks, from_tiles, pack, to_tiles   # noqa: E402

N = int(os.environ.get("FFT_BOX", "256"))
NIMG = int(os.environ.get("FFT_NIMG", "16"))
DT = os.environ.get("FFT_DT", "bf16")
REPS = 5


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    dev = ttnn.open_device(device_id=0)
    tdt, tor, nb = DTYPES[DT]
    g = dev.compute_with_storage_grid_size()
    ncores = g.x * g.y
    NT = N // 32
    ntile = 2 * NT * NT
    total = ncores * NIMG
    out = {"box": N, "dtype": DT, "images": total, "ncores": ncores, "arms": {}}

    try:
        rng = np.random.default_rng(0)
        img = (rng.standard_normal((total, N, N)) + 1j * rng.standard_normal((total, N, N))) / N
        n = np.arange(N)
        F = np.exp(-2j * np.pi * np.outer(n, n) / N)
        ref = np.fft.fft2(img[:130], axes=(-2, -1))

        # ---- arm B: the fused kernel -------------------------------------------------------------
        ftt = ttnn.from_torch(pack(dft_blocks(N, tor)), dtype=tdt, layout=ttnn.TILE_LAYOUT,
                              device=dev)
        xtt = ttnn.from_torch(pack(to_tiles(img, tor)), dtype=tdt, layout=ttnn.TILE_LAYOUT,
                              device=dev)
        ott = ttnn.from_torch(pack(torch.zeros(total * ntile, 32, 32, dtype=tor)),
                              dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)
        pd = build(dev, N, NIMG, DT, ftt, xtt, ott)
        ttnn.generic_op([ftt, xtt, ott], pd)
        ttnn.synchronize_device(dev)

        for name in ("B_fused_1", "B_fused_2"):        # the A/A pair
            best = float("inf")
            for _ in range(REPS):
                t0 = time.perf_counter()
                ttnn.generic_op([ftt, xtt, ott], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            got = from_tiles(ttnn.to_torch(ott).reshape(-1, 32, 32)[:130 * ntile], 130, N)
            out["arms"][name] = {
                "s": best, "images_per_s": total / best,
                "rel_l2": float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
                "sha256_out": sha(ttnn.to_torch(ott).to(torch.float32).numpy()),
            }
            print(f"{name:12s} {total/best:10.0f} img/s  rel_l2 "
                  f"{out['arms'][name]['rel_l2']:.3e}  sha {out['arms'][name]['sha256_out']}",
                  flush=True)
        for t in (ftt, xtt, ott):
            ttnn.deallocate(t)

        # ---- arm A: DFT-by-matmul out of stock ttnn ops -------------------------------------------
        # Arm A, implemented competently. The first version broadcast the DFT matrix to the batch
        # shape and fed ttnn.matmul a batch of 130 small 256x256 products; it ran at 588 GFLOP/s,
        # 0.18% of the matmul roof measured in S2b, and it would have inflated the speedup by more
        # than an order of magnitude. A composite path deserves its best shot, so:
        #
        #   pass 1 flattens the batch into M, giving ONE matmul of M=B*N, K=N, N=N -- the shape
        #     ttnn.matmul is good at -- against a single resident copy of F.
        #   pass 2 needs a left multiply, F.T, which that trick cannot express. The DFT matrix is
        #     symmetric, so F.T = (T^t . F)^t, and the pass becomes transpose, the same flattened
        #     matmul, transpose back.
        #
        # Those two transposes are not an artefact of the rewrite. They are the inter-pass transpose
        # that every two-pass FFT pays and that the fused kernel avoids by construction, so they
        # belong in arm A's time.
        def T2(a):
            t = torch.from_numpy(np.ascontiguousarray(a)).to(tor)
            return ttnn.from_torch(t.reshape(1, 1, *t.shape[-2:]) if t.ndim == 2
                                   else t.reshape(1, 1, -1, t.shape[-1]),
                                   dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)

        def flip(t, b):
            """Per-image transpose of a flattened [1, 1, b*N, N] stack."""
            r = ttnn.reshape(t, (1, b, N, N))
            r = ttnn.transpose(r, -2, -1)
            return ttnn.reshape(r, (1, 1, b * N, N))

        SL = int(os.environ.get("FFT_AB_SLICE", "130"))
        tot_s, first = 0.0, None
        Fr, Fi = T2(F.real), T2(F.imag)
        for s0 in range(0, total, SL):
            sl = min(SL, total - s0)
            xr, xi = T2(img[s0:s0 + sl].real), T2(img[s0:s0 + sl].imag)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            tr = ttnn.sub(ttnn.matmul(xr, Fr), ttnn.matmul(xi, Fi))
            ti = ttnn.add(ttnn.matmul(xr, Fi), ttnn.matmul(xi, Fr))
            trT, tiT = flip(tr, sl), flip(ti, sl)
            yrT = ttnn.sub(ttnn.matmul(trT, Fr), ttnn.matmul(tiT, Fi))
            yiT = ttnn.add(ttnn.matmul(trT, Fi), ttnn.matmul(tiT, Fr))
            yr, yi = flip(yrT, sl), flip(yiT, sl)
            ttnn.synchronize_device(dev)
            tot_s += time.perf_counter() - t0
            if first is None:
                first = (ttnn.to_torch(yr).to(torch.float32).numpy().reshape(sl, N, N)[0],
                         ttnn.to_torch(yi).to(torch.float32).numpy().reshape(sl, N, N)[0])
            for t in (xr, xi, tr, ti, trT, tiT, yrT, yiT, yr, yi):
                ttnn.deallocate(t)
        gotA = (first[0] + 1j * first[1])[None]
        out["arms"]["A_dft_by_matmul"] = {
            "s": tot_s, "images_per_s": total / tot_s,
            "rel_l2": float(np.linalg.norm(gotA - ref[:1]) / np.linalg.norm(ref[:1])),
            "sha256_out": sha(np.stack(first)),
        }
        a = out["arms"]["A_dft_by_matmul"]
        print(f"A_dft_matmul {total/tot_s:10.0f} img/s  rel_l2 {a['rel_l2']:.3e}  "
              f"sha {a['sha256_out']}", flush=True)

        b1, b2 = out["arms"]["B_fused_1"]["images_per_s"], out["arms"]["B_fused_2"]["images_per_s"]
        out["aa_spread_pct"] = abs(b1 - b2) / max(b1, b2) * 100
        out["speedup_B_over_A"] = max(b1, b2) / a["images_per_s"]
        print(f"A/A spread {out['aa_spread_pct']:.2f}%   B/A speedup "
              f"{out['speedup_B_over_A']:.2f}x", flush=True)
        json.dump(out, open(sys_path / "fftprobe" / "fft2d_ab.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
