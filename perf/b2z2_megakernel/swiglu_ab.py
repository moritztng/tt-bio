#!/usr/bin/env python3
"""The fused SwiGLU against the three ops it replaces: same numbers, and how much faster.

One 16-row chunk of the 512 aa pair transition -- x_norm [1,16,512,128] bf16 L1, fc1/fc2
[128,512] bf16 -- which is exactly what `Transition.__call__` hands the kernel 32 times a call.

Parity first, then a paired interleaved A/B in one process with its own A/A floor. pc card 0 is
documented to miscompute matmuls at a low, location-keyed rate (`pc-card0-512aa-fold-nondeterminism`),
so the parity leg runs the SAME arm N times and reports the incumbent's own repeat-to-repeat
agreement next to the fused-vs-incumbent one. A difference that is not larger than the incumbent's
own noise is not attributable to the kernel.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--parity-reps", type=int, default=3)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.transition_swiglu as TS

    dev = T.get_device()
    from tt_bio.af2 import compute_kernel_config
    ckc = T.trunk_compute_kernel_config(compute_kernel_config())
    grid = tuple(T.COMPUTE_GRID_MAIN)
    out = {"grid": list(grid), "cores": grid[0] * grid[1],
           "shape": [1, a.rows, a.n, a.c], "hidden": a.hidden, "reps": a.reps,
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    torch.manual_seed(0)
    xt = torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5
    w1t = torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05
    w2t = torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05
    up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=dev, memory_config=mc)
    x = up(xt, ttnn.L1_MEMORY_CONFIG)
    w1 = up(w1t, ttnn.DRAM_MEMORY_CONFIG)
    w2 = up(w2t, ttnn.DRAM_MEMORY_CONFIG)

    def incumbent():
        x1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                         memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
        x2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                         memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
        r = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        return r

    TS.set_enabled(True)
    TS.BLOCK_KEYS = {**TS.BLOCK_KEYS, **TS.DIAG_KEYS}

    def fused():
        r = TS.fused_swiglu(x, w2, w1, T._mm_ckc(ckc), grid)
        if r is None:
            raise RuntimeError("declined: " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
        return r

    # ---- parity -------------------------------------------------------------------------
    ref = []
    for _ in range(a.parity_reps):
        r = incumbent()
        ref.append(ttnn.to_torch(r).float())
        ttnn.deallocate(r)
    got = []
    for _ in range(a.parity_reps):
        r = fused()
        got.append(ttnn.to_torch(r).float())
        ttnn.deallocate(r)

    def cmp(u, v):
        d = (u - v).abs()
        num = (u * v).sum() - u.sum() * v.sum() / u.numel()
        den = ((u * u).sum() - u.sum() ** 2 / u.numel()).sqrt() * \
              ((v * v).sum() - v.sum() ** 2 / v.numel()).sqrt()
        return {"equal": bool(torch.equal(u, v)), "max_abs": float(d.max()),
                "mismatched": int((u != v).sum()), "of": int(u.numel()),
                "pcc": float(num / den) if float(den) else None}

    out["parity"] = {
        "incumbent_self": [cmp(ref[0], r) for r in ref[1:]],
        "fused_self": [cmp(got[0], g) for g in got[1:]],
        "fused_vs_incumbent": [cmp(ref[0], g) for g in got],
    }
    print("parity  incumbent-self " + json.dumps(out["parity"]["incumbent_self"]), flush=True)
    print("parity  fused-vs-incumbent " + json.dumps(out["parity"]["fused_vs_incumbent"]),
          flush=True)

    # ---- paired interleaved A/B --------------------------------------------------------
    def timed(fn, inner=4):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        return 1e3 * (time.perf_counter() - t0) / inner

    A, B, AA = [], [], []
    for _ in range(a.reps):
        A.append(timed(incumbent))
        B.append(timed(fused))
        AA.append(timed(incumbent))
    ma, mb, maa = st.median(A), st.median(B), st.median(AA)
    out["ab"] = {"incumbent_ms": round(ma, 4), "fused_ms": round(mb, 4),
                 "aa_ms": round(maa, 4), "ratio": round(ma / mb, 4),
                 "aa_floor_pct": round(100 * abs(maa - ma) / ma, 3),
                 "A": [round(v, 4) for v in A], "B": [round(v, 4) for v in B],
                 "AA": [round(v, 4) for v in AA]}
    print("A/B  incumbent %.4f  fused %.4f  ratio %.4fx  A/A floor %.2f %%"
          % (ma, mb, ma / mb, out["ab"]["aa_floor_pct"]), flush=True)
    # per-call arithmetic for the whole transition_z: 32 chunks
    out["per_transition_call_ms"] = {"incumbent": round(32 * ma, 4), "fused": round(32 * mb, 4),
                                     "delta": round(32 * (ma - mb), 4)}
    print("per transition_z call: %.4f -> %.4f ms, delta %.4f"
          % (32 * ma, 32 * mb, 32 * (ma - mb)), flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
