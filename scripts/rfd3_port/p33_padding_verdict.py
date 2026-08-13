"""The padding verdict: does tile-aligning the token axis I buy anything the shipped
program-config route does not already get at ragged I?

p14 measured the batch-into-M collapse (an explicit `ttnn.reshape` to [1,1,D*I*I,C], one
matmul, reshape back) at 0.49-0.69x on ragged I=250 and 2.09-6.65x on aligned I=256, and
concluded that padding I model-wide was the unlock. p15 argued that is the wrong framing:
the matmul's M axis is ALREADY tile-padded by TILE_LAYOUT, the 0.49x is the reshape
physically re-tiling, and the same collapse is available inside the matmul via
`fuse_batch=True` with no reshape at all -- so no shape change is needed anywhere.

This probe measures all three arms at both token counts, at the shapes and both design
batches that matter, on one card. It answers: is `min(arms) at I=250` within noise of
`min(arms) at I=256`? If yes, padding I is dead.

Every arm's output is checked bitwise against the shipped default's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.rfd3 import model as m  # noqa: E402

REPEAT = 5

# The four real per-step [D,I,I,C] @ [C,N] pair linears (p15 table).
CASES = [
    ("z_transition fc1/fc2", 128, 512),
    ("z_transition fc3", 512, 128),
    ("transition_2 fc1/fc2", 128, 256),
    ("pair_bias to_b", 128, 32),
]


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=ttnn.bfloat16)


def bench(fn):
    dev = get_device()
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPEAT * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[250, 256])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = get_device()
    kw_base = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    rows = []
    for D in a.batches:
        for I in a.tokens:
            for name, K, N in CASES:
                g = torch.Generator().manual_seed(0)
                x = tt(torch.randn(D, I, I, K, generator=g))
                w = tt(torch.randn(K, N, generator=g))
                kw = dict(kw_base)

                def f_default():
                    return ttnn.linear(x, w, core_grid=None, **kw)

                ref = f_default()
                t_def = bench(f_default)

                # arm 2: the shipped calibrator picks a fuse_batch program config, bit-exact
                # by construction. Time the calibration itself; it is the reason the flag is
                # opt-in today.
                m._TUNED_MM_CACHE.clear()
                t0 = time.perf_counter()
                pc = m._calibrate_linear(x, w, kw, None)
                t_cal = time.perf_counter() - t0
                if pc is None:
                    t_tuned, mx_tuned, pcs = float("nan"), 0.0, "none (default was fastest)"
                else:
                    t_tuned = bench(lambda: ttnn.linear(x, w, program_config=pc, **kw))
                    mx_tuned = m._mm_maxabs(ttnn.linear(x, w, program_config=pc, **kw), ref)
                    pcs = (f"bw={pc.in0_block_w} obw={pc.out_block_w} pcM={pc.per_core_M} "
                           f"pcN={pc.per_core_N}")

                # arm 3: p14's explicit reshape collapse.
                def f_collapse():
                    flat = ttnn.reshape(x, (1, 1, D * I * I, K))
                    o = ttnn.linear(flat, w, core_grid=None, **kw)
                    return ttnn.reshape(o, (D, I, I, N))

                try:
                    t_col = bench(f_collapse)
                    mx_col = m._mm_maxabs(f_collapse(), ref)
                except Exception as e:                       # noqa: BLE001
                    t_col, mx_col = float("nan"), float("nan")
                    print(f"  collapse failed: {type(e).__name__}: {str(e)[:120]}")

                rows.append(dict(case=name, K=K, N=N, D=D, I=I, default_ms=t_def,
                                 tuned_ms=t_tuned, tuned_maxabs=mx_tuned, tuned_cfg=pcs,
                                 calib_s=t_cal, collapse_ms=t_col, collapse_maxabs=mx_col))
                print(f"D={D} I={I:3d} {name:22s} K={K:3d} N={N:3d} | default {t_def:8.3f} ms | "
                      f"tuned {t_tuned:8.3f} ms (maxabs {mx_tuned:g}, calib {t_cal:5.2f} s, {pcs}) | "
                      f"collapse {t_col:8.3f} ms (maxabs {mx_col:g})", flush=True)
                for t in (x, w, ref):
                    ttnn.deallocate(t)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
