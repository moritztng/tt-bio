#!/usr/bin/env python3
"""Smallest call that exercises the fused SwiGLU descriptor, with the stage printed before it runs.

The first A/B attempt produced no stdout at all before its `timeout` killed it, because the whole
run was piped through `grep | tail` and the pipeline never flushed. Run this one unbuffered and
print BEFORE each device call, so a hang names the call it hung in instead of leaving a silent
15-minute window.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def say(m):
    print("[%7.2fs] %s" % (time.perf_counter() - T0, m), flush=True)


T0 = time.perf_counter()

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--arm", default="both", choices=["both", "fused", "incumbent"])
ap.add_argument("--probe", default="normal",
                choices=["normal", "w1zero", "w2zero", "w1big"],
                help="w1zero: silu(0)=0 so the product must be 0 everywhere -- isolates the "
                     "activation from the matmul plumbing")
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.transition_swiglu as TS                                         # noqa: E402

say("opening device")
dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
grid = tuple(T.COMPUTE_GRID_MAIN)
say(f"grid {grid}  ckc {T._mm_ckc(ckc)}")

torch.manual_seed(0)
xt = torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5
w1t = torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05
w2t = torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05
if a.probe == "w1zero":
    w1t = torch.zeros_like(w1t)
elif a.probe == "w2zero":
    w2t = torch.zeros_like(w2t)
elif a.probe == "w1big":
    w1t = w1t * 200.0        # silu(v) -> v for v >> 0, ReLU-like, so the product is ~ x@w2 * x@w1
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(xt, ttnn.L1_MEMORY_CONFIG)
w1 = up(w1t, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(w2t, ttnn.DRAM_MEMORY_CONFIG)
say("uploaded")

ref = None
if a.arm in ("both", "incumbent"):
    say("incumbent: linear silu")
    x1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    say("incumbent: linear plain")
    x2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    say("incumbent: multiply_")
    r = ttnn.multiply_(x1, x2)
    ttnn.synchronize_device(dev)
    ref = ttnn.to_torch(r).float()
    ttnn.deallocate(r)
    ttnn.deallocate(x2)
    say("incumbent done")

if a.arm in ("both", "fused"):
    TS.set_enabled(True)
    say("fused: eligibility -> " + str(TS.eligible(x, w2, w1)))
    say("fused: generic_op enqueue")
    g = TS.fused_swiglu(x, w2, w1, T._mm_ckc(ckc), grid)
    if g is None:
        say("DECLINED " + str(TS.REJECTS))
        raise SystemExit(1)
    say("fused: enqueued, syncing")
    ttnn.synchronize_device(dev)
    say("fused: synced")
    got = ttnn.to_torch(g).float()
    say("fused: read back")
    if ref is not None:
        d = (ref - got).abs()
        u, v = ref.flatten().double(), got.flatten().double()
        pcc = float(((u - u.mean()) * (v - v.mean())).mean()
                    / (u.std(unbiased=False) * v.std(unbiased=False)))
        say("equal=%s  max_abs=%.6g  pcc=%.8f  mismatched=%d/%d"
            % (torch.equal(ref, got), float(d.max()), pcc,
               int((ref != got).sum()), ref.numel()))
        # where are they: last axis is the hidden (N) axis, so a per-N-tile profile says whether
        # a whole N block is wrong or the error is spread
        bad = (ref != got)
        nt = ref.shape[-1] // 32
        per_n = [int(bad[..., i * 32:(i + 1) * 32].sum()) for i in range(nt)]
        say("mismatches per N tile (%d tiles): %s" % (nt, per_n))
        mt_rows = ref.shape[-2] // 32
        per_m = [int(bad[..., i * 32:(i + 1) * 32, :].sum()) for i in range(mt_rows)]
        say("mismatches per row tile within a row block (%d): %s" % (mt_rows, per_m))
        # torch reference, to say WHICH arm is wrong
        rt = (xt.float() @ w2t.float()) * torch.nn.functional.silu(xt.float() @ w1t.float())
        rt = rt.to(torch.bfloat16).float()
        for nm, arr in (("incumbent", ref), ("fused", got)):
            e = (arr - rt).abs()
            say("  vs torch fp32: %-9s max_abs=%.6g mean_abs=%.6g" % (nm, float(e.max()),
                                                                     float(e.mean())))
say("OK")
