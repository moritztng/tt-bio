#!/usr/bin/env python3
"""Unfused silu at the Transition chunk: what it costs, and which arm is actually closer to truth.

`b2z2-fusion-rebuild` measured `ttnn.linear(activation="silu")` at 0.11432 ms against 0.02663 ms
for the same matmul with no activation (BH, qb2 card 1) and 1.4025x for the chunk with silu split
out. It then killed the lever on a whole-molecule RMSD move at 512 aa. Two things that run never
did, and both are cheap:

  1. **Score every arm against fp32 truth, not against the incumbent.** The incumbent is not the
     reference. Boltz-2's own Transition is `silu(fc1(x)) * fc2(x)` (tt_bio/boltz2.py:1391), i.e.
     silu applied to the projection's OUTPUT tensor. "Distance from the incumbent" cannot tell a
     worse arm from a differently-rounded one; distance from the fp32 evaluation of the same
     algebra on the same operands can.
  2. **Try the fix.** Packing the projection as fp32 puts silu back on an unrounded value, which
     is what the fused activation does inside the matmul kernel. It costs bytes, so it is timed
     here in the same interleaved run rather than assumed.

Arms, paired and interleaved in one process, with an A/A floor from the incumbent run twice:

  incumbent      ttnn.linear(activation="silu"), product in place        (what ships)
  split_bf16     linear -> silu on the bf16 projection                   (TT_BIO_UNFUSED_SILU)
  split_fp32     linear(dtype=fp32) -> silu -> multiply back to bf16     (+ _FP32, the candidate fix)
  split_fp32_c   same, with an explicit typecast before the product      (fallback if mixed multiply
                                                                          is unsupported)

    chunk_probe.py --out <json> [--rows 16 --n 512 --c 128 --hidden 512]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--rows", type=int, default=16)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
ap.add_argument("--hidden", type=int, default=512)
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--inner", type=int, default=32)
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio as _TB                                                          # noqa: E402
assert Path(_TB.__file__).resolve().is_relative_to(ROOT), _TB.__file__

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
xt = (torch.randn(1, a.rows, a.n, a.c) * 0.5).bfloat16()
w1t = (torch.randn(a.c, a.hidden) * 0.05).bfloat16()
w2t = (torch.randn(a.c, a.hidden) * 0.05).bfloat16()
x = up(xt, ttnn.L1_MEMORY_CONFIG)
w1 = up(w1t, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(w2t, ttnn.DRAM_MEMORY_CONFIG)

# fp32 truth for the identical algebra on the identical (bf16) operands. Both device arms are
# approximations of this; the question is which approximation is closer.
REF = (torch.nn.functional.silu(xt.float() @ w1t.float()) * (xt.float() @ w2t.float()))


def L(w, act=None, dtype=None):
    return ttnn.linear(x, w, activation=act, compute_kernel_config=ckc,
                       memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN,
                       dtype=dtype)


def incumbent():
    """What Transition ships: silu fused into the projection, product in place."""
    x1 = L(w1, "silu")
    x2 = L(w2)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


def split_bf16():
    """TT_BIO_UNFUSED_SILU as it stands: silu on the bf16-packed projection."""
    x1 = L(w1)
    ttnn.silu(x1, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=x1)
    x2 = L(w2)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


def split_fp32():
    """The candidate fix: the projection stays fp32, so silu sees an unrounded value."""
    x1 = L(w1, dtype=ttnn.float32)
    ttnn.silu(x1, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=x1)
    x2 = L(w2)
    r = ttnn.multiply(x1, x2, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    return r


def split_fp32_c():
    """Same, rounding the activated projection back to bf16 before the product."""
    x1 = L(w1, dtype=ttnn.float32)
    ttnn.silu(x1, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=x1)
    x1b = ttnn.typecast(x1, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x1)
    x2 = L(w2)
    r = ttnn.multiply_(x1b, x2)
    ttnn.deallocate(x2)
    return r


ARMS = [("incumbent", incumbent), ("incumbent_aa", incumbent), ("split_bf16", split_bf16),
        ("split_fp32", split_fp32), ("split_fp32_c", split_fp32_c)]


def err(fn):
    """Every arm against fp32 truth, and against the incumbent, on one evaluation."""
    o = fn()
    v = ttnn.to_torch(o).float().reshape(REF.shape)
    ttnn.deallocate(o)
    d = (v - REF).abs()
    rel = d.sum() / REF.abs().sum()
    return {"max_abs_vs_fp32": round(float(d.max()), 6),
            "rms_vs_fp32": round(float((d ** 2).mean().sqrt()), 7),
            "mean_rel_vs_fp32": round(float(rel), 7),
            "pcc_vs_fp32": round(float(((REF - REF.mean()) * (v - v.mean())).mean()
                                       / (REF.std(unbiased=False) * v.std(unbiased=False))), 9)}, v


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


g = dev.compute_with_storage_grid_size()
out = {"doc": __doc__, "env": {"host": socket.gethostname(), "arch": str(dev.arch()),
                               "grid": [g.x, g.y], "core_grid_main": [T.CORE_GRID_MAIN.x,
                                                                      T.CORE_GRID_MAIN.y],
                               "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                               "shape": [1, a.rows, a.n, a.c], "hidden": a.hidden,
                               "inner": a.inner, "reps": a.reps,
                               "commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
                               "loadavg": os.getloadavg()}, "arms": {}}

qual, vals = {}, {}
for name, fn in ARMS:
    if name == "incumbent_aa":
        continue
    try:
        qual[name], vals[name] = err(fn)
    except Exception as e:                       # an arm the build does not support is a result
        qual[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"{name}: {qual[name]['error']}", flush=True)
for name in list(vals):
    if name != "incumbent":
        d = (vals[name] - vals["incumbent"]).abs()
        qual[name]["max_abs_vs_incumbent"] = round(float(d.max()), 6)
        qual[name]["bit_exact_vs_incumbent"] = bool(torch.equal(vals[name], vals["incumbent"]))

live = [(n, f) for n, f in ARMS if "error" not in qual.get(n, {})]
got = {n: [] for n, _ in live}
for _ in range(a.reps):
    for name, fn in live:
        got[name].append(timed(fn))

base = st.median(got["incumbent"])
for name, _ in live:
    m = got[name]
    row = {"ms": round(st.median(m), 5), "chunk_ratio_vs_incumbent": round(base / st.median(m), 4),
           "spread_pct": round(100 * (max(m) - min(m)) / st.median(m), 2)}
    row.update(qual.get(name, {}))
    out["arms"][name] = row
    print("%-14s %.5f ms  %.4fx  spread %.1f%%  %s" % (
        name, row["ms"], row["chunk_ratio_vs_incumbent"], row["spread_pct"],
        " ".join(f"{k}={v}" for k, v in qual.get(name, {}).items())), flush=True)
for name in qual:
    if name not in out["arms"]:
        out["arms"][name] = qual[name]

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=1))
print("wrote", a.out)
