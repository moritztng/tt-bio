#!/usr/bin/env python3
"""Is the gate multiply DRAM-bound or sigmoid-bound? The byte ranking assumed the first.

Same shapes as the fold's triangle-attention gate, out-of-place so no arm re-uploads its input and
ITERS ops inside one timed region so a 0.7 ms op is not measured against host noise -- the first
attempt at this timed one op per sync and read a 49 % A/A floor, which is no measurement at all.

  mul_sig    the op the fold runs: read o, read g, write out, with SIGMOID on b
  mul_plain  the identical op with the activation removed: the SAME 201.3 MB, no SFPU work
  sigmoid    sigmoid(g) alone, 134.2 MB

If mul_sig and mul_plain are close the op is DRAM-bound and the byte prize is real. If mul_plain is
much faster the cost is the accurate sigmoid, which no fusion can delete -- the model needs it.
"""
import json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

S, H, D = int(os.environ.get("N", "512")), 4, 32
REPS, WARM, ITERS = 7, 2, int(os.environ.get("ITERS", "10"))
dev = T.get_device()
torch.manual_seed(0)
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
o = f(torch.randn(S, H, S, D) * 0.5)
g = f(torch.randn(S, H, S, D) * 2.0)

TILE_B = 2
BYTES_RW3 = 3 * S * H * S * D * TILE_B
BYTES_RW2 = 2 * S * H * S * D * TILE_B
SIG = [ttnn.UnaryOpType.SIGMOID]


def batch(fn):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(ITERS)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) * 1e3 / ITERS
    for x in outs:
        ttnn.deallocate(x)
    return dt


ARMS = {
    "mul_sig": lambda: batch(lambda: ttnn.multiply(o, g, input_tensor_b_activations=SIG)),
    "mul_plain": lambda: batch(lambda: ttnn.multiply(o, g)),
    "mul_plain2": lambda: batch(lambda: ttnn.multiply(o, g)),
    "sigmoid": lambda: batch(lambda: ttnn.sigmoid(g)),
}
t = {n: [] for n in ARMS}
for r in range(WARM + REPS):
    for n in ARMS:
        dt = ARMS[n]()
        if r >= WARM:
            t[n].append(dt)
med = {n: st.median(t[n]) for n in ARMS}
res = {"n": S, "iters_per_region": ITERS, "reps": REPS, "arch": str(dev.arch()),
       "grid": list(T.COMPUTE_GRID_MAIN), "host": os.uname().nodename,
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "loadavg": os.getloadavg(),
       "median_ms": med, "all_ms": t,
       "spread_frac": {n: (max(t[n]) - min(t[n])) / med[n] for n in ARMS},
       "bytes_mul_MB": BYTES_RW3 / 1e6, "bytes_sigmoid_MB": BYTES_RW2 / 1e6,
       "derived": {
           "aa_floor_frac": abs(med["mul_plain"] / med["mul_plain2"] - 1.0),
           "sigmoid_extra_ms": med["mul_sig"] - med["mul_plain"],
           "sigmoid_share_of_mul": (med["mul_sig"] - med["mul_plain"]) / med["mul_sig"],
           "mul_sig_GBps": BYTES_RW3 / med["mul_sig"] * 1e-6,
           "mul_plain_GBps": BYTES_RW3 / med["mul_plain"] * 1e-6,
           "sigmoid_GBps": BYTES_RW2 / med["sigmoid"] * 1e-6,
       }}
print(json.dumps({k: res[k] for k in ("median_ms", "spread_frac", "derived")}, indent=1), flush=True)
p = REPO / f"perf/roof_gate_epilogue/sig_cost_{S}_qb2_c2.json"
p.write_text(json.dumps(res, indent=1))
print("wrote", p, flush=True)
