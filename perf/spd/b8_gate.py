"""Is a bfloat8_b pair tensor faster than a bfloat16 one, per op, at BindCraft 2's pair shape?

The gate before any bfp8 pair mode for BindCraft 2 (state/spd-gap-bind-jobs/bfp8_scope.md): nearly
every pair op on a Wormhole chip runs at 95-111 GB/s, well under the DRAM roof, so halving the bytes
need not halve the time. Each op runs on a [1, N, N, 128] tile tensor in DRAM, once with bf16
operands and once with bfp8_b operands (weights stay bf16, as they would in the mode), and reports
the median of REPS synchronised calls and the ratio. Writes one JSON object. The scope's bar: same-dtype
ops >= 1.4x faster in bfp8, or the mode is not worth building.

usage (one chip, tt-bio's device path): TT_VISIBLE_DEVICES=N python perf/spd/b8_gate.py OUT.json [N]
"""
import json
import sys
import time

import torch
import ttnn

import tt_bio.af2 as A
import tt_bio.tenstorrent as T

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 224
C, REPS = 128, 20
dev = T.get_device()
ckc = A.compute_kernel_config()
torch.manual_seed(0)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def up(t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        t0 = time.monotonic()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.monotonic() - t0)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


w128 = up(torch.randn(C, C) / C ** 0.5, ttnn.bfloat16)
w512 = up(torch.randn(C, 4 * C) / C ** 0.5, ttnn.bfloat16)
w512t = up(torch.randn(4 * C, C) / (4 * C) ** 0.5, ttnn.bfloat16)
lnw, lnb = up(torch.ones(1, C), ttnn.bfloat16), up(torch.zeros(1, C), ttnn.bfloat16)
xh, yh, hh = torch.randn(1, N, N, C), torch.randn(1, N, N, C), torch.randn(1, N, N, 4 * C)

rows = {}
for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
    x, y, h = up(xh, dt), up(yh, dt), up(hh, dt)
    ops = {
        "linear_128_128": lambda: ttnn.linear(x, w128, dtype=dt, memory_config=DRAM, compute_kernel_config=ckc),
        "linear_128_512": lambda: ttnn.linear(x, w512, dtype=dt, memory_config=DRAM, compute_kernel_config=ckc),
        "linear_512_128": lambda: ttnn.linear(h, w512t, dtype=dt, memory_config=DRAM, compute_kernel_config=ckc),
        # bfp8 read, bf16 write: the residual-stream case the scope plans (update in bfp8, sum in bf16).
        "linear_128_128_to_bf16": lambda: ttnn.linear(x, w128, dtype=ttnn.bfloat16, memory_config=DRAM,
                                                      compute_kernel_config=ckc),
        "layer_norm": lambda: ttnn.layer_norm(x, weight=lnw, bias=lnb, epsilon=1e-5, memory_config=DRAM,
                                              compute_kernel_config=ckc),
        "multiply": lambda: ttnn.multiply(x, y, dtype=dt, memory_config=DRAM),
        "add": lambda: ttnn.add(x, y, dtype=dt, memory_config=DRAM),
        "add_to_bf16": lambda: ttnn.add(x, y, dtype=ttnn.bfloat16, memory_config=DRAM),
        "permute_0213": lambda: ttnn.permute(x, (0, 2, 1, 3), memory_config=DRAM),
    }
    rows[name] = {}
    for op, fn in ops.items():
        try:
            rows[name][op] = round(timed(fn) * 1e3, 4)
        except Exception as e:  # an op that refuses bfp8 is an answer, not a crash
            rows[name][op] = f"refused: {type(e).__name__}: {str(e)[:160]}"
    for t in (x, y, h):
        ttnn.deallocate(t)

ratio = {op: round(rows["bf16"][op] / rows["bfp8"][op], 3)
         for op in rows["bf16"] if isinstance(rows["bf16"][op], float) and isinstance(rows["bfp8"][op], float)}
res = {"n": N, "c": C, "reps": REPS, "arch": "wormhole" if T.is_wormhole() else "blackhole",
       "ms": rows, "bf16_over_bfp8": ratio}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res))
