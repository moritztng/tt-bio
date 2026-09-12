"""Where do the 2.06 ms the bias fusion returns actually come from?

The DRAM-byte ledger prices that deletion at 0.70 ms and it measured 2.06. Either the byte model
is wrong by 3x at this site, or something other than a read was deleted. The bias projection is a
`[1,S,S,128] x [128,4]` matmul: one N tile, spread over a grid that wants seventeen. So this times
the three things separately instead of arguing about them —

  the stock bias `ttnn.linear` on its own,
  the fused qkv+gate matmul at 16 N tiles,
  the same matmul at 17 N tiles, which is what the fusion actually pays.

`cost of the stock op` minus `the 17th tile` is what the fusion returns, and it should account for
the measured 2.06 ms.
"""
import os
import statistics
import sys
import time

import torch
import ttnn

from tt_bio import tenstorrent as tt
from tt_bio import mm_generic as G

S = int(os.environ.get("PROBE_TOKENS", "512"))
REPS = int(os.environ.get("PROBE_REPS", "11"))
dev = tt.get_device()
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
f = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
c_z, n_heads, head_dim = 128, 4, 32
c = n_heads * head_dim

x = ttnn.reshape(f(torch.randn(1, S, S, c_z)), (S, S, c_z))
w_bias = f(torch.randn(c_z, n_heads) * 0.05)
w_qkvg = f(torch.randn(c_z, 4 * c) * 0.05)
w_qkvgb = f(torch.randn(c_z, 4 * c + 32) * 0.05)
MT = S // 32


def timed(fn):
    fn()                                    # compile
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        for o in (out if isinstance(out, (list, tuple)) else [out]):
            ttnn.deallocate(o)
    return statistics.median(ts), min(ts)


def stock_bias():
    return tt._pair_proj_linear(x, w_bias, KC, ttnn.bfloat16)


def split(w, n_widths, nb):
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([S, n_heads, S, head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG) for _ in range(4)]
    if nb:
        outs.append(ttnn.allocate_tensor_on_device(
            ttnn.Shape([S, S, n_heads]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
            ttnn.DRAM_MEMORY_CONFIG))
    G.generic_minimal_matmul(
        dev, x, w, outs, (tt._mm_block_for(w), tuple(tt.COMPUTE_GRID_MAIN)), G.ckc_args(KC),
        {"HEAD_MAJOR_MT": MT}, tt._triatt_qkv.KERNEL_DIR, n_widths=n_widths)
    return outs


rows = [
    ("stock bias ttnn.linear  [S,S,128]x[128,4]", stock_bias),
    ("qkvg matmul             16 N tiles", lambda: split(w_qkvg, None, False)),
    ("qkvgb matmul            17 N tiles", lambda: split(w_qkvgb, [4, 4, 4, 4, 1], True)),
]
got = {}
for name, fn in rows:
    med, lo = timed(fn)
    got[name] = med
    print(f"  {name:<45} median {med:7.3f} ms   min {lo:7.3f} ms", flush=True)

bias = got[rows[0][0]]
d17 = got[rows[2][0]] - got[rows[1][0]]
print(f"\n  the stock bias op costs            {bias:7.3f} ms")
print(f"  the 17th N tile costs              {d17:7.3f} ms")
print(f"  so the fusion returns              {bias - d17:7.3f} ms per attention, "
      f"{2 * (bias - d17):7.3f} ms per block")
