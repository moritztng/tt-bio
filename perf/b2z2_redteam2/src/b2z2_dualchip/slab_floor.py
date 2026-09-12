"""Why a trimul row slab stops at ~1.4x and cannot reach 2.00x, in two measurements.

`slab_perf.py` says a trimul slab is 1.38-1.43x and the bar is 2.00x. This file answers the next
question, which is whether the gap is a fallback still to be fixed or the op's own algebra.

The starting variant is `out[i,j,c] = sum_k a[i,k,c] b[j,k,c]`. A row slab shards i, and i appears
only in `a`. `b` carries the output COLUMN j and the contraction k, so on two chips the ENTIRE b
pipeline -- the b-half input projection over all of z, its gated channel move, its inner transpose
-- runs full size on BOTH chips, as does the input layer_norm that feeds it. Two readings of that:

  PART 1, the fit. `slab_ms = fixed + k*R` over R = 32..384. The intercept is everything the slab
  cannot shed, measured rather than argued.

  PART 2, the decomposition. The same quantity from the other side: time the duplicated ops
  standalone at the production shape and sum them.

Both must be read against `whole / 2`, which is what a 2.00x slab has to fit inside.

The ending variant is deliberately NOT fitted: its slab is not linear in R (R=128 measures slower
than R=256, reproducibly), because its `a` side slices z's SECOND axis and the slice width is the
gated move's column-tile count. Its R=256 point is in the table and that is the production one.

Run (card 2, pinned, under benchlock -- these are timed):

    perf/b2z2_dualchip/slab_floor.py
"""

import os
import pathlib
import statistics as st
import sys
import time

import torch

# Score the checkout this script LIVES IN; see slab_bitexact.py for why.
_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import ttnn
from tt_bio import reblock_permute as rb
from tt_bio import reference as ref
from tt_bio import tenstorrent as tt

S, C = 512, 128
REPS = int(os.environ.get("SLABFLOOR_REPS", "7"))
HEIGHTS = tuple(int(r) for r in os.environ.get("SLABFLOOR_R", "32,64,128,256,384").split(","))

dev = tt.get_device()
KC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(384, C, 16, 0.25, 32, 4, v2=True)
# Timing is not data-dependent on Tensix, but Boltz-2 zeroes both trimul output projections at
# init, so randomize anyway and this cannot be mistaken for a correctness check.
w = {k: torch.randn_like(v) * 0.05 if v.dtype.is_floating_point else v
     for k, v in rl.state_dict().items()}
layer = tt.PairformerLayer(32, 4, 24, 16, True, w, KC)
f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
z = f(torch.randn(1, S, S, C))
m1 = torch.ones(1, S)
mask = f(m1[:, :, None] * m1[:, None, :])
DRAM = ttnn.DRAM_MEMORY_CONFIG


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bench(call, keep=False):
    """Median of REPS warm calls. `keep` hands back the last result for the next stage to read."""
    out = call()
    ttnn.synchronize_device(dev)
    if not keep:
        ttnn.deallocate(out)
    last, v = None, []
    for _ in range(REPS):
        t0 = time.perf_counter()
        r = call()
        ttnn.synchronize_device(dev)
        v.append(time.perf_counter() - t0)
        if keep:
            if last is not None:
                ttnn.deallocate(last)
            last = r
        else:
            ttnn.deallocate(r)
    if keep and out is not last:
        ttnn.deallocate(out)
    return st.median(v) * 1e3, last


tm = layer.triangle_multiplication_start
log("PART 1 -- slab cost against slab height, starting trimul")
whole, _ = bench(lambda: tm(z, mask))
pts = []
for R in HEIGHTS:
    ms, _ = bench(lambda: tm(z, mask, row_slab=(0, R)))
    pts.append((R, ms))
    log(f"  R={R:4d}  {ms:7.3f} ms")
n = len(pts)
sx, sy = sum(r for r, _ in pts), sum(m for _, m in pts)
sxx, sxy = sum(r * r for r, _ in pts), sum(r * m for r, m in pts)
k = (n * sxy - sx * sy) / (n * sxx - sx * sx)
fixed = (sy - k * sx) / n
log(f"  whole {whole:.3f} ms | fit slab_ms = {fixed:.3f} + {k * 1e3:.3f} us/row")
log(f"  the R-independent part is {fixed:.3f} ms = {fixed / whole * 100:.1f} % of the whole op; "
    f"a 2.00x slab has to fit in {whole / 2:.3f} ms, leaving {whole / 2 - fixed:+.3f} ms for "
    f"256 rows of a-side work that cost {k * 256:.3f} ms")

log("PART 2 -- the duplicated ops, timed standalone")
wb = tm._gp_in_chunks(128, 1, tt._GP_B)[0]
slice_c = int(wb.shape[-1]) // 2
rows = []
t, zn = bench(lambda: ttnn.layer_norm(z, weight=tm.in_norm_weight, bias=tm.in_norm_bias,
                                      epsilon=1e-5, compute_kernel_config=KC), keep=True)
rows.append(("layer_norm over all of z", t))
t, gpb = bench(lambda: tt._in_proj_matmul(zn, wb, KC, DRAM), keep=True)
rows.append(("b-half input projection over all of z", t))
t, bmv = bench(lambda: rb.reblock_permute_gated(gpb, slice_c, 0, slice_c, memory_config=DRAM),
               keep=True)
rows.append(("b gated channel move (E6) over all of z", t))
t, _ = bench(lambda: ttnn.transpose(bmv, -2, -1, memory_config=DRAM))
rows.append(("b inner transpose over all of z", t))
for t_ in (zn, gpb, bmv):
    ttnn.deallocate(t_)
tot = sum(v for _, v in rows)
for name, v in rows:
    log(f"  {name:42s} {v:6.3f} ms")
log(f"  {'SUM: the b side plus the shared input norm':42s} {tot:6.3f} ms "
    f"= {tot / whole * 100:.1f} % of the whole op")
log(f"  even a FREE b side would leave the slab at {pts[-1][1] - tot:.3f} ms at R=384; at the "
    f"production R=256 the slab is {dict(pts).get(256, float('nan')):.3f} ms against a "
    f"{whole / 2:.3f} ms bar")

log("PART 3 -- the ending variant at the production slab height")
tme = layer.triangle_multiplication_end
we, _ = bench(lambda: tme(z, mask))
se, _ = bench(lambda: tme(z, mask, row_slab=(0, S // 2)))
log(f"  whole {we:.3f} ms | slab {se:.3f} ms | {we / se:.3f}x")
os._exit(0)
