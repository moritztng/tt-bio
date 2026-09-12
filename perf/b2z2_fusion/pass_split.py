#!/usr/bin/env python3
"""What makes the fused kernel's SECOND pass cost 3.9x its first one.

`loop_split.py` answered the question the workstream asked. On whglx card 5 (WH, 8x8) the one-pass
build costs 1.0114x a standalone `generic_op` matmul -- the two-pass LOOP is free -- and the whole
2.5x sits in pass 1, which costs 0.28815 ms against pass 0's 0.07369.

Pass 1 differs from pass 0 in exactly one thing: `copy_block` applies silu as it takes the fp32
accumulator to bf16, and it does so under a per-tile `acquire_dst()` with a per-tile
`silu_tile_init()`. This separates the three candidates and measures the two fixes at once.

  incumbent      ttnn.linear(silu) + ttnn.linear + ttnn.multiply_, what production runs
  mm_1           one standalone generic_op matmul, the reference the loop is priced against
  p1             the fused kernel, one pass, no product                (loop cost)
  p2_diag        two passes, no product, kernel as it arrived          (the 2.5x)
  real           the real kernel, product and all, as it arrived
  real_nosilu    DIAGNOSTIC, wrong answer: pass 1 without silu         (prices the activation)
  real_hoist     `silu_tile_init()` once a block instead of once a tile
  real_bN        N tiles per DST acquire in copy_block
  real_hoist_bN  both

`hoist` and `copy_batch` are both bit-exact by construction: the same tiles go through the same
SFPU function in the same order and are packed in the same order. Only an init site and a barrier
move. The harness proves it rather than asserting it -- every real arm is compared to `real` with
`torch.equal`, and `real_nosilu` is expected to FAIL that, which is what makes the check mean
anything. At MUL_MODE 2 it would NOT fail: that build packs pass 0 and never reads pass 1, so the
silu define is invisible in the output. The control caught exactly that on the first run.
"""
from __future__ import annotations

import argparse
import json
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
ap.add_argument("--block", type=str, default="8,2,2,2,2")
ap.add_argument("--grid", type=str, default="8x8")
#: name:passes/copy_batch/silu_hoist/silu/mul_mode
ap.add_argument("--arms", type=str, default=(
    "p1:1/1/0/1/2;p2_diag:2/1/0/1/2;"
    "real:2/1/0/1/1;real_nosilu:2/1/0/0/1;real_cheapsfpu:2/1/0/2/1;"
    "real_hoist:2/1/1/1/1;real_hoist_b4:2/4/1/1/1;"
    "real_lut:2/1/0/3/1;real_bf16silu:2/1/0/4/1;real_bf16silu_b4:2/4/0/4/1;real_appx:2/1/0/5/1"))
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.mm_generic as MG                                                # noqa: E402
import tt_bio.transition_swiglu as TS                                         # noqa: E402

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
ckc4 = T._mm_ckc(ckc)
block = tuple(int(v) for v in a.block.split(","))
grid = tuple(int(v) for v in a.grid.split("x"))
kt, nt = a.c // 32, a.hidden // 32

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w1 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
w2 = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
hshape = ttnn.Shape([1, a.rows, a.n, a.hidden])
TS.set_enabled(True)
TS.BLOCK_KEYS = {(kt, nt): block}
TS._block_for.cache_clear()
TS.GRID = None


def mm_1():
    o = ttnn.allocate_tensor_on_device(hshape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w1, [o], (block, grid), ckc4)
    ttnn.deallocate(o)


def mm_2():
    """The comparison the workstream states the 2.5x against: the two projections as two separate
    `generic_op` matmuls, writing two L1 tensors instead of one."""
    o1 = ttnn.allocate_tensor_on_device(hshape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                        ttnn.L1_MEMORY_CONFIG)
    o2 = ttnn.allocate_tensor_on_device(hshape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                        ttnn.L1_MEMORY_CONFIG)
    MG.generic_minimal_matmul(dev, x, w1, [o1], (block, grid), ckc4)
    MG.generic_minimal_matmul(dev, x, w2, [o2], (block, grid), ckc4)
    ttnn.deallocate(o1)
    ttnn.deallocate(o2)


ARM_SPECS = []
for tok in a.arms.split(";"):
    name, cfg = tok.split(":")
    ARM_SPECS.append((name, tuple(int(v) for v in cfg.split("/"))))


def incumbent():
    x1 = ttnn.linear(x, w1, activation="silu", compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    x2 = ttnn.linear(x, w2, compute_kernel_config=ckc,
                     memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=T.CORE_GRID_MAIN)
    r = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return r


def loop_for(spec):
    def f():
        TS.PASSES, TS.COPY_BATCH, TS.SILU_HOIST, TS.SILU, TS.MUL_MODE = spec
        o = TS.fused_swiglu(x, w2, w1, ckc4, grid)
        if o is None:
            raise RuntimeError("declined " + json.dumps({str(k): v for k, v in TS.REJECTS.items()}))
        return o
    return f


def timed(fn):
    for _ in range(2):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


# Build and read back every arm before any of them is timed, and hold p2's bytes as the reference
# every two-pass arm has to reproduce exactly.
REAL = [n for n, s in ARM_SPECS if s[0] == 2 and s[4] == 1]
#: only the arms that run the LLK silu are restructures of the same arithmetic; SILU 2 and 3 are a
#: different function and are scored, not compared bit for bit.
SAME_MATH = {n for n, s in ARM_SPECS if s[0] == 2 and s[4] == 1 and s[3] == 1}
vals, ref = {}, None
for name, spec in ARM_SPECS:
    o = loop_for(spec)()
    vals[name] = ttnn.to_torch(o).clone()
    ttnn.deallocate(o)
r = incumbent(); inc_ref = ttnn.to_torch(r).float(); ttnn.deallocate(r)
ref = vals.get("real")
exact, pccs, mx = {}, {}, {}
for name in REAL:
    exact[name] = bool(torch.equal(vals[name].float(), ref.float()))
    v = vals[name].float()
    pccs[name] = float(((inc_ref - inc_ref.mean()) * (v - v.mean())).mean()
                       / (inc_ref.std(unbiased=False) * v.std(unbiased=False)))
    mx[name] = float((inc_ref - v).abs().max())
    print("%-16s bit-exact vs real: %-5s  pcc vs incumbent %.7f  max_abs %.5f   %s"
          % (name, exact[name], pccs[name], mx[name],
             "SAME ARITHMETIC, must be bit-exact" if name in SAME_MATH
             else "different approximation, scored not compared"), flush=True)
if exact.get("real_nosilu", False):
    raise SystemExit("real_nosilu matches real bit for bit: the SILU define did nothing, "
                     "so the bit-exactness of every other arm proves nothing. Aborting.")
for name in SAME_MATH:
    if not exact[name]:
        raise SystemExit("%s is NOT bit-exact against the kernel as it arrived. It is a "
                         "restructure, not a numerics change, so this is a bug. Aborting." % name)

ARMS = ([("incumbent", incumbent), ("mm_1", mm_1), ("mm_1_aa", mm_1), ("mm_2", mm_2)]
        + [(n, loop_for(s)) for n, s in ARM_SPECS])
for _, fn in ARMS:
    r = fn()
    if r is not None:
        ttnn.deallocate(r)
ttnn.synchronize_device(dev)

got = {n: [] for n, _ in ARMS}
for _ in range(a.reps):
    for name, fn in ARMS:
        got[name].append(timed(fn))

base = st.median(got["mm_1"])
inc = st.median(got["incumbent"])
rows = {}
for name, _ in ARMS:
    m = st.median(got[name])
    rows[name] = {"ms": round(m, 5), "x_mm_1": round(m / base, 4),
                  "ratio_vs_incumbent": round(inc / m, 4),
                  "spread_pct": round(100 * (max(got[name]) - min(got[name])) / m, 2)}
    if name in exact:
        rows[name]["bit_exact_vs_real"] = exact[name]
        rows[name]["pcc_vs_incumbent"] = round(pccs[name], 7)
        rows[name]["max_abs_vs_incumbent"] = round(mx[name], 5)
        rows[name]["same_arithmetic"] = name in SAME_MATH
    print("%-14s %.5f ms  %.4fx mm_1  %.4fx incumbent  spread %.1f%%"
          % (name, rows[name]["ms"], rows[name]["x_mm_1"], rows[name]["ratio_vs_incumbent"],
             rows[name]["spread_pct"]), flush=True)

if "p2_nosilu_diag" in rows and "mm_2" in rows:
    print("\ntwo-pass loop with no activation and no product: %.5f ms against two standalone "
          "matmuls at %.5f -- %.4fx"
          % (rows["p2_nosilu_diag"]["ms"], rows["mm_2"]["ms"],
             rows["p2_nosilu_diag"]["ms"] / rows["mm_2"]["ms"]), flush=True)
    rows["twopass_vs_two_matmuls"] = round(rows["p2_nosilu_diag"]["ms"] / rows["mm_2"]["ms"], 4)
if "p1" in rows and "p2_diag" in rows:
    p1ms, p2ms = rows["p1"]["ms"], rows["p2_diag"]["ms"]
    print("\npass 1 costs %.5f ms as it arrived; a standalone matmul is %.5f (%.2fx)"
          % (p2ms - p1ms, base, (p2ms - p1ms) / base), flush=True)
    rows["second_pass_ms_p2_diag"] = round(p2ms - p1ms, 5)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(block), "grid": list(grid), "inner": a.inner,
                             "reps": a.reps, "arms": rows}, indent=1))
print("wrote", a.out)
