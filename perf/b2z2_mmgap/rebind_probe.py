#!/usr/bin/env python3
"""Is `mm_generic.rebind`'s ProgramDescriptor reconstruction needed, and what does it cost?

`best_ab.py`: the same generic_op matmul costs 0.07057 ms a call into a stable output tensor and
0.08082 into a freshly allocated one, at a 17-60 % spread instead of 2.3 %. The only difference in
the call path is `rebind()`, which rewrites the per-core runtime args AND rebuilds the whole
ProgramDescriptor. Every generic_op kernel in tt-bio takes that path on every call a fold makes,
because the fold allocates its outputs fresh.

Two questions, in order: does mutating the cached KernelDescriptors' runtime args WITHOUT
rebuilding the descriptor produce the right answer (correctness first -- a stale address here is
silent), and if it does, what is it worth.
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
ap.add_argument("--block", type=str, default="8,2,1,4,1")
ap.add_argument("--grid", type=str, default="8x8")
a = ap.parse_args()

import torch                                                                  # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.mm_generic as MG                                                # noqa: E402

dev = T.get_device()
from tt_bio.af2 import compute_kernel_config                                  # noqa: E402
ckc = T.trunk_compute_kernel_config(compute_kernel_config())
ckc4 = T._mm_ckc(ckc)
BLOCK = tuple(int(v) for v in a.block.split(","))
GRID = tuple(int(v) for v in a.grid.split("x"))

torch.manual_seed(0)
up = lambda t, mc: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=mc)
x = up(torch.randn(1, a.rows, a.n, a.c, dtype=torch.bfloat16) * 0.5, ttnn.L1_MEMORY_CONFIG)
w = up(torch.randn(a.c, a.hidden, dtype=torch.bfloat16) * 0.05, ttnn.DRAM_MEMORY_CONFIG)
shape = ttnn.Shape([1, a.rows, a.n, a.hidden])
alloc = lambda: ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                               ttnn.L1_MEMORY_CONFIG)

ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN)).float()
pcc = lambda v: float(((ref - ref.mean()) * (v - v.mean())).mean()
                      / (ref.std(unbiased=False) * v.std(unbiased=False)))

o0 = alloc()
MG.generic_minimal_matmul(dev, x, w, [o0], (BLOCK, GRID), ckc4)
entry = MG._CACHE[MG._key(x, w, [o0], (BLOCK, GRID), ckc4, (), None, None, None)]


def rebind_mutate(entry, in0_addr, in1_addr, out_addrs):
    """`rebind` without the ProgramDescriptor reconstruction."""
    n = entry["n_chunks"]
    rt = entry["rt"]
    for name, addr in (("in0_sender", in0_addr), ("in0_recv", in0_addr),
                       ("in1_sender", in1_addr), ("in1_recv", in1_addr)):
        for _, arg in rt[name]:
            arg[0] = addr
            arg[len(arg) - n:] = list(out_addrs)
    for k, name in zip(entry["kernels"][:4],
                       ("in0_sender", "in0_recv", "in1_sender", "in1_recv")):
        k.runtime_args = rt[name]
    entry["addrs"] = (in0_addr, in1_addr, tuple(out_addrs))


def run(o, how):
    """One call into output `o`, binding addresses the way `how` says."""
    addrs = (x.buffer_address(), w.buffer_address(), (o.buffer_address(),))
    if addrs != entry["addrs"]:
        (MG.rebind if how == "rebuild" else rebind_mutate)(entry, *addrs)
    ttnn.generic_op([x, w, o], entry["pd"])


# ---- correctness: alternate between two distinct output buffers ------------------------
o1, o2 = alloc(), alloc()
assert o1.buffer_address() != o2.buffer_address(), "need two distinct output addresses"
checks = {}
for how in ("rebuild", "mutate"):
    got = []
    for o in (o1, o2, o1, o2):
        ttnn.memset(o, 0.0) if hasattr(ttnn, "memset") else None
        run(o, how)
        got.append(pcc(ttnn.to_torch(o).float()))
    checks[how] = [round(v, 8) for v in got]
    print("%-8s pcc over alternating outputs: %s" % (how, checks[how]), flush=True)

ok = all(v > 0.999 for v in checks["mutate"])
print("mutate-only binding is %s" % ("CORRECT" if ok else "WRONG -- reconstruction is load-bearing"),
      flush=True)

# ---- cost -------------------------------------------------------------------------------


def timed(fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.inner):
        fn()
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / a.inner


def arm(how, fresh):
    def f():
        o = alloc() if fresh else o1
        run(o, how)
        if fresh:
            ttnn.deallocate(o)
    return f


def linear():
    ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                core_grid=T.CORE_GRID_MAIN))


ARMS = [("linear", linear), ("stable_out", arm("rebuild", False)),
        ("fresh_rebuild", arm("rebuild", True)), ("fresh_mutate", arm("mutate", True))]
res = {n: [] for n, _ in ARMS}
for _ in range(a.reps):
    for n, f in ARMS:
        res[n].append(timed(f))
base = st.median(res["linear"])
rows = {}
for n, _ in ARMS:
    m = res[n]
    rows[n] = {"ms": round(st.median(m), 5), "vs_linear": round(base / st.median(m), 4),
               "spread_pct": round(100 * (max(m) - min(m)) / st.median(m), 2)}
    print("%-14s %.5f ms  %.4fx linear  spread %.1f %%"
          % (n, rows[n]["ms"], rows[n]["vs_linear"], rows[n]["spread_pct"]), flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"block": list(BLOCK), "grid": list(GRID), "inner": a.inner,
                             "reps": a.reps, "pcc_checks": checks, "mutate_correct": ok,
                             "arms": rows}, indent=1))
print("wrote", a.out)
