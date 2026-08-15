#!/usr/bin/env python3
"""What the Transition fusion actually deletes, measured the way the fold issues it.

`opendde-beat-b200` priced the fusion off legs timed one at a time with their own sync, then found
(its 9.15) that the slice leg costs 0.0222 ms/chunk issued 52 at a time against 0.0433 timed alone,
a 95 % over-read on one of the three legs the kernel proposes to delete. The other two were never
re-measured, and the fusion's pre-committed gate (>= 4.4 ms/call) sits inside the resulting
uncertainty. This screen removes the uncertainty by measuring the marginal cost of each leg
SUBTRACTIVELY, in situ: one arm issues all 52 chunks of a real 512-row call with one sync, the
others issue the same 52 chunks with one leg taken out. The delta is what the kernel would win.

Arms:
  full            52 x (slice, layer_norm, fc1(silu), fc2, multiply_, fc3)   the shipped path
  no_mul          the same without the swiglu gate multiply
  no_slice        the same with one pre-sliced chunk reused (a fused reader indexes rows)
  no_slice_no_mul both deletions in ONE arm -- this is the fusion's own upper bound
  no_ln           the same with one pre-normed chunk reused (upper bound on the layer_norm leg;
                  the kernel deletes its traffic and dispatch, not its SFPU arithmetic, so it is
                  priced at 50 % of this delta)

PREDICTION, written before the run (state doc 3.2): full lands 28-30 ms/call reproducing the
28.8715 ms `opendde-beat-b200` 6.1 measured; deletable = (full - no_slice_no_mul) + 0.5 x
(full - no_ln) lands in 3.0-6.8 ms/call. GO iff >= 4.4.
"""
import json, os, sys, time
from pathlib import Path

WT = Path(os.environ["WT"])
sys.path.insert(0, str(WT))

import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = (ttnn.types.WormholeComputeKernelConfig
       if dev.arch() == ttnn.Arch.WORMHOLE_B0
       else ttnn.types.BlackholeComputeKernelConfig)(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

B, H, W, C = 1, 512, 512, 384
HID = 4 * C
w_eff = W
_ref = 1024 * 128
h = max(1, int(T.TRANSITION_H_CHUNK_SIZE * min(1.0, _ref / (w_eff * C))))
starts = list(range(0, H, h))
n_chunks = len(starts)
print(f"row chunk h={h} chunks/call={n_chunks} shape=[{B},{h},{W},{C}] hidden={HID}", flush=True)

tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
g = torch.Generator().manual_seed(0)
x_dev = tt(torch.randn(B, H, W, C, generator=g, dtype=torch.float32).bfloat16().float())
nw = tt(torch.randn(C, generator=g).float())
nb = tt(torch.randn(C, generator=g).float())
w1 = tt(torch.randn(C, HID, generator=g).float())
w2 = tt(torch.randn(C, HID, generator=g).float())
w3 = tt(torch.randn(HID, C, generator=g).float())

CG = T.CORE_GRID_MAIN
L1 = ttnn.L1_MEMORY_CONFIG
kw = dict(compute_kernel_config=ckc)

# persistent operands for the deletion arms, allocated once and never freed inside a call
c_fixed = x_dev[:, 0:h]
n_fixed = ttnn.layer_norm(c_fixed, weight=nw, bias=nb, epsilon=1e-5, memory_config=L1, **kw)


def call(slice_=True, ln=True, mul=True):
    for s in starts:
        c = x_dev[:, s:s + h] if slice_ else c_fixed
        if ln:
            n = ttnn.layer_norm(c, weight=nw, bias=nb, epsilon=1e-5, memory_config=L1, **kw)
        else:
            n = n_fixed
        p = ttnn.linear(n, w1, activation="silu", memory_config=L1, dtype=ttnn.bfloat16,
                        core_grid=CG, **kw)
        q = ttnn.linear(n, w2, memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
        if ln:
            ttnn.deallocate(n)
        if mul:
            ttnn.multiply_(p, q)
        ttnn.deallocate(q)
        o = ttnn.linear(p, w3, dtype=ttnn.bfloat16, core_grid=CG,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        ttnn.deallocate(p)
        if slice_:
            ttnn.deallocate(c)
        ttnn.deallocate(o)


def timeit(fn, n=12, warm=3):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


ARMS = {
    "full":            dict(slice_=True,  ln=True,  mul=True),
    "no_mul":          dict(slice_=True,  ln=True,  mul=False),
    "no_slice":        dict(slice_=False, ln=True,  mul=True),
    "no_slice_no_mul": dict(slice_=False, ln=True,  mul=False),
    "no_ln":           dict(slice_=True,  ln=False, mul=True),
    "full_2":          dict(slice_=True,  ln=True,  mul=True),   # A/A control, last
}
res = {}
for name, kwargs in ARMS.items():
    med, lo, hi = timeit(lambda kwargs=kwargs: call(**kwargs))
    res[name] = {"median_ms": round(med, 4), "min_ms": round(lo, 4), "max_ms": round(hi, 4)}
    print(f"{name:16s} {med:8.4f} ms/call  [{lo:.4f}, {hi:.4f}]", flush=True)

full = res["full"]["median_ms"]
aa = abs(res["full_2"]["median_ms"] - full)
d_mul = full - res["no_mul"]["median_ms"]
d_slice = full - res["no_slice"]["median_ms"]
d_both = full - res["no_slice_no_mul"]["median_ms"]
d_ln = full - res["no_ln"]["median_ms"]
deletable = d_both + 0.5 * d_ln
out = {
    "host": "tt-quietbox2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
    "shape": [B, h, W, C], "chunks_per_call": n_chunks,
    "arms_ms_per_call": res,
    "aa_noise_ms": round(aa, 4),
    "marginal_ms_per_call": {"multiply_": round(d_mul, 4), "slice": round(d_slice, 4),
                             "slice+multiply_ (one arm)": round(d_both, 4),
                             "layer_norm (upper bound)": round(d_ln, 4)},
    "deletable_ms_per_call": round(deletable, 4),
    "deletable_s_per_fold_at_528_calls": round(deletable * 528 / 1000.0, 4),
    "gate_ms_per_call": 4.4,
    "verdict": "GO" if deletable >= 4.4 else "NO-GO",
    "inherited_for_comparison": {"per_op_synced_deletable_ms_per_call": "4.4-5.8",
                                 "whole_call_ms_measured_by_oddeb200": 28.8715},
}
p = WT / "perf" / "odde4pd" / "screen_marginal.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
T.cleanup()
