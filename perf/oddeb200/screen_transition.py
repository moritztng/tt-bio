#!/usr/bin/env python3
"""Re-anchor the Transition fusion's gate denominator on THIS tree.

The pre-committed gate (state doc 8.4) is `torch.equal` at h_effective = 10 AND >= 4.4 ms/call off
"the MEASURED 29.096 ms". That 29.096 ms and the 0.6539 ms/chunk per-leg breakdown were measured by
`opendde-to-4x` on its own tree with `perf/odde4x/screen1.py` S5. This lineage's most expensive
lesson is that planning against an inherited number is how the 88.61 s anchor survived 14 commits
past its expiry, so the denominator is re-measured before a line of kernel is written.

Shape, derived from the module rather than assumed. tenstorrent.py:3229-3250 with the production
pair track (W = 512, c = 384) gives
    _ref = 1024*128 = 131072 ; w_eff*c = 512*384 = 196608 ; 16 * min(1, 0.6667) -> 10
so the row chunk is [1, 10, 512, 384] and a 512-row call is 52 chunks. The script asserts this
against the real constants instead of hard-coding it.

Legs timed, each with a device sync so the number is a device time and not an enqueue
(PLAYBOOKS ACCELERATE rule 1):
    slice       x[:, s:s+h]                     deletable: a fused reader indexes the rows
    layer_norm  -> L1                           partly deletable: traffic and dispatch, not the SFPU
    fc1         (silu fused)                    NOT deletable, and not what the kernel is for
    fc2                                         NOT deletable
    multiply_   the swiglu gate                 deletable: the gate can happen in dest
    fc3         -> DRAM                         NOT deletable
plus `whole`, the entire swiglu on one chunk, which is what the per-leg sum has to be compared
against: the previous pass measured the per-op screen over-reading the call by 17.7 % because in
situ the ops overlap dispatch and here each pays its own sync. That factor is re-derived here, not
inherited.
"""
import json, sys, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/opendde-beat-b200")
sys.path.insert(0, str(WT))

import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
# Exactly what TorchWrapper gives every Transition (tenstorrent.py:4672-4683). fp32_dest_acc_en
# and packer_l1_acc are the two the fused kernel must not change, so the screen must not either.
ckc = (ttnn.types.WormholeComputeKernelConfig
       if dev.arch() == ttnn.Arch.WORMHOLE_B0
       else ttnn.types.BlackholeComputeKernelConfig)(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

B, H, W, C = 1, 512, 512, 384
HID = 4 * C

# Re-derive the row chunk from the module's own constants.
w_eff = W
_ref = 1024 * 128
h = max(1, int(T.TRANSITION_H_CHUNK_SIZE * min(1.0, _ref / (w_eff * C))))
n_chunks = -(-H // h)
print(f"row chunk h={h}  chunks/call={n_chunks}  shape=[{B},{h},{W},{C}]  hidden={HID}", flush=True)

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


def sync():
    ttnn.synchronize_device(dev)


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        r = fn()
        if r is not None and hasattr(r, "deallocate"):
            r.deallocate()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        sync()
        if r is not None and hasattr(r, "deallocate"):
            r.deallocate()
    return (time.perf_counter() - t0) / n * 1e3     # ms


kw = dict(compute_kernel_config=ckc)

res = {}
# L1 is the binding constraint here, not the clock. The module frees x_norm before the gate and
# frees the gate before fc3; an isolated screen that keeps x_norm, x_1, x_2 and a clone resident at
# once has less L1 left than the real path and the next matmul's CBs clash with it (measured:
# "Statically allocated circular buffers ... clash with L1 buffers", program 7, on the first
# attempt). So every leg allocates its own inputs, is timed, and frees them before the next.
res["slice"] = timeit(lambda: x_dev[:, 0:h])

c0 = x_dev[:, 0:h]
res["layer_norm"] = timeit(lambda: ttnn.layer_norm(c0, weight=nw, bias=nb, epsilon=1e-5,
                                                   memory_config=L1, **kw))

xn = ttnn.layer_norm(c0, weight=nw, bias=nb, epsilon=1e-5, memory_config=L1, **kw)
ttnn.deallocate(c0)
res["fc1_silu"] = timeit(lambda: ttnn.linear(xn, w1, activation="silu", memory_config=L1,
                                             dtype=ttnn.bfloat16, core_grid=CG, **kw))
res["fc2"] = timeit(lambda: ttnn.linear(xn, w2, memory_config=L1, dtype=ttnn.bfloat16,
                                        core_grid=CG, **kw))

a1 = ttnn.linear(xn, w1, activation="silu", memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
a2 = ttnn.linear(xn, w2, memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
ttnn.deallocate(xn)
# In-place, exactly as the module calls it, and no clone: a clone would double the 1536-channel
# intermediate in L1, which is what broke the first attempt. Repeating it drifts a1's values and
# leaves its timing untouched, which is all this leg is for.
res["multiply_"] = timeit(lambda: ttnn.multiply_(a1, a2))
ttnn.deallocate(a2)
res["fc3"] = timeit(lambda: ttnn.linear(a1, w3, dtype=ttnn.bfloat16, core_grid=CG,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw))
ttnn.deallocate(a1)


def whole():
    c = x_dev[:, 0:h]
    n = ttnn.layer_norm(c, weight=nw, bias=nb, epsilon=1e-5, memory_config=L1, **kw)
    p = ttnn.linear(n, w1, activation="silu", memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
    q = ttnn.linear(n, w2, memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
    ttnn.deallocate(n)
    ttnn.multiply_(p, q)
    ttnn.deallocate(q)
    o = ttnn.linear(p, w3, dtype=ttnn.bfloat16, core_grid=CG, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
    ttnn.deallocate(p)
    ttnn.deallocate(c)
    return o


res["whole_chunk"] = timeit(whole, n=20)

legs = ["slice", "layer_norm", "fc1_silu", "fc2", "multiply_", "fc3"]
per_leg_sum = sum(res[k] for k in legs)
over_read = per_leg_sum / res["whole_chunk"] - 1.0
deletable = res["slice"] + res["multiply_"] + 0.5 * res["layer_norm"]
out = {
    "host": "tt-quietbox2", "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else None,
    "shape": [B, h, W, C], "hidden": HID, "row_chunk_h": h, "chunks_per_call": n_chunks,
    "ms_per_chunk": {k: round(v, 4) for k, v in res.items()},
    "per_leg_sum_ms": round(per_leg_sum, 4),
    "whole_chunk_ms": round(res["whole_chunk"], 4),
    "per_op_over_read": round(over_read, 4),
    "whole_call_ms": round(res["whole_chunk"] * n_chunks, 4),
    "deletable_ms_per_chunk": round(deletable, 4),
    "deletable_ms_per_call": round(deletable * n_chunks / (1 + over_read), 4),
    "inherited": {"whole_call_ms": 29.096, "per_leg_sum_per_chunk_ms": 0.6539,
                  "over_read": 0.177, "gate_ms_per_call": 4.4},
}
out["gate_denominator_moved"] = round(out["whole_call_ms"] - 29.096, 4)
p = WT / "perf" / "oddeb200" / "screen_transition.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
T.cleanup()
