#!/usr/bin/env python3
"""Can the Transition row slice land in L1 instead of DRAM, and is it bit-exact?

The fusion in section 8.4 deletes the slice by writing a custom reader, which is multi-day. Before
that: the slice is a PURE COPY, so where its result lives cannot change a single output value, only
the traffic. `c = x[:, s:s+h]` currently inherits x's DRAM memory config and costs 0.0433 ms/chunk
moving 3.93 MB in and 3.93 MB out at 195 GB/s, 49 % of the measured copy roof. The very next op
(layer_norm) writes its own output to L1 already, so the chunk is L1-sized by construction.

If ttnn.slice can target L1, the 3.93 MB write and the layer_norm's 3.93 MB read both move off DRAM
for a one-line change and no kernel at all.

Two questions, in the order that decides it:
  1. BIT-EXACT? torch.equal on the full swiglu output, DRAM-sliced vs L1-sliced. A copy cannot
     change values, but "cannot" is what the screen is for -- ttnn may pad, re-tile or re-layout on
     a memory-config change, and the previous h-chunk attempt was torch.equal False for a reason
     nobody predicted either.
  2. HOW MUCH? the slice leg alone, and the whole chunk end to end, both ways.

Ceiling if it works, from screen_transition.json: 0.0433 ms/chunk x 52 chunks / 1.2113 over-read
= 1.86 ms/call, x 528 calls = 0.98 s/fold. Below section 8.4's 4.4 ms/call gate, which is a gate on
the multi-day fusion, not on a one-line memory-config change -- this is screened on its own terms.
"""
import json, sys, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/opendde-beat-b200")
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
h = max(1, int(T.TRANSITION_H_CHUNK_SIZE * min(1.0, (1024 * 128) / (W * C))))
n_chunks = -(-H // h)
CG, L1 = T.CORE_GRID_MAIN, ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
kw = dict(compute_kernel_config=ckc)

tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
g = torch.Generator().manual_seed(0)
x_dev = tt(torch.randn(B, H, W, C, generator=g, dtype=torch.float32).bfloat16().float())
nw, nb = tt(torch.randn(C, generator=g).float()), tt(torch.randn(C, generator=g).float())
w1 = tt(torch.randn(C, HID, generator=g).float())
w2 = tt(torch.randn(C, HID, generator=g).float())
w3 = tt(torch.randn(HID, C, generator=g).float())

print("x memory_config:", x_dev.memory_config(), flush=True)
base = x_dev[:, 0:h]
print("x[:, 0:h] memory_config:", base.memory_config(), flush=True)
ttnn.deallocate(base)


def slice_dram():
    return x_dev[:, 0:h]


def slice_l1():
    return ttnn.slice(x_dev, [0, 0, 0, 0], [B, h, W, C], memory_config=L1)


def swiglu(c):
    n = ttnn.layer_norm(c, weight=nw, bias=nb, epsilon=1e-5, memory_config=L1, **kw)
    p = ttnn.linear(n, w1, activation="silu", memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
    q = ttnn.linear(n, w2, memory_config=L1, dtype=ttnn.bfloat16, core_grid=CG, **kw)
    ttnn.deallocate(n)
    ttnn.multiply_(p, q)
    ttnn.deallocate(q)
    o = ttnn.linear(p, w3, dtype=ttnn.bfloat16, core_grid=CG, memory_config=DRAM, **kw)
    ttnn.deallocate(p)
    return o


out = {"shape": [B, h, W, C], "chunks_per_call": n_chunks}

# ---- 1. bit-exactness, before any timing -------------------------------------
try:
    cA = slice_dram(); oA = swiglu(cA); ttnn.deallocate(cA)
    tA = torch.Tensor(ttnn.to_torch(oA)).float(); ttnn.deallocate(oA)
    cB = slice_l1()
    out["l1_slice_memory_config"] = str(cB.memory_config())
    oB = swiglu(cB); ttnn.deallocate(cB)
    tB = torch.Tensor(ttnn.to_torch(oB)).float(); ttnn.deallocate(oB)
    out["torch_equal"] = bool(torch.equal(tA, tB))
    out["max_abs_diff"] = float((tA - tB).abs().max())
    print(f"torch.equal={out['torch_equal']}  max_abs_diff={out['max_abs_diff']:.3e}", flush=True)
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"
    print("FAILED:", out["error"], flush=True)
    (WT / "perf" / "oddeb200" / "screen_slice_l1.json").write_text(json.dumps(out, indent=1) + "\n")
    T.cleanup(); sys.exit(1)


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        ttnn.synchronize_device(dev)
        if r is not None:
            ttnn.deallocate(r)
    return (time.perf_counter() - t0) / n * 1e3


out["slice_dram_ms"] = round(timeit(slice_dram), 4)
out["slice_l1_ms"] = round(timeit(slice_l1), 4)
out["whole_dram_ms"] = round(timeit(lambda: swiglu(slice_dram())), 4)
out["whole_l1_ms"] = round(timeit(lambda: swiglu(slice_l1())), 4)
out["slice_delta_ms_per_chunk"] = round(out["slice_l1_ms"] - out["slice_dram_ms"], 4)
out["whole_delta_ms_per_chunk"] = round(out["whole_l1_ms"] - out["whole_dram_ms"], 4)
out["whole_delta_ms_per_call"] = round(out["whole_delta_ms_per_chunk"] * n_chunks, 4)
out["projected_s_per_fold"] = round(out["whole_delta_ms_per_call"] * 528 / 1000.0, 4)
p = WT / "perf" / "oddeb200" / "screen_slice_l1.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
T.cleanup()
