#!/usr/bin/env python3
"""Token-encoder pair Transition microbench: the shipped row-chunked SwiGLU vs whole-tensor variants.

    gap_pt_bench.py OUT.json [I=685] [REPS=10]

The pair Transitions (2 x hidden 256 + 2 x hidden 512 per token-encoder call, 2 calls a step) run as
~13 row chunks x 3 linears each, which is most of the token encoder's 312 linears/step (Tracy, fast+rs,
p150a: 50.3 ms/step). The chunk height is part of the arithmetic, so the chunking keeps the default path
reproducible; --fast does not need that. Arms, all on the same random weights and input, [1,I,I,128]:

  A   shipped Transition.__call__ (chunked, L1-resident), bf16 hidden
  A8  A with the hidden in bfp8 (what --fast runs today, RFD3_TRANSITION_B8)
  W   whole tensor, one _swiglu call in DRAM, bf16 hidden
  W8  W with bfp8 hidden
  F8  whole tensor, fc1|fc2 as ONE linear (weights concatenated), slice, silu * b, fc3; bfp8 hidden

Error is against a float64 host reference of the same weights and the bf16-rounded input: max and mean
abs over the output, and the rel L2. ms/call synchronized.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                       # noqa: E402

ensure_p300_mesh_descriptor()
import torch                                                              # noqa: E402
import ttnn                                                               # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402

OUT = sys.argv[1]
I = int(sys.argv[2]) if len(sys.argv) > 2 else 685
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
C = 128
ckc = M._default_compute_kernel_config()
dev = M.get_device()
res = {"I": I, "reps": REPS, "card": os.environ.get("TT_VISIBLE_DEVICES"), "rows": {}}


def ref64(x, sd):
    """float64 Transition: rms_norm -> silu(x W1) * (x W2) -> W3, in 64-row blocks to bound memory."""
    out = torch.empty(x.shape[:-1] + (C,), dtype=torch.float64)
    g, w1, w2, w3 = (sd[k].double() for k in ("layer_norm_1.weight", "linear_1.weight",
                                                "linear_2.weight", "linear_3.weight"))
    for s in range(0, x.shape[1], 64):
        xb = x[:, s:s + 64].double()
        xn = xb * torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        out[:, s:s + 64] = (torch.nn.functional.silu(xn @ w1.T) * (xn @ w2.T)) @ w3.T
    return out


def timed(fn):
    y = fn()
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return y, 1e3 * (time.perf_counter() - t0) / REPS


for n in (2, 4):
    torch.manual_seed(n)
    hid = n * C
    sd = {"layer_norm_1.weight": 1 + 0.1 * torch.randn(C),
          "linear_1.weight": torch.randn(hid, C) / C ** 0.5,
          "linear_2.weight": torch.randn(hid, C) / C ** 0.5,
          "linear_3.weight": torch.randn(C, hid) / hid ** 0.5}
    tr = M.Transition(sd, ckc, C, n=n)
    x_host = torch.randn(1, I, I, C).bfloat16().float()
    x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref = ref64(x_host, sd)
    w12 = ttnn.from_torch(torch.cat([sd["linear_1.weight"], sd["linear_2.weight"]]).T.contiguous(),
                          layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def fused8():
        xn = ttnn.rms_norm(x, weight=tr.norm_w, epsilon=1e-6, compute_kernel_config=ckc)
        ab = ttnn.linear(xn, w12, compute_kernel_config=ckc, dtype=ttnn.bfloat8_b, core_grid=M.BATCH_INVARIANT_GRID)
        ttnn.deallocate(xn)
        a = ttnn.silu(ttnn.slice(ab, [0, 0, 0, 0], [1, I, I, hid]))
        b = ttnn.slice(ab, [0, 0, 0, hid], [1, I, I, 2 * hid])
        ttnn.deallocate(ab)
        m = ttnn.multiply(a, b, dtype=ttnn.bfloat8_b)
        out = ttnn.linear(m, tr.fc3_w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=M.CORE_GRID_MAIN)
        for t in (a, b, m):
            ttnn.deallocate(t)
        return out

    arms = {"A": (False, lambda: tr(x)), "A8": (True, lambda: tr(x)),
            "W": (False, lambda: tr._swiglu(x, None)), "W8": (True, lambda: tr._swiglu(x, None)),
            "F8": (True, fused8)}
    row = {}
    for name, (b8, fn) in arms.items():
        M._TRANSITION_B8 = b8
        try:
            y, ms = timed(fn)
        except Exception as exc:                                          # one arm failing must not void the rest
            row[name] = {"error": repr(exc)[:300]}
            print(n, name, row[name], flush=True)
            continue
        d = ttnn.to_torch(y).double()[..., :C] - ref
        ttnn.deallocate(y)
        row[name] = {"ms_per_call": ms, "max_abs": d.abs().max().item(), "mean_abs": d.abs().mean().item(),
                     "rel_l2": (d.norm() / ref.norm()).item()}
        print(n, name, row[name], flush=True)
    M._TRANSITION_B8 = False
    res["rows"]["hidden%d" % hid] = row
    for t in (x, w12):
        ttnn.deallocate(t)

json.dump(res, open(OUT, "w"), indent=1)
