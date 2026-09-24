#!/usr/bin/env python3
"""The dest-carry guard's host cost per call: many small matmuls dispatched back to back.

[1,1,M,K] x [K,N] at a fold's small shapes, N calls per arm without a sync between them, then one
sync: `off` is ttnn's own plan, `on` goes through the guard, `pinned` passes the same k1 plan
explicitly (the guard returns at its first test). on - pinned is the guard's own Python.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
ckc = ttnn.types.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
calls = 2000
rows = []
for m, k, n in [(64, 128, 128), (1536, 128, 128), (1536, 384, 384)]:
    x = ttnn.from_torch(torch.randn(1, 1, m, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(torch.randn(k, n).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    pin = T._k1_program_config(m // 32, n // 32)
    arms = {"off": (False, None), "on": (True, None), "pinned": (True, pin)}
    r = {"m": m, "k": k, "n": n}
    for rep in range(2):
        for name, (guard, pc) in arms.items():
            T._DEST_CARRY_GUARD = guard
            kw = {} if pc is None else {"program_config": pc}
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(calls):
                ttnn.deallocate(ttnn.linear(x, w, compute_kernel_config=ckc, **kw))
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            if rep:
                r[name] = {"host_us_per_call": round((t1 - t0) / calls * 1e6, 1),
                           "wall_us_per_call": round((t2 - t0) / calls * 1e6, 1)}
    rows.append(r)
    print(json.dumps(r), flush=True)
out = Path(__file__).resolve().parent / "results" / "host_probe.json"
out.write_text(json.dumps({"rows": rows}, indent=1) + "\n")
