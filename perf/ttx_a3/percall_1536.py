#!/usr/bin/env python3
"""Why a 4.2x op win can arrive at the fold as a 2.7x LOSS: per-call cost with fresh operands.

`perf/bgsdpa/sdpa_ab.py` times the same q/k/v/bias tensors three times over. A fold does not: each
of its 560 triangle-attention calls arrives with operands at different device addresses. If
anything on the fused path is rebuilt per call when the buffers change -- a program, a plan, a
kernel hash -- a fixed-tensor micro-benchmark cannot see it and a fold pays it 560 times.

Three arms at 1536 aa, h=4 d=32, batch 1536 (the real triangle-attention shape), each arm timed per
call with a device sync around it:

  reuse   one operand set, called N times. This is what sdpa_ab.py measures.
  rotate  R operand sets, called round-robin, so consecutive calls differ in address.
  fresh   a newly allocated bias every call, q/k/v reused. Isolates the bias, which is the tensor
          this kernel treats specially (it is the one held persistently in L1).

Run for both routes. If `on` degrades from reuse to rotate/fresh and `off` does not, the fold
regression is per-call overhead on the fused path, not the kernel's arithmetic.

    TT_VISIBLE_DEVICES=3 python3 perf/ttx_a3/percall_1536.py --out perf/ttx_a3/percall_1536.json
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.environ.get("WT", os.getcwd()))
from tt_bio import tenstorrent as T            # noqa: E402
from tt_bio import triatt_sdpa as TS           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--batch", type=int, default=None, help="default: seq, the real shape")
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--sets", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, H, D = a.seq, a.heads, a.head_dim
    B = a.batch or S
    dev = T.get_device()
    torch.manual_seed(0)
    scale = D ** -0.5

    def mk(shape):
        t = (torch.randn(shape, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        return ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    sets = []
    for _ in range(a.sets):
        sets.append(tuple(mk([B, H, S, D]) for _ in range(3)) + (mk([1, H, S, S]),))
    ttnn.synchronize_device(dev)

    res = {"seq": S, "heads": H, "head_dim": D, "batch": B, "calls": a.calls, "sets": a.sets,
           "grid": list(T.COMPUTE_GRID_MAIN), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "commit": os.popen("git rev-parse HEAD").read().strip(),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": {}}

    def timed(q, k, v, bias):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = T._tri_att_sdpa_at(q, k, v, bias, scale)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(o)
        return dt

    for route in (False, True):
        T._SDPA_FUSED_LARGE_S = route
        tag = "on" if route else "off"
        for mode in ("reuse", "rotate", "fresh"):
            T.SDPA_CHUNK_PICKS.clear()
            TS.STATS[:] = [0, 0]
            timed(*sets[0])                      # warmup, discarded
            ms = []
            for i in range(a.calls):
                if mode == "reuse":
                    ms.append(timed(*sets[0]))
                elif mode == "rotate":
                    ms.append(timed(*sets[i % a.sets]))
                else:
                    q, k, v, _ = sets[0]
                    b = mk([1, H, S, S])
                    ms.append(timed(q, k, v, b))
                    ttnn.deallocate(b)
            res["arms"][f"{tag}:{mode}"] = {
                "ms": [round(x, 3) for x in ms],
                "median_ms": round(st.median(ms), 3),
                "min_ms": round(min(ms), 3), "max_ms": round(max(ms), 3),
                "pick": {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()},
                "served": TS.STATS[0], "declined": TS.STATS[1]}
            print(f"{tag}:{mode:7s} median {st.median(ms):9.3f} ms  min {min(ms):9.3f}  "
                  f"max {max(ms):9.3f}  served {TS.STATS[0]}", flush=True)

    for mode in ("reuse", "rotate", "fresh"):
        o = res["arms"][f"off:{mode}"]["median_ms"]
        n = res["arms"][f"on:{mode}"]["median_ms"]
        res[f"speedup_{mode}"] = round(o / n, 4)
        print(f"speedup {mode:7s} {o / n:.4f}x", flush=True)

    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    ttnn.close_device(dev)
    return 0


sys.exit(main())
