#!/usr/bin/env python3
"""The DRAM leak, without the model: a weight stack, a tape, a backward and AdamW.

`of3t-p10trainout` measured ~0.98 GB a step over six steps of the real training composition
and could not run a seventh. This reproduces the same shape in a loop that costs a second,
so the cause can be found by bisection instead of by a 60 s step. It prints the DRAM
allocated after every phase of every iteration, and names who still holds a pre-step
weight handle.
"""
from __future__ import annotations

import argparse
import gc
import sys
import weakref
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def dram(dev):
    import ttnn
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--referrers", action="store_true",
                    help="hold one pre-step handle and name what still refers to it")
    a = ap.parse_args()

    from perf.of3t_p10leak import census as CZ
    import numpy as np
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    from tt_bio.train.optim import AdamW

    dev = get_device()
    d = a.dim
    raws = [ttnn.from_torch(torch.randn(d, d) * 0.02, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(a.layers)]
    params = {f"w{i}": ag.parameter(r) for i, r in enumerate(raws)}
    del raws
    x_raw = ttnn.from_torch(torch.randn(d, d) * 0.02, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
    weight_bytes = a.layers * d * d * 2
    print(f"{a.layers} weights of {d}x{d} bf16 = {weight_bytes/1e9:.3f} GB", flush=True)
    opt = AdamW(params, lr=a.lr)
    prev_cz = {}
    base = dram(dev)
    print(f"after construction {base/1e9:.3f} GB", flush=True)

    for it in range(a.iters):
        for t in params.values():
            t.grad = None
        with ag.tape():
            h = ag.Tensor(x_raw)
            for i in range(a.layers):
                h = ag.matmul(h, params[f"w{i}"])
            after_fwd = dram(dev)
            ag.backward([h], [ttnn.ones_like(h.value)])
        ttnn.synchronize_device(dev)
        after_bwd = dram(dev)
        watch = None
        if a.referrers:
            watch = params["w0"].value
        opt.step()
        ag.release_pins()
        del h
        gc.collect()
        ttnn.synchronize_device(dev)
        after_opt = dram(dev)
        cz = CZ.census(dev)
        print("  census:", CZ.diff(prev_cz, cz), f"total {CZ.total(cz)/1e9:.3f} GB", flush=True)
        prev_cz = cz
        print(f"  writes_skipped {opt.last_writes_skipped} of {len(params)}", flush=True)
        print(f"[iter {it}] fwd {after_fwd/1e9:.3f}  bwd {after_bwd/1e9:.3f}  "
              f"opt {after_opt/1e9:.3f} GB  (+{(after_opt-base)/1e9:.3f} since construction, "
              f"{(after_opt-base)/max(weight_bytes,1):.2f} weight copies)", flush=True)
        if watch is not None:
            refs = gc.get_referrers(watch)
            print(f"  pre-step w0 handle still has {len(refs)} referrers", flush=True)
            for r in refs[:8]:
                info = type(r).__name__
                if isinstance(r, dict):
                    ks = [k for k, v in r.items() if v is watch]
                    info += f" keys={ks[:3]} len={len(r)}"
                elif isinstance(r, list):
                    info += f" len={len(r)}"
                print(f"    - {info}", flush=True)
            del refs, watch
    print("done", flush=True)


if __name__ == "__main__":
    main()
