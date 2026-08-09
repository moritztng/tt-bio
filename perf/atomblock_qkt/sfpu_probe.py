#!/usr/bin/env python3
"""What holds SDPA at 30.7% of its mask-read floor: the SFPU, or the NOC?

SDPA's unavoidable traffic is the 130 MB bf16 mask: 335 us at this card's 388.4 GB/s read roof.
It measures 1091 us. Before asserting a mechanism, price the two candidates on the same bytes:

  clone DRAM->DRAM   moves the bytes with no maths      -> the NOC/DRAM cost alone
  ttnn.exp           moves the same bytes plus one SFPU exponential per element
  ttnn.multiply      moves the same bytes plus one FPU multiply per element

The exp-minus-clone difference is the SFPU exponential's own cost for H*M*N elements, which is
exactly the work SDPA's online softmax has to do on top of reading the mask.
"""
import json, statistics as st, time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
dev = get_device()
H, M, N = 4, 4032, 4032
elems = H * M * N
nbytes = elems * 2
x = ttnn.from_torch(torch.randn(1, H, M, N) * 0.1, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
print(f"{elems/1e6:.1f} M elements, {nbytes/1e6:.1f} MB bf16", flush=True)


def timed(fn, warm=2, pipe=2, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


res = {}
for name, fn in (("clone DRAM->DRAM", lambda: ttnn.clone(x, memory_config=DRAM)),
                 ("multiply (FPU)", lambda: ttnn.multiply(x, 1.5)),
                 ("exp (SFPU)", lambda: ttnn.exp(x)),
                 ("softmax bf16", lambda: ttnn.softmax(x, dim=-1))):
    t = timed(lambda: ttnn.deallocate(fn()))
    res[name] = {"us": round(t * 1e6, 1), "eff_GBs": round(2 * nbytes / t / 1e9, 1),
                 "G_elem_per_s": round(elems / t / 1e9, 1)}
    print(f"  {name:20s} {t*1e6:9.1f} us   {2*nbytes/t/1e9:6.1f} GB/s r+w   "
          f"{elems/t/1e9:6.1f} G elem/s", flush=True)

clone, exp = res["clone DRAM->DRAM"]["us"], res["exp (SFPU)"]["us"]
res["sfpu_exp_cost_us"] = round(exp - clone, 1)
res["sfpu_exp_G_elem_per_s"] = round(elems / ((exp - clone) * 1e-6) / 1e9, 1) if exp > clone else None
print(f"\nSFPU exponential alone: {exp - clone:.1f} us for {elems/1e6:.1f} M elements "
      f"= {res['sfpu_exp_G_elem_per_s']} G elem/s", flush=True)
print(f"SDPA measures 1091 us; mask read floor 335 us; exp floor {exp - clone:.1f} us; "
      f"sum {335 + (exp - clone):.1f} us", flush=True)
json.dump(res, open("perf/atomblock_qkt/sfpu_qb1c2.json", "w"), indent=2)
