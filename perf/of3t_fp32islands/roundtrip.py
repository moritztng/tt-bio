#!/usr/bin/env python3
"""What a host round trip costs, per island shape, at the fold's own sizes.

An island we cannot supply on device can be evaluated on the host in real fp32/float64.
The question is only whether the transfer fits the site's call count. Measures
device -> host -> (torch fp32 op) -> device for each island's tensor shape.
"""
import json
import os
import statistics
import sys
import time

import torch
import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from tt_bio.autograd import precise_config  # noqa: E402

REPS = 7
CASES = [
    ("layer_norm, single track s", (1, 384, 384), 384, 48 * 4),
    ("layer_norm, pair track z", (1, 384, 384, 128), 128, 48 * 4),
    ("softmax, trunk APB scores", (1, 16, 384, 384), None, 48),
    ("softmax, triangle attention scores (one i-slice batch)", (16, 4, 384, 384), None, 48 * 2),
    ("distogram logits (loss, once per step)", (1, 384, 384, 64), None, 1),
]


def main():
    dev = ttnn.open_device(device_id=0)
    cfg = precise_config()
    rows = []
    try:
        for name, shp, c, calls in CASES:
            x = torch.randn(*shp, dtype=torch.float32)
            t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ttnn.to_torch(t)  # warm
            ts = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                h = ttnn.to_torch(t).float()
                h = (torch.nn.functional.layer_norm(h, (c,), eps=1e-5) if c
                     else torch.softmax(h, dim=-1))
                back = ttnn.from_torch(h.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=dev)
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
                ttnn.deallocate(back)
            ms = statistics.median(ts) * 1e3
            mb = x.numel() * 2 / 1e6
            rows.append(dict(site=name, shape=list(shp), mb_bf16=round(mb, 2),
                             ms_median=round(ms, 3), calls_per_step=calls,
                             s_per_step=round(ms * calls / 1e3, 3)))
            print(f"{name:52s} {mb:8.2f} MB  {ms:8.2f} ms  x{calls:4d} "
                  f"= {ms*calls/1e3:7.2f} s/step", flush=True)
            ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)
    with open(os.path.join(HERE, "roundtrip.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
