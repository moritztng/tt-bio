#!/usr/bin/env python3
"""Is the kernel's intermediate actually wide? Pack the DEST into a float32 result and look.

The four-arm grade came back matching `ttnn.add(bf16, bf16)` element for element (97,234 of
331,776), which is the signature of a NARROW add -- but the kernel compiles with
`DST_ACCUM_MODE == 1`, so DEST is 32-bit. One of those two readings is wrong and only the
intermediate can say which: this packs the DEST straight into a float32 tensor and compares it
to the exact float64 sum, before any narrowing has a chance to happen.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bcx_p10_calls"))

import ttnn                                                             # noqa: E402
from tt_bio import rne_add                                              # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                     # noqa: E402
from rne_add_probe import bf16, tie_pair, N                             # noqa: E402


def main():
    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=0)
    try:
        rne_add.set_enabled(True)
        g = torch.Generator().manual_seed(7)
        n = N * N
        x = bf16(torch.randn(n, generator=g)).reshape(1, 1, N, N)
        u = bf16(torch.randn(n, generator=g)).reshape(1, 1, N, N)
        tx, th = tie_pair(n)
        tx, th = tx.reshape(1, 1, N, N), th.reshape(1, 1, N, N)

        for label, (a_t, b_t) in (("random", (x, u)), ("ties", (tx, th))):
            exact = a_t.double() + b_t.double()
            for arm in [(0, 0), (1, 0)]:
                rne_add.ADD_MODE, rne_add.ROUND_MODE = arm
                rne_add.OUT_DTYPE = ttnn.float32
                rne_add._CACHE.clear()
                a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                wide = ttnn.to_torch(rne_add.rne_add(a, b)).to(torch.float64)
                same_as_exact = int((wide == exact).sum())
                # Is the wide result already a bfloat16 value? That is the narrowing test.
                is_bf16 = int((wide == wide.to(torch.bfloat16).to(torch.float64)).sum())
                print("%-8s arm=%s  f32 out == exact f64 sum : %6d / %d   "
                      "f32 out is bf16-valued : %6d / %d"
                      % (label, arm, same_as_exact, wide.numel(), is_bf16, wide.numel()))
                for t in (a, b):
                    ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
