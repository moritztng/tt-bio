#!/usr/bin/env python3
"""Per-call cost of `TriangleMultiplication.__call__` below its row-blocking gate.

`e0257fb82` is bit-exact at the panel median and 1.8 % slower there (138.5 -> 141.0 s over 18
serial legs with an A/A control). Below `SEQ_LEN_MORE_CHUNKING` the row-blocked branch is not taken,
so the executed path should be identical, and a fold-level bisect attributed only about 40 % of the
delta to the one hunk that does differ -- reading `H` and `batch` off `x` before the layer_norm
instead of off `x_norm_in` after it. 40 % at n=3 is too weak to engineer against.

A fold is a 140 s instrument for a per-call question. There are ~104 trimul calls in a 9ncy fold, so
2.5 s is ~24 ms a call, which a microbenchmark resolves in seconds. This times the op directly on
random weights at the shapes a 9ncy fold actually uses, with a device sync either side, and reports
the median so one slow call cannot carry the result.

    TT_VISIBLE_DEVICES=26 python3 perf/trimul_kernel/inproj_shape_read.py --n 505 --cz 64
    TT_VISIBLE_DEVICES=26 python3 perf/trimul_kernel/inproj_shape_read.py --n 977 --cz 384
"""
import argparse
import statistics
import time

import torch

import ttnn

from tt_bio.tenstorrent import TriangleMultiplication, get_device


def build(cz, hidden, ckc):
    """A TriangleMultiplication on random weights. Values are irrelevant here -- the question is
    dispatch cost per call, and the op's shapes and control flow depend only on cz/hidden/H."""
    g = torch.Generator().manual_seed(0)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    sd = {
        "norm_in.weight": rnd(cz), "norm_in.bias": rnd(cz),
        "norm_out.weight": rnd(cz), "norm_out.bias": rnd(cz),
        "g_in.weight": rnd(2 * hidden, cz), "p_in.weight": rnd(2 * hidden, cz),
        "g_out.weight": rnd(cz, cz), "p_out.weight": rnd(cz, hidden),
    }
    return TriangleMultiplication(False, sd, ckc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=977)
    ap.add_argument("--cz", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()
    hidden = args.hidden if args.hidden is not None else args.cz

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    tm = build(args.cz, hidden, ckc)

    x_host = torch.randn(1, args.n, args.n, args.cz, dtype=torch.float32).bfloat16()

    times = []
    for r in range(args.reps + 3):          # 3 warm-up reps, discarded
        x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = tm(x)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        if r >= 3:
            times.append(dt * 1e3)

    print(f"H={args.n} c_z={args.cz} hidden={hidden} reps={len(times)}")
    print(f"  median {statistics.median(times):8.3f} ms   "
          f"min {min(times):8.3f}   max {max(times):8.3f}   "
          f"stdev {statistics.stdev(times):6.3f}")


if __name__ == "__main__":
    main()
