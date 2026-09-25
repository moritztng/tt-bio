#!/usr/bin/env python3
"""Reproduce the trimul circular-buffer clash in a taped Evoformer backward, in about a minute.

One taped Evoformer block, forward and backward with checkpoint recompute, at a chosen length.
On main's tt_bio before `a09c43f42` it fails at n=224 and n=288: `tri_mul_in` narrows its chunk
64 -> 32, falls back to DRAM, and the static circular buffers still clash with an L1 buffer
(`program.cpp:1052`). With `a09c43f42` it passes at 224, 288 and 320 with zero ladder clashes.

The cause is the tape, not the trimul. An L1-resident intermediate that the forward merely
stopped using stays allocated, because the tape holds its handle, and the trimul's CBs in the
recompute collide with it. `a09c43f42` ("autograd: evict an L1 intermediate once its consumer has
read it", from OF3T's `130f8b1a9`) moves such an intermediate to DRAM once its consumer has read
it. It acts only under a tape.

Two earlier attributions in this file's history are withdrawn: "a size defect on main's
backward" (every failing control shared levers-off and main's code) and "the 352-token residency
threshold over-promises" (with the eviction fix, L1 fits at all three sizes).

    PYTHONPATH=.:perf/bcx_afgrad TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> \
        python3 perf/bcx_predictor/repro_trimul_288.py --seq 288
"""
import argparse
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=288)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--params", default=os.path.expanduser(
        "~/.boltz/af2/params/params_model_1_ptm.npz"))
    ap.add_argument("--multimer", action="store_true",
                    help="load a multimer_v3 checkpoint instead; the point of the flag is that "
                         "it changes nothing about whether this reproduces")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bcx_afgrad"))
    import afgrad as A

    dm, _ = A.load_models(args.params, multimer=args.multimer, refs=False)
    dev = A.Dev(dm.to_device())
    n = args.seq
    torch.manual_seed(0)
    m = torch.randn(1, n, 256) * 0.1
    z = torch.randn(n, n, 128) * 0.1

    mt = dev.ag.Tensor(dev.up(m), requires_grad=True)
    zt = dev.ag.Tensor(dev.up(z), requires_grad=True)
    t0 = time.time()
    with dev.tt.tape():
        mo, zo = dev.stack(mt, zt, 0, args.blocks, ckpt=True)
        dev.ag.backward([mo, zo], [dev.seed(torch.ones_like(m), mo),
                                   dev.seed(torch.ones_like(z), zo)])
    dev.sync()
    print(f"seq {n}: {args.blocks} block(s) forward+backward OK in {time.time() - t0:.1f}s "
          f"(multimer={args.multimer})")


if __name__ == "__main__":
    main()
