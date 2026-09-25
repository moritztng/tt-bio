#!/usr/bin/env python3
"""Reproduce the seq-288 trimul clash in the Evoformer backward, without a BindCraft 2 campaign.

Found by `bcx-multimer` while running the shipped five-model pool end to end. It is NOT a
multimer defect and NOT a pool defect: the monomer trunk fails identically at the same length,
and one resident trunk fails the same as five. It is main's backward at a length BindCraft 2
draws from every trajectory (binder 60..180 against a 115-residue target reaches seq 288), so it
blocks the campaign's GO condition for every variant.

What happens. `tri_mul_in` narrows its channel chunk 64 -> 32, then takes the designed fallback
to pair tensors in DRAM, and throws inside that fallback:

    tt_bio/tenstorrent.py:7189  host_acc_after_refusal
    ttnn::experimental::minimal_matmul
    TT_THROW program.cpp:1052  Statically allocated circular buffers ... clash with L1 buffers

    PYTHONPATH=.:perf/bcx_afgrad TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> \
        python3 perf/bcx_predictor/repro_trimul_288.py --seq 288

`--seq 224` is the control and passes; the campaign's published device trajectories ran there.
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
