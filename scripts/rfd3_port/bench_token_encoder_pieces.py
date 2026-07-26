"""Price the pieces of DiffusionTokenEncoder's pair-stack entry at D=1 and D=8.

p11's amortization profile named `token_encoder` as the whole batching shortfall
(at 3359 atoms it costs 1.65x what eight D=1 calls cost, +444 ms of a +376 ms
total excess per step). This replays its real arguments in steady state, with
fast runtime mode on, so each piece can be priced in ms and in bytes-moved
against the bandwidth an elementwise device pass over the same tensor achieves.

    TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/bench_token_encoder_pieces.py \
        --tokens 250 --batches 1 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

C_Z, N_BINS = 128, 65


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=250)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--reps", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    import ttnn
    from tt_bio.tenstorrent import get_device
    import tt_bio.rfd3 as rfd3

    dev = get_device()
    dt = ttnn.bfloat16
    I = args.tokens
    rows = []

    def timed(label, batch, fn, nbytes=None):
        fn()
        ttnn.synchronize_device(dev)
        best = float("inf")
        for _ in range(args.reps):
            start = time.perf_counter_ns()
            out = fn()
            ttnn.synchronize_device(dev)
            best = min(best, (time.perf_counter_ns() - start) / 1e6)
        rows.append((label, batch, best, nbytes))
        gb = f"{nbytes / best / 1e6:8.1f}" if nbytes else "        -"
        print(f"{label:<34}D={batch:<3d}{best:9.3f} ms  {gb} GB/s", flush=True)
        return out

    for batch in args.batches:
        print(f"--- I={I} batch={batch} ---", flush=True)
        R_ca = torch.randn(batch, I, 3) * 2.0
        Z1 = torch.randn(1, I, I, C_Z)
        # host: the one-hot distogram, as shipped
        onehot = timed("host bucketize (one_hot)", batch,
                       lambda: rfd3._bucketize_scaled_distogram(R_ca, n_bins=N_BINS))
        # host: what the device path would need instead -- just the bin indices
        def bin_only():
            D = torch.linalg.norm(R_ca.unsqueeze(-2) - R_ca.unsqueeze(-3), dim=-1)
            bins = torch.linspace(1.0 / 16.0, 30.0 / 16.0, N_BINS - 1)
            return torch.bucketize(D, bins).to(torch.int32)
        idx = timed("host bucketize (indices only)", batch, bin_only)
        # host: replicating the batch-invariant Z over the batch dim
        Zb = timed("host Z expand+contiguous", batch,
                   lambda: Z1.expand(batch, -1, -1, -1).contiguous(),
                   batch * I * I * C_Z * 4)
        timed("upload Z [B,I,I,128] bf16", batch,
              lambda: rfd3._tt(Zb, dev, dt), batch * I * I * C_Z * 2)
        timed("upload distogram [B,I,I,65]", batch,
              lambda: rfd3._tt(onehot, dev, dt), batch * I * I * N_BINS * 2)
        timed("upload bin idx [B,I,I] uint32", batch,
              lambda: rfd3._tt(idx, dev, ttnn.uint32), batch * I * I * 4)
        # device one-hot from the uploaded indices: gather rows of an identity
        eye = rfd3._tt(torch.eye(N_BINS), dev, dt)
        idx_dev = rfd3._tt(idx.reshape(1, -1), dev, ttnn.uint32)

        def dev_onehot():
            oh = ttnn.embedding(idx_dev, eye, layout=ttnn.ROW_MAJOR_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            oh = ttnn.reshape(oh, (batch, I, I, N_BINS))
            return ttnn.to_layout(oh, ttnn.TILE_LAYOUT)

        timed("device one_hot (embedding)", batch, dev_onehot,
              batch * I * I * N_BINS * 2)
        z_dev = rfd3._tt(Zb, dev, dt)
        d_dev = rfd3._tt(onehot, dev, dt)
        s_dev = rfd3._tt(onehot, dev, dt)
        cat = timed("device concat -> 258 ch", batch,
                    lambda: ttnn.concat([z_dev, d_dev, s_dev], dim=-1),
                    batch * I * I * (C_Z + 2 * N_BINS) * 2 * 2)
        # a bandwidth reference on the same tensor
        timed("device add on 258-ch tensor", batch,
              lambda: ttnn.add(cat, 1.0), batch * I * I * 258 * 2 * 2)

    print("\nlabel,batch,ms,bytes")
    for label, batch, ms, nbytes in rows:
        print(f"{label},{batch},{ms:.3f},{nbytes or ''}")


if __name__ == "__main__":
    main()
