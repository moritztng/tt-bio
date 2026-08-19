"""Screen: sparse neighbour gather vs dense masked message pass on Blackhole.

The dominant op of any k-nearest-neighbour message-passing stage (ProteinMPNN /
SolubleMPNN, and the same shape recurs in other auxiliary design-pipeline stages)
is fetching neighbour node features. Two ways to express it on a Tenstorrent card:

  sparse  h[N,H] gathered by E_idx[N,k]  ->  [N,k,H]     N*k*H elements via ttnn.embedding
  dense   h[1,N,H] * mask[N,N,1]         ->  [N,N,H]     N*N*H elements via broadcast multiply

Gather runs at ~1.2 G elem/s on this chip, elementwise at ~69.5 G elem/s, so dense
should win whenever 57.9 * k / N > 1, i.e. up to N ~ 2800 at k=48.

Timing runs REPS iterations with a single sync at the end (isolated per-op timing
over-syncs and roughly doubles the apparent cost).
"""
import argparse, json, os, time

import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor


def bench(fn, reps, warmup=3):
    for _ in range(warmup):
        out = fn()
    ttnn.synchronize_device(out.device())
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(out.device())
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 300, 848])
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rec = {"chip": "p150a (Blackhole)", "k": args.k, "hidden": args.hidden,
           "reps": args.reps, "cells": []}

    ensure_p300_mesh_descriptor()
    t0 = time.perf_counter()
    device = ttnn.open_device(device_id=0)
    rec["device_open_s"] = time.perf_counter() - t0

    H = args.hidden
    try:
        for B in args.batch:
            for N in args.sizes:
                cell = {"batch": B, "n_residues": N}

                # sparse: one embedding lookup per design, N*k indices into an [N,H] table
                table = ttnn.from_torch(torch.randn(N, H), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device)
                idx = ttnn.from_torch(torch.randint(0, N, (B, N * args.k)),
                                      dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=device)
                g = lambda: ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT)
                cell["sparse_s"] = bench(g, args.reps)
                cell["sparse_elems"] = B * N * args.k * H
                ttnn.deallocate(table); ttnn.deallocate(idx)

                # control: the same row fetch expressed as ttnn.gather, which indexes
                # per element rather than per row. This is the op the 1.2 G elem/s
                # per-element gather rate was measured on. It is deliberately NOT
                # scaled by batch -- its job is to pin the per-element rate, so the
                # B=8 rows repeat the B=1 shape and should not be read as a batched arm.
                try:
                    src = ttnn.from_torch(torch.randn(N, H), dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=device)
                    gidx = ttnn.from_torch(
                        torch.randint(0, N, (N * args.k, 1)).expand(N * args.k, H).contiguous(),
                        dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
                    gg = lambda: ttnn.gather(src, 0, gidx)
                    cell["gather_s"] = bench(gg, max(3, args.reps // 4))
                    cell["gather_G_elem_s"] = N * args.k * H / cell["gather_s"] / 1e9
                    ttnn.deallocate(src); ttnn.deallocate(gidx)
                except Exception as exc:
                    cell["gather_s"] = None
                    cell["gather_error"] = str(exc)[:200]

                # dense: broadcast node features against an [N,N,1] neighbour mask
                h = ttnn.from_torch(torch.randn(B, 1, N, H), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=device)
                mask = ttnn.from_torch(torch.randint(0, 2, (B, N, N, 1)).float(),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=device)
                d = lambda: ttnn.multiply(h, mask)
                cell["dense_s"] = bench(d, args.reps)
                cell["dense_elems"] = B * N * N * H
                ttnn.deallocate(h); ttnn.deallocate(mask)

                cell["sparse_G_elem_s"] = cell["sparse_elems"] / cell["sparse_s"] / 1e9
                cell["dense_G_elem_s"] = cell["dense_elems"] / cell["dense_s"] / 1e9
                cell["dense_speedup"] = cell["sparse_s"] / cell["dense_s"]
                cell["predicted_speedup"] = 57.9 * args.k / N
                rec["cells"].append(cell)
                print("B=%d N=%4d  sparse %8.3f ms (%5.2f G/s)  dense %8.3f ms (%5.2f G/s)"
                      "  dense wins %6.2fx  (predicted %.2fx)  ttnn.gather %s"
                      % (B, N, cell["sparse_s"] * 1e3, cell["sparse_G_elem_s"],
                         cell["dense_s"] * 1e3, cell["dense_G_elem_s"],
                         cell["dense_speedup"], cell["predicted_speedup"],
                         ("%.3f ms (%.2f G/s)" % (cell["gather_s"] * 1e3, cell["gather_G_elem_s"]))
                         if cell.get("gather_s") else "n/a"), flush=True)
    finally:
        t0 = time.perf_counter()
        ttnn.close_device(device)
        rec["device_close_s"] = time.perf_counter() - t0

    print("device open %.2f s, close %.2f s" % (rec["device_open_s"], rec["device_close_s"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
