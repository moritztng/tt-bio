#!/usr/bin/env python3
"""TRIANGLE_MULT_CHUNK_SIZE A/B: block time, bit-exactness and DRAM peak.

The trimul splits its hidden channels into chunks of TRIANGLE_MULT_CHUNK_SIZE and runs one
minimal_matmul + chunk + 2 gates + 3 layout moves + 1 matmul + 1 concat per chunk. At the
117-aa shape the two trimuls are 52% of a Pairformer block and most of that is per-op
overhead, so the chunk size is really an op-count knob.

Channels are independent in the triangle product, so a different chunking is a different
partition of the same sum and must be bit-exact. This checks that rather than assuming it,
on the real layer-0 weights, and records the device DRAM peak because a bigger chunk is a
bigger live allocation and the 12 GiB parts are the constraint.

    TT_VISIBLE_DEVICES=3 python3 perf/trunk_layout/trimul_chunk_ab.py --n 128 --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, action="append", default=None)
    ap.add_argument("--chunks", type=int, action="append", default=None)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--pipe", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ns = args.n or [128]
    chunks = args.chunks or [32, 64, 128]

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    results = []
    for N in ns:
        torch.manual_seed(0)
        s_h = torch.randn(1, N, 384)
        ref_z = None
        for C in chunks:
            # C == 0 means \"whatever _trimul_chunk_size picks\"; a positive C pins the width
            # by zeroing the L1 budget so the doubling loop never runs.
            T.TRIANGLE_MULT_CHUNK_SIZE = 32 if C == 0 else C
            T.TRIANGLE_MULT_L1_CHUNK_BUDGET = (64 * 320 * 320) if C == 0 else 0
            layer, c_z = build_layer(ckc)
            tm = layer.triangle_multiplication_start
            eff = T._trimul_chunk_size(N, tm._hidden)
            n_pairs = tm._hidden // eff
            # A chunk size that does not divide the hidden width silently drops channels.
            hidden2 = tm._hidden * 2
            covered = n_pairs * eff * 2
            exact_cover = covered == hidden2
            torch.manual_seed(0)
            z_h = torch.randn(1, N, N, c_z)
            s = ttnn.from_torch(s_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            z = ttnn.from_torch(z_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            # one call from a fixed input for the numerics check
            s1, z1 = layer(s, z)
            out = ttnn.to_torch(z1).float()
            if C == chunks[0]:
                ref_z = out
                bit_exact, maxabs = True, 0.0
            else:
                bit_exact = bool(torch.equal(out, ref_z))
                maxabs = float((out - ref_z).abs().max())
            # timing: keep feeding the layer its own output, as the trunk does
            st = {"s": s1, "z": z1}

            def call():
                st["s"], st["z"] = layer(st["s"], st["z"])
            for _ in range(args.warm):
                call()
            ttnn.synchronize_device(dev)
            ser = []
            for _ in range(args.iters):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                call()
                ttnn.synchronize_device(dev)
                ser.append((time.perf_counter() - t0) * 1e3)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.pipe):
                call()
            ttnn.synchronize_device(dev)
            pipe = (time.perf_counter() - t0) * 1e3 / args.pipe
            med = sorted(ser)[len(ser) // 2]
            peak = None
            try:
                peak = round(dev.get_memory_view(ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
                             * dev.num_dram_channels() / 2 ** 30, 4)
            except Exception:
                try:
                    peak = round(ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
                                 .total_bytes_allocated_per_bank / 2 ** 20, 3)
                except Exception:
                    pass
            row = dict(n=N, chunk=C, effective_chunk=eff, n_pairs=n_pairs, exact_cover=exact_cover,
                       hidden2=int(hidden2), covered=int(covered),
                       serial_ms=round(med, 3), pipe_ms=round(pipe, 3),
                       bit_exact_vs_chunk32=bit_exact, max_abs_vs_chunk32=maxabs,
                       dram_gib=peak)
            results.append(row)
            print(f"  N={N} C={C:3d}->{eff:3d} n_pairs={n_pairs} cover={covered}/{hidden2} "
                  f"serial {med:6.2f} ms  pipe {pipe:6.2f} ms  "
                  f"bit_exact={bit_exact} maxabs={maxabs:.3g} dram={peak}", flush=True)
            for t in (s, z, s1, z1, st["s"], st["z"]):
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
            del layer
    T.TRIANGLE_MULT_CHUNK_SIZE = 32
    T.TRIANGLE_MULT_L1_CHUNK_BUDGET = 64 * 320 * 320
    base = {r["n"]: r for r in results if r["chunk"] == chunks[0]}
    for r in results:
        r["speedup_vs_chunk32"] = round(base[r["n"]]["pipe_ms"] / r["pipe_ms"], 4)
    print()
    for r in results:
        print(f"N={r['n']} C={r['chunk']}: {r['speedup_vs_chunk32']}x", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
