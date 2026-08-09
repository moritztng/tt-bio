#!/usr/bin/env python3
"""W4 milestone 1: how fast can a kernel do the sub-tile exchange, as a function of
transaction size?

The cube transpose that turns `gp_in_fused` into a channel-major matmul operand needs one
tile-index <-> intra-tile-row exchange (probe 2b: unavoidable by construction, and 78% of
the transform's cost). ttnn's `permute(0,2,1,3)` does it at 205.6 GB/s = 18% of the L1
copy roof on this card. A tile row of 32 bf16 spans two 16-wide faces, so it is two 32 B
pieces; if ttnn is transaction-issue-bound at 32 B, then a kernel that moves 64 B or wider
pieces should scale with the piece size, and the fused input op clears its gate. If the
curve is flat, the exchange is bandwidth- or latency-bound, ttnn is already near the
achievable number, and input-side fusion is worth ~1.04x on the block -- below this leg's
declared 1.05x stop gate.

So: a `generic_op` whose reader streams whole tiles into L1 and whose writer does a
local-L1 piece exchange (`noc_async_read_one_packet_with_state`, the low-overhead
state-based path) before writing whole tiles out. One knob: PIECE_BYTES. The exchange is
the same stride permutation the real kernel needs, `m -> (m%32)*(M/32) + m/32`, on the
production shape [1,320,320,64] = 13.1 MB, L1 -> L1.

Arms:
  copy        reader + writer only, no exchange. The floor for this kernel structure, and
              the correctness check: must be `torch.equal` with the input.
  R=<bytes>   with the exchange. Correctness here is weaker by design (a bandwidth probe,
              not the real op): the result must be a permutation of the input (same sorted
              values) and must differ from it, which catches a no-op or a partial write.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/gather_granularity.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import get_device

TILE_BYTES = 2048

READER = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr   = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t TPB   = TILES_PER_BLOCK;
    constexpr auto src_args  = TensorAccessorArgs<0>();
    const uint32_t tile_bytes = get_local_cb_interface(cb_in).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);
    uint32_t tid = start_tile;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_reserve_back(cb_in, TPB);
        uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t t = 0; t < TPB; ++t) {
            noc_async_read(s.get_noc_addr(tid + t), l1, tile_bytes);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, TPB);
        tid += TPB;
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in  = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t TPB    = TILES_PER_BLOCK;
    constexpr auto dst_args   = TensorAccessorArgs<0>();
    const uint32_t tile_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    const auto d = TensorAccessor(dst_args, dst_addr, tile_bytes);
#ifdef SCATTER
    constexpr uint32_t R   = PIECE_BYTES;
    constexpr uint32_t BLK = TPB * 2048;
    constexpr uint32_t M   = BLK / R;      // pieces per block
    constexpr uint32_t S   = 32;           // exchange span (32 tiles <-> 32 rows)
    constexpr uint32_t MS  = M / S;
#endif
    uint32_t tid = start_tile;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_wait_front(cb_in, TPB);
        const uint32_t src = get_read_ptr(cb_in);
#ifdef SCATTER
        const uint32_t dst = get_write_ptr(cb_out);
        noc_async_read_one_packet_set_state(get_noc_addr(src), R);
        for (uint32_t m = 0; m < M; ++m) {
            const uint32_t sm = (m % S) * MS + (m / S);
            noc_async_read_one_packet_with_state(src + sm * R, dst + m * R);
        }
        noc_async_read_barrier();
        const uint32_t out = dst;
#else
        const uint32_t out = src;
#endif
        for (uint32_t t = 0; t < TPB; ++t) {
            noc_async_write(out + t * tile_bytes, d.get_noc_addr(tid + t), tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_in, TPB);
        tid += TPB;
    }
}
"""


def build(dev, inp, out, tpb, grid_x, grid_y, piece=None):
    n_tiles = inp.volume() // (32 * 32)
    n_blocks = n_tiles // tpb
    ncores = grid_x * grid_y
    assert n_blocks % ncores == 0, f"{n_blocks} blocks over {ncores} cores is uneven"
    bpc = n_blocks // ncores
    cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))]
    )
    fmt = lambda idx: ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16,
                                              page_size=TILE_BYTES)
    cbs = [
        ttnn.CBDescriptor(total_size=2 * tpb * TILE_BYTES, core_ranges=cores,
                          format_descriptors=[fmt(0)]),
        ttnn.CBDescriptor(total_size=tpb * TILE_BYTES, core_ranges=cores,
                          format_descriptors=[fmt(16)]),
    ]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for y in range(grid_y):
        for x in range(grid_x):
            start = c * bpc * tpb
            r_rt[x][y] = [inp.buffer_address(), bpc, start]
            w_rt[x][y] = [out.buffer_address(), bpc, start]
            c += 1
    defines = [("TILES_PER_BLOCK", str(tpb))]
    if piece is not None:
        defines += [("SCATTER", "1"), ("PIECE_BYTES", str(piece))]
    K = ttnn.KernelDescriptor
    reader = K(kernel_source=READER, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
               compile_time_args=list(ttnn.TensorAccessorArgs(inp).get_compile_time_args()),
               defines=defines, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor())
    writer = K(kernel_source=WRITER, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
               compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
               defines=defines, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--tpb", type=int, default=32)
    ap.add_argument("--grid", default="10x10")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gx, gy = (int(v) for v in args.grid.split("x"))
    dev = get_device()
    N, C = args.n, args.c
    torch.manual_seed(0)
    L1 = ttnn.L1_MEMORY_CONFIG
    shape = [1, N, N, C]
    tin = torch.randn(shape)
    inp = ttnn.from_torch(tin, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                          memory_config=L1)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                        dev, L1)
    mb = 2 * (N * N * C * 2) / 1e6
    ref = tin.to(torch.bfloat16)
    ref_sorted = torch.sort(ref.flatten().float()).values
    rows = []
    print(f"\n=== sub-tile exchange, [1,{N},{N},{C}] bf16, L1->L1, {mb:.1f} MB r+w, "
          f"grid {gx}x{gy}, {args.tpb} tiles/block ===", flush=True)
    print(f"    reference points on this card: clone (copy roof) 1168.9 GB/s, "
          f"ttnn permute(0,2,1,3) 205.6 GB/s", flush=True)

    for piece in [None, 32, 64, 128, 256, 512, 1024, 2048]:
        label = "copy (no exchange)" if piece is None else f"exchange R={piece}B"
        try:
            pd = build(dev, inp, out, args.tpb, gx, gy, piece)
            got_t = ttnn.to_torch(ttnn.generic_op([inp, out], pd))
            if piece is None:
                note = f"exact_copy={torch.equal(got_t, ref)}"
            else:
                perm_ok = torch.equal(torch.sort(got_t.flatten().float()).values, ref_sorted)
                note = f"is_permutation={perm_ok} differs={not torch.equal(got_t, ref)}"
            for _ in range(2):
                for _ in range(args.reps):
                    ttnn.generic_op([inp, out], pd)
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(5):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(args.reps):
                    ttnn.generic_op([inp, out], pd)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3 / args.reps)
            ms = sorted(ts)[len(ts) // 2]
            gbs = mb / ms
            rows.append(dict(arm=label, ms=round(ms, 4), eff_gbs=round(gbs, 1), note=note))
            print("  %-22s %8.4f ms  %7.1f GB/s   %s" % (label, ms, gbs, note), flush=True)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            rows.append(dict(arm=label, error=msg))
            print("  %-22s FAILED %s" % (label, msg), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            dict(n=N, c=C, tpb=args.tpb, grid=args.grid, reps=args.reps, mb=mb, rows=rows),
            indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
