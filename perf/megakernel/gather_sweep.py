#!/usr/bin/env python3
"""W4 milestone 1b: raise the fused op's ceiling. The exchange is settled (7% at 256 B);
what limits the kernel now is its own copy floor, 467.6 GB/s against ttnn.clone's 1168.9
on the same shape. That floor caps the whole fused op, so sweep the two things that plausibly
set it: how much work a core takes per block (deeper NOC pipelining before the barrier) and
how many blocks the input CB can hold (how far the reader may run ahead of the writer).

Arms: pure copy (bit-exact against the input) and the exchange at 256 B, which is the piece
size the real kernel will use.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/gather_sweep.py
"""
import argparse, json, sys, time
from pathlib import Path
import torch, ttnn
from tt_bio.tenstorrent import get_device
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gg", str(Path(__file__).resolve().parent / "gather_granularity.py"))
gg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gg)

TILE_BYTES = 2048


def build(dev, inp, out, tpb, depth, gx, gy, piece=None):
    n_tiles = inp.volume() // (32 * 32)
    n_blocks = n_tiles // tpb
    ncores = gx * gy
    assert n_blocks % ncores == 0
    bpc = n_blocks // ncores
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                              ttnn.CoreCoord(gx - 1, gy - 1))])
    fmt = lambda i: ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16,
                                            page_size=TILE_BYTES)
    cbs = [ttnn.CBDescriptor(total_size=depth * tpb * TILE_BYTES, core_ranges=cores,
                             format_descriptors=[fmt(0)]),
           ttnn.CBDescriptor(total_size=tpb * TILE_BYTES, core_ranges=cores,
                             format_descriptors=[fmt(16)])]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for y in range(gy):
        for x in range(gx):
            r_rt[x][y] = [inp.buffer_address(), bpc, c * bpc * tpb]
            w_rt[x][y] = [out.buffer_address(), bpc, c * bpc * tpb]
            c += 1
    defines = [("TILES_PER_BLOCK", str(tpb))]
    if piece is not None:
        defines += [("SCATTER", "1"), ("PIECE_BYTES", str(piece))]
    K = ttnn.KernelDescriptor
    reader = K(kernel_source=gg.READER, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
               compile_time_args=list(ttnn.TensorAccessorArgs(inp).get_compile_time_args()),
               defines=defines, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor())
    writer = K(kernel_source=gg.WRITER, source_type=K.SourceType.SOURCE_CODE, core_ranges=cores,
               compile_time_args=list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
               defines=defines, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--grid", default="10x10")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gx, gy = (int(v) for v in a.grid.split("x"))
    dev = get_device()
    print("compute grid:", dev.compute_with_storage_grid_size(), flush=True)
    N, C = a.n, a.c
    torch.manual_seed(0)
    L1 = ttnn.L1_MEMORY_CONFIG
    tin = torch.randn(1, N, N, C)
    inp = ttnn.from_torch(tin, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                          memory_config=L1)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N, N, C]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, L1)
    mb = 2 * (N * N * C * 2) / 1e6
    ref = tin.to(torch.bfloat16)
    rows = []
    print(f"\n=== copy floor + 256 B exchange, [1,{N},{N},{C}] L1->L1 {mb:.1f} MB r+w, "
          f"grid {gx}x{gy}; roof: ttnn.clone 1168.9 GB/s ===", flush=True)
    for tpb in (4, 8, 16, 32, 64):
        for depth in (2, 4):
            if depth * tpb * TILE_BYTES > 700 * 1024:
                continue
            for piece, tag in ((None, "copy"), (256, "R=256B")):
                if piece is not None and (tpb * TILE_BYTES) % (piece * 32) != 0:
                    continue
                label = f"tpb={tpb:2d} depth={depth} {tag}"
                try:
                    pd = build(dev, inp, out, tpb, depth, gx, gy, piece)
                    got = ttnn.to_torch(ttnn.generic_op([inp, out], pd))
                    note = ("exact" if torch.equal(got, ref) else "MISMATCH") if piece is None \
                        else ("perm" if torch.equal(torch.sort(got.flatten().float()).values,
                                                   torch.sort(ref.flatten().float()).values)
                              else "NOT-PERM")
                    for _ in range(2):
                        for _ in range(a.reps):
                            ttnn.generic_op([inp, out], pd)
                    ttnn.synchronize_device(dev)
                    ts = []
                    for _ in range(5):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        for _ in range(a.reps):
                            ttnn.generic_op([inp, out], pd)
                        ttnn.synchronize_device(dev)
                        ts.append((time.perf_counter() - t0) * 1e3 / a.reps)
                    ms = sorted(ts)[len(ts) // 2]
                    rows.append(dict(arm=label, ms=round(ms, 4), eff_gbs=round(mb / ms, 1),
                                     note=note))
                    print("  %-26s %8.4f ms  %7.1f GB/s  %s" % (label, ms, mb / ms, note),
                          flush=True)
                except Exception as e:
                    print("  %-26s FAILED %s: %s" % (label, type(e).__name__, str(e)[:120]),
                          flush=True)
                    rows.append(dict(arm=label, error=f"{type(e).__name__}: {str(e)[:200]}"))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(n=N, c=C, grid=a.grid, mb=mb, rows=rows),
                                         indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
