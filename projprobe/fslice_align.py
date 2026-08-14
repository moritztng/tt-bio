#!/usr/bin/env python3
"""What byte alignment does a per-row NoC read actually require? This is a hard design constraint.

The first build of stage 2 returned the source window at offset 0 when it asked for offset 3
elements (6 bytes): the read silently ignored the misalignment. Every earlier screen masked its
offsets with `& ~0x1F`, so none of them could have caught this -- S1's "the per-row offset is free"
is true only for offsets the NoC will honour.

This matters because the design needs an ARBITRARY per-row integer offset, floor(B*r + C), and if the
NoC only honours multiples of 16 bf16 elements then 15 of every 16 offsets are unreachable and the
residual shift has to be handled somewhere else. So: sweep the alignment of the requested offset and
find the smallest granularity that comes back bit-exact.

Arm `align = a` bytes forces every per-row offset to a multiple of a. Bit-exact at a means the NoC
honours a-byte offsets; a failure means it does not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "tt_bio" / "kernels" / "fslice"
CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT = 0, 4, 8, 12, 16
NROWS, SRC_TILES = 32, 2
WIN = 32 * SRC_TILES
ELEM = 2
TILE_B = 32 * 32 * ELEM
SRC_W = 512
BARRIER_EVERY = 4
ALIGNS = (2, 4, 8, 16, 32, 64, 128)


def build(dev, x, out, offs_bytes, row0, match_dst=0):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    # mode 0 needs no selection matrices, so the sel/frac accessors are still declared (the kernel's
    # compile-time layout is fixed) but nothing is loaded through them.
    rct = ([CB_SRC, WIN * ELEM, NROWS, SRC_W * ELEM, SRC_TILES, BARRIER_EVERY, 0,
            CB_SEL, CB_FRAC, 3 * SRC_TILES, TILE_B]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, 0]
    wct = [CB_OUT, TILE_B, SRC_TILES] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rrt[0][0] = ([x.buffer_address(), 1, row0] + [int(o) for o in offs_bytes]
                 + [out.buffer_address(), out.buffer_address()])
    crt[0][0] = [1]
    wrt[0][0] = [out.buffer_address(), 1, 0]
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_fslice.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_fslice.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_fslice.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(CB_SRC, TILE_B, 4 * BARRIER_EVERY * SRC_TILES),
                           cb(CB_TIL, TILE_B, 2 * SRC_TILES), cb(CB_SEL, TILE_B, 3 * SRC_TILES),
                           cb(CB_FRAC, TILE_B, 2), cb(CB_OUT, TILE_B, 2 * SRC_TILES)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"aligns": {}, "src_w": SRC_W, "win": WIN}
    try:
        rng = np.random.default_rng(11)
        src_i = rng.integers(-120, 120, size=(256, SRC_W)).astype(np.float32)
        src_t = torch.from_numpy(src_i).to(torch.bfloat16)
        srcn = src_t.to(torch.float64).numpy()
        srcs = {}
        for tag, bt in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
            srcs[tag] = ttnn.from_torch(
                src_t.reshape(1, 1, 256, SRC_W), dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt))
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * SRC_TILES, 32).to(torch.bfloat16),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        row0 = 8
        # Offsets must stay DISTINCT at every alignment. With a narrow span a coarse arm collapses
        # them all to 0, and then "bit-exact" only means "offset 0 works" -- a false pass, which is
        # exactly what the first run of this probe reported at 64 B.
        base = np.array([12 * r + 3 for r in range(NROWS)], dtype=np.int64)  # elements, 3..375
        for tag, md in (("dram", 0), ("l1", 0), ("dram", 1)):
          x = srcs[tag]
          for a in ALIGNS:
            # Requested per-row offsets, in bytes, forced to a multiple of `a`.
            ob = (base * ELEM) // a * a
            pd = build(dev, x, out, ob, row0, md)
            ttnn.generic_op([x, out], pd)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(out).reshape(SRC_TILES, 32, 32).to(torch.float64).numpy()
            got = np.concatenate([got[i] for i in range(SRC_TILES)], axis=1)
            exp = np.stack([srcn[row0 + r, ob[r] // ELEM: ob[r] // ELEM + WIN] for r in range(NROWS)])
            nbad = int((got != exp).sum())
            distinct = len(set(int(o) for o in ob))
            res["aligns"][f"{tag}/md{md}/{a}"] = {"mismatches": nbad, "bit_exact": nbad == 0,
                                     "distinct_offsets": distinct,
                                     "offsets_bytes": [int(o) for o in ob[:8]]}
            verdict = "BIT-EXACT" if nbad == 0 else f"{nbad:5d} mismatches"
            if nbad == 0 and distinct < 4:
                verdict += "  (INCONCLUSIVE: too few distinct offsets)"
            print(f"{tag}/md{md} align {a:3d} B  ({distinct:2d} distinct)  {verdict}", flush=True)
            json.dump(res, open(HERE / "fslice_align.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    ok = [k for k, v in res["aligns"].items()
          if v["bit_exact"] and v["distinct_offsets"] >= 4]
    print("\nconclusive passes (>=4 distinct offsets):")
    for k in ok:
        print("   ", k)
    res["conclusive_passes"] = ok
    json.dump(res, open(HERE / "fslice_align.json", "w"), indent=1)


main()
