#!/usr/bin/env python3
"""Phase 1(ii) -- the exact-trilinear gather, graded against numpy.

Phase 1(i) emits float(voxel_index) + 2^23 and the sentinel 2^23-1 for pixels outside the radius.
This arm checks that the reader recovers the index with one AND, skips the sentinel, issues eight
8 B reads at RELION's own corner offsets, and lands each one at the right element of the right slot
tile -- the three things §4.4 and §8.7 assert and nothing has yet verified.

The model here is synthetic and small enough to sit in one core's L1, so the arm tests the reader's
addressing rather than the 130-core sharding, and the addresses are synthesised directly (including
sentinels) rather than taken from the dense stage, so a failure is attributable to this kernel alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
CB_A, CB_MDL, CB_SCR, CB_SLOT = 0, 1, 2, 16
TB = 4096
MDLX, MDLY, MDLZ = 8, 8, 8          # 512 complex voxels, 4 kB, comfortably L1-resident
MDLXY = MDLX * MDLY
NVOX = MDLXY * MDLZ
N_BLOCKS = 3
SENTINEL = 0x7FFFFF
OFF = (0, 1, MDLX, MDLX + 1, MDLXY, MDLXY + 1, MDLXY + MDLX, MDLXY + MDLX + 1)


def tile_to_faces(a):
    """[32, 32] logical -> the 4x(16x16) face order tt-metal stores a tile in."""
    return np.concatenate([a[r:r + 16, c:c + 16].reshape(-1)
                           for r in (0, 16) for c in (0, 16)])


def faces_to_tile(v):
    out = np.zeros((32, 32), dtype=v.dtype)
    i = 0
    for r in (0, 16):
        for c in (0, 16):
            out[r:r + 16, c:c + 16] = v[i:i + 256].reshape(16, 16)
            i += 256
    return out


def main():
    rng = np.random.default_rng(0)
    # Diagnostic model: each voxel's value IS its index, so a wrong result names the voxel that was
    # actually fetched instead of just being wrong by an opaque amount.
    v = np.arange(NVOX, dtype=np.float32)
    mdl = np.stack([v, -v], axis=1).astype(np.float32)

    # One address per pair, duplicated across the pair's two columns, some of them sentinels.
    idx_f = np.zeros((N_BLOCKS, 1024), dtype=np.uint32)
    want = np.zeros((N_BLOCKS, 8, 1024), dtype=np.float32)
    top = NVOX - MDLXY - MDLX - 2
    for b in range(N_BLOCKS):
        base = rng.integers(0, top, size=512).astype(np.uint32)   # arbitrary parity, as the real kernel has
        sent = rng.random(512) < 0.25
        base[sent] = SENTINEL
        for k in range(512):
            idx_f[b, 2 * k] = base[k]
            idx_f[b, 2 * k + 1] = base[k]
            if base[k] == SENTINEL:
                continue
            for s, o in enumerate(OFF):
                v = mdl[base[k] + o]
                want[b, s, 2 * k] = v[0]
                want[b, s, 2 * k + 1] = v[1]

    # float(idx) + 2^23 -- the mantissa is the index, so the reader's recovery is one AND.
    addr_np = np.zeros((N_BLOCKS, 32, 32), dtype=np.float32)
    for b in range(N_BLOCKS):
        addr_np[b] = faces_to_tile((idx_f[b].astype(np.float32) + np.float32(8388608.0)))
    want_t = np.stack([np.stack([faces_to_tile(want[b, s]) for s in range(8)])
                       for b in range(N_BLOCKS)])

    dev = ttnn.open_device(device_id=0)
    try:
        l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        ta = ttnn.from_torch(torch.from_numpy(addr_np).reshape(1, 1, -1, 32), dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)
        # The model as raw interleaved complex floats, L1-resident, read by raw offset.
        # One 4 kB page, staged into the reader's own L1 so the gather can address it by raw byte
        # offset. Tiled layout would permute the voxels, so this stays row-major and the page is
        # simply the raw complex array.
        tm = ttnn.from_torch(torch.from_numpy(mdl.reshape(-1)).reshape(1, 1, 1, 1024),
                             dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                             memory_config=dram)
        tout = ttnn.from_torch(torch.zeros(1, 1, N_BLOCKS * 8 * 32, 32), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)

        cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        def cb(i, depth):
            f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
            return ttnn.CBDescriptor(total_size=depth * TB, core_ranges=cg, format_descriptors=[f])

        aa = list(ttnn.TensorAccessorArgs(ta).get_compile_time_args())
        ma = list(ttnn.TensorAccessorArgs(tm).get_compile_time_args())
        da = list(ttnn.TensorAccessorArgs(tout).get_compile_time_args())
        rrt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rrt[0][0] = [ta.buffer_address(), tm.buffer_address()]
        wrt[0][0] = [tout.buffer_address()]
        mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
        pd = ttnn.ProgramDescriptor(kernels=[
            mk(KDIR / "reader_p1_gather.cpp",
               [CB_A, CB_SLOT, TB, N_BLOCKS, MDLX, MDLXY, CB_MDL, 1, CB_SCR] + aa + ma, rrt,
               ttnn.ReaderConfigDescriptor()),
            mk(KDIR / "writer_p1_slots.cpp", [CB_SLOT, TB, N_BLOCKS] + da, wrt,
               ttnn.WriterConfigDescriptor()),
        ], semaphores=[], cbs=[cb(CB_A, 2), cb(CB_MDL, 1), cb(CB_SCR, 1), cb(CB_SLOT, 16)])
        ttnn.generic_op([ta, tm, tout], pd)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(tout).numpy().reshape(N_BLOCKS, 8, 32, 32)
    finally:
        ttnn.close_device(dev)

    # Show what actually landed where, for the first few pairs of block 0, before summarising.
    g0 = np.array([tile_to_faces(got[0, s]) for s in range(8)])
    w0 = np.array([tile_to_faces(want_t[0, s]) for s in range(8)])
    shown = 0
    for k in range(512):
        if shown >= 4:
            break
        if abs(g0[0, 2 * k] - w0[0, 2 * k]) > 0:
            print("pair k=%d addr=%d  slot0 got %.0f want %.0f   slot1 got %.0f want %.0f"
                  % (k, idx_f[0, 2 * k], g0[0, 2 * k], w0[0, 2 * k],
                     g0[1, 2 * k], w0[1, 2 * k]), flush=True)
            shown += 1
    # The no-op trap: a gather that wrote nothing would match an all-zero expectation and report
    # bit-exact. Both sides have to be substantially non-zero before the comparison means anything.
    nz_want = float((want_t != 0).mean())
    nz_got = float((got != 0).mean())
    print("non-zero fraction: want %.3f  got %.3f" % (nz_want, nz_got), flush=True)
    assert nz_want > 0.5 and nz_got > 0.5, "vacuous comparison -- one side is mostly zero"

    res, ok = {}, True
    for s in range(8):
        e = float(np.abs(got[:, s] - want_t[:, s]).max())
        ok &= (e == 0.0)
        res["slot%d" % s] = e
        print("slot %d  max_abs_err %.6e" % (s, e), flush=True)
    res["all_bit_exact"] = bool(ok)
    res["nonzero_fraction"] = {"want": nz_want, "got": nz_got}
    print("ALL BIT-EXACT: %s" % ok, flush=True)
    (HERE / "p1_gather.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
