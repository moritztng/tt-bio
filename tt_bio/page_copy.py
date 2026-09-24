"""Write a block of a pair tensor back into the pair it updates, on the device.

A row-blocked pair op whose update is ``z + f(z)`` computes row block ``I`` from ``z[I]`` alone
(the triangle attentions once their bias is built, the transition, the triangle multiplication's
output tail once its hidden exists). So ``z[I]`` is dead the moment block ``I`` is done, and the
block can overwrite it. Without that, the result is a second pair tensor beside ``z``, and past
``concat_host_bytes()`` its blocks were sent to the host and uploaded again. At OpenDDE's 1536
residue refiner (2987 structural tokens, a 6.95 GB pair) that was 452 GB each way over four
Pairformer blocks and ~950 s of an 1201 s seam on whglx (`perf/mgx_wide_seq/`).

In TILE layout a ``[1, S, S, C]`` tensor stores each row ``i`` as one contiguous run of
``(S/32) * (C/32)`` pages, so a row block is one page range and a column strip is ``S`` equal
runs. The kernel (``tt_bio/kernels/page_copy/copy_pages.cpp``) moves whole pages: the bytes that
land are the bytes the block holds, which is why this is bit-exact by construction and measured
with ``torch.equal`` rather than argued.

Run through ``ttnn.generic_op`` like the reblock kernels; the descriptor is cached per shape and
only the two addresses and the two page offsets change per call.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

KERNEL = Path(__file__).resolve().parent / "kernels" / "page_copy" / "copy_pages.cpp"
TILE = 32
CB = 0
DEPTH = 8                 # pages per half of the L1 window, 32 KiB per core in bf16

# (calls, pages moved)
STATS = [0, 0]
_CACHE: dict = {}


def _tiles(n):
    return -(-int(n) // TILE)


def ok(z, blk) -> bool:
    """Both tensors are bf16 TILE interleaved DRAM, rank 4, with the same channel width."""
    for t in (z, blk):
        mc = t.memory_config()
        if (len(t.shape) != 4 or int(t.shape[0]) != 1 or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or mc.buffer_type != ttnn.BufferType.DRAM
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return False
    return int(z.shape[3]) == int(blk.shape[3]) and int(z.shape[3]) % TILE == 0


def _build(src, dst, n_pages, run_len, src_stride, dst_stride):
    dev = dst.device()
    g = dev.compute_with_storage_grid_size()
    page = TILE * TILE * 2
    num_cores, core_grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, n_pages)
    rt = ttnn.RuntimeArgs()
    t0 = 0
    for group, per in ((cg1, w1), (cg2, w2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [per, t0 // run_len, t0 % run_len]
                    t0 += per
    assert t0 == n_pages, (t0, n_pages)
    ct = [CB, DEPTH, run_len, src_stride, dst_stride, page]
    ct.extend(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    ct.extend(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
    fmt = ttnn.CBFormatDescriptor(buffer_index=CB, data_format=ttnn.bfloat16, page_size=page)
    cbs = [ttnn.CBDescriptor(total_size=2 * DEPTH * page, core_ranges=core_grid,
                             format_descriptors=[fmt])]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=ct, runtime_args=rt,
        common_runtime_args=[0, 0, 0, 0], config=ttnn.ReaderConfigDescriptor())
    return {"k": k, "cbs": cbs}


def _copy(src, dst, n_pages, run_len, src_stride, dst_stride, src_off, dst_off):
    key = (dst.device().id(), n_pages, run_len, src_stride, dst_stride,
           tuple(ttnn.TensorAccessorArgs(src).get_compile_time_args()),
           tuple(ttnn.TensorAccessorArgs(dst).get_compile_time_args()))
    e = _CACHE.get(key)
    if e is None:
        e = _CACHE[key] = _build(src, dst, n_pages, run_len, src_stride, dst_stride)
    k = e["k"]
    k.common_runtime_args = [src.buffer_address(), dst.buffer_address(), src_off, dst_off]
    STATS[0] += 1
    STATS[1] += n_pages
    ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=e["cbs"]))


def write_rows(z, blk, start: int) -> None:
    """``z[:, start:start + R] = blk`` for ``blk`` of shape ``[1, R, S, C]``."""
    S, C = int(z.shape[2]), int(z.shape[3])
    assert int(blk.shape[2]) == S and int(blk.shape[0]) == 1, (z.shape, blk.shape)
    row = _tiles(S) * (C // TILE)
    n = int(blk.shape[1]) * row
    _copy(blk, z, n, n, n, n, 0, start * row)


def write_cols(z, blk, start: int) -> None:
    """``z[:, :, start:start + R] = blk`` for ``blk`` of shape ``[1, S, R, C]``, R and start
    multiples of 32 (the strip must own whole tiles along its width)."""
    S, R, C = int(z.shape[1]), int(blk.shape[2]), int(z.shape[3])
    assert start % TILE == 0 and (R % TILE == 0 or start + R == int(z.shape[2])), (start, R)
    assert int(blk.shape[1]) == S, (z.shape, blk.shape)
    ct = C // TILE
    run = _tiles(R) * ct
    _copy(blk, z, S * run, run, run, _tiles(z.shape[2]) * ct, 0, (start // TILE) * ct)
