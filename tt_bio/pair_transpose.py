"""``permute(t, (1, 0, 2))`` of a TILE pair tensor ``[S1, S2, C]``, as one data-movement program.

ttnn runs it as untilize, row permute, tilize through ROW_MAJOR (`tenstorrent._pair_transpose_impl`,
0.62 ms at the BindCraft 2 round's [288, 288, 128]: 126 + 302 + 195 us) or as a tiled permute whose
writes are row-granular scatter into DRAM (19 % of the copy roof). Tiling covers the last two axes,
so swapping the untiled S1 with the tile-row axis S2 moves single rows between tiles; done in L1
instead, the DRAM side is one read and one write of whole tiles. A unit is the 32 x 32 tile block
(It, Jt) of one channel tile: the reader brings the 32 input tiles of row-block It, the writer
gathers row q of each into output tile q with local NoC reads and writes the 32 out. No compute
kernel; bit-exact by construction. The work split is rne_add's.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import ops
from .envflags import env_flag
from .rne_add import _split_plan

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_transpose"
IN_CB, OUT_CB = 0, 16
_ELEM = {ttnn.bfloat16: 2, ttnn.float32: 4}

#: The lever. On by default: bit-exact by construction, graded in BindCraft 2's round on Blackhole and
#: digest-equal over Protenix-v2's 730-token fold on Wormhole.
PAIR_TRANSPOSE_FUSED = env_flag("TT_BIO_PAIR_TRANSPOSE_FUSED", True)

#: (calls served, calls declined), cumulative; sample at a round boundary.
STATS = [0, 0]
_CACHE: dict = {}


def _dims(t):
    """(lead, S1, S2, C) of a rank-3 or rank-4 tensor whose extra leading axis is 1, else None."""
    s = [int(d) for d in t.shape]
    if len(s) == 4 and s[0] == 1:
        s = s[1:]
    if len(s) != 3:
        return None
    return s


#: Fewest 32x32-tile units worth a program: below it the grid idles and the stock permute wins
#: (64x96x64: 12 units, 0.087 ms against 0.037; 288x288x32: 81 units, 0.124 against 0.135).
MIN_UNITS = 64


def shape_ok(t) -> bool:
    """bf16 or float32 TILE interleaved, whole tiles on every axis that moves, enough units."""
    s = _dims(t)
    return (s is not None and t.layout == ttnn.TILE_LAYOUT and t.dtype in _ELEM
            and all(d % 32 == 0 for d in s)
            and s[0] * s[1] * s[2] // 32 ** 3 >= MIN_UNITS
            and t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


def eligible(t) -> bool:
    """The lever, a tape entry when a tape is recording, and `shape_ok`."""
    if not PAIR_TRANSPOSE_FUSED or ops.declines_under_tape("pair_transpose"):
        return False
    ok = shape_ok(t)
    if not ok:
        STATS[1] += 1
    return ok


def _cb(idx, depth, dtype, core_grid):
    page = 32 * 32 * _ELEM[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=depth * page, core_ranges=core_grid,
                             format_descriptors=[fmt])


def _build(t, out, device, S1t, S2t, Ct):
    U = S1t * S2t * Ct
    num_cores, core_grid, cg1, cg2, work1, work2 = _split_plan(device, U)
    rt = ttnn.RuntimeArgs()
    placed = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rt[cx][cy] = [placed, per_core]
                    placed += per_core
    assert placed == U, (placed, U)
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_pair_transpose.cpp"), source_type=src,
        core_ranges=core_grid,
        compile_time_args=[S1t, S2t, Ct] + list(ttnn.TensorAccessorArgs(t).get_compile_time_args()),
        runtime_args=rt, common_runtime_args=[0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_pair_transpose.cpp"), source_type=src,
        core_ranges=core_grid,
        compile_time_args=[S1t, S2t, Ct, 16 * _ELEM[t.dtype]]
        + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
        runtime_args=rt, common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    return {"kernels": [reader, writer],
            "cbs": [_cb(IN_CB, 64, t.dtype, core_grid), _cb(OUT_CB, 32, t.dtype, core_grid)]}


@ops.fused_kernel("pair_transpose")
def pair_transpose(t, memory_config=None):
    """``permute(t, (1, 0, 2))`` (rank 3) or ``(0, 2, 1, 3)`` (rank 4), same dtype, new tensor."""
    device = t.device()
    S1, S2, C = _dims(t)
    shape = [S2, S1, C] if len(t.shape) == 3 else [1, S2, S1, C]
    out = ttnn.empty(ttnn.Shape(shape), t.dtype, ttnn.TILE_LAYOUT, device,
                     memory_config or ttnn.DRAM_MEMORY_CONFIG)
    g_ = device.compute_with_storage_grid_size()
    key = (device.id(), S1, S2, C, str(t.dtype), t.memory_config().buffer_type,
           out.memory_config().buffer_type, g_.x, g_.y)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(t, out, device, S1 // 32, S2 // 32, C // 32)
    reader, writer = entry["kernels"]
    reader.common_runtime_args = [t.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=entry["cbs"])
    STATS[0] += 1
    ttnn.generic_op([t, out], pd)
    return out
