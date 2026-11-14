"""``z[:, start:start + R] += blk`` on the device, in place, for a row block of a pair tensor.

A row-blocked pair op whose result is ``z + f(z)`` (the transition) used to assemble its blocks
with ``ttnn.concat`` and then add the whole update into ``z`` with ``add_``: the concat reads and
writes one pair tensor P, the add reads 2P and writes P. Adding each block straight into its rows
of ``z`` reads 2 blocks and writes 1, so the join and the separate add are gone (5P -> 3P per call,
and the concat ran well under the DRAM roof).

The sum is ``add_tiles`` into a 32-bit DEST packed to bfloat16, the arithmetic ``ttnn.add`` does on
two bfloat16 tensors (``kernels/rne_add/compute_rne_add.cpp`` measures them equal element for
element), so the bytes are the ones ``add_`` writes. ``perf/spd_pair/ops.py --groups trans`` checks
that with ``torch.equal`` on the device.

Run through ``ttnn.generic_op``; the descriptor is cached per shape and only the two addresses and
the page offset change per call.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split

_DIR = Path(__file__).resolve().parent / "kernels" / "add_rows"
TILE = 32
GRAN = 4                  # tiles per batch: a 32-bit DEST holds four
DEPTH = 4                 # batches each circular buffer holds

# (calls, tiles added)
STATS = [0, 0]
_CACHE: dict = {}


def _tiles(n):
    return -(-int(n) // TILE)


def ok(z, blk) -> bool:
    """Both bf16 TILE interleaved DRAM, rank 4, batch 1, the same row width and channels."""
    for t in (z, blk):
        mc = t.memory_config()
        if (len(t.shape) != 4 or int(t.shape[0]) != 1 or t.dtype != ttnn.bfloat16
                or t.layout != ttnn.TILE_LAYOUT or mc.buffer_type != ttnn.BufferType.DRAM
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return False
    return (int(z.shape[2]) == int(blk.shape[2]) and int(z.shape[3]) == int(blk.shape[3])
            and int(z.shape[3]) % TILE == 0)


def _build(z, blk, n_tiles):
    dev = z.device()
    g = dev.compute_with_storage_grid_size()
    page = TILE * TILE * 2
    num_cores, core_grid, cg1, cg2, w1, w2 = core_split.split_work_to_cores(g, n_tiles)
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    t0 = 0
    for group, per in ((cg1, w1), (cg2, w2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = [t0, per]
                    wr[cx][cy] = [t0, per]
                    cp[cx][cy] = [per]
                    t0 += per
    assert t0 == n_tiles, (t0, n_tiles)
    za = list(ttnn.TensorAccessorArgs(z).get_compile_time_args())
    ba = list(ttnn.TensorAccessorArgs(blk).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_DIR / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[GRAN, page] + za + ba, runtime_args=rd, common_runtime_args=[0, 0, 0],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[GRAN, page] + za, runtime_args=wr, common_runtime_args=[0, 0],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[GRAN], runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                            math_approx_mode=False, fp32_dest_acc_en=True))

    def cb(i):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=DEPTH * GRAN * page, core_ranges=core_grid,
                                 format_descriptors=[fmt])
    return {"ks": [reader, writer, compute], "cbs": [cb(0), cb(1), cb(16)]}


def add_rows(z, blk, start: int) -> None:
    """``z[:, start:start + R] += blk`` for ``blk`` of shape ``[1, R, S, C]``; ``blk`` is kept."""
    S, C = int(z.shape[2]), int(z.shape[3])
    row = _tiles(S) * (C // TILE)
    n = int(blk.shape[1]) * row
    key = (z.device().id(), n, tuple(ttnn.TensorAccessorArgs(z).get_compile_time_args()),
           tuple(ttnn.TensorAccessorArgs(blk).get_compile_time_args()))
    e = _CACHE.get(key)
    if e is None:
        e = _CACHE[key] = _build(z, blk, n)
    reader, writer, _ = e["ks"]
    reader.common_runtime_args = [z.buffer_address(), blk.buffer_address(), start * row]
    writer.common_runtime_args = [z.buffer_address(), start * row]
    STATS[0] += 1
    STATS[1] += n
    ttnn.generic_op([blk, z], ttnn.ProgramDescriptor(kernels=e["ks"], semaphores=[], cbs=e["cbs"]))
