"""``z += permute(u, (1, 0, 2))`` in one program: the ending triangle attention's way back.

The ending triangle attention runs on the transposed pair, so its update comes out transposed. Today it
goes back through ``pair_transpose`` (read u, write a new pair) and then the layer's ``add_`` (read z and
that pair, write z): 5P of DRAM traffic per call, P = one bf16 pair (277 MB at 736 tokens). This reads
u and z once and writes z once, 3P.

Same units and the same reader as ``pair_transpose``'s split kernels (the reader is that file, unchanged).
The writer reads the z tiles a unit lands on and passes the shuffled slot to a compute kernel instead of
writing it; the compute kernel adds with ``add_tiles`` into a 32-bit DEST packed to bfloat16, which is
``ttnn.add``'s arithmetic on two bfloat16 tensors. So the bytes are those of pair_transpose + add_, and
``perf/spd_pair/ops.py --groups tadd`` checks it with torch.equal.

``TT_BIO_PAIR_TRANSPOSE_ADD=0`` restores the transpose and the separate add.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import ops
from . import pair_transpose as PT
from .envflags import env_flag
from .rne_add import _split_plan

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_transpose_add"
PAIR_TRANSPOSE_ADD = env_flag("TT_BIO_PAIR_TRANSPOSE_ADD", True)
GRAN = 4                  # sums per DEST batch: a 32-bit DEST holds four tiles
QR = 8                    # output tiles the reader shuffles; the writer also carries the z traffic

# (calls served, calls declined)
STATS = [0, 0]
_CACHE: dict = {}


def eligible(u, z) -> bool:
    """u [S1, S2, C] or [1, S1, S2, C] and z its transpose, both bf16 TILE interleaved DRAM."""
    ok = (PAIR_TRANSPOSE_ADD and not ops.taping() and PT.shape_ok(u) and PT.shape_ok(z)
          and u.dtype == ttnn.bfloat16 and z.dtype == ttnn.bfloat16)
    if ok:
        s, t = PT._dims(u), PT._dims(z)
        ok = t == [s[1], s[0], s[2]] and all(
            x.memory_config().buffer_type == ttnn.BufferType.DRAM for x in (u, z))
    if not ok:
        STATS[1] += 1
    return ok


def _build(u, z, device, S1t, S2t, Ct):
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
    row = 16 * 2
    reader = ttnn.KernelDescriptor(
        kernel_source=str(PT.KERNEL_DIR / "reader_pair_transpose_split.cpp"), source_type=src,
        core_ranges=core_grid,
        compile_time_args=[S1t, S2t, Ct, row, QR] + list(ttnn.TensorAccessorArgs(u).get_compile_time_args()),
        runtime_args=rt, common_runtime_args=[0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[S1t, S2t, Ct, row, QR, GRAN] + list(ttnn.TensorAccessorArgs(z).get_compile_time_args()),
        runtime_args=rt, common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[GRAN], runtime_args=rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                            math_approx_mode=False, fp32_dest_acc_en=True))
    b = ttnn.bfloat16
    cbs = [PT._cb(PT.IN_CB, 64, b, core_grid), PT._cb(PT.OUT_CB, 64, b, core_grid),
           PT._cb(PT.HALF_CB, 2, b, core_grid), PT._cb(PT.FREE_CB, 2, b, core_grid),
           PT._cb(3, 32, b, core_grid), PT._cb(17, 2 * GRAN, b, core_grid)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def transpose_add(u, z):
    """``z += permute(u)`` in place; returns z. Call only where ``eligible(u, z)``."""
    device = u.device()
    S1, S2, C = PT._dims(u)
    g_ = device.compute_with_storage_grid_size()
    key = (device.id(), S1, S2, C, g_.x, g_.y, QR, GRAN)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(u, z, device, S1 // 32, S2 // 32, C // 32)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [u.buffer_address()]
    writer.common_runtime_args = [z.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    STATS[0] += 1
    ttnn.generic_op([u, z], pd)
    return z
