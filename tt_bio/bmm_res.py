"""``a [1, C, M, K] @ b [1, C, K, N]`` per channel with ``b[c]`` resident in L1, for the trimul einsum on Wormhole.

The production plan (`tenstorrent._triangle_mul_program_config`, a 2D multicast with a 2x2-tile block per core at
512 tokens) runs the [1, 256, 512, 512] einsum at 3.39 ms on a Wormhole chip at 1000 MHz, 2.5x its ~1.4 ms DRAM floor
(402 MB at 288 GB/s), and neither fidelity nor any ttnn program config moves it (perf/af2_trimul_bmm_sweep.py at 512:
whole-channel plans overflow L1, row-block reuse re-reads b per block). Here each core owns whole channels: it loads
b[c] into L1 once (512 KB bf16 at 512 tokens, two channels deep when L1 allows), streams a[c]'s 32-row tiles through
it and writes out[c], so a, b and out each cross DRAM once. The reader, compute and writer are pair_linear's, keyed
per channel. One fp32 dest pass over all of K.

PROTOTYPE: perf/spd_trimul/bmm_res.py is its first timing and accuracy check; no model path calls it yet.
`matmul` returns the product or None, and None means the caller keeps its own path.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import core_split
from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "bmm_res"
WRITER = Path(__file__).resolve().parent / "kernels" / "pair_swiglu_ws" / "writer.cpp"
A_CB, B_CB, OUT_CB = 0, 1, 16
A_DEPTH = 2            # row tiles of a in flight per core
#: L1 a core may spend on CBs: b B_DEPTH * kt * nt + a A_DEPTH * kt + out 2 * nt tiles.
L1_BUDGET = 1200 * 1024
_CACHE: dict = {}


def _b_depth(kt, nt, a_bytes, b_bytes, budget=L1_BUDGET):
    """Channels of b in flight (2, 1), or 0 if one does not fit."""
    rest = A_DEPTH * kt * a_bytes + 2 * nt * MG.tile_bytes(ttnn.bfloat16)
    for depth in (2, 1):
        if depth * kt * nt * b_bytes + rest <= budget:
            return depth
    return 0


def _build(a, b, out, C, mt, kt, nt, depth, fidelity, max_cores):
    g = a.device().compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, C, max_cores)

    def cb(idx, tiles, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
        return ttnn.CBDescriptor(total_size=tiles * MG.tile_bytes(dtype), core_ranges=grid,
                                 format_descriptors=[fmt])

    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = [first, per]
                    wr[cx][cy] = [first * mt, per * mt]
                    first += per
    assert first == C, (first, C)
    rct = [mt, kt, nt]
    for t in (a, b):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [nt] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 2,
                                   defines=[], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(WRITER), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=[], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=path, core_ranges=grid,
        compile_time_args=[mt, kt, nt], runtime_args=cp, defines=[],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True))
    cbs = [cb(A_CB, A_DEPTH * kt, a.dtype), cb(B_CB, depth * kt * nt, b.dtype), cb(OUT_CB, 2 * nt, ttnn.bfloat16)]
    return [reader, writer, compute], cbs


def matmul(a, b, compute_kernel_config=None, budget=L1_BUDGET, max_cores=0, depth=None):
    """``a @ b`` over the channel axis as a new bf16 DRAM tensor, or None if the call is outside what the kernel does.

    It sums K in an fp32 dest at the caller's fidelity (HiFi4 without a config); a bf16-dest config declines.
    `max_cores` caps the channel split (0 = the whole grid) and `depth` forces b's channel depth; both are bench
    knobs that move scheduling, never a number.
    """
    if compute_kernel_config is not None and not compute_kernel_config.fp32_dest_acc_en:
        return None
    fidelity = ttnn.MathFidelity.HiFi4 if compute_kernel_config is None else compute_kernel_config.math_fidelity
    for t in (a, b):
        mc = t.memory_config()
        if (t.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b) or t.layout != ttnn.TILE_LAYOUT
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or len(t.shape) != 4
                or int(t.shape[0]) != 1):
            return None
    sa, sb = [int(d) for d in a.padded_shape], [int(d) for d in b.padded_shape]
    if sa[1] != sb[1] or sa[3] != sb[2]:
        return None
    C, mt, kt, nt = sa[1], sa[2] // 32, sa[3] // 32, sb[3] // 32
    fit = _b_depth(kt, nt, MG.tile_bytes(a.dtype), MG.tile_bytes(b.dtype), budget)
    depth = min(depth or fit, fit)
    if not depth:
        return None
    dev = a.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([int(d) for d in a.shape][:-1] + [int(b.shape[-1])]),
                                         ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    key = (dev.id(), tuple(sa), tuple(sb), a.dtype, b.dtype, a.memory_config().buffer_type,
           b.memory_config().buffer_type, depth, fidelity, max_cores)
    if key not in _CACHE:
        _CACHE[key] = _build(a, b, out, C, mt, kt, nt, depth, fidelity, max_cores)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [a.buffer_address(), b.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    return ttnn.generic_op([a, b, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
