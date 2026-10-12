"""``LayerNorm(x)^T`` over a pair tensor in one Tensix program: x read once, the normed transpose written once.

The ending-node triangle attention normalises z^T. Today that is `pair_transpose` (read z, write z^T) and then
`ttnn.layer_norm` (read z^T, write xn): 4P of DRAM traffic for 2P of data (P = one bf16 pair tensor). LayerNorm
is per row, so LN(z)^T = LN(z^T) row for row: this program reads 32x32 (i, j) blocks of z, normalises each row
tile in L1 (kernels/common/ln_prologue.hpp, as trimul T1/T2) and writes the block transposed with
kernels/pair_transpose's face-row shuffle (kernels/pair_ln_transpose). The `triatt_tr` lever.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from . import core_split
from .pair_swiglu_ws import _ln_vec_ok

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_ln_transpose"
_TILE = 32 * 32 * 2
_L1_BUDGET = 1100 * 1024
_CACHE: dict = {}
STATS = [0, 0]   # served, declined


def _dims(x):
    s = [int(d) for d in x.padded_shape]
    if len(s) == 4 and s[0] == 1:
        s = s[1:]
    return s if len(s) == 3 else None


def _l1_bytes(kt) -> int:
    # raw x (2 kt), normed unit (32 kt), scratch (32), gamma / beta (kt each), scaler; fp32 eps, mean, var,
    # rstd, xc (kt), tmp (kt)
    return (2 * kt + 32 * kt + 32 + 2 * kt + 1) * _TILE + (4 + 2 * kt) * 2 * _TILE


def eligible(x, gamma, beta) -> bool:
    s = _dims(x)
    ok = (s is not None and x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
          and x.memory_config() == ttnn.DRAM_MEMORY_CONFIG and all(d % 32 == 0 for d in s)
          and all(_ln_vec_ok(v, s[2]) for v in (gamma, beta)) and _l1_bytes(s[2] // 32) <= _L1_BUDGET)
    if not ok:
        STATS[1] += 1
    return ok


def _build(x, gamma, beta, out, device, s1t, s2t, kt, eps, fidelity):
    g = device.compute_with_storage_grid_size()
    num_cores, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, s1t * s2t, 0)

    def cb(idx, tiles, dtype=ttnn.bfloat16):
        size = 2 * _TILE if dtype == ttnn.float32 else _TILE
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(5, 2 * kt), cb(8, kt), cb(9, kt), cb(10, 1), cb(16, 32 * kt), cb(24, 32)]
    cbs += [cb(i, n, ttnn.float32) for i, n in ((11, 1), (12, 1), (13, 1), (14, 1), (17, kt), (18, kt))]
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / (32 * kt)))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = wr[cx][cy] = [first, per]
                    cp[cx][cy] = [per]
                    first += per
    assert first == s1t * s2t, (first, s1t, s2t)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=[kt, s2t, scaler, eps_bits] + acc(x) + acc(gamma) + acc(beta),
                                   runtime_args=rd, common_runtime_args=[0] * 3, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=[kt, s1t, s2t, _TILE // 64] + acc(out),
                                   runtime_args=wr, common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid, compile_time_args=[kt],
        runtime_args=cp, config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True))
    return [reader, writer, compute], cbs


def ln_transpose(x, gamma, beta, eps=1e-5, fidelity=ttnn.MathFidelity.HiFi4):
    """LayerNorm over the channels of x [S1, S2, C] (or [1, S1, S2, C]), transposed: a new bf16 DRAM tensor
    [S2, S1, C] of the same rank. Call only where `eligible(x, gamma, beta)`."""
    device = x.device()
    s1, s2, c = _dims(x)
    shape = [s2, s1, c] if len(x.shape) == 3 else [1, s2, s1, c]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                         ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), s1, s2, c, float(eps), fidelity)
    if key not in _CACHE:
        _CACHE[key] = _build(x, gamma, beta, out, device, s1 // 32, s2 // 32, c // 32, eps, fidelity)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [x.buffer_address(), gamma.buffer_address(), beta.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    STATS[0] += 1
    return ttnn.generic_op([x, gamma, beta, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
