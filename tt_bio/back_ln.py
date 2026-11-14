"""The trimul's back channel move and its output layer norm as one kernel.

The triangle multiplication's einsum leaves its result channel-major, `[1, C, N, N]`, in channel
chunks; `reblock_permute_back` moves each chunk to the last axis of one `[1, N, N, H]` tensor and
`ttnn.layer_norm` then reads and writes that tensor again. This kernel serves the LAST chunk: it
moves it like `reblock_permute_back` does, reads the channels the earlier chunks already put in the
output, and writes the normalized rows in place, so the layer norm's own pass over the pair is gone.
The move is bound by the reader's gather (64 NOC packets per tile), which leaves the compute
engine free for the statistics: float32 intermediates and float32 DEST reductions, the forward
half of kernels/lnbw.

Off unless armed (TT_BIO_TRIMUL_BACK_LN); a value change against `ttnn.layer_norm`, graded before
it ships.
"""
from __future__ import annotations

import collections
import pathlib
import struct

import torch
import ttnn

from tt_bio import core_split
from tt_bio.envflags import env_flag

KERNEL_DIR = pathlib.Path(__file__).parent / "kernels" / "back_ln"

ENABLED = env_flag("TT_BIO_TRIMUL_BACK_LN", False)

#: Served and declined calls by reason, cumulative.
REACH: collections.Counter = collections.Counter()

_TILE = 32
_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}
_CACHE: dict = {}
_ROWS: dict = {}       # (id of the weight tensor, device id) -> (gamma, beta) as one-row tiles


def set_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global ENABLED
    prev, ENABLED = ENABLED, bool(on)
    return prev


def _decline(reason: str) -> bool:
    REACH["declined: " + reason] += 1
    return False


def eligible(x_chunk, out, c_off: int) -> bool:
    """A bf16 channel-major `[1, C, N, N]` last chunk landing at `c_off` of a bf16 `[1, N, N, H]`
    DRAM tensor that ends with it, N a multiple of 32, H a power-of-two multiple of 32 (1/H exact in
    the bf16 reduce scaler)."""
    if not ENABLED:
        return False
    xs, os_ = [int(d) for d in x_chunk.shape], [int(d) for d in out.shape]
    if len(xs) != 4 or xs[0] != 1 or xs[2] != xs[3] or xs[2] % _TILE or xs[1] % _TILE:
        return _decline("chunk shape")
    H = os_[-1]
    if os_ != [1, xs[2], xs[2], H] or c_off + xs[1] != H or c_off % _TILE or H & (H - 1):
        return _decline("out shape")
    for t in (x_chunk, out):
        if (t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT
                or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            return _decline("dtype/layout/memory")
    return True


def _cb(idx, depth, dtype, grid, page=None):
    page = page or _BYTES[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=depth * page, core_ranges=grid, format_descriptors=[fmt])


def _parts(device, units_per_part: int) -> int:
    """Row parts per (it, jt) block: the fewest that give every core at least four units."""
    cores = device.compute_with_storage_grid_size()
    cores = cores.x * cores.y
    for p in (1, 2, 4):
        if units_per_part * p >= 4 * cores:
            return p
    return 4


def _build(x, out, gamma, beta, c_off, eps, device, parts):
    N, H = int(x.shape[2]), int(out.shape[-1])
    Nt, Ct, Cp = N // _TILE, H // _TILE, c_off // _TILE
    P = parts or _parts(device, Nt * Nt)
    units = Nt * Nt * P
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(
        device.compute_with_storage_grid_size(), units, 0)
    f32, bf16 = ttnn.float32, ttnn.bfloat16
    clt = Ct - Cp
    stage_bytes = clt * _TILE * 2 * (32 // P) * 32     # each channel tile's part: two faces' rows
    cbs = [_cb(0, 2, bf16, grid), _cb(2, Ct, bf16, grid), _cb(3, Ct, bf16, grid),
           _cb(4, 1, bf16, grid), _cb(5, 1, f32, grid), _cb(16, 2 * Ct, bf16, grid),
           _cb(24, 1, bf16, grid, page=stage_bytes)]
    if Cp:
        cbs.append(_cb(1, 2 * Cp, bf16, grid))
    cbs += [_cb(i, 1, f32, grid) for i in (6, 7, 8)]
    cbs += [_cb(i, Ct, f32, grid) for i in (9, 10, 11)]

    inv = struct.unpack("<I", struct.pack("<f", 1.0 / H))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_ct = [Nt, Ct, Cp, P, scaler, eps_bits] + acc(x) + acc(out) + acc(gamma) + acc(beta)
    writer_ct = [Nt, Ct, P] + acc(out)

    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for rng in group.ranges():
            for cx in range(rng.start.x, rng.end.x + 1):
                for cy in range(rng.start.y, rng.end.y + 1):
                    rr[cx][cy] = [first, per]
                    wr[cx][cy] = [first, per]
                    cr[cx][cy] = [per * (32 // P)]
                    first += per
    assert first == units, (first, units)

    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=reader_ct, runtime_args=rr, common_runtime_args=[0, 0, 0, 0],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=writer_ct, runtime_args=wr, common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute.cpp"), source_type=src, core_ranges=grid,
        compile_time_args=[Ct, Cp], runtime_args=cr,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                            fp32_dest_acc_en=True))
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def weight_rows(weight, bias, device):
    """gamma and beta as `[32, H]` bf16 tiles, row 0 the weights (the kernel broadcasts row 0).
    Cached per weight tensor."""
    key = (id(weight), id(bias), device.id())
    hit = _ROWS.get(key)
    if hit is None:
        rows = []
        for w in (weight, bias):
            v = ttnn.to_torch(w).reshape(-1).to(torch.bfloat16)
            t = torch.zeros(_TILE, v.numel(), dtype=torch.bfloat16)
            t[0] = v
            rows.append(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        hit = _ROWS[key] = (rows, weight, bias)    # the refs keep the ids from being reused
    return hit[0]


def back_ln(x_chunk, out, c_off, gamma, beta, eps, parts=None):
    """Move `x_chunk` ([1, C, N, N]) into `out[..., c_off:]` and layer-norm every row of `out` over
    its H channels in place: `out = layer_norm(out) * gamma + beta`, gamma/beta from `weight_rows`.
    Returns `out`. The caller checks `eligible` first."""
    device = x_chunk.device()
    key = (device.id(), tuple(int(d) for d in x_chunk.shape), int(out.shape[-1]), c_off, float(eps),
           parts)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(x_chunk, out, gamma, beta, c_off, eps, device, parts)
    reader, writer, compute = entry["kernels"]
    reader.common_runtime_args = [x_chunk.buffer_address(), out.buffer_address(),
                                  gamma.buffer_address(), beta.buffer_address()]
    writer.common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=entry["cbs"])
    REACH["served"] += 1
    ttnn.generic_op([x_chunk, gamma, beta, out], pd)
    return out
