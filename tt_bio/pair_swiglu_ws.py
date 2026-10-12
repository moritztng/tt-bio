"""``fc3(silu(x @ w1) * (x @ w2))`` over a pair tensor, weights resident in L1, one Tensix program.

RFD3's token pair Transition (c_z 128, hidden 256 or 512) runs as fc1, fc2, a gate multiply and fc3 over 64-row
chunks: about 6.3 ms a call in fast mode against ~2 ms of work at the roof. Every core here holds `[w1 | w2]` and
`w3` in L1 for the whole call and takes whole 32-row tiles of the flattened pair tensor: x is read once, the
output written once, and fc1, fc2 and their product never leave the core (kernels/pair_swiglu_ws).

The engine's shared `_transition_swiglu_sharded` does not serve this shape: it needs the output width to split
over 8 core columns, and 128 channels are 4 tiles.

The output is bfloat16; x bfloat16 or bfloat8_b; the weights bfloat16, or bfloat8_b where bf16 weights would not fit L1 (c_z 256,
hidden 1024: 768 weight tiles are 1.5 MB in bf16, 0.8 MB in bfp8). The norm stays the caller's op.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from . import core_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_swiglu_ws"
X_CB, W12_CB, W3_CB, GATE_CB, HID_CB, OUT_CB = 0, 1, 2, 3, 4, 16
_TILE = 32 * 32 * 2
_SW = 4                   # hidden tiles per dest pass (one fp32 dest half)
# Bytes of a core's L1 the program may take: weights plus its streaming CBs. A Wormhole core has 1464 KiB,
# a Blackhole core 1536 KiB; the rest is the firmware, the kernel binaries and ttnn's own reserve.
_L1_BUDGET = 1100 * 1024


SILU_POLY = 2  # ws5 on WH 1000 MHz: c_z 128 fast 5.64 -> 4.77 ms, c_z 256 13.08 -> 11.28, rel_rms to float64 unchanged


def _tile_bytes(dtype) -> int:
    return 1088 if dtype == ttnn.bfloat8_b else _TILE
_CACHE: dict = {}


def weights(w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, device, dtype=ttnn.bfloat16):
    """`[w1 | w2]` ([C, 2H]) and `w3` ([H, C]) as `dtype` TILE tensors, from torch [in, out] matrices."""
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=device)
    return up(torch.cat([w1, w2], 1)), up(w3)


def eligible(x, w12, w3) -> bool:
    """bfloat16 TILE interleaved x, bf16/bfp8 TILE interleaved weights, hidden a multiple of 4 tiles, output
    width at most 4 tiles or a multiple of 4, and the weights plus the streaming CBs inside `_L1_BUDGET`."""
    if x.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b) or w12.dtype != w3.dtype \
            or w12.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
        return False
    for t in (x, w12, w3):
        if t.layout != ttnn.TILE_LAYOUT or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return False
    kt, ct = int(x.padded_shape[-1]) // 32, int(w3.shape[-1]) // 32
    nh = int(w3.shape[0]) // 32
    if not (kt == int(w12.shape[0]) // 32 and int(w12.shape[1]) == 2 * int(w3.shape[0])
            and nh % _SW == 0 and (ct <= 4 or ct % 4 == 0)):
        return False
    return _l1_bytes(kt, nh, ct, w12.dtype, x.dtype) <= _L1_BUDGET


def _l1_bytes(kt, nh, ct, wdtype, xdtype=ttnn.bfloat16) -> int:
    return ((kt * 2 * nh + nh * ct) * _tile_bytes(wdtype) + 2 * kt * _tile_bytes(xdtype)
            + (2 * _SW + 2 * nh + 2 * ct) * _TILE)


def _build(x, w12, w3, out, device, mt, kt, nh, ct, fidelity, defines, silu_poly):
    g = device.compute_with_storage_grid_size()
    num_cores, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, mt, 0)

    def cb(idx, tiles, dtype=ttnn.bfloat16):
        size = _tile_bytes(dtype)
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(X_CB, 2 * kt, x.dtype), cb(W12_CB, kt * 2 * nh, w12.dtype), cb(W3_CB, nh * ct, w3.dtype), cb(GATE_CB, 2 * _SW),
           cb(HID_CB, 2 * nh), cb(OUT_CB, 2 * ct)]
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == mt, (first, mt)
    rct = [kt, nh, ct]
    for t in (x, w12, w3):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [ct] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0, 0, 0],
                                   defines=list(defines), config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=list(defines), config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid, compile_time_args=[kt, nh, ct, silu_poly],
        runtime_args=cp, defines=list(defines), config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity,
                                                             fp32_dest_acc_en=True))
    return [reader, writer, compute], cbs


def pair_swiglu(x, w12, w3, fidelity=ttnn.MathFidelity.HiFi4, defines=(), silu_poly=None):
    """`fc3(silu(x @ w1) * (x @ w2))` for a TILE `x` [..., C]; a new bfloat16 DRAM tensor [..., C_out].

    Accumulation is fp32 in dest at every fidelity; `fidelity` is the caller's matmul fidelity. `defines` are
    the kernels' WS_NO_* time-attribution switches (perf/spd_fastmode/ws_attr.py), empty in production.
    `silu_poly` picks the silu: 0 metal's silu_tile, whose fp32-dest exp + reciprocal was ~85 % of the kernel on WH
    (c_z 128: 8.97 ms with it, 1.37 ms without), 1 sigmoid_poly.hpp's degree-8 polynomial (sigmoid within 3e-4),
    2 its degree-6 one with fp16 coefficients (7.4e-4, a third fewer SFPU instructions). None takes `SILU_POLY`."""
    silu_poly = SILU_POLY if silu_poly is None else silu_poly
    device = x.device()
    xs = [int(d) for d in x.padded_shape]
    mt = 1
    for d in xs[:-1]:
        mt *= d
    mt //= 32
    kt, nh, ct = xs[-1] // 32, int(w3.shape[0]) // 32, int(w3.shape[-1]) // 32
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([int(d) for d in x.shape][:-1] + [int(w3.shape[-1])]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), tuple(xs), nh, ct, x.memory_config().buffer_type, fidelity, w12.dtype, x.dtype,
           tuple(defines), silu_poly)
    if key not in _CACHE:
        _CACHE[key] = _build(x, w12, w3, out, device, mt, kt, nh, ct, fidelity, defines, silu_poly)
    kernels, cbs = _CACHE[key]
    kernels[0].common_runtime_args = [x.buffer_address(), w12.buffer_address(), w3.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([x, w12, w3, out], pd)
