"""``fc3(silu(x @ w1) * (x @ w2))`` over a pair tensor, weights resident in L1, one Tensix program.

RFD3's token pair Transition (c_z 128, hidden 256 or 512) runs as fc1, fc2, a gate multiply and fc3 over 64-row
chunks: about 6.3 ms a call in fast mode against ~2 ms of work at the roof. Every core here holds `[w1 | w2]` and
`w3` in L1 for the whole call and takes whole 32-row tiles of the flattened pair tensor: x is read once, the
output written once, and fc1, fc2 and their product never leave the core (kernels/pair_swiglu_ws).

The engine's shared `_transition_swiglu_sharded` does not serve this shape: it needs the output width to split
over 8 core columns, and 128 channels are 4 tiles.

The output is bfloat16; x bfloat16 or bfloat8_b; the weights bfloat16, or bfloat8_b where bf16 weights would not fit L1 (c_z 256,
hidden 1024: 768 weight tiles are 1.5 MB in bf16, 0.8 MB in bfp8). The norm is the caller's op, or with `ln` the
kernel's prologue (kernels/common/ln_prologue.hpp, as trimul T1/T2), so the normalised x never reaches DRAM.

`pair_relu` is the same program for AF2's Transition, ``fc2(relu(x @ w1 + b1)) + b2``: one weight matrix in front of
the activation instead of two, and each bias seeds its matmul's dest.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from . import core_split
from . import ops

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "pair_swiglu_ws"
X_CB, W12_CB, W3_CB, GATE_CB, HID_CB, OUT_CB = 0, 1, 2, 3, 4, 16
XRAW_CB, GAMMA_CB, BETA_CB, SCALER_CB = 5, 8, 9, 10     # the LayerNorm prologue's bf16 CBs
LN_F32_CBS = ((11, 1), (12, 1), (13, 1), (14, 1), (17, None), (18, None))   # eps, mean, var, rstd, xc, tmp (None: kt)
ACC_CB = 19               # fp32, _SW tiles: a matmul's K chunks summed by the packer (`l1acc`)
RES_CB = 20               # bf16, 2 row tiles of the residual fc3 adds into (`resid`)
B1_CB, B3_CB = 21, 22     # bf16, `pair_relu`'s biases as row-replicated tiles
DY_CB = 23                # bf16, 2 row tiles of `pair_relu_bw`'s cotangent
_TILE = 32 * 32 * 2
_SW = 4                   # hidden tiles per dest pass (one fp32 dest half)
# Bytes of a core's L1 the program may take: weights plus its streaming CBs. A Wormhole core has 1464 KiB,
# a Blackhole core 1536 KiB; the rest is the firmware, the kernel binaries and ttnn's own reserve.
_L1_BUDGET = 1100 * 1024


SILU_POLY = 2  # ws5 on WH 1000 MHz: c_z 128 fast 5.64 -> 4.77 ms, c_z 256 13.08 -> 11.28, rel_rms to float64 unchanged
# The polynomial silu on the PACK thread over the fc1 dest half while MATH runs fc2 (kernel WS_PACK_SILU). Same SFPU
# code on the same dest, so the output does not move; the silu was ~1 ms of a 4.6 ms c_z 384 chunk on WH (wsb).
# A/B knob while it is measured.
PACK_SILU = False


def _tile_bytes(dtype) -> int:
    return 1088 if dtype == ttnn.bfloat8_b else 2 * _TILE if dtype == ttnn.float32 else _TILE
_CACHE: dict = {}


def weights(w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, device, dtype=ttnn.bfloat16):
    """`[w1 | w2]` ([C, 2H]) and `w3` ([H, C]) as `dtype` TILE tensors, from torch [in, out] matrices, in as few
    hidden chunks as keep each chunk's weights inside the L1 budget: a list of (w12, w3) pairs whose
    `pair_swiglu` outputs sum to the whole. One pair wherever the weights fit (c_z 128; c_z 256 in bfp8); c_z 256
    in bf16 and OpenDDE's c_z 384 take two or more, each re-reading x."""
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=device)
    kt, nh, ct = w1.shape[0] // 32, w1.shape[1] // 32, w3.shape[1] // 32
    parts = next((p for p in range(1, nh // _SW + 1) if nh % (p * _SW) == 0
                  and _l1_bytes(kt, nh // p, ct, dtype) <= _L1_BUDGET), 1)
    h = w1.shape[1] // parts
    return [(up(torch.cat([w1[:, i:i + h], w2[:, i:i + h]], 1)), up(w3[i:i + h])) for i in range(0, w1.shape[1], h)]


def ln_bytes(kt) -> int:
    """The LayerNorm prologue's extra L1: raw x (2 kt), gamma, beta (kt each), the scaler, and the fp32 CBs."""
    return (4 * kt + 1) * _TILE + (4 + 2 * kt) * 2 * _TILE


def _ln_vec_ok(v, k) -> bool:
    """gamma / beta as `torch_to_tt` leaves a bf16 norm parameter: TILE, interleaved, k values in row 0."""
    return (v.dtype == ttnn.bfloat16 and v.layout == ttnn.TILE_LAYOUT and int(v.shape[-1]) == k
            and all(int(d) == 1 for d in list(v.shape)[:-1])
            and v.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


def eligible(x, w12, w3, ln=(), l1acc=False, resid=None) -> bool:
    """bfloat16 TILE interleaved x, bf16/bfp8 TILE interleaved weights, hidden a multiple of 4 tiles, output
    width at most 4 tiles or a multiple of 4, and the weights plus the streaming CBs inside `_L1_BUDGET`. With
    `ln` = (gamma, beta) also the LayerNorm prologue (beta None: RMSNorm): bf16 x, bf16 [1, C] gamma / beta, and its
    CBs in budget. `l1acc` adds its fp32 accumulator CB; `resid` (bf16, TILE, interleaved DRAM, the output's padded
    shape) its two row tiles."""
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
    if ln and (x.dtype != ttnn.bfloat16 or not all(_ln_vec_ok(v, 32 * kt) for v in ln if v is not None)):
        return False
    if resid is not None and not (
            resid.dtype == ttnn.bfloat16 and resid.layout == ttnn.TILE_LAYOUT
            and resid.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and resid.memory_config().buffer_type == ttnn.BufferType.DRAM
            and [int(d) for d in resid.padded_shape] == [int(d) for d in x.padded_shape][:-1] + [32 * ct]):
        return False
    return (_l1_bytes(kt, nh, ct, w12.dtype, x.dtype) + (ln_bytes(kt) if ln else 0)
            + (_SW * 2 * _TILE if l1acc else 0) + (2 * ct * _TILE if resid is not None else 0) <= _L1_BUDGET)


def _l1_bytes(kt, nh, ct, wdtype, xdtype=ttnn.bfloat16) -> int:
    return ((kt * 2 * nh + nh * ct) * _tile_bytes(wdtype) + 2 * kt * _tile_bytes(xdtype)
            + (2 * _SW + 2 * nh + 2 * ct) * _TILE)


def _build(x, w12, w3, out, device, mt, kt, nh, ct, fidelity, defines, silu_poly, ln, bias=None, bw=False):
    g = device.compute_with_storage_grid_size()
    num_cores, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, mt, 0)

    def cb(idx, tiles, dtype=ttnn.bfloat16):
        size = _tile_bytes(dtype)
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(X_CB, 2 * kt, x.dtype), cb(W3_CB, nh * ct, w3.dtype), cb(HID_CB, 2 * nh), cb(OUT_CB, 2 * ct)]
    if bias is None or bw:
        cbs += [cb(W12_CB, kt * 2 * nh, w12.dtype), cb(GATE_CB, 2 * _SW)]
    if bw:
        cbs += [cb(B1_CB, nh), cb(DY_CB, 2 * kt)]
        defines = tuple(defines) + (("WS_RELU_BW", "1"),)
    elif bias is not None:
        cbs += [cb(W12_CB, kt * nh, w12.dtype), cb(B1_CB, nh), cb(B3_CB, ct)]
        defines = tuple(defines) + (("WS_RELU", "1"),)
    if ("WS_L1ACC", "1") in defines:
        cbs.append(cb(ACC_CB, _SW, ttnn.float32))
    if ("WS_RESID", "1") in defines:
        cbs.append(cb(RES_CB, 2 * ct))
    scaler = eps_bits = 0
    if ln is not None:
        import struct
        cbs[0] = cb(X_CB, 2 * kt)   # the normalised x the matmuls read, bf16
        cbs += [cb(XRAW_CB, 2 * kt), cb(GAMMA_CB, kt), cb(BETA_CB, kt), cb(SCALER_CB, 1)]
        cbs += [cb(i, kt if n is None else n, ttnn.float32) for i, n in LN_F32_CBS]
        inv = struct.unpack("<I", struct.pack("<f", 1.0 / (32 * kt)))[0] >> 16
        scaler = (inv << 16) | inv          # 1/K as a packed bf16 pair, exact for K a power of two
        eps_bits = struct.unpack("<I", struct.pack("<f", float(ln[2])))[0]
        defines = tuple(defines) + (("WS_LN", "1"),)
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == mt, (first, mt)
    rct = [kt, nh, ct, scaler, eps_bits]
    for t in ((x, w12, w3) + (() if ln is None else ln[:2]) + ((out,) if ("WS_RESID", "1") in defines else ())
              + (() if bias is None else bias)):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [ct] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    src = lambda n: str(KERNEL_DIR / n)  # noqa: E731
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=src("reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 8,
                                   defines=list(defines), config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=src("writer.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=list(defines), config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=src("compute.cpp"), source_type=path, core_ranges=grid, compile_time_args=[kt, nh, ct, silu_poly],
        runtime_args=cp, defines=list(defines), config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity,
                                                             fp32_dest_acc_en=True))
    return [reader, writer, compute], cbs


def pair_swiglu(x, w12, w3, fidelity=ttnn.MathFidelity.HiFi4, defines=(), silu_poly=None, ln=None, l1acc=False,
                resid=None):
    """`fc3(silu(x @ w1) * (x @ w2))` for a TILE `x` [..., C]; a new bfloat16 DRAM tensor [..., C_out].

    Accumulation is fp32 in dest at every fidelity; `fidelity` is the caller's matmul fidelity. `defines` are
    the kernels' WS_NO_* time-attribution switches (perf/spd_fastmode/ws_attr.py), empty in production.
    `silu_poly` picks the silu: 0 metal's silu_tile, whose fp32-dest exp + reciprocal was ~85 % of the kernel on WH
    (c_z 128: 8.97 ms with it, 1.37 ms without), 1 sigmoid_poly.hpp's degree-8 polynomial (sigmoid within 3e-4),
    2 its degree-6 one with fp16 coefficients (7.4e-4, a third fewer SFPU instructions). None takes `SILU_POLY`.
    `ln` = (gamma, beta, eps) runs the LayerNorm over the raw bf16 `x` in the kernel first; gamma / beta bf16 TILE
    [1, C] as `torch_to_tt` leaves a norm parameter. beta None makes it an RMSNorm, as ttnn.rms_norm.
    `l1acc`: every matmul sums at most 2 K tiles in dest and the packer adds the chunks in an fp32 CB. On Wormhole an
    fp32 dest summing all of K writes exact +-1 / 2 / 4 into ~1e-8 of the outputs (tt_bio/dest_guard.py).
    `resid`: fc3's result is added to `resid` in the fp32 dest and written back over it, one bf16 rounding, and
    `resid` is returned; no new tensor. `resid` may be `x` itself (`x + f(x)`, with `ln` reading the raw x): each
    core's reader takes a row tile's x and residual pages before the writer can put that row tile back, and no row
    tile is read by two cores. With hidden chunks (`weights` returning more than one pair) only the last call may
    take `resid is x`, the others would read the updated rows."""
    if resid is not None:
        defines = tuple(defines) + (("WS_RESID", "1"),)
    silu_poly = SILU_POLY if silu_poly is None else silu_poly
    if PACK_SILU and silu_poly in (1, 2):
        defines = tuple(defines) + (("WS_PACK_SILU", "1"),)
    if l1acc:
        defines = tuple(defines) + (("WS_L1ACC", "1"),)
    if ln is not None and ln[1] is None:
        ln = (ln[0], ln[0], ln[2])   # the reader loads a beta either way; the RMS prologue never reads it
        defines = tuple(defines) + (("WS_RMS", "1"),)
    return _run(x, w12, w3, fidelity, defines, silu_poly, ln, resid)


@ops.fused_kernel("pair_relu")
def pair_relu(x, w1, b1, w2, b2, fidelity=ttnn.MathFidelity.HiFi4, ln=None, bw_weights=None):
    """AF2's Transition after its norm, `relu(x @ w1 + b1) @ w2 + b2`, for a bf16 TILE `x` [..., C]; a new bf16 DRAM
    tensor [..., C_out]. Weights as `relu_weights` returns them. `ln` = (gamma, beta, eps) runs the LayerNorm in the
    kernel first, as `pair_swiglu`. Same fp32 dest accumulation, the bias summed into it before the products.

    Under a tape it runs only through its entry in `taped_ttnn` (`pair_relu`), which needs `bw_weights`
    (`relu_bw_weights`) and `ln`; otherwise it returns None and the caller's composed ops run."""
    if ops.declines_under_tape("pair_relu"):
        return None
    return _run(x, w1, w2, fidelity, (), SILU_POLY, ln, None, (b1, b2))


def relu_weights(w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, device):
    """(w1, b1, w2, b2) for `pair_relu` from torch [in, out] matrices and [out] biases: bf16 TILE tensors, each bias
    repeated over the 32 rows of its tiles (dest is seeded with whole tiles)."""
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=device)
    return up(w1), up(b1.reshape(1, -1).expand(32, -1)), up(w2), up(b2.reshape(1, -1).expand(32, -1))


def relu_eligible(x, w1, w2, ln=()) -> bool:
    """What `pair_relu` serves: bf16 TILE interleaved x and weights, hidden a multiple of 4 tiles, output width at most
    4 tiles or a multiple of 4, and everything resident inside `_L1_BUDGET`."""
    if x.dtype != ttnn.bfloat16 or w1.dtype != ttnn.bfloat16 or w2.dtype != ttnn.bfloat16:
        return False
    for t in (x, w1, w2):
        if t.layout != ttnn.TILE_LAYOUT or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            return False
    kt, nh, ct = int(x.padded_shape[-1]) // 32, int(w2.shape[0]) // 32, int(w2.shape[-1]) // 32
    if not (kt == int(w1.shape[0]) // 32 and int(w1.shape[1]) == 32 * nh and nh % _SW == 0
            and (ct <= 4 or ct % 4 == 0)):
        return False
    if ln and not all(_ln_vec_ok(v, 32 * kt) for v in ln):
        return False
    return ((kt * nh + nh * ct + nh + ct + 2 * kt + 2 * nh + 2 * ct) * _TILE + (ln_bytes(kt) if ln else 0)
            <= _L1_BUDGET)


def pair_relu_bw(x, dy, w12, b1, w3, fidelity=ttnn.MathFidelity.HiFi4, ln=None):
    """The cotangent of `pair_relu`'s normed input: `((xn @ w1 + b1 > 0) * (dy @ w2^T)) @ w1^T`, xn the LayerNorm of x
    under `ln` (else x itself), from `relu_bw_weights`. The pre-activation is recomputed from x, never stored. The
    LayerNorm's own backward is the caller's (`lnbw.layer_norm_bw`)."""
    return _run(x, w12, w3, fidelity, (), SILU_POLY, ln, None, (b1, dy), bw=True)


def relu_bw_weights(w1: torch.Tensor, w2: torch.Tensor, device):
    """(w12, w3) for `pair_relu_bw` from the torch [in, out] matrices: `[w1 | w2^T]` and `w1^T`, bf16 TILE."""
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=device)
    return up(torch.cat([w1, w2.t()], 1)), up(w1.t())


def _run(x, w12, w3, fidelity, defines, silu_poly, ln, resid, bias=None, bw=False):
    device = x.device()
    xs = [int(d) for d in x.padded_shape]
    mt = 1
    for d in xs[:-1]:
        mt *= d
    mt //= 32
    kt, nh, ct = xs[-1] // 32, int(w3.shape[0]) // 32, int(w3.shape[-1]) // 32
    out = resid if resid is not None else ttnn.allocate_tensor_on_device(
        ttnn.Shape([int(d) for d in x.shape][:-1] + [int(w3.shape[-1])]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG)
    key = (device.id(), tuple(xs), nh, ct, x.memory_config().buffer_type, fidelity, w12.dtype, x.dtype,
           tuple(defines), silu_poly, None if ln is None else float(ln[2]), bias is not None, bw)
    if key not in _CACHE:
        _CACHE[key] = _build(x, w12, w3, out, device, mt, kt, nh, ct, fidelity, defines, silu_poly, ln, bias, bw)
    kernels, cbs = _CACHE[key]
    g, b = (0, 0) if ln is None else (ln[0].buffer_address(), ln[1].buffer_address())
    b1, b3 = (0, 0) if bias is None else (bias[0].buffer_address(), bias[1].buffer_address())
    kernels[0].common_runtime_args = [x.buffer_address(), w12.buffer_address(), w3.buffer_address(), g, b,
                                      out.buffer_address(), b1, b3]
    kernels[1].common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([x, w12, w3] + ([] if ln is None else list(ln[:2])) + ([] if bias is None else list(bias))
                           + [out], pd)
