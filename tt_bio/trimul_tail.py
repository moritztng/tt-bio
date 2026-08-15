"""F1: the trimul output tail's two projections and its gate as one `generic_op`.

    production   p_out = x_norm @ Wp ; g_out = x_norm_in @ Wg
                 multiply_(p_out, g_out, SIGMOID on b)              4P read, 3P write
    F1           one kernel over both (activation, weight) pairs    2P read, 1P write

At 512 aa `P` is 134.22 MB, so this deletes 268.4 MB of DRAM reads and 268.4 MB of writes per
trimul call. The kernels are generated from the wheel's own `minimal_matmul` sources by
`kernels/trimul_tail/patch_trimul_tail.py`, which also carries the rounding argument; the
descriptor is `mm_generic`'s transcription with three circular buffers and one runtime address
added per data-movement kernel.

Scoped to the class the fold actually issues at 512 aa: bf16 in and out, interleaved DRAM, both
activations and both weights the same shape/dtype/layout, one K block, no bias, no ternary. Outside
that `fused_tail` returns None and the caller keeps today's three ops.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import mm_generic as MG

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "trimul_tail"

TILE = 32
PASSES = 2

# How the product reaches bf16 before it is packed. MEASURED at N=128 against
# multiply_(p, g, SIGMOID) (perf/trimul_f1/f1_round_diag.py, qb1 card 3, ttnn 0.67.4):
#     0  leave it to the packer         38004/4194304 elements miss (0.906%)
#     1  float_to_fp16b + reinterpret   38004/4194304, byte-identical to 0 -- a silent NO-OP
#     2  the same rounding in integers  0/4194304, torch.equal
# 1 does nothing because under fp32 DST the SFPSTOCHRND result does not land where
# reinterpret<vFloat> reads it; it only works in the 16-bit DST the LLK's own sigmoid runs in.
# SKIP_SIGMOID drops the gate so the multiply can be scored alone. Diagnostic only.
ROUND = 2
SKIP_SIGMOID = 0

# Same swept block config the two projections use today (`tenstorrent._MM_BLOCK[8]`), so each pass
# runs the identical `matmul_blocks` over the identical single K block in the identical order.
BLOCK = (4, 8, 1, 4, 1)

STATS = [0, 0]          # served, declined
REJECTS: dict = {}      # (reason, shape) -> count, so a decline is diagnosable from the fold JSON


def _reject(why, shape=""):
    STATS[1] += 1
    REJECTS[(why, shape)] = REJECTS.get((why, shape), 0) + 1
    return None


def eligible(xa, xb, wa, wb):
    """None when F1's descriptor covers this call, else the reason it does not.

    Every clause is a real assumption of the fork, so a decline names which one.
    """
    if ttnn.bfloat16 not in (xa.dtype, xb.dtype, wa.dtype, wb.dtype):
        return "dtype"
    if xa.dtype != xb.dtype or wa.dtype != wb.dtype:
        return "dtype_pair"
    if tuple(xa.padded_shape) != tuple(xb.padded_shape):
        return "act_shape_pair"
    if len(wa.shape) != 2 or tuple(wa.shape) != tuple(wb.shape):
        return "weight_shape_pair"
    if str(xa.memory_config()) != str(xb.memory_config()):
        return "act_memcfg_pair"
    if str(wa.memory_config()) != str(wb.memory_config()):
        return "weight_memcfg_pair"
    kt = (int(wa.shape[-2]) + TILE - 1) // TILE
    nt = (int(wa.shape[-1]) + TILE - 1) // TILE
    M, K, N, _, _ = BLOCK
    if kt != K:
        # One K block is the fusion's whole simplification. At 512 aa this declines the
        # narrow-hidden trimuls (c_hidden 64, kt = 2), which production does not put through
        # `minimal_matmul` either: `_MM_BLOCK` has no entry for a 2-tile output.
        return f"k_tiles={kt}"
    if nt % N:
        return f"n_tiles={nt}"
    mt = 1
    for d in [int(d) for d in xa.padded_shape][:-1]:
        mt *= d
    mt = (mt + TILE - 1) // TILE
    if mt % M:
        return f"m_tiles={mt}"
    if mt <= nt:
        return "m_le_n"               # `transpose`, the only core-grid orientation this is run on
    return None


def _cb(idx, core_grid, tiles):
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=idx, data_format=ttnn.bfloat16, page_size=MG._TILE_BYTES[ttnn.bfloat16])
    return ttnn.CBDescriptor(
        total_size=tiles * MG._TILE_BYTES[ttnn.bfloat16], core_ranges=core_grid,
        format_descriptors=[fmt])


def _build(device, xa, xb, wa, wb, out, grid, ckc):
    defs = {"TRIMUL_TAIL_PASSES": PASSES, "TRIMUL_TAIL_ROUND": ROUND,
            "TRIMUL_TAIL_SKIP_SIGMOID": SKIP_SIGMOID}
    entry = MG.build(device, xa, wa, [out], (BLOCK, grid), ckc,
                     defines=defs, kernel_dir=KERNEL_DIR)

    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    out_block = BLOCK[0] * BLOCK[2]
    # c_4 / c_5: the two bf16 GEMM results, double buffered. c_6: the rounded gate, one tile live.
    entry["cbs"] += [_cb(4, core_grid, out_block * 2),
                     _cb(5, core_grid, out_block * 2),
                     _cb(6, core_grid, 2)]

    # The compute kernel is the fork's, not the wheel's, and it needs the pass count too.
    compute = entry["kernels"][4]
    compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
    compute.defines = [(k, str(v)) for k, v in defs.items()]

    _bind_b(entry, xb.buffer_address(), wb.buffer_address())
    _repack(entry)
    return entry


def _bind_b(entry, xb_addr, wb_addr):
    """Pass 1's two addresses ride in the unused `in2_addr` runtime slot, index 1 in both kernels."""
    for name in ("in0_sender", "in0_recv"):
        for _, a in entry["rt"][name]:
            a[1] = xb_addr
    for name in ("in1_sender", "in1_recv"):
        for _, a in entry["rt"][name]:
            a[1] = wb_addr
    entry["b_addrs"] = (xb_addr, wb_addr)


def _repack(entry):
    for k, name in zip(entry["kernels"][:4],
                       ("in0_sender", "in0_recv", "in1_sender", "in1_recv")):
        k.runtime_args = entry["rt"][name]
    entry["pd"] = ttnn.ProgramDescriptor(
        kernels=entry["kernels"], semaphores=entry["semaphores"], cbs=entry["cbs"])


_CACHE: dict = {}


def fused_tail(xa, xb, wa, wb, ckc, grid):
    """`p * sigmoid(g)` for `p = xa @ wa`, `g = xb @ wb`, in one kernel. None if out of scope."""
    why = eligible(xa, xb, wa, wb)
    if why is not None:
        return _reject(why, "x".join(str(int(d)) for d in xa.padded_shape)
                       + "@" + "x".join(str(int(d)) for d in wa.shape))
    device = xa.device()
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = (spec(xa), spec(wa), tuple(grid), tuple(str(c) for c in ckc), ROUND, SKIP_SIGMOID)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([int(d) for d in xa.shape][:-1] + [int(wa.shape[-1])]),
        ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(device, xa, xb, wa, wb, out, grid, ckc)
    else:
        # `MG.rebind` repacks the descriptor itself, so only bind B separately when it does not run.
        addrs = (xa.buffer_address(), wa.buffer_address(), (out.buffer_address(),))
        b = (xb.buffer_address(), wb.buffer_address())
        stale_b = b != entry["b_addrs"]
        if stale_b:
            _bind_b(entry, *b)
        if addrs != entry["addrs"]:
            MG.rebind(entry, *addrs)
        elif stale_b:
            _repack(entry)

    ttnn.generic_op([xa, wa, xb, wb, out], entry["pd"])
    STATS[0] += 1
    return out
