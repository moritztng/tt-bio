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
activations and both weights the same shape/dtype/layout, one K block whose (kt, nt) key is in
`F1_BLOCK_KEYS`, no bias, no ternary. Outside that `fused_tail` returns None and the caller keeps
today's three ops.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import ttnn

from . import dest_guard
from . import mm_generic as MG
from .envflags import env_flag

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
# TRIMUL_TAIL_ABL, the kernels' stage ablation (see the patch script's HEADER). Diagnostic only.
ABL = 0

# The epilogue (`TRIMUL_TAIL_EPI` in compute.cpp). 0 is the bit-exact production order above. 1
# packs each GEMM pass straight out of DST into bf16 (sigmoid applied in DST on the gate pass) and
# gates with the FPU multiply, so no fp32 accumulator copy, no gate copy, no SFPU multiply and no
# integer rounding: a numerics change at the bf16-ULP level (sigmoid of the unrounded g, the
# packer's tie rule). spd-trimul A/B, `perf/spd_trimul/bench.py` arm `epi1`. 2 is 1 plus the
# residual add, for a caller that passes `resid` (the pair tensor the update is added to): the
# product is added to z in DST and written back into z, and the caller's `add_` is skipped. Without
# `resid` a call runs as 1.
EPI = int(os.environ.get("TT_BIO_TRIMUL_TAIL_EPI", "0"))


def set_epi(v: int) -> int:
    """A/B switch for the paired harness. Returns the previous value."""
    global EPI
    prev, EPI = EPI, int(v)
    return prev


def _epi() -> int:
    """EPI, or 2 under Protenix's `trimul_tail` precision lever."""
    from .tenstorrent import lever
    return EPI or (2 if lever("trimul_tail") else 0)

# The swept block config each pass runs, resolved per call from the weight's (kt, nt) key through
# the same `tenstorrent._MM_BLOCK` table production's own projections read, so a served call folds
# the identical single K block in the identical order the ops it replaces would.
#
# The key set is an allow-list, not the whole table, and (8, 8) is the only entry that is SAFE.
# MEASURED on qb2 card 2, 2026-08-15: at (12, 12) -- `_MM_BLOCK`'s opendde c_z = 384 entry,
# (8, 12, 1, 2, 1) -- this descriptor builds and runs, returns wrong numbers at N = 32 and 64
# (max_abs_diff 3.19 and 4.75 against the ops it replaces, on an O(1) distribution) and then
# HANGS THE DEVICE at N = 128: `generic_op` enqueues, and the sync never returns. The kernels are
# a transcription of `minimal_matmul` swept only at (4, 8, 1, 4, 1); `out_block` doubles to 8
# tiles and `subblock_h` halves to 2 at the wider key, and the circular buffers do not follow.
#
# So this is an allow-list and not `_mm_block_for`, for two independent reasons. `_MM_BLOCK` also
# holds (4, 4) and (2, 2), so a general lookup would switch F1 ON for boltz2 and openfold3, which
# decline 100 % of their calls today -- a four-model default flip inside a one-model change. And
# it would hand those models a block this kernel does not correctly implement. Widening the set
# is a kernel fix first, then a five-model release gate, and never a one-line table swap.
#
# A trimul tail's weights are square ([c_z, c_z]), so the key is always (kt, kt).
F1_BLOCK_KEYS = {(8, 8)}

# P7 step 3: (4, 4) is boltz2's and openfold3's key at c_z = 128, and it is the reason F1 ships on
# by default while declining 100 % of both models' calls. `_MM_BLOCK` maps it to (4, 4, 1, 4, 1)
# against (8, 8)'s (4, 8, 1, 4, 1): same M_block, same subblock_h, same subblock_w, only K_block
# halving, and kt = 4 still means exactly one K block, which is this kernel's stated precondition.
# Neither of the two things that broke (12, 12) -- out_block doubling, subblock_h halving -- moves
# here, so the entry itself is most likely correct.
#
# It is still a loss, and now a measured one. `fused_tail` allocates its output in
# DRAM_MEMORY_CONFIG, while the incumbent tail at c_z = 128 is `_pair_proj_linear(l1_out=True)` and
# keeps both projections and their product in L1, so F1 at this key turns an L1-resident product
# into a DRAM round trip: +2.000 Z per trimul for -2 ops, +2.019 ms/trimul, 1.066 s on a 512 aa
# Boltz-2 fold. The general statement is the useful one: F1 deletes bytes wherever its incumbent is
# DRAM-resident and ADDS them wherever the incumbent is L1-resident, and nothing in its eligibility
# check looks at where the ops it replaces put their output.
# perf/b2x_trimul/fold_ab_512_qb2c2.json, state/b2x-trimul-fusion-unlock.md.
#
# OFF by default, and it stays off, on both counts: it is slower here, and turning it on would be a
# two-model flip (boltz2 AND openfold3 share the key) needing openfold3's own parity leg.
def set_f1_cz128(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global F1_BLOCK_KEYS
    prev = (4, 4) in F1_BLOCK_KEYS
    F1_BLOCK_KEYS = {(8, 8), (4, 4)} if on else {(8, 8)}
    _block_for.cache_clear()
    return prev


if env_flag("TT_BIO_TRIMUL_TAIL_F1_CZ128", False):
    F1_BLOCK_KEYS = {(8, 8), (4, 4)}


def _tiles(n):
    return (int(n) + TILE - 1) // TILE


@lru_cache(maxsize=None)
def _block_for(kt, nt):
    from . import tenstorrent as TT    # late: `tenstorrent` imports this module at its own import
    return TT._MM_BLOCK[(kt, nt)] if (kt, nt) in F1_BLOCK_KEYS else None


# A/B only (spd-trimul): the tail's own (M, K, N, subblock_h, subblock_w) in place of `_MM_BLOCK`'s
# for an allow-listed key. K must stay the whole contraction (one K block), so the order of the sum
# and the numerics do not move; only how output tiles map to cores and how often a core reads each
# activation tile. None is production's entry.
BLOCK = None


def set_block(b):
    """A/B switch for the paired harness. Returns the previous value."""
    global BLOCK
    prev, BLOCK = BLOCK, (tuple(b) if b else None)
    return prev


def _block(w):
    """F1's block config for this weight, or None when its (kt, nt) key is not allow-listed."""
    b = _block_for(_tiles(w.shape[-2]), _tiles(w.shape[-1]))
    return b if b is None or BLOCK is None else BLOCK

STATS = [0, 0]          # served, declined
OUT_L1_STATS = [0, 0]   # products packed straight into L1, products that went to DRAM
REJECTS: dict = {}      # (reason, shape) -> count, so a decline is diagnosable from the fold JSON


def _reject(why, shape=""):
    STATS[1] += 1
    REJECTS[(why, shape)] = REJECTS.get((why, shape), 0) + 1
    return None


def eligible(xa, xb, wa, wb):
    """None when F1's descriptor covers this call, else the reason it does not.

    Every clause is a real assumption of the fork, so a decline names which one.
    """
    # Membership widened, the pair tests below untouched: this clause used to be "bf16 must
    # appear somewhere", and a fast dtype appearing somewhere is a strict superset of it.
    if not (MG.FAST_DTYPES & {xa.dtype, xb.dtype, wa.dtype, wb.dtype}):
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
    kt = _tiles(wa.shape[-2])
    nt = _tiles(wa.shape[-1])
    block = _block(wa)
    if block is None:
        # One K block is the fusion's whole simplification, and the block has to be the one
        # production folds with or the fusion is a numerics change rather than a rewrite. At 512 aa
        # this declines the narrow-hidden trimuls (c_hidden 64, kt = 2), which production does not
        # put through `minimal_matmul` either: `_MM_BLOCK` has no entry for a 2-tile output.
        return f"k_tiles={kt}"
    M, K, N, _, _ = block
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
        buffer_index=idx, data_format=ttnn.bfloat16, page_size=MG.tile_bytes(ttnn.bfloat16))
    return ttnn.CBDescriptor(
        total_size=tiles * MG.tile_bytes(ttnn.bfloat16), core_ranges=core_grid,
        format_descriptors=[fmt])


def _build(device, xa, xb, wa, wb, outs, grid, ckc, block, epi, shared):
    defs = {"TRIMUL_TAIL_PASSES": PASSES, "TRIMUL_TAIL_ROUND": ROUND,
            "TRIMUL_TAIL_SKIP_SIGMOID": SKIP_SIGMOID, "TRIMUL_TAIL_EPI": epi,
            "TRIMUL_TAIL_SHARED_IN0": int(shared), "TRIMUL_TAIL_ABL": ABL}
    # EPI >= 1 applies the sigmoid to the DST a GEMM pass packs from, which is the finished sum
    # only when the contraction is one K block.
    assert epi == 0 or block[1] == _tiles(wa.shape[-2]), (epi, block, tuple(wa.shape))
    entry = MG.build(device, xa, wa, list(outs), (block, grid), ckc,
                     defines=defs, kernel_dir=KERNEL_DIR)

    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    out_block = block[0] * block[2]
    # c_4 / c_5: the two bf16 GEMM results, double buffered. c_6: the rounded gate, one tile live.
    entry["cbs"] += [_cb(4, core_grid, out_block * 2),
                     _cb(5, core_grid, out_block * 2),
                     _cb(6, core_grid, 2)]
    if epi == 2:
        # c_7: the residual block, read by the non-writer DM kernel, double buffered.
        entry["cbs"].append(_cb(7, core_grid, out_block * 2))

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


# The weights-resident program (`kernels/trimul_tail_res`): every core keeps both [K, N] weights
# in L1 and streams only its own activation rows, so no core forwards anything to another. The
# stage ablation found the 2D program's in0 multicast chain and CB handshakes binding (~2.8 ms of
# a 3.4 ms GEMM pass at 736 with every stage's work removed). Same epilogue as EPI 1 / 2, op for
# op, so the output is the same bits (torch.equal at 128 and 736, EPI 1 and 2, shared and unshared
# in0). At 736 on WH it takes the EPI 2 tail from 8.66 to 6.72 ms; what is left is the bf16 SFPU
# sigmoid (~2.0 ms) and the GEMM math (~1.8 ms) serialized on the math thread, over a 3.7 ms data
# floor (perf/spd_trimul/tail_res.py --abl). bf16, one K block, DRAM output, no split.
RES = env_flag("TT_BIO_TRIMUL_TAIL_RES", True)
# The gated in-projection's pair mask folded into the resident epilogue (`mask_ok`).
MASK_FOLD = env_flag("TT_BIO_TRIMUL_MASK_FOLD", True)
MASK_STATS = [0]     # calls that folded the mask


if env_flag("TT_BIO_TRIMUL_STATS", False):
    # One stderr line at exit: which trimul routes a whole run took (fold-level proof a lever fired).
    import atexit, json, sys

    def _print_stats():
        from . import tenstorrent as _T
        print("TRIMUL_STATS " + json.dumps({
            "tail": STATS, "res": RES_STATS, "mask_fold": MASK_STATS, "resid": RESID_STATS, "t1": T1_STATS, "t1_why": T1_WHY, "t2": T2_STATS,
            "rejects": {f"{k[0]}:{k[1]}": v for k, v in REJECTS.items()},
            "routes": {f"{k[0]}/{k[1]}": v for k, v in _T.TRIMUL_MM_TRANSPOSE_STATS.items()}}),
            file=sys.stderr, flush=True)
    atexit.register(_print_stats)


def set_mask_fold(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global MASK_FOLD
    prev, MASK_FOLD = MASK_FOLD, bool(on)
    return prev
# The gates' sigmoid as a degree-8 polynomial in gin_moved and the resident tail
# (kernels/trimul_gin_moved/sigmoid_poly.hpp), max |error| 3e-4 against exp + reciprocal's bf16 rounding.
# Resident tail at 736 on WH (1000 MHz): 6.53 -> 5.51 ms, rel_rms to float64 0.00307 -> 0.00304.
# triatt_tail ships the same polynomial (TT_BIO_TRIATT_TAIL_SIGPOLY).
SIGPOLY = env_flag("TT_BIO_TRIMUL_SIGPOLY", True)
# Round DST to bf16 nearest-even in those two kernels: bit 0 before the output pack, bit 1 before the p/g
# packs, bit 2 (resident tail, EPI 2) the product before the residual add, whose move into srcA truncates.
# Bits 0 and 1 measured null on the outputs (gm8): the packs round already. Bit 2 brings EPI 2 to one rounding
# (tests/test_trimul_tail_epi.py's bound holds, rel_rms 0.00240 -> 0.00224) for +5 % tail time (gm9/gm10), so it
# stays off: the fold grades pass without it.
RNE = int(os.environ.get("TT_BIO_TRIMUL_RNE", "0"))
RES_ABL = 0          # the resident compute's stage ablation (see its compute.cpp). Diagnostic only.
RES_STATS = [0, 0]   # served by the resident program, declined to the 2D one


def set_res(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global RES
    prev, RES = RES, bool(on)
    return prev


def _res_ok(xa, wa, epi, split, mem):
    return (RES and epi >= 1 and split in (1, 2) and mem == ttnn.DRAM_MEMORY_CONFIG
            and xa.dtype == ttnn.bfloat16 and wa.dtype == ttnn.bfloat16
            and xa.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and wa.memory_config() == ttnn.DRAM_MEMORY_CONFIG)


def mask_ok(mask, xa, wa):
    """Whether `fused_tail(..., split=2, mask=mask)` can fold the pair mask into the resident
    epilogue: a bf16 tiled DRAM [B, S, S] mask, S a whole number of tiles, and the activation
    [B, S, S, K] so its flattened rows are the mask's (b, x, y)."""
    if not (RES and MASK_FOLD) or mask is None or len(mask.shape) != 3:
        return False
    B, S, S2 = (int(d) for d in mask.shape)
    return (S == S2 and S % TILE == 0 and tuple(int(d) for d in mask.padded_shape) == (B, S, S)
            and mask.dtype == ttnn.bfloat16 and mask.layout == ttnn.TILE_LAYOUT
            and mask.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and [int(d) for d in xa.shape][:-1] in ([B, S, S], [1, B, S, S])
            and _tiles(wa.shape[-1]) % 8 == 0)


def _build_res(xa, xb, wa, wb, outs, grid, ckc, block, epi, shared, mask=None):
    Mb = block[0]
    sbw = block[4]
    kt, nt = _tiles(wa.shape[-2]), _tiles(wa.shape[-1])
    mt = 1
    for d in [int(d) for d in xa.padded_shape][:-1]:
        mt *= d
    mt //= TILE
    nb = mt // Mb
    gx, gy = grid
    ncores = gx * gy
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    b0 = 0
    for c in range(ncores):
        n = nb // ncores + (c < nb % ncores)
        x, y = c % gx, c // gx
        rd[x][y] = [b0 * Mb, n]
        wr[x][y] = [b0 * Mb, n]
        cp[x][y] = [n]
        b0 += n
    assert b0 == nb, (b0, nb)
    resid = int(epi == 2)
    S_t = int(mask.shape[-1]) // TILE if mask is not None else 0
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    d = KERNEL_DIR.parent / "trimul_tail_res"
    reader = ttnn.KernelDescriptor(
        kernel_source=str(d / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, Mb, int(shared), resid] + acc(xa) + acc(wa) + acc(xb)
        + acc(wb) + acc(outs[0]) + [int(mask is not None), S_t] + acc(mask if mask is not None else xa),
        runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[nt, Mb, len(outs)] + acc(outs[0]) + acc(outs[-1]),
        runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, fp32, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, Mb, sbw, int(shared), resid, RES_ABL, int(mask is not None),
                           int(SIGPOLY), RNE],
        runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32,
            dst_full_sync_en=full))
    blk = Mb * nt
    cbs = [_cb(0, core_grid, 2 * Mb * kt), _cb(1, core_grid, 2 * kt * nt),
           _cb(2, core_grid, 2 * blk), _cb(4, core_grid, blk), _cb(5, core_grid, blk)]
    if resid:
        cbs.append(_cb(7, core_grid, 2 * blk))
    if mask is not None:
        cbs += [_cb(3, core_grid, 1), _cb(6, core_grid, 2 * Mb)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def _run_res(entry, xa, xb, wa, wb, outs, mask=None):
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [xa.buffer_address(), wa.buffer_address(), xb.buffer_address(),
                                  wb.buffer_address(), outs[0].buffer_address(),
                                  (mask if mask is not None else xa).buffer_address()]
    writer.common_runtime_args = [outs[0].buffer_address(), outs[-1].buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([xa, wa, xb, wb, *outs] + ([mask] if mask is not None else []), pd)


_CACHE: dict = {}


RESID_STATS = [0, 0]    # residual folded into the tail, offered but not taken


def _resid_ok(resid, xa, wa, mem):
    """Whether `resid` can be the tail's in-place destination: the output's exact shape, bf16,
    tiled, interleaved DRAM, and the product not headed for L1."""
    return (resid is not None and mem == ttnn.DRAM_MEMORY_CONFIG
            and resid.dtype == ttnn.bfloat16 and resid.layout == ttnn.TILE_LAYOUT
            and resid.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and [int(d) for d in resid.shape] == [int(d) for d in xa.shape][:-1] + [int(wa.shape[-1])]
            and tuple(resid.padded_shape) == tuple(xa.padded_shape)[:-1] + (int(wa.padded_shape[-1]),))


def _alloc_out(shape, device, mem):
    """The product's buffer and where it landed: `mem`, or DRAM if the allocator refuses L1."""
    try:
        out = ttnn.allocate_tensor_on_device(
            shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    except Exception:                                                      # noqa: BLE001
        if mem == ttnn.DRAM_MEMORY_CONFIG:
            raise
        mem = ttnn.DRAM_MEMORY_CONFIG
        OUT_L1_STATS[1] += 1
        out = ttnn.allocate_tensor_on_device(
            shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    else:
        OUT_L1_STATS[0 if mem != ttnn.DRAM_MEMORY_CONFIG else 1] += 1
    return out, mem


def fused_tail(xa, xb, wa, wb, ckc, grid, out_memory_config=None, resid=None, split=1, mask=None):
    """`p * sigmoid(g)` for `p = xa @ wa`, `g = xb @ wb`, in one kernel. None if out of scope.

    `xa is xb` (the trimul in-projection: p and g of one activation) reads each activation block
    once for both passes. `split` > 1 writes the product as that many equal column chunks, separate
    tensors, and returns them as a list (no residual then). With `split=2` and `mask` (only where
    `mask_ok`) the first chunk comes out multiplied by the pair mask; None if the resident program
    does not serve the call, so a mask is never silently dropped.

    With `resid` and EPI == 2 it computes `resid + p * sigmoid(g)` into `resid` itself and returns
    `resid` (the caller's in-place add is then already done: `_add_input` sees `u is x`). Any
    other case ignores `resid` and returns the bare product.

    `out_memory_config` is where the product lands. It is a real perf decision and not a
    detail: the three ops this replaces put their product wherever `_trimul_out_proj` put
    `p_out`, which at c_z = 128 is L1. Allocating in DRAM there turns an L1-resident product
    into a round trip and F1 ADDS bytes instead of deleting them (measured +2.019 ms/trimul,
    state/b2x-trimul-fusion-unlock.md). The writer reads its buffer type out of
    `TensorAccessorArgs`, so an L1 destination needs no kernel change -- only its own cache
    entry, which is why the memory config is part of the key. Falls back to DRAM if the
    allocator refuses, which is the only test that knows what the block is already holding.
    """
    from . import ops
    if ops.taping():
        return None   # no backward for `generic_op`; the three composed ops run instead
    if dest_guard.exposed_args(ckc):
        return None   # the kernel keeps all of K in dest; see dest_guard

    why = eligible(xa, xb, wa, wb)
    if why is not None:
        return _reject(why, "x".join(str(int(d)) for d in xa.padded_shape)
                       + "@" + "x".join(str(int(d)) for d in wa.shape))
    device = xa.device()
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    mem = out_memory_config or ttnn.DRAM_MEMORY_CONFIG
    nt = _tiles(wa.shape[-1])
    if split > 1 and (nt % split or (nt // split) % _block(wa)[2]):
        return _reject(f"split={split}", f"n_tiles={nt}")
    shared = xa.buffer_address() == xb.buffer_address()
    want = _epi()
    epi = min(want, 1)
    if want == 2 and resid is not None and split == 1:
        if _resid_ok(resid, xa, wa, mem):
            epi = 2
            RESID_STATS[0] += 1
        else:
            RESID_STATS[1] += 1
    shape = ttnn.Shape([int(d) for d in xa.shape][:-1] + [int(wa.shape[-1]) // split])
    if epi == 2:
        outs = [resid]
    else:
        outs = []
        for _ in range(split):
            out, mem = _alloc_out(shape, device, mem)
            outs.append(out)
    key = (spec(xa), spec(wa), tuple(grid), tuple(str(c) for c in ckc), ROUND, SKIP_SIGMOID,
           ABL, epi, str(mem), _block(wa), shared, split)
    if mask is not None and not (split == 2 and mask_ok(mask, xa, wa)
                                 and _res_ok(xa, wa, epi, split, mem) and not ABL):
        for o in outs:
            ttnn.deallocate(o)
        return _reject("mask", "x".join(str(int(d)) for d in mask.padded_shape))
    if _res_ok(xa, wa, epi, split, mem) and not ABL:
        key = ("res", RES_ABL, SIGPOLY, RNE, None if mask is None else str(mask.padded_shape)) + key
        entry = _CACHE.get(key)
        if entry is None:
            entry = _CACHE[key] = _build_res(xa, xb, wa, wb, outs, grid, ckc, _block(wa), epi,
                                             shared, mask)
        _run_res(entry, xa, xb, wa, wb, outs, mask)
        MASK_STATS[0] += mask is not None
        STATS[0] += 1
        RES_STATS[0] += 1
        return outs if split > 1 else outs[0]
    if RES:
        RES_STATS[1] += 1

    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(device, xa, xb, wa, wb, outs, grid, ckc, _block(wa), epi,
                                     shared)
    else:
        # `MG.rebind` repacks the descriptor itself, so only bind B separately when it does not run.
        addrs = (xa.buffer_address(), wa.buffer_address(), tuple(o.buffer_address() for o in outs))
        b = (xb.buffer_address(), wb.buffer_address())
        stale_b = b != entry["b_addrs"]
        if stale_b:
            _bind_b(entry, *b)
        if addrs != entry["addrs"]:
            MG.rebind(entry, *addrs)
        elif stale_b:
            _repack(entry)

    ttnn.generic_op([xa, wa, xb, wb, *outs], entry["pd"])
    STATS[0] += 1
    return outs if split > 1 else outs[0]


# The gated in-projection with its channel move fused in (`kernels/trimul_gin_moved`). The
# resident program above writes a and b as [B, S, S, C] and two plain moves then bring C to the
# batch axis (2.0 ms each at 736 on WH, gather-transaction bound). Here the projection is computed
# transposed, W^T @ X^T per row tile, so each product tile is [channel x y] and the writer gathers
# the moved [B, C, S, S] tiles straight out of L1: the moves' DRAM round trip is gone and their
# gather runs on the writer, which has slack under the resident kernel's compute bound.
# Default on wherever trimul_gin runs: at 736 on WH (HiFi3, 1000 MHz) it is 7.33 ms against the 10.43 ms
# of the resident split plus its two moves, rel_rms to float64 unchanged (perf/spd_trimul/gin_move.py,
# BOARD 2026-10-09 04:55Z). TT_BIO_TRIMUL_GIN_MOVE=0 takes the split route back.
GIN_MOVE = env_flag("TT_BIO_TRIMUL_GIN_MOVE", True)
GIN_MOVE_STATS = [0, 0]   # served, declined
GIN_MOVE_ABL = 0          # the three kernels' stage ablation bits (see each .cpp). Diagnostic only.
GIN_MOVE_NB = int(os.environ.get("TT_BIO_TRIMUL_GIN_MOVE_NB", "2"))   # writer tiles per read barrier


def _gin_move_tiles(kt, ct2, mask):
    return 2 * ct2 * kt + 32 * kt + 64 + 2 * 32 + 2 * GIN_MOVE_NB + (33 if mask else 0)


def gin_move_split(x, c, mask=None):
    """How many channel slices `gin_moved` runs `c` output channels of `a` (and of `b`) in, or None.

    One launch keeps the whole [2c, K] weight pair resident, which is 256 of the 677 tiles at
    ESMFold2's c_z 256 / hidden 256 and over the 0.85-of-L1 budget. Each slice holds 1/n of the
    weights and reads the activations again; per channel the arithmetic is the one-launch kernel's,
    so the output is the same bits. The smallest n that fits, with every slice whole tiles of a and b.
    """
    shp = [int(d) for d in x.shape]
    if c % TILE or shp[-1] % TILE:
        return None
    from .tenstorrent import _l1_bank_bytes
    ct, kt = c // TILE, shp[-1] // TILE
    for n in (1, 2, 4):
        if ct % n == 0 and (_gin_move_tiles(kt, 2 * ct // n, mask is not None) * MG.tile_bytes(ttnn.bfloat16)
                            <= 0.85 * _l1_bank_bytes()):
            return n
    return None


def gin_moved_ok(x, wpT, mask=None):
    """Whether `gin_moved` serves this call: bf16 tiled interleaved DRAM pair tensor [B?, S, S, K]
    with S a whole number of tiles, transposed weights [2C, K] with C a whole number of tiles, the
    mask (if any) as `mask_ok` wants it, and the program's circular buffers inside L1."""
    shp = [int(d) for d in x.shape]
    if not (GIN_MOVE and len(shp) in (3, 4) and shp[-2] == shp[-3] and shp[-2] % TILE == 0
            and x.dtype == ttnn.bfloat16 and wpT.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and wpT.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and int(wpT.shape[-1]) == shp[-1] and shp[-1] % TILE == 0
            and int(wpT.shape[-2]) % (2 * TILE) == 0):
        return False
    if mask is not None:
        S = shp[-2]
        B = shp[0] if len(shp) == 4 else 1
        if not (tuple(int(d) for d in mask.padded_shape) == (B, S, S) and mask.dtype == ttnn.bfloat16
                and mask.layout == ttnn.TILE_LAYOUT and mask.memory_config() == ttnn.DRAM_MEMORY_CONFIG):
            return False
    from .tenstorrent import _l1_bank_bytes
    tiles = _gin_move_tiles(shp[-1] // TILE, int(wpT.shape[-2]) // TILE, mask is not None)
    return tiles * MG.tile_bytes(ttnn.bfloat16) <= 0.85 * _l1_bank_bytes()


def _build_gin_move(x, wpT, wgT, outs, grid, ckc, mask, cto=0, ctt=None):
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    St, kt, ct2 = shp[-2] // TILE, shp[-1] // TILE, int(wpT.shape[-2]) // TILE
    nu = B * St * St
    gx, gy = grid
    ncores = gx * gy
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        n = nu // ncores + (c < nu % ncores)
        x_, y_ = c % gx, c // gx
        rd[x_][y_] = [u0, n]
        wr[x_][y_] = [u0, n]
        cp[x_][y_] = [n]
        u0 += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    d = KERNEL_DIR.parent / "trimul_gin_moved"
    reader = ttnn.KernelDescriptor(
        kernel_source=str(d / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, St, int(mask is not None), GIN_MOVE_ABL] + acc(x) + acc(wpT) + acc(wgT)
        + acc(mask if mask is not None else x),
        runtime_args=rd, common_runtime_args=[0] * 4, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[ct2, St, GIN_MOVE_ABL, GIN_MOVE_NB, cto, ct2 // 2 if ctt is None else ctt]
        + acc(outs[0]) + acc(outs[1]),
        runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, fp32, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, 4, int(mask is not None), GIN_MOVE_ABL, int(SIGPOLY), RNE],
        runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32,
            dst_full_sync_en=full))
    cbs = [_cb(0, core_grid, 2 * ct2 * kt), _cb(1, core_grid, 32 * kt), _cb(2, core_grid, 64),
           _cb(4, core_grid, 32), _cb(5, core_grid, 32), _cb(24, core_grid, 2 * GIN_MOVE_NB)]
    if mask is not None:
        cbs += [_cb(3, core_grid, 1), _cb(6, core_grid, 32)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def gin_moved(x, wpT, wgT, ckc, grid, mask=None):
    """`a, b` of the gated in-projection, already channel-moved: [B, C, S, S] each, for
    `[a | b] = (x @ Wp) * sigmoid(x @ Wg)` with `wpT`, `wgT` the transposed [2C, K] weights, and
    `a` times the pair mask when `mask` is given. None where `gin_moved_ok` declines.

    `wpT` and `wgT` may also be equal-length sequences of channel slices (`gin_move_split`): slice
    s holds rows s of `[p_a | p_b]`'s n equal parts of each half, and writes channels
    [s * C/n, (s+1) * C/n) of `a` and of `b`."""
    from . import ops
    wps = list(wpT) if isinstance(wpT, (list, tuple)) else [wpT]
    wgs = list(wgT) if isinstance(wgT, (list, tuple)) else [wgT]
    if ops.taping() or dest_guard.exposed_args(ckc) or not all(gin_moved_ok(x, w, mask) for w in wps):
        GIN_MOVE_STATS[1] += 1
        return None
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    n = len(wps)
    S, Cs = shp[-2], int(wps[0].shape[-2]) // 2
    device = x.device()
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([B, n * Cs, S, S]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(2)]
    for s_, (wp, wg) in enumerate(zip(wps, wgs)):
        key = ("gin_move", str(x.padded_shape), str(wp.padded_shape), tuple(grid),
               tuple(str(c) for c in ckc), mask is not None, GIN_MOVE_ABL, GIN_MOVE_NB, SIGPOLY, RNE,
               s_, n)
        entry = _CACHE.get(key)
        if entry is None:
            entry = _CACHE[key] = _build_gin_move(x, wp, wg, outs, grid, ckc, mask,
                                                  cto=s_ * Cs // TILE, ctt=n * Cs // TILE)
        reader, writer, _ = entry["kernels"]
        reader.common_runtime_args = [x.buffer_address(), wp.buffer_address(), wg.buffer_address(),
                                      (mask if mask is not None else x).buffer_address()]
        writer.common_runtime_args = [outs[0].buffer_address(), outs[1].buffer_address()]
        pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
        ttnn.generic_op([x, wp, wg, *outs] + ([mask] if mask is not None else []), pd)
    GIN_MOVE_STATS[0] += 1
    MASK_STATS[0] += mask is not None
    return outs


# gin_moved for a ragged pair grid with the projection biases: AF2's trimul at 195 tokens, whose
# in-projection carries biases and whose S is not a whole number of tiles, runs today as a biased
# minimal_matmul into L1 and two reblock_permute_gated moves (~1.09 ms per 64-channel chunk on WH).
# kernels/trimul_gin_ragged: x rows past S re-read row S - 1, the bias tile seeds DST before the K
# loop, the writer zeroes rows and columns >= S, and a unit is (cube, channel range) so 49 cubes still
# fill the grid. x and the outputs may sit in L1. Opt-in (perf/spd_gap_bind/gin195.py grades it).
GIN_RAGGED_QS = int(os.environ.get("TT_BIO_TRIMUL_GIN_RAGGED_QS", "0"))   # 0: one split
GIN_RAGGED_STATS = [0, 0]   # served, declined


def gin_ragged_ok(x, wpT, S, mask=None):
    shp = [int(d) for d in x.shape]
    if not (len(shp) in (3, 4) and shp[-2] == shp[-3] == S and S % TILE
            and x.dtype == ttnn.bfloat16 and wpT.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT and not x.memory_config().is_sharded()
            and wpT.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            and int(wpT.shape[-1]) == shp[-1] and shp[-1] % TILE == 0
            and int(wpT.shape[-2]) % (2 * TILE) == 0):
        return False
    if mask is not None:
        St = -(-S // TILE)
        B = shp[0] if len(shp) == 4 else 1
        if not (tuple(int(d) for d in mask.padded_shape) == (B, St * TILE, St * TILE)
                and mask.dtype == ttnn.bfloat16 and mask.layout == ttnn.TILE_LAYOUT
                and mask.memory_config() == ttnn.DRAM_MEMORY_CONFIG):
            return False
    from .tenstorrent import _l1_bank_bytes
    tiles = _gin_move_tiles(shp[-1] // TILE, int(wpT.shape[-2]) // TILE, mask is not None) \
        + int(wpT.shape[-2]) // TILE * 2 + 2
    return tiles * MG.tile_bytes(ttnn.bfloat16) <= 0.85 * _l1_bank_bytes()


def gin_ragged_bias(bp, bg, S, device):
    """The bias tiles gin_ragged takes, from the [2C] value and gate biases (torch): [bp | bg], each
    row c of channel tile t carrying bias[32 t + c] in every column, then one column-mask tile for
    the last column tile of an S-wide grid (1 in columns < S - 32 * (ceil(S / 32) - 1))."""
    import torch
    rows = torch.cat([bp.reshape(-1), bg.reshape(-1)]).float()
    last = S - TILE * (-(-S // TILE) - 1)
    cmask = (torch.arange(TILE) < last).float()[None, :].expand(TILE, -1)
    t = torch.cat([rows[:, None].expand(-1, TILE), cmask])
    return ttnn.from_torch(t.contiguous().to(torch.bfloat16),
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _gin_ragged_qs(ct2):
    # One unit per cube: splitting the channels re-reads x per split. At 195 x 64 channels on the
    # Wormhole 8x9 grid qs 1 read 0.49 ms (DRAM) / 0.52 (L1) against 0.58 / 0.80 at qs 4.
    return GIN_RAGGED_QS if GIN_RAGGED_QS and ct2 % GIN_RAGGED_QS == 0 else 1


def _build_gin_ragged(x, wpT, wgT, bias, outs, grid, ckc, mask, S, qs):
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    St, kt, ct2 = -(-S // TILE), shp[-1] // TILE, int(wpT.shape[-2]) // TILE
    nu = B * St * St * qs
    gx, gy = grid
    ncores = gx * gy
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        n = nu // ncores + (c < nu % ncores)
        x_, y_ = c % gx, c // gx
        rd[x_][y_] = [u0, n]
        wr[x_][y_] = [u0, n]
        cp[x_][y_] = [u0, n]
        u0 += n
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    d = KERNEL_DIR.parent / "trimul_gin_ragged"
    reader = ttnn.KernelDescriptor(
        kernel_source=str(d / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, St, S, int(mask is not None), qs] + acc(x) + acc(wpT) + acc(wgT)
        + acc(mask if mask is not None else x) + acc(bias),
        runtime_args=rd, common_runtime_args=[0] * 5, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[ct2, St, S, GIN_MOVE_NB, qs] + acc(outs[0]) + acc(outs[1]),
        runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, fp32, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, 4, int(mask is not None), GIN_MOVE_ABL, int(SIGPOLY), RNE, qs, St],
        runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32,
            dst_full_sync_en=full))
    cbs = [_cb(0, core_grid, 2 * ct2 * kt), _cb(1, core_grid, 32 * kt), _cb(2, core_grid, 64),
           _cb(4, core_grid, 32), _cb(5, core_grid, 32), _cb(7, core_grid, 2 * ct2 + 1),
           _cb(24, core_grid, 2 * GIN_MOVE_NB), _cb(25, core_grid, 1)]
    if mask is not None:
        cbs += [_cb(3, core_grid, 1), _cb(6, core_grid, 32)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def gin_ragged(x, wpT, wgT, bias, ckc, grid, S, mask=None, memory_config=None):
    """`a, b` of the biased gated in-projection, channel-moved, for a pair grid of S rows that need
    not be whole tiles: [B, C, S, S] each with zero padding, for `[a | b] = (x @ Wp + bp) *
    sigmoid(x @ Wg + bg)` [* m on a]. `bias` from `gin_ragged_bias` at the same S. None where it declines."""
    from . import ops
    if ops.taping() or dest_guard.exposed_args(ckc) or not gin_ragged_ok(x, wpT, S, mask):
        GIN_RAGGED_STATS[1] += 1
        return None
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    C = int(wpT.shape[-2]) // 2
    ct2 = int(wpT.shape[-2]) // TILE
    St = -(-S // TILE)
    qs = _gin_ragged_qs(ct2)
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    device = x.device()
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([B, C, S, S]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, device, mc)
            for _ in range(2)]
    key = ("gin_ragged", str(x.padded_shape), str(x.memory_config().buffer_type), str(wpT.padded_shape),
           tuple(grid), tuple(str(c) for c in ckc), mask is not None, GIN_MOVE_NB, SIGPOLY, RNE, S, qs, GIN_MOVE_ABL,
           str(mc.buffer_type))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build_gin_ragged(x, wpT, wgT, bias, outs, grid, ckc, mask, S, qs)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), wpT.buffer_address(), wgT.buffer_address(),
                                  (mask if mask is not None else x).buffer_address(), bias.buffer_address()]
    writer.common_runtime_args = [outs[0].buffer_address(), outs[1].buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, wpT, wgT, bias, *outs] + ([mask] if mask is not None else []), pd)
    GIN_RAGGED_STATS[0] += 1
    return outs


# T2: the whole c_z = 128 output tail in the resident program (`kernels/trimul_tail_res`,
# compute_ln.cpp / reader_ln.cpp, its writer): z += (layer_norm(x) @ Wp) * sigmoid(g) with g the
# fused in-projection's g_out read from DRAM, z written in place. Today that is four ops: at
# N = 512 layer_norm 511 us, the projection 287 us, multiply_ with SIGMOID 522 us, add_ 348 us
# (1.67 ms); T2's floor is reading x, g, z and writing z, 4 x 67 MB = 600 us at 448 GB/s. Each
# core holds whole rows of K, so the channel LayerNorm is a per-core prologue. NEVER RUN ON A
# DEVICE: OFF by default, perf/spd_trimul/tail_t2.py is its first accuracy and timing check.
# The model path takes it under the `trimul_t2` lever; T2 here only forces it on for the benches.
T2 = False
T2_STATS = [0, 0]       # served, declined
T2_KEYS = {(4, 4)}      # (kt, nt): c_z = 128 only, the shape the kernel was written for


def set_t2(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global T2
    prev, T2 = T2, bool(on)
    return prev


def _cb_fmt(idx, core_grid, tiles, dtype):
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=idx, data_format=dtype, page_size=MG.tile_bytes(dtype))
    return ttnn.CBDescriptor(
        total_size=tiles * MG.tile_bytes(dtype), core_ranges=core_grid,
        format_descriptors=[fmt])


def _vec_ok(v, K):
    """gamma / beta as `torch_to_tt` leaves a norm parameter: bf16, tiled, K values in row 0."""
    return (v is not None and v.dtype == ttnn.bfloat16 and v.layout == ttnn.TILE_LAYOUT
            and int(v.shape[-1]) == K and all(int(d) == 1 for d in list(v.shape)[:-1])
            and v.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)


def t2_eligible(x, ln_w, ln_b, wp, g, resid):
    """None when T2 serves this call, else the reason it does not."""
    from .tenstorrent import lever
    if not (T2 or lever("trimul_t2")):
        return "off"
    for t in (x, wp, g, resid):
        if t is None or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return "memcfg"
    return _t2_shape_why([int(d) for d in x.shape], [int(d) for d in x.padded_shape],
                         ln_w, ln_b, wp, g, resid)


def _t2_shape_why(xs, xps, ln_w, ln_b, wp, g, resid):
    """T2's checks on everything but x's own dtype / memory: `xs` / `xps` are x's logical and padded
    shape in the [.., S, S, K] layout T2 reads (T2CM passes its channel-major x's shape permuted)."""
    if len(wp.shape) != 2:
        return "weight_shape"
    K, N = int(wp.shape[-2]), int(wp.shape[-1])
    kt, nt = _tiles(K), _tiles(N)
    if (kt, nt) not in T2_KEYS or K % TILE or N % TILE:
        return f"key={kt}x{nt}"
    if xs[-1] != K or xps[-1] != K:
        return "act_k"
    if not (_vec_ok(ln_w, K) and _vec_ok(ln_b, K)):
        return "ln_params"
    want = xs[:-1] + [N]
    if [int(d) for d in g.shape] != want or [int(d) for d in resid.shape] != want:
        return "out_shape"
    if (tuple(g.padded_shape) != tuple(resid.padded_shape)
            or tuple(resid.padded_shape) != tuple(xps[:-1]) + (N,)):
        return "padded_shape"
    from . import tenstorrent as TT    # late: `tenstorrent` imports this module at its own import
    block = TT._MM_BLOCK.get((kt, nt))
    if block is None:
        return "block"
    mt = 1
    for d in xps[:-1]:
        mt *= d
    mt //= TILE
    if mt % block[0] or (block[0] * nt) % 4 or nt % block[4]:
        return f"m_tiles={mt}"
    return None


def _t2_tiles(kt, nt, Mb):
    """bf16-tile equivalents of T2's circular buffers per core (an fp32 tile counts 2)."""
    blk = Mb * nt
    bf16 = 2 * Mb * kt + kt * nt + 2 * blk + Mb * kt + blk + 2 * blk + 2 * blk + 2 * kt + 1
    fp32 = 1 + 3 + 2 * kt
    return bf16 + 2 * fp32


def _build_t2(x, ln_w, ln_b, wp, g, resid, grid, ckc, block, eps, cm=False):
    """T2's program. `cm` (T2CM): x is the einsum's channel-major [1, K, S, S] and the work unit is one
    (it, jt) cube of 32 row tiles (`t2cm_split`), read through reader_ln_cm / writer_ln_cm and transposed
    in compute; everything downstream of compute's x CB is the same program."""
    import struct
    Mb, sbw = block[0], block[4]
    K = int(wp.shape[-2])
    kt, nt = _tiles(K), _tiles(wp.shape[-1])
    gx, gy = grid
    ncores = gx * gy
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if cm:
        St = int(x.padded_shape[-1]) // TILE
        for c, (u0, n) in enumerate(t2cm_split(St * St, ncores)):
            cx, cy = c % gx, c // gx
            rd[cx][cy] = [u0, n]
            wr[cx][cy] = [u0, n]
            cp[cx][cy] = [n * (TILE // Mb)]
    else:
        mt = 1
        for d in [int(d) for d in x.padded_shape][:-1]:
            mt *= d
        mt //= TILE
        nb = mt // Mb
        b0 = 0
        for c in range(ncores):
            n = nb // ncores + (c < nb % ncores)
            cx, cy = c % gx, c // gx
            rd[cx][cy] = [b0 * Mb, n]
            wr[cx][cy] = [b0 * Mb, n]
            cp[cx][cy] = [n]
            b0 += n
        assert b0 == nb, (b0, nb)
    # 1/K as a packed bf16 pair (exact: K = 128) and eps as fp32 bits, as lnbw generates them.
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / K))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    d = KERNEL_DIR.parent / "trimul_tail_res"
    if cm:
        reader = ttnn.KernelDescriptor(
            kernel_source=str(d / "reader_ln_cm.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[kt, nt, Mb, scaler, eps_bits, St] + acc(x) + acc(wp) + acc(g)
            + acc(resid) + acc(ln_w) + acc(ln_b),
            runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(d / "writer_ln_cm.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[nt, Mb, St] + acc(resid),
            runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    else:
        reader = ttnn.KernelDescriptor(
            kernel_source=str(d / "reader_ln.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[kt, nt, Mb, scaler, eps_bits] + acc(x) + acc(wp) + acc(g) + acc(resid)
            + acc(ln_w) + acc(ln_b),
            runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[nt, Mb, 1] + acc(resid) + acc(resid),
            runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute_ln.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, Mb, sbw, int(SIGPOLY), RNE, int(cm)],
        runtime_args=cp,
        # fp32 DEST always: the LayerNorm statistics are packed to fp32 CBs and the 4-tile
        # epilogue fits half-sync fp32 DEST. The fidelity stays the caller's, so the GEMM is
        # computed as the incumbent projection computes it; the LN multiplies share it.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=True,
            dst_full_sync_en=full))
    blk = Mb * nt
    f32 = ttnn.float32
    cbs = [_cb(0, core_grid, 2 * Mb * kt), _cb(1, core_grid, kt * nt),
           _cb(2, core_grid, 2 * blk), _cb(3, core_grid, Mb * kt), _cb(4, core_grid, blk),
           _cb(5, core_grid, 2 * blk), _cb(7, core_grid, 2 * blk),
           _cb(8, core_grid, kt), _cb(9, core_grid, kt), _cb(10, core_grid, 1),
           _cb_fmt(11, core_grid, 1, f32), _cb_fmt(12, core_grid, 1, f32),
           _cb_fmt(13, core_grid, 1, f32), _cb_fmt(14, core_grid, 1, f32),
           _cb_fmt(16, core_grid, kt, f32), _cb_fmt(17, core_grid, kt, f32)]
    if cm:
        cbs += [_cb(T2CM_CB_GATHER, core_grid, 2 * Mb * kt), _cb(T2CM_CB_SCRATCH, core_grid, TILE * kt)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def fused_tail_ln(x, ln_w, ln_b, wp, g, resid, ckc, grid, eps=1e-5):
    """`resid += (layer_norm(x; ln_w, ln_b, eps) @ wp) * sigmoid(g)` in one kernel, `resid`
    written in place and returned (the caller's `_add_input` then sees `u is x`). None when T2 is
    off or does not cover the call; nothing has been touched then. `x` is not deallocated."""
    from . import ops
    if ops.taping() or dest_guard.exposed_args(ckc):
        T2_STATS[1] += 1
        return None
    why = t2_eligible(x, ln_w, ln_b, wp, g, resid)
    if why is not None:
        T2_STATS[1] += 1
        if why != "off":
            REJECTS[("t2:" + why, "x".join(str(int(d)) for d in x.padded_shape))] = \
                REJECTS.get(("t2:" + why, "x".join(str(int(d)) for d in x.padded_shape)), 0) + 1
        return None
    from . import tenstorrent as TT
    kt, nt = _tiles(wp.shape[-2]), _tiles(wp.shape[-1])
    block = TT._MM_BLOCK[(kt, nt)]
    if _t2_tiles(kt, nt, block[0]) * MG.tile_bytes(ttnn.bfloat16) > 0.85 * TT._l1_bank_bytes():
        T2_STATS[1] += 1
        return None
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = ("t2", spec(x), spec(wp), spec(ln_w), spec(ln_b), tuple(grid),
           tuple(str(c) for c in ckc), float(eps), SIGPOLY, RNE)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build_t2(x, ln_w, ln_b, wp, g, resid, grid, ckc, block, eps)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), wp.buffer_address(), g.buffer_address(),
                                  resid.buffer_address(), ln_w.buffer_address(),
                                  ln_b.buffer_address()]
    writer.common_runtime_args = [resid.buffer_address(), resid.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, wp, g, ln_w, ln_b, resid], pd)
    T2_STATS[0] += 1
    return resid


# T2CM (the `trimul_t2cm` lever): T2 reading the einsum's channel-major output [1, K, S, S] directly, so
# the trimul's `reblock_permute_back` into [1, S, S, K] (0.58 ms a call on Blackhole p150a at 512, 384
# calls a fold) never runs. A work unit is one (it, jt) cube: the 32 row tiles m = (it*32 + il)*St + jt,
# il in [0, 32), of the [S*S, K] matrix T2 reads. The reader takes the cube's K input tiles
# (channel c: page c*St*St + it*St + jt) into an L1 scratch window and gathers, per row il and channel
# tile ct, the tile T[cl, jl] = x[ct*32 + cl, it*32 + il, jt*32 + jl] exactly as
# reader_reblock_permute_back.cpp does; compute applies the same transpose_wh (fp32 DEST, exact for
# bf16) that `reblock_permute_back`'s compute applies, so the x tile it hands the LayerNorm is the
# byte-for-byte tile the permute would have written to DRAM, and from there it is T2's program.
# Bit-exact against T2 on the permuted tensor by construction.
T2CM = False            # bench switch, as T2
T2CM_STATS = [0, 0]     # served, declined (lever on)
T2CM_CB_GATHER = 6      # gathered (untransposed) x tiles, reader -> compute
T2CM_CB_SCRATCH = 18    # the cube's K input tiles, the reader's private staging window


def set_t2cm(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global T2CM
    prev, T2CM = T2CM, bool(on)
    return prev


def t2cm_split(units, ncores):
    """(first unit, units) per core: contiguous, sizes differ by at most one."""
    out, u0 = [], 0
    for c in range(ncores):
        n = units // ncores + (c < units % ncores)
        out.append((u0, n))
        u0 += n
    return out


def t2cm_tiles(kt, nt, Mb):
    """bf16-tile equivalents of T2CM's circular buffers per core: T2's plus the gather CB and the
    scratch window of one cube's K input tiles."""
    return _t2_tiles(kt, nt, Mb) + 2 * Mb * kt + TILE * kt


def t2cm_eligible(xcm, ln_w, ln_b, wp, g, resid, ckc=None):
    """None when T2CM serves `resid += T2(xcm permuted to [1, S, S, K])`, else the reason it does not.
    Pure host checks; nothing is touched."""
    from .tenstorrent import lever
    from . import ops
    if not ((T2CM or lever("trimul_t2cm")) and (T2 or lever("trimul_t2"))):
        return "off"
    if ops.taping():
        return "tape"
    if ckc is not None and dest_guard.exposed_args(ckc):
        return "dest_guard"
    for t in (xcm, wp, g, resid):
        if t is None or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return "memcfg"
    xs, xps = [int(d) for d in xcm.shape], [int(d) for d in xcm.padded_shape]
    if len(xs) != 4 or xs[0] != 1 or xs[2] != xs[3] or xs != xps or xs[2] % TILE:
        return "cm_shape"
    why = _t2_shape_why([1, xs[2], xs[3], xs[1]], [1, xps[2], xps[3], xps[1]],
                        ln_w, ln_b, wp, g, resid)
    if why is not None:
        return why
    from . import tenstorrent as TT
    kt, nt = _tiles(wp.shape[-2]), _tiles(wp.shape[-1])
    Mb = TT._MM_BLOCK[(kt, nt)][0]
    if TILE % Mb:
        return f"cm_mb={Mb}"
    if t2cm_tiles(kt, nt, Mb) * MG.tile_bytes(ttnn.bfloat16) > 0.85 * TT._l1_bank_bytes():
        return "l1"
    return None


def fused_tail_ln_cm(xcm, ln_w, ln_b, wp, g, resid, ckc, grid, eps=1e-5):
    """`fused_tail_ln` on the channel-major [1, K, S, S] x: `resid += (layer_norm(x) @ wp) *
    sigmoid(g)` with x[0, i, j, c] = xcm[0, c, i, j], `resid` written in place and returned. None when
    T2CM is off or does not cover the call; nothing has been touched then. `xcm` is not deallocated."""
    why = t2cm_eligible(xcm, ln_w, ln_b, wp, g, resid, ckc)
    if why is not None:
        if why != "off":
            T2CM_STATS[1] += 1
            k = ("t2cm:" + why, "x".join(str(int(d)) for d in xcm.padded_shape))
            REJECTS[k] = REJECTS.get(k, 0) + 1
        return None
    from . import tenstorrent as TT
    kt, nt = _tiles(wp.shape[-2]), _tiles(wp.shape[-1])
    block = TT._MM_BLOCK[(kt, nt)]
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = ("t2cm", spec(xcm), spec(wp), spec(ln_w), spec(ln_b), tuple(grid),
           tuple(str(c) for c in ckc), float(eps), SIGPOLY, RNE)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build_t2(xcm, ln_w, ln_b, wp, g, resid, grid, ckc, block, eps, cm=True)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [xcm.buffer_address(), wp.buffer_address(), g.buffer_address(),
                                  resid.buffer_address(), ln_w.buffer_address(),
                                  ln_b.buffer_address()]
    writer.common_runtime_args = [resid.buffer_address(), resid.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([xcm, wp, g, ln_w, ln_b, resid], pd)
    T2CM_STATS[0] += 1
    return resid


# T1: the input side of the c_z = 128 trimul in one resident program (`kernels/trimul_in_t1`):
# layer_norm(z) per row tile, then gin_moved's transposed gated projection (a = p_a * sigmoid(g_a)
# [* m], b alike, written straight into the einsum's (0,3,1,2) layout), and the tail's gate
# g_out = layer_norm(z) @ Wg_out in z's own layout, for T2 or the incumbent tail. z is read once.
# Today that is layer_norm (510 us at N = 512), the fused in-projection with g_out appended, two
# gated moves (645 us each) and the moved mask multiply: 14 P of DRAM traffic, 16 P with the mask,
# P = one bf16 [S, S, 128] pair tensor (67.1 MB at 512). T1's is 4 P plus every core's copy of the
# weights. The model path takes it under the `trimul_t1` lever; T1 here only forces it on for the
# benches (perf/spd_trimul/in_t1.py). Below MIN_N tokens the incumbent ops are faster (N <= 128 on
# Blackhole), so both kernels decline there.
T1 = False
MIN_N = 160
T1_STATS = [0, 0]                 # served, declined
T1_WHY = {}                       # decline reason -> calls (tenstorrent._t1_in)
T1_KEYS = {(4, 8, 4)}             # (kt, ct2, ntg): c_z 128, hidden 128, g_out 128 (openfold3, boltz2)
T1_RB = 8                         # z row tiles per reader barrier
T1_SBW = 4                        # DST tiles per matmul block (half-sync fp32 DST holds 4)
T1_ABL = 0                        # bench-only stage-skip bits (kernels/trimul_in_t1 ABL); 0 = the real T1
BF16_TILE, FP32_TILE = 2048, 4096


def set_t1(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global T1
    prev, T1 = T1, bool(on)
    return prev


def t1_cb_plan(kt, ct2, ntg, mask, rb=None, nb=None):
    """T1's circular buffers per core, ([(index, bf16 tiles)], [(index, fp32 tiles)]). Independent
    of N: N only sets how many units each core runs."""
    nb = GIN_MOVE_NB if nb is None else nb
    rb = T1_RB if rb is None else rb
    bf = [(0, 2 * ct2 * kt),     # [Wp^T | Wg^T], resident
          (1, 32 * kt),          # layer_norm(z) of one unit, slot dt * 32 + il
          (2, 64),               # a / b tiles to the writer, two channel tiles
          (4, 32), (5, 32),      # p and sigmoid(g) of one channel tile
          (7, 2 * rb * kt),      # raw z rows, double buffered
          (8, kt), (9, kt),      # gamma, beta
          (10, 1),               # 1/K reduce scaler
          (24, 2 * nb)]          # the writer's gather staging
    if ntg:
        bf += [(18, kt * ntg),   # Wg_out [K, N], resident
               (19, 2 * ntg)]    # g_out of one row tile, double buffered
    if mask:
        bf += [(3, 1), (6, 32)]  # the mask tile and its 32 row-broadcast cuts
    f32 = [(11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt)]   # eps, mean, var, rstd, xc, tmp
    return bf, f32


def t1_cb_bytes(kt, ct2, ntg, mask, rb=None, nb=None):
    bf, f32 = t1_cb_plan(kt, ct2, ntg, mask, rb, nb)
    return sum(t for _, t in bf) * BF16_TILE + sum(t for _, t in f32) * FP32_TILE


def t1_host_weights(g_in_t, p_in_t, g_out_t):
    """T1's three weights from the checkpoint's [K, 2 * hidden] g_in / p_in (already transposed, as
    `TriangleMultiplication` keeps them) and [K, c_z] g_out: wpT = [p_a | p_b]^T and
    wgT = [g_a | g_b]^T, each [2 * hidden, K] with the hidden channels in their natural order (one
    chunk, so no gp_roles reordering applies), and Wg_out as is."""
    h = int(g_in_t.shape[-1]) // 2
    assert int(p_in_t.shape[-1]) == 2 * h
    assert int(g_in_t.shape[0]) == int(p_in_t.shape[0]) == int(g_out_t.shape[0])
    return p_in_t.t().contiguous(), g_in_t.t().contiguous(), g_out_t.contiguous()


def t1_split(nu, ncores):
    """Units (b, xt, yt) per core, as gin_moved splits them: [(first unit, count)] per core."""
    out, u0 = [], 0
    for c in range(ncores):
        n = nu // ncores + (c < nu % ncores)
        out.append((u0, n))
        u0 += n
    assert u0 == nu
    return out


# The kernels' page arithmetic, mirrored here so the CPU test can check it against torch.
def t1_x_page(b, xt, yt, il, dt, S_t, kt):
    """reader.cpp: z tile of row tile (b, x = xt * 32 + il, yt), channel tile dt."""
    return ((b * S_t * 32 + xt * 32 + il) * S_t + yt) * kt + dt


def t1_ab_page(b, q, c, xt, yt, ct, S_t):
    """writer.cpp: [B, C, S, S] tile of channel (q % ct) * 32 + c at (xt, yt); q < ct is a."""
    return ((b * ct + q % ct) * 32 + c) * S_t * S_t + xt * S_t + yt


def t1_g_page(b, xt, yt, il, n, S_t, ntg):
    """writer.cpp: g_out tile of row tile (b, xt * 32 + il, yt), channel tile n, in z's layout."""
    return ((b * S_t * 32 + xt * 32 + il) * S_t + yt) * ntg + n


def t1_mask_ok(mask, x):
    """Whether T1 folds `mask` into `a`: a bf16 tiled DRAM [B, S, S] pair mask for x [B?, S, S, K],
    the shape gin_moved's reader cuts (m[b, x, y], applied before the move)."""
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    return (mask is not None and len(shp) in (3, 4)
            and tuple(int(d) for d in mask.padded_shape) == (B, shp[-2], shp[-2])
            and tuple(int(d) for d in mask.shape) == (B, shp[-2], shp[-2])
            and mask.dtype == ttnn.bfloat16 and mask.layout == ttnn.TILE_LAYOUT
            and mask.memory_config() == ttnn.DRAM_MEMORY_CONFIG)


def t1_eligible(x, ln_w, ln_b, wpT, wgT, wgo, mask=None):
    """None when T1 serves this call, else the reason it does not."""
    from .tenstorrent import lever
    if not (T1 or lever("trimul_t1")):
        return "off"
    for t in (x, wpT, wgT, wgo):
        if t is None or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return "memcfg"
    shp = [int(d) for d in x.shape]
    if len(shp) not in (3, 4) or shp[-2] != shp[-3] or shp[-2] % TILE or shp[-1] % TILE:
        return "act_shape"
    if [int(d) for d in x.padded_shape] != shp:
        return "padded_shape"
    K = shp[-1]
    if (len(wpT.shape) != 2 or [int(d) for d in wpT.shape] != [int(d) for d in wgT.shape]
            or int(wpT.shape[-1]) != K or int(wpT.shape[-2]) % (2 * TILE)):
        return "weight_shape"
    if len(wgo.shape) != 2 or int(wgo.shape[-2]) != K or int(wgo.shape[-1]) % TILE:
        return "gout_shape"
    key = (K // TILE, int(wpT.shape[-2]) // TILE, int(wgo.shape[-1]) // TILE)
    if key not in T1_KEYS:
        return "key=" + "x".join(map(str, key))
    if not (_vec_ok(ln_w, K) and _vec_ok(ln_b, K)):
        return "ln_params"
    if mask is not None and not t1_mask_ok(mask, x):
        return "mask"
    from .tenstorrent import _l1_bank_bytes
    if t1_cb_bytes(*key, mask is not None) > 0.85 * _l1_bank_bytes():
        return "l1"
    return None


def _build_t1(x, ln_w, ln_b, wpT, wgT, wgo, outs, grid, ckc, mask, eps):
    import struct
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    St, kt, ct2 = shp[-2] // TILE, shp[-1] // TILE, int(wpT.shape[-2]) // TILE
    ntg = int(wgo.shape[-1]) // TILE
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (u0, n) in enumerate(t1_split(B * St * St, gx * gy)):
        cx, cy = c % gx, c // gx
        rd[cx][cy] = [u0, n]
        wr[cx][cy] = [u0, n]
        cp[cx][cy] = [n]
    # 1/K as a packed bf16 pair (exact for K a power of two) and eps as fp32 bits, as T2 has them.
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / shp[-1]))[0] >> 16
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    d = KERNEL_DIR.parent / "trimul_in_t1"
    m = mask if mask is not None else x
    abl = [("ABL", str(T1_ABL))] if T1_ABL else []
    reader = ttnn.KernelDescriptor(
        kernel_source=str(d / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, St, int(mask is not None), ntg, T1_RB, (inv << 16) | inv,
                           eps_bits]
        + acc(x) + acc(wpT) + acc(wgT) + acc(wgo) + acc(ln_w) + acc(ln_b) + acc(m),
        defines=abl, runtime_args=rd, common_runtime_args=[0] * 7, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[ct2, St, GIN_MOVE_NB, ntg] + acc(outs[0]) + acc(outs[1]) + acc(outs[2]),
        defines=abl, runtime_args=wr, common_runtime_args=[0] * 3, config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, T1_SBW, int(mask is not None), int(SIGPOLY), RNE, ntg],
        defines=abl, runtime_args=cp,
        # fp32 DEST always, as T2: the LayerNorm statistics live in fp32 CBs. The fidelity stays
        # the caller's, so the projections run as the incumbent in-projection runs them.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=True,
            dst_full_sync_en=full))
    bf, f32 = t1_cb_plan(kt, ct2, ntg, mask is not None)
    cbs = ([_cb(i, core_grid, t) for i, t in bf]
           + [_cb_fmt(i, core_grid, t, ttnn.float32) for i, t in f32])
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def fused_in_ln(x, ln_w, ln_b, wpT, wgT, wgo, ckc, grid, mask=None, eps=1e-5):
    """`[a, b, g_out]` of the trimul input side from the raw pair `x`, one kernel:
    xn = layer_norm(x; ln_w, ln_b, eps), [a | b] = (xn @ wpT^T) * sigmoid(xn @ wgT^T), `a` times
    the pair mask when `mask` is given, both [B, C, S, S] (gin_moved's layout), and
    g_out = xn @ wgo as [*x.shape[:-1], N]. None when T1 is off or does not cover the call;
    nothing is allocated then. `x` is not touched."""
    from . import ops
    fid, approx, _, full = ckc
    if ops.taping() or dest_guard.exposed_args((fid, approx, True, full)):
        T1_STATS[1] += 1
        return None
    why = t1_eligible(x, ln_w, ln_b, wpT, wgT, wgo, mask)
    if why is not None:
        T1_STATS[1] += 1
        if why != "off":
            k = ("t1:" + why, "x".join(str(int(d)) for d in x.padded_shape))
            REJECTS[k] = REJECTS.get(k, 0) + 1
        return None
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    S, C = shp[-2], int(wpT.shape[-2]) // 2
    device = x.device()
    alloc = lambda s: ttnn.allocate_tensor_on_device(  # noqa: E731
        ttnn.Shape(s), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    outs = [alloc([B, C, S, S]), alloc([B, C, S, S]), alloc(shp[:-1] + [int(wgo.shape[-1])])]
    key = ("t1", str(x.padded_shape), str(wpT.padded_shape), str(wgo.padded_shape), tuple(grid),
           tuple(str(c) for c in ckc), mask is not None, float(eps), SIGPOLY, RNE, GIN_MOVE_NB,
           T1_RB, T1_SBW, T1_ABL)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build_t1(x, ln_w, ln_b, wpT, wgT, wgo, outs, grid, ckc, mask, eps)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), wpT.buffer_address(), wgT.buffer_address(),
                                  wgo.buffer_address(), ln_w.buffer_address(),
                                  ln_b.buffer_address(),
                                  (mask if mask is not None else x).buffer_address()]
    writer.common_runtime_args = [o.buffer_address() for o in outs]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, wpT, wgT, wgo, ln_w, ln_b] + ([mask] if mask is not None else []) + outs,
                    pd)
    T1_STATS[0] += 1
    MASK_STATS[0] += mask is not None
    return outs
