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
            "tail": STATS, "res": RES_STATS, "mask_fold": MASK_STATS, "resid": RESID_STATS,
            "rejects": {f"{k[0]}:{k[1]}": v for k, v in REJECTS.items()},
            "routes": {f"{k[0]}/{k[1]}": v for k, v in _T.TRIMUL_MM_TRANSPOSE_STATS.items()}}),
            file=sys.stderr, flush=True)
    atexit.register(_print_stats)


def set_mask_fold(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global MASK_FOLD
    prev, MASK_FOLD = MASK_FOLD, bool(on)
    return prev
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
        compile_time_args=[kt, nt, Mb, sbw, int(shared), resid, RES_ABL, int(mask is not None)],
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
        key = ("res", RES_ABL, None if mask is None else str(mask.padded_shape)) + key
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
GIN_MOVE = env_flag("TT_BIO_TRIMUL_GIN_MOVE", False)
GIN_MOVE_STATS = [0, 0]   # served, declined


def _gin_move_tiles(kt, ct2, mask):
    return 2 * ct2 * kt + 32 * kt + 64 + 2 * 32 + 2 + (33 if mask else 0)


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


def _build_gin_move(x, wpT, wgT, outs, grid, ckc, mask):
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
        compile_time_args=[kt, ct2, St, int(mask is not None)] + acc(x) + acc(wpT) + acc(wgT)
        + acc(mask if mask is not None else x),
        runtime_args=rd, common_runtime_args=[0] * 4, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[ct2, St] + acc(outs[0]) + acc(outs[1]),
        runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, fp32, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, 4, int(mask is not None)], runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32,
            dst_full_sync_en=full))
    cbs = [_cb(0, core_grid, 2 * ct2 * kt), _cb(1, core_grid, 32 * kt), _cb(2, core_grid, 64),
           _cb(4, core_grid, 32), _cb(5, core_grid, 32), _cb(24, core_grid, 2)]
    if mask is not None:
        cbs += [_cb(3, core_grid, 1), _cb(6, core_grid, 32)]
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def gin_moved(x, wpT, wgT, ckc, grid, mask=None):
    """`a, b` of the gated in-projection, already channel-moved: [B, C, S, S] each, for
    `[a | b] = (x @ Wp) * sigmoid(x @ Wg)` with `wpT`, `wgT` the transposed [2C, K] weights, and
    `a` times the pair mask when `mask` is given. None where `gin_moved_ok` declines."""
    from . import ops
    if ops.taping() or not gin_moved_ok(x, wpT, mask):
        GIN_MOVE_STATS[1] += 1
        return None
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    S, C = shp[-2], int(wpT.shape[-2]) // 2
    device = x.device()
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([B, C, S, S]), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(2)]
    key = ("gin_move", str(x.padded_shape), str(wpT.padded_shape), tuple(grid),
           tuple(str(c) for c in ckc), mask is not None)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build_gin_move(x, wpT, wgT, outs, grid, ckc, mask)
    reader, writer, _ = entry["kernels"]
    reader.common_runtime_args = [x.buffer_address(), wpT.buffer_address(), wgT.buffer_address(),
                                  (mask if mask is not None else x).buffer_address()]
    writer.common_runtime_args = [outs[0].buffer_address(), outs[1].buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, wpT, wgT, *outs] + ([mask] if mask is not None else []), pd)
    GIN_MOVE_STATS[0] += 1
    MASK_STATS[0] += mask is not None
    return outs
