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


def _sigpoly():
    """The kernels' SIGPOLY arg: 0 metal's sigmoid, 1 the degree-8 polynomial, 2 the degree-6 one (lever
    trimul_sig6, max |error| 7.4e-4 vs 3.0e-4, ~1/3 fewer SFPU instructions per gate)."""
    return 0 if not SIGPOLY else 2 if _lever("trimul_sig6") else 1
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
        compile_time_args=[nt, Mb, len(outs), 0] + acc(outs[0]) + acc(outs[-1]),
        runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, fp32, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, Mb, sbw, int(shared), resid, RES_ABL, int(mask is not None),
                           _sigpoly(), RNE],
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


def f1_folds_resid(xa, xb, wa, wb, resid, ckc_args, mem):
    """Whether `fused_tail(xa, xb, wa, wb, ckc_args, ..., out_memory_config=mem, resid=resid)` serves the
    call with the residual folded in (EPI 2, product in DRAM). T2's GATE form computes exactly those bits
    (`perf/spd_trifuse/t2probe.py`: `torch.equal` at 512 and 1024 under ESMFold2's sets), and different ones
    where F1 lands its product in L1 and adds it separately, so `trimul_t2g` takes only these calls."""
    from . import ops
    return (not ops.taping() and not dest_guard.exposed_args(ckc_args) and eligible(xa, xb, wa, wb) is None
            and _epi() == 2 and _resid_ok(resid, xa, wa, mem or ttnn.DRAM_MEMORY_CONFIG))


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
        key = ("res", RES_ABL, _sigpoly(), RNE, None if mask is None else str(mask.padded_shape)) + key
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


def _gin_fuse(masked):
    """Whether gin_moved runs its fused gate (lever trimul_gin_fuse): p and g side by side in DST, multiplied
    there, no p/g buffers in L1. Unmasked calls on the polynomial sigmoid only."""
    return not masked and _sigpoly() != 0 and _lever("trimul_gin_fuse")


def _gin_out_tiles(kt, ct2, masked):
    """The output buffer: two channel tiles' worth (compute runs one ahead of the writer's gather), or one when
    the fused gate gets the call into a single launch only that way (ESMFold2's c_z 256 on Wormhole)."""
    if not _gin_fuse(masked):
        return 64
    from .tenstorrent import _l1_bank_bytes
    full = (2 * ct2 * kt + 32 * kt + 64 + 2 * GIN_MOVE_NB) * MG.tile_bytes(ttnn.bfloat16)
    return 64 if full <= 0.85 * _l1_bank_bytes() else 32


def _gin_move_tiles(kt, ct2, mask):
    if _gin_fuse(mask):
        return 2 * ct2 * kt + 32 * kt + _gin_out_tiles(kt, ct2, False) + 2 * GIN_MOVE_NB
    return 2 * ct2 * kt + 32 * kt + 64 + 2 * 32 + 2 * GIN_MOVE_NB + (33 if mask else 0)


def _gin_move_bytes(kt, ct2, mask, wdt=ttnn.bfloat16):
    """L1 bytes of a gin_moved launch whose resident weights are `wdt` (bf16, or bfp8 under trimul_gin_w8)."""
    bf = MG.tile_bytes(ttnn.bfloat16)
    return _gin_move_tiles(kt, ct2, mask) * bf - 2 * ct2 * kt * (bf - MG.tile_bytes(wdt))


def gin_move_split(x, c, mask=None, wdt=ttnn.bfloat16):
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
        if ct % n == 0 and (_gin_move_bytes(kt, 2 * ct // n, mask is not None, wdt)
                            <= 0.85 * _l1_bank_bytes()):
            return n
    return None


def gin_moved_ok(x, wpT, mask=None):
    """Whether `gin_moved` serves this call: bf16 tiled interleaved DRAM pair tensor [B?, S, S, K]
    with S a whole number of tiles, transposed weights [2C, K] with C a whole number of tiles, the
    mask (if any) as `mask_ok` wants it, and the program's circular buffers inside L1."""
    shp = [int(d) for d in x.shape]
    if not (GIN_MOVE and len(shp) in (3, 4) and shp[-2] == shp[-3] and shp[-2] % TILE == 0
            and x.dtype == ttnn.bfloat16 and wpT.dtype in (ttnn.bfloat16, ttnn.bfloat8_b)
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
    return (_gin_move_bytes(shp[-1] // TILE, int(wpT.shape[-2]) // TILE, mask is not None, wpT.dtype)
            <= 0.85 * _l1_bank_bytes())


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
    # The fused gate holds a subblock's p and g in DST at once: half of DST each.
    fuse = _gin_fuse(mask is not None)
    sbw = min(4, (4 if fp32 else 8) * (2 if full else 1) // 2) if fuse else 4
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, ct2, sbw, int(mask is not None), GIN_MOVE_ABL, _sigpoly(), RNE, int(fuse)],
        runtime_args=cp,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32,
            dst_full_sync_en=full))
    if fuse:
        cbs = [_cb_fmt(0, core_grid, 2 * ct2 * kt, wpT.dtype), _cb(1, core_grid, 32 * kt),
               _cb(2, core_grid, _gin_out_tiles(kt, ct2, False)), _cb(24, core_grid, 2 * GIN_MOVE_NB)]
        return {"kernels": [reader, writer, compute], "cbs": cbs}
    cbs = [_cb_fmt(0, core_grid, 2 * ct2 * kt, wpT.dtype), _cb(1, core_grid, 32 * kt), _cb(2, core_grid, 64),
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
        key = ("gin_move", str(x.padded_shape), str(wp.padded_shape), str(wp.dtype), tuple(grid),
               tuple(str(c) for c in ckc), mask is not None, GIN_MOVE_ABL, GIN_MOVE_NB, _sigpoly(), RNE,
               s_, n, _gin_fuse(mask is not None))
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
        compile_time_args=[kt, ct2, 4, int(mask is not None), GIN_MOVE_ABL, _sigpoly(), RNE, qs, St],
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
           tuple(grid), tuple(str(c) for c in ckc), mask is not None, GIN_MOVE_NB, _sigpoly(), RNE, S, qs, GIN_MOVE_ABL,
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
# T2's stage ablation: compute_ln.cpp's bits 1 LN, 2 p GEMM, 4 gate GEMM + sigmoid, 8 epilogue (the reader and
# writer still move every byte); 16 the reader skips its per-block DRAM reads, 32 the writer its writes (non-moved
# T2 only). Diagnostic only, 0 in every model path.
T2_ABL = 0
T2_KEYS = {(4, 4), (8, 8)}   # (kt, nt): c_z 128 and 256 (ESMFold2)
T2_C64 = (2, 2)              # the template pair stacks' c_t 64, under lever trimul_t1_c64 (T1's key (2, 4, 2))
# c_z 384 (OpenDDE) under the `trimul_t2w` lever, at T2's own block: `_MM_BLOCK`'s (8, 12, 1, 2, 1) does not
# fit L1 and declines. (2, 12, 1, 2, 1) keeps the single K block, so the contraction order is the projection's.
# WH chip, perf/spd_trimul/tail_t2.py --c 384 vs float64 (10-13): N 512 13.63 -> 6.19 ms, rel_rms 0.0028
# (four ops it replaces 0.0025), max_abs 2.94 (3.30); every N 32-512 finite, no hang.
T2_WIDE_KEYS = {(12, 12)}
T2_BLOCK = {(12, 12): (2, 12, 1, 2, 1)}


def _t2_block(kt, nt):
    from . import tenstorrent as TT
    return T2_BLOCK.get((kt, nt)) or TT._MM_BLOCK.get((kt, nt))


# T2m: T2 reading x where the einsum left it, channel-major, instead of after reblock_permute_back
# (reader_moved.cpp). The gather that bounds the back move (64 NOC packets per tile) runs on the
# reader while compute does T2's LayerNorm, two GEMMs and sigmoid, and the moved x is never written.
# Under the `trimul_t2m` lever (in no mode set: it lost to plain T2 once trimul_bres fed the tail); T2M forces it
# for the benches.
T2M = False
T2M_STATS = [0, 0]      # served, declined
T2M_WHY = {}


def set_t2m(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global T2M
    prev, T2M = T2M, bool(on)
    return prev


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


def t2_eligible(x, ln_w, ln_b, wp, g, resid, wg=None):
    """None when T2 serves this call, else the reason it does not. With `wg`, `g` is the trimul's
    input norm and T2 computes sigmoid(g @ wg) itself (the GATE form)."""
    from .tenstorrent import lever
    if not (T2 or lever("trimul_t2" if wg is None else "trimul_t2g")):
        return "off"
    for t in (x, wp, g, resid) + ((wg,) if wg is not None else ()):
        if t is None or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        # A plain-form g_out may be bfp8 (trimul_b8in writes it so): CB 5 takes g's format and
        # compute unpacks it into DEST like the bf16 tile, so b8in and T2 stack.
        if t.dtype != ttnn.bfloat16 and not (t is g and wg is None and t.dtype == ttnn.bfloat8_b):
            return "dtype"
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return "memcfg"
    return _t2_shape_why([int(d) for d in x.shape], [int(d) for d in x.padded_shape],
                         ln_w, ln_b, wp, g, resid, wg)


def _t2_shape_why(xs, xps, ln_w, ln_b, wp, g, resid, wg=None):
    """T2's checks on everything but x's own dtype / memory: `xs` / `xps` are x's logical and padded
    shape in the [.., S, S, K] layout T2 reads (T2CM passes its channel-major x's shape permuted)."""
    if len(wp.shape) != 2:
        return "weight_shape"
    K, N = int(wp.shape[-2]), int(wp.shape[-1])
    kt, nt = _tiles(K), _tiles(N)
    from .tenstorrent import lever
    if ((kt, nt) not in T2_KEYS and not ((kt, nt) in T2_WIDE_KEYS and lever("trimul_t2w"))
            and not ((kt, nt) == T2_C64 and lever("trimul_t1_c64")) or K % TILE or N % TILE):
        return f"key={kt}x{nt}"
    if xs[-1] != K or xps[-1] != K:
        return "act_k"
    if not (_vec_ok(ln_w, K) and _vec_ok(ln_b, K)):
        return "ln_params"
    if wg is not None and [int(d) for d in wg.shape] != [K, N]:
        return "gate_weight_shape"
    want = xs[:-1] + [N]
    g_want, g_pad = (want, tuple(xps[:-1]) + (N,)) if wg is None else (xs, tuple(xps))
    if [int(d) for d in g.shape] != g_want or [int(d) for d in resid.shape] != want:
        return "out_shape"
    if (tuple(g.padded_shape) != g_pad
            or tuple(resid.padded_shape) != tuple(xps[:-1]) + (N,)):
        return "padded_shape"
    from . import tenstorrent as TT    # late: `tenstorrent` imports this module at its own import
    block = _t2_block(kt, nt)
    if block is None:
        return "block"
    mt = 1
    for d in xps[:-1]:
        mt *= d
    mt //= TILE
    if mt % block[0] or (block[0] * nt) % 4 or nt % block[4]:
        return f"m_tiles={mt}"
    return None


def _t2_tiles(kt, nt, Mb, gate=False, moved=None):
    """bf16-tile equivalents of T2's circular buffers per core (an fp32 tile counts 2). The GATE
    form adds Wg and the double-buffered xb rows, and computes g single-buffered. `moved` = (Cp, P):
    T2m assembles one x row at a time (Kt tiles instead of 2 Mb Kt), plus the gather's two-tile
    input, the Cp prev tiles and the stage, (Kt - Cp) * 32 channel parts of 2048 / P bytes."""
    blk = Mb * nt
    bf16 = 2 * Mb * kt + kt * nt + 2 * blk + Mb * kt + blk + 2 * blk + 2 * blk + 2 * kt + 1
    if gate:
        bf16 += kt * nt + 2 * Mb * kt - blk
    if moved is not None:
        cp, P = moved
        bf16 += kt - 2 * Mb * kt + 2 + 2 * cp + -(-(kt - cp) * 32 // P)
    fp32 = 1 + 3 + 2 * kt
    return bf16 + 2 * fp32


def _build_t2(x, ln_w, ln_b, wp, g, resid, grid, ckc, block, eps, wg=None, moved=None):
    """T2's program. `wg`: the GATE form. `moved` (T2m, and T2CM with c_off 0): see `fused_tail_ln`."""
    import struct
    Mb, sbw = block[0], block[4]
    K = int(wp.shape[-2])
    kt, nt = _tiles(K), _tiles(wp.shape[-1])
    gate = wg is not None
    gx, gy = grid
    ncores = gx * gy
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if moved is None:
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
    else:
        # T2m: work is (it, jt, part) units of R = 32 / P rows i, R / Mb blocks each.
        chunk, c_off, P = moved
        St, Cp = int(chunk.shape[2]) // TILE, c_off // TILE
        units = St * St * P
        u0 = 0
        for c in range(ncores):
            n = units // ncores + (c < units % ncores)
            cx, cy = c % gx, c // gx
            rd[cx][cy] = [u0, n]
            wr[cx][cy] = [u0, n]
            cp[cx][cy] = [n * (TILE // P) // Mb]
            u0 += n
        assert u0 == units, (u0, units)
    # 1/K as a packed bf16 pair (exact: K is a power of two) and eps as fp32 bits, as lnbw generates them.
    inv = struct.unpack("<I", struct.pack("<f", 1.0 / K))[0] >> 16
    scaler = (inv << 16) | inv
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    d = KERNEL_DIR.parent / "trimul_tail_res"
    if moved is None:
        reader = ttnn.KernelDescriptor(
            kernel_source=str(d / "reader_ln.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[kt, nt, Mb, scaler, eps_bits, int(gate), T2_ABL] + acc(x) + acc(wp) + acc(g)
            + acc(resid) + acc(ln_w) + acc(ln_b) + acc(wg if gate else wp),
            runtime_args=rd, common_runtime_args=[0] * 7, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[nt, Mb, 1, T2_ABL] + acc(resid) + acc(resid),
            runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    else:
        reader = ttnn.KernelDescriptor(
            kernel_source=str(d / "reader_moved.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[kt, nt, St, Cp, P, scaler, eps_bits, int(gate)] + acc(chunk) + acc(x)
            + acc(wp) + acc(wg if gate else wp) + acc(ln_w) + acc(ln_b),
            runtime_args=rd, common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=str(d / "writer_moved.cpp"), source_type=src, core_ranges=core_grid,
            compile_time_args=[kt, nt, St, Mb, P, int(gate)] + acc(g) + acc(resid),
            runtime_args=wr, common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute_ln.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, nt, Mb, sbw, _sigpoly(), RNE, int(gate), int(moved is not None),
                           0 if moved is None else Cp, T2_ABL],
        runtime_args=cp,
        # fp32 DEST always: the LayerNorm statistics are packed to fp32 CBs and the 4-tile
        # epilogue fits half-sync fp32 DEST. The fidelity stays the caller's, so the GEMM is
        # computed as the incumbent projection computes it; the LN multiplies share it.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=True,
            dst_full_sync_en=full))
    blk = Mb * nt
    f32 = ttnn.float32
    cbs = [_cb(0, core_grid, kt if moved is not None else 2 * Mb * kt),
           _cb(1, core_grid, (1 + gate) * kt * nt),
           _cb(2, core_grid, 2 * blk), _cb(3, core_grid, Mb * kt), _cb(4, core_grid, blk),
           _cb_fmt(5, core_grid, (2 - gate) * blk, g.dtype if moved is None else ttnn.bfloat16),
           _cb(7, core_grid, 2 * blk),
           _cb(8, core_grid, kt), _cb(9, core_grid, kt), _cb(10, core_grid, 1),
           _cb_fmt(11, core_grid, 1, f32), _cb_fmt(12, core_grid, 1, f32),
           _cb_fmt(13, core_grid, 1, f32), _cb_fmt(14, core_grid, 1, f32),
           _cb_fmt(16, core_grid, kt, f32), _cb_fmt(17, core_grid, kt, f32)]
    if gate:
        cbs.append(_cb(6, core_grid, 2 * Mb * kt))
    if moved is not None:
        cbs.append(_cb(15, core_grid, 2))
        if Cp:
            cbs.append(_cb(18, core_grid, 2 * Cp))
        stage = (kt - Cp) * TILE * 2 * (TILE // P) * TILE     # each channel's part: two faces' rows
        fmt = ttnn.CBFormatDescriptor(buffer_index=24, data_format=ttnn.bfloat16, page_size=stage)
        cbs.append(ttnn.CBDescriptor(total_size=stage, core_ranges=core_grid, format_descriptors=[fmt]))
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def t2m_parts(kt, nt, Mb, gate, cp, min_parts=1):
    """Row parts per (it, jt) block for T2m: the fewest (>= min_parts) whose stage fits beside T2's
    buffers, or None. Fewer parts read each channel tile in fewer, longer pieces."""
    from . import tenstorrent as TT
    for P in (p for p in (1, 2, 4) if p >= min_parts):
        if ((TILE // P) % Mb == 0 and _t2_tiles(kt, nt, Mb, gate, (cp, P)) * MG.tile_bytes(ttnn.bfloat16)
                <= 0.85 * TT._l1_bank_bytes()):
            return P
    return None


def t2m_eligible(x_chunk, x, c_off):
    """None when T2m can read `x` as the einsum left it: `x_chunk` a bf16 channel-major
    [1, C, N, N] DRAM chunk that ends the channels of `x` ([1, N, N, K], channels [0, c_off) already
    moved there), else the reason."""
    if not (T2M or _lever("trimul_t2m")):
        return "off"
    cs, xs = [int(d) for d in x_chunk.shape], [int(d) for d in x.shape]
    if len(cs) != 4 or cs[0] != 1 or cs[2] != cs[3] or cs[2] % TILE or cs[1] % TILE:
        return "chunk_shape"
    if xs != [1, cs[2], cs[2], c_off + cs[1]] or c_off % TILE:
        return "x_shape"
    if (x_chunk.dtype != ttnn.bfloat16 or x_chunk.layout != ttnn.TILE_LAYOUT
            or x_chunk.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
        return "chunk_dtype"
    return None


def _lever(name):
    from .tenstorrent import lever
    return lever(name)


def fused_tail_ln(x, ln_w, ln_b, wp, g, resid, ckc, grid, eps=1e-5, wg=None, moved=None):
    """`resid += (layer_norm(x; ln_w, ln_b, eps) @ wp) * sigmoid(g)` in one kernel (with `wg`:
    `* sigmoid(g @ wg)`, g the trimul's input norm), `resid`
    written in place and returned (the caller's `_add_input` then sees `u is x`). None when T2 is
    off or does not cover the call; nothing has been touched then. `x` is not deallocated.
    `moved` = (x_chunk, c_off) is T2m: x's channels from c_off on are still in the einsum's
    channel-major chunk and are read from there (`x` holds only [0, c_off), and its shape)."""
    from . import ops
    if ops.taping() or dest_guard.exposed_args(ckc):
        T2_STATS[1] += 1
        return None
    why = t2_eligible(x, ln_w, ln_b, wp, g, resid, wg)
    if why is not None:
        T2_STATS[1] += 1
        if why != "off":
            REJECTS[("t2:" + why, "x".join(str(int(d)) for d in x.padded_shape))] = \
                REJECTS.get(("t2:" + why, "x".join(str(int(d)) for d in x.padded_shape)), 0) + 1
        return None
    from . import tenstorrent as TT
    kt, nt = _tiles(wp.shape[-2]), _tiles(wp.shape[-1])
    block = _t2_block(kt, nt)
    P = None
    if moved is not None:
        why = t2m_eligible(moved[0], x, moved[1]) or ("g_dtype" if g.dtype != ttnn.bfloat16 else None)
        P = None if why else t2m_parts(kt, nt, block[0], wg is not None, moved[1] // TILE)
        if P is None:
            T2M_STATS[1] += 1
            if why != "off":
                why = why or "l1"
                T2M_WHY[why] = T2M_WHY.get(why, 0) + 1
            T2_STATS[1] += 1
            return None
    elif _t2_tiles(kt, nt, block[0], wg is not None) * MG.tile_bytes(ttnn.bfloat16) > 0.85 * TT._l1_bank_bytes():
        T2_STATS[1] += 1
        k = ("t2:l1", "x".join(str(int(d)) for d in x.padded_shape))
        REJECTS[k] = REJECTS.get(k, 0) + 1
        return None
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = ("t2", spec(x), spec(wp), str(g.dtype), spec(ln_w), spec(ln_b), wg is not None, tuple(grid),
           tuple(str(c) for c in ckc), float(eps), _sigpoly(), RNE, T2_ABL,
           None if moved is None else (spec(moved[0]), moved[1], P))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build_t2(x, ln_w, ln_b, wp, g, resid, grid, ckc, block, eps, wg,
                                        None if moved is None else (moved[0], moved[1], P))
    reader, writer, _ = entry["kernels"]
    wg_ = wg if wg is not None else wp
    if moved is None:
        reader.common_runtime_args = [x.buffer_address(), wp.buffer_address(), g.buffer_address(),
                                      resid.buffer_address(), ln_w.buffer_address(),
                                      ln_b.buffer_address(), wg_.buffer_address()]
        writer.common_runtime_args = [resid.buffer_address(), resid.buffer_address()]
    else:
        reader.common_runtime_args = [moved[0].buffer_address(), x.buffer_address(), wp.buffer_address(),
                                      wg_.buffer_address(), ln_w.buffer_address(), ln_b.buffer_address()]
        writer.common_runtime_args = [g.buffer_address(), resid.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
    ttnn.generic_op([x, wp, g, ln_w, ln_b, resid] + ([wg] if wg is not None else [])
                    + ([moved[0]] if moved is not None else []), pd)
    T2_STATS[0] += 1
    if moved is not None:
        T2M_STATS[0] += 1
    return resid


# T2CM (the `trimul_t2cm` lever): T2 reading the einsum's channel-major output [1, K, S, S] directly, so
# the trimul's `reblock_permute_back` into [1, S, S, K] (0.58 ms a call on Blackhole p150a at 512, 384
# calls a fold) never runs. It is T2m with the whole hidden in the one chunk (c_off 0): reader_moved.cpp
# gathers each x row out of the channel-major tiles and compute transposes it, as reblock_permute_back
# does, so the x tile the LayerNorm reads is the tile the permute would have written. Bit-exact against
# T2 on the permuted tensor; T2CM had its own reader / writer until T2m's whole-tile staging read
# matched it (Blackhole c128, N 512 4.51 vs 4.64 ms, N 1024 level, bit-exact).
T2CM = False            # bench switch, as T2
T2CM_STATS = [0, 0]     # served, declined (lever on)


def set_t2cm(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global T2CM
    prev, T2CM = T2CM, bool(on)
    return prev


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
    return t2cm_fit(wp)


# The most row parts T2CM's stage may take. Above it each channel tile is read in 256-byte pieces and the
# gather loses to reblock_permute_back + T2: OpenDDE (c_z 384, P = 4) on a Wormhole Galaxy chip at 1000 MHz,
# cdk2x2 512 x 1, trimul_t2cm on vs off (spd-m-opendde cmab, 2026-10-18): normal 102.81 -> 99.88 s, fast
# 96.93 -> 94.91 s, CIFs byte-identical; in the fold census the tail read 10.53 ms a call against 4.68 + 2.65.
# At c_z 128 (P = 1) it wins (OpenFold3 WH 43.76 -> 43.45 s). On Wormhole P = 2 loses too: Protenix-v2
# (c_z 256, P = 2) on a Galaxy chip at 1000 MHz, cdk2x2 512 x 1 normal, 5 paired warm folds, trimul_t2cm on vs
# off (spd-m-protenix cmn, 2026-10-19): 60.39 -> 59.38 s median, 4 of 5 pairs faster, CIFs byte-identical.
# Blackhole keeps P = 2 until measured there.
T2CM_MAX_P = {"wormhole_b0": 1, "blackhole": 2}


def t2cm_fit(wp):
    """None when T2CM's stage fits L1 in at most T2CM_MAX_P parts for the [K, K] weight `wp`, else the reason."""
    from .tenstorrent import arch_name
    P = _t2cm_parts(wp)
    if P is None:
        return "l1"
    return "parts" if P > T2CM_MAX_P.get(arch_name(), 1) else None


def _t2cm_parts(wp, min_parts=1):
    from . import tenstorrent as TT
    kt, nt = _tiles(wp.shape[-2]), _tiles(wp.shape[-1])
    return t2m_parts(kt, nt, _t2_block(kt, nt)[0], False, 0, min_parts)


def balance_split(units, ncores):
    """2 when cutting every work unit in two lifts the round balance (units per core-round) by at
    least 10 %, else 1. At 512 tokens 256 units on 110 cores run at 78 %, 512 halves at 93 %."""
    eff = lambda u: u / (-(-u // ncores) * ncores)  # noqa: E731
    return 2 if eff(2 * units) >= 1.1 * eff(units) else 1


def t1_rows(plan_il, units, ncores):
    """Row tiles per T1 work item: the plan's (T1_PLANS), except that whole 32-row units run as two
    16-row halves where balance_split says so (`units` of them on `ncores`). T1_ROWS forces it."""
    if T1_ROWS:
        return min(plan_il, T1_ROWS)
    return plan_il // balance_split(units, ncores) if plan_il == 32 else plan_il


# A/B knob: the fewest row parts T2CM splits each (it, jt) block into; 0 = balance_split's choice
# (production). Each part's rows compute exactly as in the whole block, so the choice is bit-exact
# (bht1f: identical CIFs at 512 on both models, OpenBind-0 --fast -0.14 s on p150a).
T2CM_MIN_P = 0


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
    P = _t2cm_parts(wp, T2CM_MIN_P or balance_split(_tiles(xcm.shape[2]) ** 2, grid[0] * grid[1])) or _t2cm_parts(wp)
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = ("t2cm", spec(xcm), spec(wp), spec(ln_w), spec(ln_b), tuple(grid),
           tuple(str(c) for c in ckc), float(eps), _sigpoly(), RNE, P)
    entry = _CACHE.get(key)
    if entry is None:
        # x is only read for the channels before c_off, so with c_off 0 xcm stands in for it.
        entry = _CACHE[key] = _build_t2(xcm, ln_w, ln_b, wp, g, resid, grid, ckc, _t2_block(kt, nt),
                                        eps, moved=(xcm, 0, P))
    reader, writer, _ = entry["kernels"]
    a = xcm.buffer_address()
    reader.common_runtime_args = [a, a, wp.buffer_address(), wp.buffer_address(),
                                  ln_w.buffer_address(), ln_b.buffer_address()]
    writer.common_runtime_args = [g.buffer_address(), resid.buffer_address()]
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
# (kt, ct2, ntg) -> (channel groups, row tiles per unit). c_z 128 (openfold3, boltz2) is one launch of
# whole 32-row units. c_z 256 (protenix-v2, opendde, esmfold2) holds twice the weights and twice the
# normalised rows per unit, ~1.9 MB of CBs a core; as two channel groups of 16-row units it is ~1.0 MB,
# at the price of reading z twice (5 pair passes instead of the incumbent's 14).
# (2, 4, 2) is OpenFold3's template pair stack (c_t 64): it fell back on all 80 template trimuls a fold.
T1_PLANS = {(2, 4, 2): (1, 32), (4, 8, 4): (1, 32), (8, 16, 8): (2, 16), (12, 24, 12): (3, 8)}
T1_KEYS = set(T1_PLANS)
# The plans when the resident weights are bfp8 (lever trimul_t1w8): half the weight bytes per core, so c_z 384
# fits its [a | b] and g_out weights in two channel groups instead of three, one launch and one xn pass fewer.
# c_z 256 fits ONE launch of 16-row units (1072 KB of CBs, what its two bf16 groups take now): z is read once.
T1_PLANS_W8 = {(8, 16, 8): (1, 16), (12, 24, 12): (2, 8)}


def t1_plan(key, w8=False):
    """(channel groups, row tiles per unit) for a (kt, ct2, ntg) key; `w8`: the weights are bfp8."""
    return (w8 and T1_PLANS_W8.get(key)) or T1_PLANS.get(key, (1, 32))
T1_RB = 8                         # z row tiles per reader barrier
T1_SBW = 4                        # DST tiles per matmul block (half-sync fp32 DST holds 4)
T1_ABL = 0                        # bench-only stage-skip bits (kernels/trimul_in_t1 ABL); 0 = the real T1
T1_B8 = 0                         # bench knob: a and b as bfp8 (lever trimul_t1b8 in a model)
T1_C256 = 0                       # bench knob: the c_z 256 / 384 plans without their levers
# Plans with channel groups normalise z once: the first launch stores xn (bf16, z's layout) beside its outputs
# and the later launches read it instead of re-running the layer_norm. The matmuls read the same bf16 tiles,
# so a, b and g_out are bit-identical. At OpenDDE's c_z 384 (3 launches) the layer_norm was 4.67 of T1's 16.18 ms
# at 512 on a Wormhole Galaxy chip, 1000 MHz (t1abl, ABL 1). A/B knob; True is production.
T1_LN1 = True
T1_WIDE = {(2, 4, 2): "trimul_t1_c64", (8, 16, 8): "trimul_t1_c256", (12, 24, 12): "trimul_t1_c384"}   # plan key -> the lever that turns T1 on for it
# Row tiles per work item: 32 = one unit (b, xt, yt) per item; 16 = each unit as two items, rows 0-15 and
# 16-31, which are faces 0-1 and 2-3 of every [B, C, S, S] tile the writer gathers, so the two halves
# write disjoint bytes. Every row is computed exactly as in the whole unit (same LN, same K order).
# The point is balance: at 512 tokens 256 units on 110 cores is 3 rounds for 36 cores and 2 for the
# rest (78 %), and T1 is compute-bound (t1abl2-512: 2.08 of 2.18 ms with every DRAM stage skipped);
# 512 halves is 5 rounds against 4.65 (93 %). Bit-exact (same digest at 64/128/512, identical CIFs);
# the op runs 2.07 -> 1.78 ms at 512 on a p150a. A/B knob; 0 = balance_split's choice (production).
T1_ROWS = 0
BF16_TILE, FP32_TILE = 2048, 4096


def set_t1(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global T1
    prev, T1 = T1, bool(on)
    return prev


def t1_cb_plan(kt, ct2, ntg, mask, rb=None, nb=None, il=None, w8=False):
    """T1's circular buffers per core for one launch of the (kt, ct2, ntg) plan,
    ([(index, bf16 tiles)], [(index, fp32 tiles)]). Independent of N: N only sets how many units
    each core runs."""
    nb = GIN_MOVE_NB if nb is None else nb
    rb = T1_RB if rb is None else rb
    groups, plan_il = t1_plan((kt, ct2, ntg), w8)
    il = plan_il if il is None else il
    rb = min(rb, il // 4)           # 8 at 32-row units (as before), 4 at 16, 2 at 8
    qn, ng = ct2 // groups, ntg // groups
    bf = [(0, 2 * qn * kt),      # [Wp^T | Wg^T] of the group, resident
          (1, il * kt),          # layer_norm(z) of one unit, slot dt * il + row
          (2, 2 * il),           # a / b tiles to the writer, two channel tiles
          (4, il), (5, il),      # p and sigmoid(g) of one channel tile
          (7, 2 * rb * kt),      # raw z rows, double buffered
          (8, kt), (9, kt),      # gamma, beta
          (10, 1),               # 1/K reduce scaler
          (24, 2 * nb)]          # the writer's gather staging
    if ng:
        bf += [(18, kt * ng),    # Wg_out [K, N] of the group, resident
               (19, 2 * ng)]     # g_out of one row tile, double buffered
    if mask:
        bf += [(3, 1), (6, il),  # the mask tile and its row-broadcast cuts
               (15, 4)]          # g * m of one block
    if T1_LN1 and groups > 1:
        bf += [(20, 2 * kt)]     # xn of one row tile to the writer (T1_LN1), double buffered
    f32 = [(11, 1), (12, 1), (13, 1), (14, 1), (16, kt), (17, kt)]   # eps, mean, var, rstd, xc, tmp
    return bf, f32


T1_W_CBS = (0, 18)                # the resident weight CBs, bfp8 under trimul_t1w8


def t1_cb_bytes(kt, ct2, ntg, mask, rb=None, nb=None, w8=False):
    bf, f32 = t1_cb_plan(kt, ct2, ntg, mask, rb, nb, w8=w8)
    wb = MG.tile_bytes(ttnn.bfloat8_b) if w8 else BF16_TILE
    return (sum(t * (wb if i in T1_W_CBS else BF16_TILE) for i, t in bf)
            + sum(t for _, t in f32) * FP32_TILE)


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


def t1_any() -> bool:
    """T1 may serve some plan: the knob, trimul_t1 (every plan's master), or a wide plan's own lever."""
    from .tenstorrent import lever
    return T1 or lever("trimul_t1") or any(lever(v) for v in T1_WIDE.values())


def t1_eligible(x, ln_w, ln_b, wpT, wgT, wgo, mask=None):
    """None when T1 serves this call, else the reason it does not."""
    from .tenstorrent import lever
    on = T1 or lever("trimul_t1")
    if not t1_any():
        return "off"
    for t in (x, wpT, wgT, wgo):
        if t is None or t.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b) or t.layout != ttnn.TILE_LAYOUT:
            return "dtype"
        if t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return "memcfg"
    w8 = t1_w8(wpT, wgT, wgo)
    if x.dtype != ttnn.bfloat16 or w8 is None:
        return "dtype"
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
    if key in T1_WIDE:
        if not (lever(T1_WIDE[key]) or (on and T1_C256)):
            return T1_WIDE[key][len("trimul_t1_"):] + "_off"
    elif not on:
        return "off"
    if not (_vec_ok(ln_w, K) and _vec_ok(ln_b, K)):
        return "ln_params"
    if mask is not None and not t1_mask_ok(mask, x):
        return "mask"
    from .tenstorrent import _l1_bank_bytes
    if t1_cb_bytes(*key, mask is not None, w8=w8) > 0.85 * _l1_bank_bytes():
        return "l1"
    return None


def t1_w8(wpT, wgT, wgo):
    """True when all three weights are bfp8, False when all are bf16, None when mixed."""
    d = {t.dtype for t in (wpT, wgT, wgo) if t is not None}
    return True if d == {ttnn.bfloat8_b} else False if d == {ttnn.bfloat16} else None


def _build_t1(x, ln_w, ln_b, wpT, wgT, wgo, outs, grid, ckc, mask, eps, il, b8=False, group=0, ln1=0, xn=None):
    """The program of channel group `group` of the T1_PLANS entry for this call, `il` row tiles per item.
    `ln1` (T1_LN1): 1 the launch also writes xn into `xn`, 2 it reads xn as `x` and runs no layer_norm."""
    import struct
    shp = [int(d) for d in x.shape]
    B = shp[0] if len(shp) == 4 else 1
    St, kt, ct2 = shp[-2] // TILE, shp[-1] // TILE, int(wpT.shape[-2]) // TILE
    ntg = int(wgo.shape[-1]) // TILE
    w8 = bool(t1_w8(wpT, wgT, wgo))
    groups = t1_plan((kt, ct2, ntg), w8)[0]
    qn, ng = ct2 // groups, ntg // groups
    rb = min(T1_RB, il // 4)
    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (u0, n) in enumerate(t1_split(B * St * St * (32 // il), gx * gy)):
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
    xout, pre = [("T1XOUT", "1")] if ln1 == 1 else [], [("T1PRE", "1")] if ln1 == 2 else []
    reader = ttnn.KernelDescriptor(
        kernel_source=str(d / "reader.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, qn, St, int(mask is not None), ng, rb, (inv << 16) | inv,
                           eps_bits, group * qn, group * ng, ntg, il]
        + acc(x) + acc(wpT) + acc(wgT) + acc(wgo) + acc(ln_w) + acc(ln_b) + acc(m),
        defines=abl + pre, runtime_args=rd, common_runtime_args=[0] * 7, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(d / "writer.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[ct2, St, GIN_MOVE_NB, ng, group * qn, qn, group * ng, ntg, il] + acc(outs[0]) + acc(outs[1]) + acc(outs[2])
        + ([kt] + acc(xn) if ln1 == 1 else []),
        defines=abl + xout + ([("T1B8", "1")] if b8 else []), runtime_args=wr,
        common_runtime_args=[0] * (4 if ln1 == 1 else 3), config=ttnn.WriterConfigDescriptor())
    fid, approx, _, full = ckc
    compute = ttnn.KernelDescriptor(
        kernel_source=str(d / "compute.cpp"), source_type=src, core_ranges=core_grid,
        compile_time_args=[kt, qn, T1_SBW, int(mask is not None), _sigpoly(), RNE, ng, group * qn, ct2, il],
        defines=abl + xout + pre, runtime_args=cp,
        # fp32 DEST always, as T2: the LayerNorm statistics live in fp32 CBs. The fidelity stays
        # the caller's, so the projections run as the incumbent in-projection runs them.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=True,
            dst_full_sync_en=full))
    bf, f32 = t1_cb_plan(kt, ct2, ntg, mask is not None, il=il, w8=w8)
    cbs = ([_cb_fmt(i, core_grid, t, ttnn.bfloat8_b) if (b8 and i == 2) or (w8 and i in T1_W_CBS)
            else _cb(i, core_grid, t) for i, t in bf]
           + [_cb_fmt(i, core_grid, t, ttnn.float32) for i, t in f32])
    return {"kernels": [reader, writer, compute], "cbs": cbs}


def fused_in_ln(x, ln_w, ln_b, wpT, wgT, wgo, ckc, grid, mask=None, eps=1e-5, b8=False):
    """`[a, b, g_out]` of the trimul input side from the raw pair `x`, one kernel:
    xn = layer_norm(x; ln_w, ln_b, eps), [a | b] = (xn @ wpT^T) * sigmoid(xn @ wgT^T), `a` times
    the pair mask when `mask` is given, both [B, C, S, S] (gin_moved's layout), and
    g_out = xn @ wgo as [*x.shape[:-1], N]. None when T1 is off or does not cover the call;
    nothing is allocated then. `x` is not touched. `b8` (or T1_B8) writes a and b as bfp8."""
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
    w8 = bool(t1_w8(wpT, wgT, wgo))
    groups, il = t1_plan((int(x.shape[-1]) // TILE, 2 * C // TILE, int(wgo.shape[-1]) // TILE), w8)
    il = t1_rows(il, B * (S // TILE) ** 2, grid[0] * grid[1])
    b8 = bool(b8 or T1_B8) and il == 32   # writer.cpp: whole tiles
    alloc = lambda s, dt=ttnn.bfloat16: ttnn.allocate_tensor_on_device(  # noqa: E731
        ttnn.Shape(s), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    ab = ttnn.bfloat8_b if b8 else ttnn.bfloat16
    outs = [alloc([B, C, S, S], ab), alloc([B, C, S, S], ab), alloc(shp[:-1] + [int(wgo.shape[-1])])]
    key = ("t1", str(x.padded_shape), str(wpT.padded_shape), str(wgo.padded_shape), tuple(grid),
           tuple(str(c) for c in ckc), mask is not None, float(eps), _sigpoly(), RNE, GIN_MOVE_NB,
           T1_RB, T1_SBW, T1_ABL, b8, il, w8)
    xn = alloc(shp) if T1_LN1 and groups > 1 else None
    for g in range(groups):
        ln1 = 0 if xn is None else 1 if g == 0 else 2
        entry = _CACHE.get(key + (g, ln1))
        if entry is None:
            entry = _CACHE[key + (g, ln1)] = _build_t1(x, ln_w, ln_b, wpT, wgT, wgo, outs, grid, ckc, mask,
                                                       eps, il, b8, g, ln1, xn)
        reader, writer, _ = entry["kernels"]
        src = xn if ln1 == 2 else x
        reader.common_runtime_args = [src.buffer_address(), wpT.buffer_address(), wgT.buffer_address(),
                                      wgo.buffer_address(), ln_w.buffer_address(),
                                      ln_b.buffer_address(),
                                      (mask if mask is not None else x).buffer_address()]
        writer.common_runtime_args = [o.buffer_address() for o in outs] + ([xn.buffer_address()] if ln1 == 1 else [])
        pd = ttnn.ProgramDescriptor(kernels=entry["kernels"], semaphores=[], cbs=entry["cbs"])
        ttnn.generic_op([src, wpT, wgT, wgo, ln_w, ln_b] + ([mask] if mask is not None else []) + outs
                        + ([xn] if ln1 == 1 else []), pd)
    if xn is not None:
        ttnn.deallocate(xn)
    T1_STATS[0] += 1
    MASK_STATS[0] += mask is not None
    return outs
