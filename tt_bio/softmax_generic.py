"""`ttnn.softmax` re-driven through `ttnn.generic_op`, so the output CB can pack bf16.

A transcription of `softmax_program_factory_attention_optimized.cpp` at the `v0.68.0` tag (the
wheel's version) into a Python `ttnn.ProgramDescriptor`, for exactly the call RFD3's attention
makes and nothing else:

    plain softmax over the last dim of a rank-4 interleaved-DRAM tensor, no mask, no fused scale,
    no causal mask, `numeric_stable`, last dim already tile-aligned.

Under those conditions the factory's mask half is dead and what is left is a CB table, three of
the wheel's own unmodified kernels and a work split. The one thing this module changes is
`cb_out0`'s data format.

Why that is worth a transcription. `ttnn.softmax` writes its output at the input dtype -- there is
no `dtype` and no `output_tensor` argument on this wheel -- so every attention site in RFD3 spends
a `ttnn.typecast` immediately afterwards turning fp32 scores into bf16. At the decoder's
`[1, 4, 6051, 6080]` that typecast reads 588.6 MB and writes 294.3, and softmax itself has already
written 588.6 MB it did not need to. Packing bf16 straight out of DST deletes the typecast outright
and halves softmax's own write.

Why it stays bit-exact. Everything upstream of the final pack is the wheel's kernel with the wheel's
CB formats and the wheel's reduction order. The shipped chain packs DST to an fp32 CB (exact) and
`ttnn.typecast` then rounds fp32 -> bf16; this packs the same DST value to bf16 once. One rounding
either way, and `S2` in `scripts/rfd3_port/p74_softmax_generic.py` is the gate that proves the
packer's rounding is the same rounding.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from .envflags import env_flag
from .mm_generic import ckc_args, tile_bytes, ttnn_cpp_root

TILE = 32

# `device->l1_size_per_core()` is not exposed to Python. Blackhole is 1464 KB. The value only
# feeds the factory's `use_large_kernel` trip at `0.9 * l1_size_per_core`, and both production
# shapes sit far from it -- the DiT's 22 Wt needs 307 KB and the atom path's 190 Wt needs
# 2.26 MB -- so no plausible value moves either verdict. A wrong choice would fail S1 anyway.
L1_PER_CORE = 1499136

_CACHE: dict = {}


def _kdir():
    return (ttnn_cpp_root()
            / "cpp/ttnn/operations/normalization/softmax/device/kernels/attention")


def _div_up(a, b):
    return (a + b - 1) // b


def find_max_divisor(val, start_max_div):
    """`tt_metal::find_max_divisor` (`work_split.cpp:52`). It skips 7 and 5."""
    for d in range(start_max_div, 0, -1):
        if d in (7, 5):
            continue
        if val % d == 0:
            return d
    return 1


def plan(x, out, grid, fp32_dest_acc, numeric_stable=True):
    """Everything the descriptor needs, derived the way the factory derives it."""
    padded = [int(d) for d in x.padded_shape]
    logical = [int(d) for d in x.shape]
    assert len(padded) == 4, padded
    vol = 1
    for d in padded:
        vol *= d
    NC, W = padded[0], padded[-1]
    H = vol // (padded[0] * padded[-1])
    Wt, Ht = W // TILE, H // TILE
    # The padded-last-dim path pulls in cb_mask_padded and a different first compute loop. Both
    # RFD3 sites pass a key axis that is already `align_tile`d, so it is asserted, not handled.
    assert W == logical[-1], ("padded last dim %d != logical %d; the mask_padded_data path is "
                              "not transcribed" % (W, logical[-1]))

    block_size = find_max_divisor(Wt, 4 if fp32_dest_acc else 8)
    in0_t = _div_up(Wt, block_size) * block_size if numeric_stable else block_size * 2
    out0_t = block_size * 2
    im4_t = _div_up(Wt, block_size) * block_size
    im0_t = block_size * _div_up(Wt, block_size)
    assert im0_t == Wt, (im0_t, Wt)
    cb_length = in0_t

    in0_ts = tile_bytes(x.dtype)
    im_fmt = ttnn.float32 if fp32_dest_acc else ttnn.bfloat16
    im_ts = tile_bytes(im_fmt)

    # The factory sizes this estimate off the OUTPUT tile size. Using the input's keeps the fp32
    # and bf16 arms on the same kernel, which is the point: they must differ in exactly one field.
    cb_sum = in0_t * in0_ts + im0_t * im_ts + out0_t * in0_ts + 5 * im_ts
    if numeric_stable:
        cb_sum += im4_t * im_ts
    use_large = (L1_PER_CORE * 0.9) < cb_sum
    if use_large:
        cb_length = (80 // block_size) * block_size
        in0_t = im4_t = im0_t = cb_length

    gx, gy = grid
    max_cores = gx * gy
    units = NC * Ht
    target = max_cores if units >= max_cores else units
    per1 = units // target
    rem = units % target
    per2 = 0
    if rem:
        per2 = per1
        per1 = per1 + 1

    return dict(NC=NC, W=W, H=H, Wt=Wt, Ht=Ht, block_size=block_size, in0_t=in0_t, out0_t=out0_t,
                im0_t=im0_t, im4_t=im4_t, cb_length=cb_length, use_large=use_large,
                im_fmt=im_fmt, im_ts=im_ts, in0_ts=in0_ts, out0_ts=tile_bytes(out.dtype),
                numeric_stable=numeric_stable, fp32_dest_acc=fp32_dest_acc,
                gx=gx, gy=gy, max_cores=max_cores, units=units, target=target,
                per1=per1, per2=per2, rem=rem)


# --- L5b: the PV matmul folded into the normalise pass ----------------------------------------

# The three CB indices `softmax_large_tensor.cpp` leaves free. Kept here and in the kernel only.
PV_CB_VV, PV_CB_NORM, PV_CB_ACC = 1, 13, 14


def _cb_bytes(cbs):
    return sum(c.total_size for c in cbs)


def pv_l1_bytes(p, vv_ts=2048):
    """CB bytes per core for the fused program, from `plan`'s own table.

    Value residency is the only term that grows with the key axis: `cb_in0`, `cb_exps` and the
    reductions are all capped at `cb_length`, so the fused program's footprint is
    `fixed + Wt * 2048`. That is what bounds L5b's addressable size range from above.
    """
    fixed = (p["in0_t"] * p["in0_ts"]          # c_0   scores
             + p["out0_t"] * p["out0_ts"]      # c_11  output
             + p["im_ts"]                      # c_7   recip sum
             + 2048                            # c_2   bcast scaler
             + p["im0_t"] * p["im_ts"]         # c_6   exps
             + 2048                            # c_5   mask_padded
             + 3 * p["im_ts"]                  # c_12 c_15 c_16
             + (p["im_ts"] if p["numeric_stable"] else 0)   # c_8 max
             + 2 * p["block_size"] * vv_ts     # c_13  normalised row, double buffered
             + 4096)                           # c_14  the fp32 partial, one tile deep
    # c_10 (cb_x) is NOT in that list. It is only touched by `pad_input` and the fused-scale-mask
    # path, both of which are dead here -- `plan` asserts the last dim is already tile-aligned, so
    # `mask_padded_data` is 0 on every core -- and dropping its `cb_length` tiles is most of what
    # pays for value residency.
    return fixed + p["Wt"] * vv_ts


def build(device, x, out, grid, ckc, numeric_stable=True, vv=None, extra_defines=()):
    """The ProgramDescriptor. `ckc` is `(math_fidelity, math_approx, fp32_dest_acc, dst_full_sync)`.

    `softmax_init_compute_kernel_config` defaults it to
    `(HiFi4, approx=True, fp32_dest_acc=input_is_fp32, l1_acc=False)` on Blackhole, and
    `dst_full_sync` defaults False.

    With `vv` given (L5b), `out` is `attn @ vv` rather than the attention itself, and the program
    never writes the attention to DRAM at all. `pv_classify` decides whether a shape may take this
    path; `build` assumes it already said yes.
    """
    fidelity, approx, fp32_dest_acc, dst_full_sync = ckc
    pv = vv is not None
    p = plan(x, out, grid, fp32_dest_acc, numeric_stable)
    gx, gy = p["gx"], p["gy"]
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    def cb(idx, n_tiles, page, fmt):
        f = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)
        return ttnn.CBDescriptor(total_size=n_tiles * page, core_ranges=core_grid,
                                 format_descriptors=[f])

    im, im_ts = p["im_fmt"], p["im_ts"]
    scalar_ts = 2048                      # scalar_cb_data_format is always Float16_b
    cbs = [
        cb(0, p["in0_t"], p["in0_ts"], x.dtype),           # c_in0
        cb(11, p["out0_t"], p["out0_ts"], out.dtype),      # c_out0  <-- the one change
        cb(7, 1, im_ts, im),                               # c_recipsumexps
        cb(2, 1, scalar_ts, ttnn.bfloat16),                # c_bcast_scaler
        cb(6, p["im0_t"], im_ts, im),                      # c_exps
        cb(5, 1, scalar_ts, ttnn.bfloat16),                # c_mask_padded (always created)
    ]
    if p["use_large"]:
        cbs += [cb(12, 1, im_ts, im), cb(15, 1, im_ts, im), cb(16, 1, im_ts, im)]
    if numeric_stable:
        cbs.append(cb(8, 1, im_ts, im))                    # c_max
    if not pv and (numeric_stable or p["use_large"]):
        cbs.append(cb(10, p["im4_t"], im_ts, im))          # c_x
    if pv:
        vv_ts = tile_bytes(vv.dtype)
        # `cb_acc` is the reuse factory's `interm0`, and that factory makes it Float32 whenever
        # `fp32_dest_acc_en` is set, one tile deep so the packer's accumulate always lands on the
        # same L1 address (`matmul_multicore_reuse_optimized_program_factory.cpp`).
        cbs += [cb(PV_CB_VV, p["Wt"], vv_ts, vv.dtype),
                cb(PV_CB_NORM, 2 * p["block_size"], vv_ts, out.dtype),
                cb(PV_CB_ACC, 1, 4096, ttnn.float32)]

    def acc(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_ct = acc(x) + (acc(vv) if pv else [])
    writer_ct = [0, TILE * TILE] + acc(out)                # num_datum_padded, tile_hw
    dm_defines = [("NUMERIC_STABLE", "1")] if numeric_stable else []
    compute_defines = list(dm_defines) + [("EXP_APPROX", "1" if approx else "0"),
                                          ("ENABLE_FP32_DEST_ACC", "1" if fp32_dest_acc else "0")]
    # `extra_defines` is a diagnostic knob, empty on every production path: it lets a probe compile
    # a variant of the same kernel in the same process (p134 scores two `typecast_tile_init`
    # placements that way). It is part of `softmax_into`'s cache key, because a compile-time define
    # that is not in the key hands the second arm the first arm's program -- the same unkeyed-cache
    # bug §16.5 fixed in `_tuned_key`.
    compute_defines += [(str(k), str(v)) for k, v in extra_defines]

    # When the output dtype differs from the input's, the packer's own fp32 -> bf16 rounding is
    # NOT the rounding `ttnn.typecast` does (measured 0.00195 maxabs, one bf16 ULP, p74 S2), so the
    # compute kernel has to do typecast's SFPU conversion itself and pack a DST that is already
    # bf16-valued. `tt_bio/kernels/rfd3_softmax/` holds the wheel's two compute kernels with that
    # one insertion behind `PACK_BF16_TYPECAST`; everything else is byte-identical to the wheel.
    cast_in_sfpu = out.dtype != x.dtype
    if cast_in_sfpu:
        assert x.dtype == ttnn.float32 and out.dtype == ttnn.bfloat16, (x.dtype, out.dtype)
        compute_defines.append(("PACK_BF16_TYPECAST", "1"))
    if pv:
        compute_defines.append(("PV_FUSED", "1"))
        dm_defines = list(dm_defines) + [("PV_FUSED", "1")]

    kd = _kdir()
    if p["use_large"]:
        reader_src = str(kd / "dataflow/reader_unary_interleaved_sm_large_tensor.cpp")
        compute_src = str(kd / "compute/softmax_large_tensor.cpp")
    else:
        reader_src = str(kd / "dataflow/reader_unary_interleaved_sm.cpp")
        compute_src = str(kd / "compute/softmax.cpp")
    local = Path(__file__).resolve().parent / "kernels" / "rfd3_softmax"
    if cast_in_sfpu:
        compute_src = str(local / ("softmax_large_tensor.cpp" if p["use_large"]
                                   else "softmax.cpp"))
    if pv:
        assert p["use_large"] and cast_in_sfpu, (p["use_large"], cast_in_sfpu)
        reader_src = str(local / "reader_sm_large_tensor.cpp")
    writer_src = str(kd / "dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp")

    src_a, out_a = x.buffer_address(), out.buffer_address()
    vv_a = vv.buffer_address() if pv else 0
    Wt, Ht, blk = p["Wt"], p["Ht"], p["block_size"]
    # `plan` folds every dim between the first and the last into H, so `Ht` counts row-tiles across
    # ALL heads and is the wrong stride for anything per-head. The value tiles are per-head, so the
    # fused program needs the per-head row-tile count separately -- `Ht // heads`, taken off the
    # padded row axis directly so it does not depend on how many dims got folded.
    mt = int(x.padded_shape[-2]) // TILE
    assert not pv or (Ht % mt == 0 and mt > 0), (Ht, mt)
    # The fused program emits one output tile per row-tile instead of Wt of them, and the writer
    # drains it a tile at a time. Everything else about the work split is untouched.
    out_w, out_blk = (1, 1) if pv else (Wt, blk)
    pv_pad = [0, 0, 0] if pv else []
    rr, wr, cr = [], [], []
    curr_row = 0
    for i in range(p["max_cores"]):
        core = ttnn.CoreCoord(i % gx, i // gx)
        if i >= p["target"]:
            rr.append((core, [0] * 10 + [0x3F803F80, 0] + pv_pad))
            cr.append((core, [0] * (9 if pv else 7)))
            wr.append((core, [0] * 6))
            continue
        n = p["per1"] if (p["rem"] == 0 or i < p["rem"]) else p["per2"]
        tile_offset = curr_row * Wt
        curr_ht = curr_row % Ht
        mask_id = curr_row // Ht * Wt
        rr.append((core, [src_a, blk, 0x3F800000, n, tile_offset, Wt, Ht, 0, curr_ht, mask_id,
                          0x3F803F80, p["in0_t"]]
                        + ([vv_a, curr_row, mt] if pv else [])))
        cr.append((core, [n, Ht, Wt, blk, curr_ht, 0, p["cb_length"]]
                        + ([curr_row, mt] if pv else [])))
        wr.append((core, [out_a, n * out_w, curr_row * out_w, out_blk, 0, 0]))
        curr_row += n
    assert curr_row == p["units"], (curr_row, p["units"])

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=reader_src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=reader_ct, defines=dm_defines,
            runtime_args=rr, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=writer_src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=writer_ct, defines=[],
            runtime_args=wr, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=compute_src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid, compile_time_args=[], defines=compute_defines,
            runtime_args=cr,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity, math_approx_mode=approx,
                fp32_dest_acc_en=fp32_dest_acc, dst_full_sync_en=dst_full_sync)),
    ]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return {"pd": pd, "kernels": kernels, "cbs": cbs, "plan": p, "rt": (rr, wr, cr),
            "addrs": (src_a, out_a, vv_a), "pv": pv, "l1_bytes": _cb_bytes(cbs)}


def softmax_into(device, x, out, grid=None, ckc=None, numeric_stable=True, vv=None,
                 extra_defines=()):
    """Run the transcribed program, writing into `out` (any dtype). Returns `out`.

    With `vv`, `out` is `softmax(x) @ vv` and the attention is never written to DRAM (L5b).
    """
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = (g.x, g.y)
    if ckc is None:
        ckc = (ttnn.MathFidelity.HiFi4, True, x.dtype == ttnn.float32, False)
    extra_defines = tuple(sorted((str(k), str(v)) for k, v in extra_defines))
    key = (str(x.padded_shape), str(x.shape), str(x.dtype), str(out.dtype), grid,
           tuple(str(c) for c in ckc), numeric_stable,
           None if vv is None else (str(vv.padded_shape), str(vv.dtype)), extra_defines)
    e = _CACHE.get(key)
    if e is None:
        e = _CACHE[key] = build(device, x, out, grid, ckc, numeric_stable, vv, extra_defines)
    addrs = (x.buffer_address(), out.buffer_address(),
             vv.buffer_address() if vv is not None else 0)
    if addrs != e["addrs"]:
        rr, wr, _ = e["rt"]
        for _c, a in rr:
            if a[3]:
                a[0] = addrs[0]
                if e["pv"]:
                    a[12] = addrs[2]      # same index build() emits vv_a at
        for _c, a in wr:
            if a[1]:
                a[0] = addrs[1]
        e["kernels"][0].runtime_args = rr
        e["kernels"][1].runtime_args = wr
        e["pd"] = ttnn.ProgramDescriptor(kernels=e["kernels"], semaphores=[], cbs=e["cbs"])
        e["addrs"] = addrs
    ttnn.generic_op([x, out] if vv is None else [x, vv, out], e["pd"])
    return out


# --- the model-facing entry point ------------------------------------------------------------

SSTATS = [0]                       # calls served, so an A/B arm cannot silently decline

_ENABLED = env_flag("RFD3_SOFTMAX_BF16", True)


def set_enabled(on: bool) -> bool:
    """Flip the lever, returning the previous state. For A/B harnesses."""
    global _ENABLED
    prev, _ENABLED = _ENABLED, bool(on)
    return prev


# `softmax_large_tensor.cpp`'s bf16 pack is bit-exact against `typecast(softmax(x))`
# (p74 S2, maxabs 0 at [1,4,6051,6080]). The small kernel's is not yet, so shapes that do not
# cross the factory's large-kernel trip fall back to the shipped pair until p74's S2 column is
# green at the DiT rungs too. Flip with `RFD3_SOFTMAX_BF16_SMALL=1` once it is.
_SMALL_OK = env_flag("RFD3_SOFTMAX_BF16_SMALL", False)


def set_small_enabled(on: bool) -> bool:
    global _SMALL_OK
    prev, _SMALL_OK = _SMALL_OK, bool(on)
    return prev


def eligible(x, dtype) -> bool:
    """Only the shape family this module transcribes: rank-4, fp32 in, bf16 out, tile-aligned W."""
    if not _ENABLED:
        return False
    if dtype != ttnn.bfloat16 or x.dtype != ttnn.float32:
        return False
    if len(x.shape) != 4:
        return False
    if int(x.shape[-1]) != int(x.padded_shape[-1]):
        return False
    if _SMALL_OK:
        return True
    try:
        g = x.device().compute_with_storage_grid_size()
        return bool(plan(x, x, (g.x, g.y), True, True)["use_large"])
    except Exception:
        return False


def softmax_bf16(x, dtype):
    """`ttnn.typecast(ttnn.softmax(x, dim=-1), dtype)` in one kernel, bit-exactly.

    Falls back to the shipped pair whenever the shape is outside what the transcription covers.
    """
    if not eligible(x, dtype):
        # The off arm has to be the shipped chain exactly, down to freeing the fp32 intermediate
        # before the caller continues -- at [1,4,6051,6080] that tensor is 588.6 MB.
        sm = ttnn.softmax(x, dim=-1)
        out = ttnn.typecast(sm, dtype, memory_config=sm.memory_config())
        ttnn.deallocate(sm)
        return out
    out = ttnn.empty(list(x.shape), dtype, ttnn.TILE_LAYOUT, x.device(), x.memory_config())
    softmax_into(x.device(), x, out)
    SSTATS[0] += 1
    return out


# --- L5b, the model-facing entry point --------------------------------------------------------

PVSTATS = [0]                      # calls served fused, so an A/B arm cannot silently decline
PVDECLINES = {}                    # reason -> count, so a decline is readable rather than silent
PVSERVED = {}                      # key width -> count, the other half of the same census

_PV_ENABLED = env_flag("RFD3_SOFTMAX_PV_FUSED", False)


def set_pv_enabled(on: bool) -> bool:
    """Flip L5b, returning the previous state. For A/B harnesses."""
    global _PV_ENABLED
    prev, _PV_ENABLED = _PV_ENABLED, bool(on)
    return prev


def pv_classify(x, vv, dtype, ckc, grid=None):
    """Whether `softmax(x) @ vv` may run fused, and if not, exactly which condition said no.

    Four independent conditions, none of which is a tolerance:

    * **the kernel has to engage** -- below the factory's `use_large` trip the fold runs the
      shipped `softmax` + `typecast` pair and there is no kernel to extend (`p124`, ~3425 atoms);
    * **the two K-blockings have to agree** -- `apply_recip` walks the row in
      `find_max_divisor(Wt, 4)` tiles and the shipped PV matmul rounds its partial to bf16 every
      `in0_block_w` tiles, so the fused kernel reproduces the shipped arithmetic only where the
      two coincide. They coincide on half the engaged widths (`p124`);
    * **the fidelity has to match the model's** -- the fused kernel runs ONE compute config for
      both halves, so the matmul half only reproduces the shipped `attn_value_matmul` when that
      config carries the model's `HiFi4`, `fp32_dest_acc_en` AND `packer_l1_acc`
      (`tt_bio/rfd3/model.py::_default_compute_kernel_config`). All three are load-bearing:
      `MatmulMultiCoreReuseProgramConfig` runs
      `bmm_large_block_zm_fused_bias_activation.cpp` through the OPTIMIZED reuse factory, which
      turns `PACKER_L1_ACC` on for `packer_l1_acc && num_blocks > 2` and makes the partial Float32
      whenever `fp32_dest_acc_en`. Drop `packer_l1_acc` and the partial is spilled and reloaded
      instead of packer-accumulated, which is a different fp32 grouping;
    * **value residency has to fit L1** -- the fused program keeps `Wt` value tiles resident for a
      whole head, because re-reading them per row would move as much DRAM traffic as the write it
      deletes. That is the one term that grows with the key axis, and it caps L5b from above.

    Returns a dict; `ok` is the verdict and `why` names the first failing condition.
    """
    def no(why, **kw):
        return dict(ok=False, why=why, **kw)

    if not eligible(x, dtype):
        return no("softmax kernel does not engage at this shape")
    if vv is None or len(vv.shape) != 4 or len(x.shape) != 4:
        return no("not the rank-4 attention shape")
    if vv.dtype != dtype:
        return no("value dtype %s is not the attention dtype %s" % (vv.dtype, dtype))
    if list(x.shape)[:2] != list(vv.shape)[:2]:
        return no("batch/head dims differ")

    if grid is None:
        g = x.device().compute_with_storage_grid_size()
        grid = (g.x, g.y)
    p = plan(x, vv, grid, True, True)
    Wt = p["Wt"]
    if int(vv.padded_shape[-2]) != Wt * TILE:
        return no("value K axis %d is not the key axis %d" % (vv.padded_shape[-2], Wt * TILE))
    n_tiles = int(vv.padded_shape[-1]) // TILE
    if n_tiles != 1:
        return no("head_dim %d is more than one tile; the mirror is derived at N=1"
                  % vv.padded_shape[-1])

    ibw = 2 if Wt % 2 == 0 else 1
    if p["block_size"] != ibw:
        return no("K-blockings differ", blk=p["block_size"], in0_block_w=ibw)

    from .tenstorrent import _attn_value_program_config
    m_tiles = -(-int(x.shape[-2]) // TILE)
    batch = 1
    for d in list(x.shape)[:-2]:
        batch *= int(d)
    # elem_bytes, not tile bytes: `attn_value_matmul` passes 4 for fp32 and 2 for bf16.
    if _attn_value_program_config(m_tiles, Wt, n_tiles, batch,
                                  4 if dtype == ttnn.float32 else 2) is None:
        return no("the shipped PV matmul is on ttnn's heuristic, so there is no pinned blocking")

    fidelity, _approx, fp32_dest_acc, _dst_full_sync = ckc_args(ckc)
    packer_l1_acc = bool(getattr(ckc, "packer_l1_acc", False))
    if fidelity != ttnn.MathFidelity.HiFi4 or not fp32_dest_acc or not packer_l1_acc:
        return no("compute config is not the model's HiFi4 / fp32_dest_acc / packer_l1_acc",
                  fidelity=str(fidelity), fp32_dest_acc=bool(fp32_dest_acc),
                  packer_l1_acc=packer_l1_acc)
    if Wt // p["block_size"] <= 2:
        # Below this the factory stops enabling PACKER_L1_ACC and the partial changes format.
        return no("fewer than three K blocks, so the shipped matmul is not packer-accumulating")

    l1 = pv_l1_bytes(p, tile_bytes(vv.dtype))
    budget = int(L1_PER_CORE * 0.9)
    if l1 > budget:
        return no("value residency does not fit L1", l1_bytes=l1, l1_budget=budget)

    return dict(ok=True, why="", Wt=Wt, blk=p["block_size"], in0_block_w=ibw,
                l1_bytes=l1, l1_budget=budget)


def softmax_pv_fused(x, vv, dtype, ckc):
    """`attn_value_matmul(softmax_bf16(x, dtype), vv, ...)` in one kernel, or None to decline.

    Declining is the default-safe answer and the caller runs the shipped pair. Nothing here
    widens what `softmax_bf16` already covers; it only removes the trip to DRAM between the two.
    """
    if not _PV_ENABLED:
        return None
    v = pv_classify(x, vv, dtype, ckc)
    if not v["ok"]:
        # Key the census by shape as well as reason: "the kernel does not engage" at a 1024 key
        # width and at a 6080 one mean different things for how much of the site L5b actually
        # serves, and a bare reason count cannot tell them apart.
        k = "%s @ key %d" % (v["why"], int(x.padded_shape[-1]))
        PVDECLINES[k] = PVDECLINES.get(k, 0) + 1
        return None
    # Record the width served, not just the count. A decline census alone cannot say WHICH site a
    # fold served, so "the lever declined at this design size" and "it served a different site of
    # the same fold" are indistinguishable without this (root-caused on the R2 fold, §12.1).
    kw = int(x.padded_shape[-1])
    PVSERVED[kw] = PVSERVED.get(kw, 0) + 1
    out = ttnn.empty([int(d) for d in list(x.shape)[:-1]] + [int(vv.shape[-1])],
                     dtype, ttnn.TILE_LAYOUT, x.device(), x.memory_config())
    # The softmax half's own config, unchanged: `approx=True` is what `EXP_APPROX` rides on and
    # the model's config sets `math_approx_mode=False`, so the two must NOT be unified. Only
    # `math_fidelity` and `fp32_dest_acc_en` are shared between the halves, and `pv_classify` has
    # already refused the call if the model's differ from these.
    softmax_into(x.device(), x, out, ckc=(ttnn.MathFidelity.HiFi4, True, True, False), vv=vv)
    PVSTATS[0] += 1
    SSTATS[0] += 1
    return out
