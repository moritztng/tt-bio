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

TILE = 32
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}

# `device->l1_size_per_core()` is not exposed to Python. Blackhole is 1464 KB. The value only
# feeds the factory's `use_large_kernel` trip at `0.9 * l1_size_per_core`, and both production
# shapes sit far from it -- the DiT's 22 Wt needs 307 KB and the atom path's 190 Wt needs
# 2.26 MB -- so no plausible value moves either verdict. A wrong choice would fail S1 anyway.
L1_PER_CORE = 1499136

_CACHE: dict = {}


def _kdir():
    from .mm_generic import ttnn_cpp_root
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
    # RFD3 sites pass a key axis that is already `_align_tile`d, so it is asserted, not handled.
    assert W == logical[-1], ("padded last dim %d != logical %d; the mask_padded_data path is "
                              "not transcribed" % (W, logical[-1]))

    block_size = find_max_divisor(Wt, 4 if fp32_dest_acc else 8)
    in0_t = _div_up(Wt, block_size) * block_size if numeric_stable else block_size * 2
    out0_t = block_size * 2
    im4_t = _div_up(Wt, block_size) * block_size
    im0_t = block_size * _div_up(Wt, block_size)
    assert im0_t == Wt, (im0_t, Wt)
    cb_length = in0_t

    in0_ts = _TILE_BYTES[x.dtype]
    im_fmt = ttnn.float32 if fp32_dest_acc else ttnn.bfloat16
    im_ts = _TILE_BYTES[im_fmt]

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
                im_fmt=im_fmt, im_ts=im_ts, in0_ts=in0_ts, out0_ts=_TILE_BYTES[out.dtype],
                numeric_stable=numeric_stable, fp32_dest_acc=fp32_dest_acc,
                gx=gx, gy=gy, max_cores=max_cores, units=units, target=target,
                per1=per1, per2=per2, rem=rem)


def build(device, x, out, grid, ckc, numeric_stable=True):
    """The ProgramDescriptor. `ckc` is `(math_fidelity, math_approx, fp32_dest_acc, dst_full_sync)`.

    `softmax_init_compute_kernel_config` defaults it to
    `(HiFi4, approx=True, fp32_dest_acc=input_is_fp32, l1_acc=False)` on Blackhole, and
    `dst_full_sync` defaults False.
    """
    fidelity, approx, fp32_dest_acc, dst_full_sync = ckc
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
    if numeric_stable or p["use_large"]:
        cbs.append(cb(10, p["im4_t"], im_ts, im))          # c_x

    def acc(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_ct = acc(x)
    writer_ct = [0, TILE * TILE] + acc(out)                # num_datum_padded, tile_hw
    dm_defines = [("NUMERIC_STABLE", "1")] if numeric_stable else []
    compute_defines = list(dm_defines) + [("EXP_APPROX", "1" if approx else "0"),
                                          ("ENABLE_FP32_DEST_ACC", "1" if fp32_dest_acc else "0")]

    # When the output dtype differs from the input's, the packer's own fp32 -> bf16 rounding is
    # NOT the rounding `ttnn.typecast` does (measured 0.00195 maxabs, one bf16 ULP, p74 S2), so the
    # compute kernel has to do typecast's SFPU conversion itself and pack a DST that is already
    # bf16-valued. `tt_bio/kernels/rfd3_softmax/` holds the wheel's two compute kernels with that
    # one insertion behind `PACK_BF16_TYPECAST`; everything else is byte-identical to the wheel.
    cast_in_sfpu = out.dtype != x.dtype
    if cast_in_sfpu:
        assert x.dtype == ttnn.float32 and out.dtype == ttnn.bfloat16, (x.dtype, out.dtype)
        compute_defines.append(("PACK_BF16_TYPECAST", "1"))

    kd = _kdir()
    if p["use_large"]:
        reader_src = str(kd / "dataflow/reader_unary_interleaved_sm_large_tensor.cpp")
        compute_src = str(kd / "compute/softmax_large_tensor.cpp")
    else:
        reader_src = str(kd / "dataflow/reader_unary_interleaved_sm.cpp")
        compute_src = str(kd / "compute/softmax.cpp")
    if cast_in_sfpu:
        local = Path(__file__).resolve().parent / "kernels" / "rfd3_softmax"
        compute_src = str(local / ("softmax_large_tensor.cpp" if p["use_large"]
                                   else "softmax.cpp"))
    writer_src = str(kd / "dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp")

    src_a, out_a = x.buffer_address(), out.buffer_address()
    Wt, Ht, blk = p["Wt"], p["Ht"], p["block_size"]
    rr, wr, cr = [], [], []
    curr_row = 0
    for i in range(p["max_cores"]):
        core = ttnn.CoreCoord(i % gx, i // gx)
        if i >= p["target"]:
            rr.append((core, [0] * 10 + [0x3F803F80, 0]))
            cr.append((core, [0] * 7))
            wr.append((core, [0] * 6))
            continue
        n = p["per1"] if (p["rem"] == 0 or i < p["rem"]) else p["per2"]
        tile_offset = curr_row * Wt
        curr_ht = curr_row % Ht
        mask_id = curr_row // Ht * Wt
        rr.append((core, [src_a, blk, 0x3F800000, n, tile_offset, Wt, Ht, 0, curr_ht, mask_id,
                          0x3F803F80, p["in0_t"]]))
        cr.append((core, [n, Ht, Wt, blk, curr_ht, 0, p["cb_length"]]))
        wr.append((core, [out_a, n * Wt, tile_offset, blk, 0, 0]))
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
            "addrs": (src_a, out_a)}


def softmax_into(device, x, out, grid=None, ckc=None, numeric_stable=True):
    """Run the transcribed program, writing into `out` (any dtype). Returns `out`."""
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = (g.x, g.y)
    if ckc is None:
        ckc = (ttnn.MathFidelity.HiFi4, True, x.dtype == ttnn.float32, False)
    key = (str(x.padded_shape), str(x.shape), str(x.dtype), str(out.dtype), grid,
           tuple(str(c) for c in ckc), numeric_stable)
    e = _CACHE.get(key)
    if e is None:
        e = _CACHE[key] = build(device, x, out, grid, ckc, numeric_stable)
    addrs = (x.buffer_address(), out.buffer_address())
    if addrs != e["addrs"]:
        rr, wr, _ = e["rt"]
        for _c, a in rr:
            if a[3]:
                a[0] = addrs[0]
        for _c, a in wr:
            if a[1]:
                a[0] = addrs[1]
        e["kernels"][0].runtime_args = rr
        e["kernels"][1].runtime_args = wr
        e["pd"] = ttnn.ProgramDescriptor(kernels=e["kernels"], semaphores=[], cbs=e["cbs"])
        e["addrs"] = addrs
    ttnn.generic_op([x, out], e["pd"])
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
