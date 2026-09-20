#!/usr/bin/env python3
"""Route the fold's 1D mcast_in1 matmuls through `tt_bio/mm1d_generic.py`, with or without the split.

Three arms, because two would conflate two different changes:

  ship   nothing installed -- ttnn.linear/ttnn.matmul exactly as they run today
  tr     every covered call through the transcription, split OFF
  split  the same calls through the transcription, split ON

`tr` vs `ship` prices the transcription and whatever the block-config chooser does differently;
`split` vs `tr` prices the split alone, with the config held fixed between them.  Reporting only
`split` vs `ship` would charge the split for a config difference it did not cause.

This is a harness, not a product change: it wraps the ttnn entry points for the length of a
measurement rather than editing `tt_bio/tenstorrent.py`, so an arm is installed and removed in
one process and nothing about the shipped fold changes.
"""
from __future__ import annotations

import collections

import ttnn

from tt_bio.mm1d_generic import generic_mm1d, systolic_1d_config

TILE = 32
_ORIG = {}
_STATS = collections.Counter()
_SHAPES = collections.Counter()
_SPEC = {}
_WIDE = [False]


def stats():
    return dict(_STATS), dict(_SHAPES)


def _batch(shape):
    n = 1
    for d in shape[:-2]:
        n *= d
    return n


def _unsharded(t):
    """What mm1d_generic actually requires.

    The factory's only memory branches are `in0_is_sharded` and `output_is_sharded`; an
    L1-INTERLEAVED operand takes neither, and its page size reaches the kernel through
    TensorAccessorArgs like a DRAM one does.  Requiring DRAM here as well was this harness being
    stricter than the transcription, and it cost 7,532 routable calls a fold.
    """
    return not t.memory_config().is_sharded()


def _ckc_args(ckc):
    """(math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc) off a kernel config."""
    return (getattr(ckc, "math_fidelity", ttnn.MathFidelity.HiFi4),
            bool(getattr(ckc, "math_approx_mode", False)),
            bool(getattr(ckc, "fp32_dest_acc_en", False)),
            bool(getattr(ckc, "packer_l1_acc", False)))


def _out_spec(name, x, w, kw, key):
    """The output's spec, taken from ONE native call per distinct signature.

    Recomputing it here was wrong: `compute_matmul_output_shape` broadcasts the batch dims and
    follows the higher-rank operand, and the default memory config and dtype are resolved deeper
    still.  A native call costs one program build per signature, eighteen in a fold, and it makes
    the routed op a drop-in by construction rather than by argument.
    """
    if key in _SPEC:
        return _SPEC[key]
    ref = _ORIG[name](x, w, **kw)
    spec = ([int(d) for d in ref.shape], ref.dtype, ref.memory_config())
    ttnn.deallocate(ref)
    _SPEC[key] = spec
    return spec


def _route(name, device, split, x, w, kw):
    """The routed output, or None to fall through to the shipped call.

    Every rejection is counted under its own reason, so a null result reads as a reason rather
    than as silence.
    """
    if kw.get("program_config") is not None:
        _STATS["skip:explicit_program_config"] += 1
        return None
    if kw.get("bias") is not None or kw.get("activation") is not None:
        _STATS["skip:bias_or_activation"] += 1
        return None
    # tenstorrent.py:6814 issues matmuls with transpose_a / transpose_b set, which change every
    # stride the factory computes.  mm1d_generic asserts them out; routing one here would have
    # silently computed a different matmul, and did, until mm1dcalib tripped over
    # a=[16384,64] b=[16384,64] -- a pair whose k dimensions do not even match as a plain matmul.
    if kw.get("transpose_a") or kw.get("transpose_b"):
        _STATS["skip:transpose"] += 1
        return None
    if x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        _STATS["skip:dtype"] += 1
        return None
    out_dtype = kw.get("dtype") or ttnn.bfloat16
    if out_dtype != ttnn.bfloat16:
        _STATS["skip:out_dtype"] += 1
        return None
    if not (_unsharded(x) and _unsharded(w)):
        _STATS["skip:operand_sharded"] += 1
        return None
    # An L1-interleaved operand is not a factory branch, and routing it reaches 15,451 calls a
    # fold instead of 9,247.  It is off by default anyway, because it MOVES THE FOLD DIGEST and
    # the reason is not yet established: with --wide the tr and split arms agree with each other
    # and differ from ship, which is the same signature the transposed-matmul defect had, so it
    # is a chooser or coverage question on the newly reached shapes rather than anything the
    # split does.  mm1dcalib.py has only been run over the 17 DRAM shapes; run it over the wide
    # set before turning this on.
    if not _WIDE[0] and not (x.memory_config().buffer_type == ttnn.BufferType.DRAM
                             and w.memory_config().buffer_type == ttnn.BufferType.DRAM):
        _STATS["skip:operand_not_dram"] += 1
        return None
    mc = kw.get("memory_config")
    if mc is not None and mc.is_sharded():
        _STATS["skip:out_sharded"] += 1
        return None
    a = [int(d) for d in x.padded_shape]
    b = [int(d) for d in w.padded_shape]
    if len(b) < 2 or _batch(b) > 1:
        _STATS["skip:b_batched"] += 1
        return None
    if a[-1] % TILE or a[-2] % TILE or b[-1] % TILE or b[-2] % TILE:
        _STATS["skip:untiled"] += 1
        return None

    cg = kw.get("core_grid")
    grid = (cg.x, cg.y) if cg is not None else (device.compute_with_storage_grid_size().x,
                                                device.compute_with_storage_grid_size().y)
    ckc = kw.get("compute_kernel_config")
    if ckc is None:
        _STATS["skip:no_compute_kernel_config"] += 1
        return None
    fidelity, approx, fp32, packer = _ckc_args(ckc)

    bm = _batch(a) * a[-2] // TILE
    nt = b[-1] // TILE
    if bm <= nt:
        _STATS["skip:not_tall"] += 1          # mcast_in0, a different factory
        return None
    # The chooser only reaches the 1D config for a narrow or within-a-tile shape; anything else
    # is the 2D factory and this module does not transcribe it.
    k, m, n = a[-1], a[-2], b[-1]
    h, wd = bm * TILE, n
    narrow = (h // wd if h > wd else wd // h) > 8
    if not (narrow or k <= TILE or m <= TILE or n <= TILE):
        _STATS["skip:mcast_2d"] += 1
        return None

    pc, mcast_in0 = systolic_1d_config(x, w, grid, fp32, out_dtype)
    if pc is None or mcast_in0:
        _STATS["skip:no_config"] += 1
        return None
    num_blocks_y = (bm - 1) // pc[6] + 1
    num_blocks_x = (nt - 1) // pc[7] + 1
    if num_blocks_y * num_blocks_x > grid[0] * grid[1]:
        _STATS["skip:blocks_exceed_cores"] += 1
        return None

    key = (tuple(a), tuple(b), str(x.dtype), str(w.dtype), str(out_dtype), grid,
           str(x.memory_config()), str(w.memory_config()), str(kw.get("memory_config")))
    shape, ref_dtype, ref_mc = _out_spec(name, x, w, kw, key)
    if ref_mc.is_sharded() or ref_dtype != out_dtype:
        _STATS["skip:native_out_spec"] += 1
        return None
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(shape), ref_dtype, ttnn.TILE_LAYOUT,
                        ref_mc.buffer_type), device)
    try:
        r = generic_mm1d(device, x, w, out, pc, (fidelity, approx, fp32, packer),
                         writer_on_in0=split)
    except Exception as e:
        _STATS["skip:build_failed:" + type(e).__name__] += 1
        ttnn.deallocate(out)
        return None
    _STATS["routed"] += 1
    _SHAPES[(tuple(a), tuple(b), pc[1], pc[6], pc[7])] += 1
    return r


def install(device, split, wide=False):
    """Wrap ttnn.matmul and ttnn.linear.  Idempotent; `remove()` puts the originals back."""
    remove()
    _STATS.clear()
    _SHAPES.clear()
    _SPEC.clear()
    _WIDE[0] = bool(wide)
    for name in ("matmul", "linear"):
        _ORIG[name] = getattr(ttnn, name)

        def wrapped(*args, _name=name, **kw):
            if len(args) >= 2 and len(args) == 2:
                r = _route(_name, device, split, args[0], args[1], kw)
                if r is not None:
                    return r
            else:
                _STATS["skip:positional_extra_args"] += 1
            return _ORIG[_name](*args, **kw)

        setattr(ttnn, name, wrapped)


def remove():
    for name, fn in list(_ORIG.items()):
        setattr(ttnn, name, fn)
    _ORIG.clear()
