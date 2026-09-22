"""`silu(x @ W1) * (x @ W2)` as one `generic_op`: the pair Transition's SwiGLU in one kernel.

    production   x_1 = silu(x_norm @ W1) ; x_2 = x_norm @ W2 ; multiply_(x_1, x_2)
                 3 ops, 2 output-sized L1 tensors live, 3 output-sized L1 round trips
    fused        one kernel over (x_norm, W1) and (x_norm, W2), silu and the multiply in the core
                 1 op, 1 output-sized L1 tensor, 1 write

c14-radical measured why this site needs a kernel and not a rewrite. Merging W1 and W2 into one
[128, 1024] projection is 1.340x on the matmuls themselves (0.06317 against 0.08466 ms), so the
per-op fixed term is 50.8 % of a single N=512 call at this key. But ttnn has no view, so
`multiply_` needs the halves as separate operands and the merged arm pays two `ttnn.slice` calls
worth 0.03302 ms against the 0.02149 ms the merge returns: 1.076x SLOWER end to end, interleaved,
against a 0.13 % A/A floor. The fused kernel is the only route that keeps the merge's saving
without paying for an un-concatenate, and it deletes the silu and the multiply's L1 round trips
on top.

The kernels are generated from the wheel's own minimal_matmul sources by
`kernels/swiglu_fused/patch_swiglu_fused.py`, which starts from `trimul_tail`'s two-pass fork --
that fork already reads a second (activation, weight) pair out of the unused `in2_addr` runtime
slot -- and diverges on the one thing SwiGLU does differently: both passes contract the SAME
activation. Pass 1 therefore does not read in0 at all and the compute kernel holds pass 0's block
across the pass boundary. That is a correctness fix before it is a saving; the patch script records
the CB leak that deadlocked the card when the wheel's in0 reuse was left alone.

Scoped to the class the fold issues at 512 aa: bf16 in and out, both weights the same shape and
layout, no bias, one K block whose (kt, nt) key is in `SWIGLU_BLOCK_KEYS`. Outside that
`fused_swiglu` returns None and the caller keeps today's three ops.

NOTHING IN THE MODEL CALLS THIS, deliberately. The kernel is correct -- closer to a float64
reference than the three ops it replaces -- and the fusion works: 0.5018-0.5134x against its own
unfused two-matmul control, in two sessions at a during-sampled 1350 MHz. It is still
1.377-1.409x SLOWER than production, because `generic_op` can only re-drive kernels the wheel
already ships and `minimal_matmul`, the only forkable matmul, is about 6.5x off the multicast
matmul `ttnn.linear` picks at this shape: mt = 256, nt = 16, kt = 4 over 110 cores is 24 M tiles
and 2 N tiles per core, and its semaphore-chained dataflow never amortises there. Kept as the
record of a measured NO-GO, not as a lever. `perf/c14_swiglu/`,
`~/.coworker/state/c14-swiglu-kernel.md`.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

from . import mm_generic as MG
from . import trimul_tail as TTAIL

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "swiglu_fused"

TILE = 32

#: How the product reaches bf16 before it is packed. 2 is binary_ng's round-to-nearest-even done
#: in integers, which is what `multiply_` does; 0 leaves it to the packer, which breaks ties away
#: from zero under fp32 DST. See `kernels/trimul_tail/patch_trimul_tail.ROUND_BF16`.
ROUND = 2

#: (kt, nt) of one weight, both weights being the same shape. (4, 16) is the pair Transition at
#: c_z = 128 with hidden 512 -- boltz2, openfold3 -- and `_MM_BLOCK` maps it to (4, 4, 1, 4, 1),
#: the same block trimul_tail's (4, 4) entry uses: one K block, which is this kernel's whole
#: precondition, since the two GEMMs must not interact through the cross-block L1 accumulator.
#: (4, 16) is a FUSED width, so `_MM_BLOCK` no longer writes it down -- `_mm_block_at`
#: derives it from (4, 12) + (4, 4), the same value the literal carried.
#: An allow-list and not `_MM_BLOCK` itself, for trimul_tail's reason: a general lookup would hand
#: models a block these kernels have not been swept at, and (12, 12) is the recorded case where
#: that returns wrong numbers at N = 32 and then hangs the device.
SWIGLU_BLOCK_KEYS = {(4, 16)}

STATS = [0, 0]          # served, declined
REJECTS: dict = {}      # (reason, shape) -> count, so a decline is diagnosable from the fold JSON


def _tiles(n):
    return (int(n) + TILE - 1) // TILE


def _reject(why, shape=""):
    STATS[1] += 1
    REJECTS[(why, shape)] = REJECTS.get((why, shape), 0) + 1
    return None


def _block(w):
    """This weight's block config, or None when its (kt, nt) key is not allow-listed."""
    from . import tenstorrent as TT      # late: `tenstorrent` imports this module at its import
    key = (_tiles(w.shape[-2]), _tiles(w.shape[-1]))
    return TT._mm_block_at(*key) if key in SWIGLU_BLOCK_KEYS else None


def eligible(x, w1, w2):
    """None when the descriptor covers this call, else the reason it does not.

    Every clause is a real assumption of the kernel, so a decline names which one.
    """
    from . import ops
    if ops.taping():
        return "taping"   # generic_op has no backward; the composed path runs instead

    if x.dtype != ttnn.bfloat16 or w1.dtype != ttnn.bfloat16 or w2.dtype != ttnn.bfloat16:
        return "dtype"
    if len(w1.shape) != 2 or tuple(w1.shape) != tuple(w2.shape):
        return "weight_shape_pair"
    if str(w1.memory_config()) != str(w2.memory_config()):
        return "weight_memcfg_pair"
    block = _block(w1)
    if block is None:
        return f"key={_tiles(w1.shape[-2])}x{_tiles(w1.shape[-1])}"
    M, _, N, _, _ = block
    nt = _tiles(w1.shape[-1])
    if nt % N:
        return f"n_tiles={nt}"
    mt = 1
    for d in [int(d) for d in x.padded_shape][:-1]:
        mt *= d
    mt = _tiles(mt)
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


def _build(device, x, w1, w2, out, grid, ckc, block):
    # TRIMUL_TAIL_PASSES is the data-movement fork's own macro and reaches only its two kernels.
    entry = MG.build(device, x, w1, [out], (block, grid), ckc,
                     defines={"TRIMUL_TAIL_PASSES": 2}, kernel_dir=KERNEL_DIR)
    # One K block is the kernel's stated precondition: the two GEMMs must not interact through the
    # cross-block L1 accumulator, and in0 is held across the pass boundary. The compute kernel
    # static_asserts it too, but raising here costs no JIT build.
    assert entry["dims"]["K_blocks"] == 1, entry["dims"]

    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    out_block = block[0] * block[2]
    # c_4: silu(fc1). c_5: fc2. Both bf16, one output block, double buffered.
    entry["cbs"] += [_cb(4, core_grid, out_block * 2), _cb(5, core_grid, out_block * 2)]

    compute = entry["kernels"][4]
    compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
    compute.defines = [("SWIGLU_ROUND", str(ROUND))]

    _bind_pass1(entry, x.buffer_address(), w2.buffer_address())
    TTAIL._repack(entry)
    return entry


def _bind_pass1(entry, x_addr, w2_addr):
    """Pass 1's two addresses ride in the unused `in2_addr` slot, index 1 in both DM kernels.

    Pass 1's activation IS pass 0's, which is the whole reason this fusion is cheaper than
    trimul_tail's: one activation, two weights.
    """
    for name in ("in0_sender", "in0_recv"):
        for _, a in entry["rt"][name]:
            a[1] = x_addr
    for name in ("in1_sender", "in1_recv"):
        for _, a in entry["rt"][name]:
            a[1] = w2_addr
    entry["b_addrs"] = (x_addr, w2_addr)


_CACHE: dict = {}


def fused_swiglu(x, w1, w2, ckc, grid, out_memory_config=None):
    """`silu(x @ w1) * (x @ w2)` in one kernel, or None when the call is out of scope.

    `out_memory_config` is where the product lands and it is a real perf decision, not a detail:
    production keeps both projections and their product in L1, so a DRAM destination would turn
    the round trip this deletes into one it adds. trimul_tail measured that mistake at
    +2.019 ms/trimul. Defaults to L1 and falls back to DRAM only if the allocator refuses.
    """
    why = eligible(x, w1, w2)
    if why is not None:
        return _reject(why, "x".join(str(int(d)) for d in x.padded_shape)
                       + "@" + "x".join(str(int(d)) for d in w1.shape))
    device = x.device()
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    mem = out_memory_config or ttnn.L1_MEMORY_CONFIG
    shape = ttnn.Shape([int(d) for d in x.shape][:-1] + [int(w1.shape[-1])])
    try:
        out = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    except Exception:                                                      # noqa: BLE001
        if mem == ttnn.DRAM_MEMORY_CONFIG:
            raise
        mem = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
    key = (spec(x), spec(w1), tuple(grid), tuple(str(c) for c in ckc), ROUND, str(mem))

    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(device, x, w1, w2, out, grid, ckc, _block(w1))
    else:
        # `MG.rebind` repacks the descriptor itself, so only bind pass 1 separately when it does
        # not run.
        addrs = (x.buffer_address(), w1.buffer_address(), (out.buffer_address(),))
        b = (x.buffer_address(), w2.buffer_address())
        stale_b = b != entry["b_addrs"]
        if stale_b:
            _bind_pass1(entry, *b)
        if addrs != entry["addrs"]:
            MG.rebind(entry, *addrs)
        elif stale_b:
            TTAIL._repack(entry)

    ttnn.generic_op([x, w1, x, w2, out], entry["pd"])
    STATS[0] += 1
    return out
