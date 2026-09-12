"""The SwiGLU transition's two projections and its product as one ``generic_op``.

    production   x_1 = silu(x_norm @ fc1) ; x_2 = x_norm @ fc2 ; multiply_(x_1, x_2)
                 2 activation reads, 2 hidden writes, 2 hidden reads, 1 hidden write
    here         one kernel over both (activation, weight) pairs
                 2 activation reads, 1 hidden write

At 512 aa with c_z = 128 and hidden = 512 the hidden tensor is 4,096 tiles a 16-row chunk, so this
deletes **16,384 tile passes a chunk, 524,288 a `transition_z` call** -- the largest single fusible
adjacency in the whole PairformerLayer block (`perf/b2z2_megakernel/LEDGER.md`).

Every one of those passes is an L1 pass, which is the regime where deleting a pass pays: the three
ops it replaces all ask for ``L1_MEMORY_CONFIG``, so none of this traffic is on the byte axis that
has already lost twice on the trimul. **The output therefore has to stay in L1 too.** That is the
one thing `trimul_tail` got wrong at c_z = 128: it allocated in DRAM against an incumbent the
allocator was granting L1, and turned a 2-op saving into +2.019 ms a call.

The kernels are `tt_bio/kernels/trimul_tail/`'s with two changes: silu is applied on pass 1's fp32
accumulator inside `copy_block`, where ``ttnn.linear(activation="silu")`` applies it, and the
epilogue is a plain product instead of a sigmoid gate.

Scoped to the class the fold issues: bf16 in and out, both activations the same tensor, both
weights the same shape, one K block, no bias, no ternary. Outside that `fused_swiglu` returns None
and the caller keeps today's three ops.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import ttnn

from . import mm_generic as MG
from .envflags import env_flag

KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "transition_swiglu"

TILE = 32
PASSES = 2
ROUND = 2          # the product's rounding to bf16; see kernels/trimul_tail/compute.cpp
MUL_BATCH = 1      # output tiles folded per DST acquire in the epilogue product; ceiling 2 on fp32 DST
MUL_MODE = 0       # 0 = two copy_tile + SFPU product, 1 = mul_tiles (one FPU unpack, 1 DST slot)

#: (kt, nt) -> the block config the fused kernel folds with. Deliberately a LOCAL table and not
#: `tenstorrent._MM_BLOCK`: a lookup into the shared table would switch this on for every model
#: whose transition happens to share a key, which is a multi-model default flip inside a one-site
#: change. (4, 16) is c_z = 128 with hidden = 512, i.e. Boltz-2's and OpenFold3's pair transition.
#: The tuple is `_MM_BLOCK[(4, 4)]`'s, unchanged: K_block = 4 is still the whole contraction in one
#: block, and N_block = 1 means the wider output is more N blocks rather than a bigger one, so
#: neither of the two things that broke `trimul_tail` at (12, 12) -- `out_block` doubling and
#: `subblock_h` halving -- moves here. The 4-tile subblock is also exactly the fp32 half-sync DST
#: budget, so the chain does not need `dst_full_sync_en` and keeps the math/pack double buffer.
BLOCK_KEYS = {(4, 16): (4, 4, 1, 4, 1)}

#: Diagnostic only: narrower hidden widths, to separate a bug in the kernel from a bug that only
#: appears once the output is many N blocks wide. Never served in production.
DIAG_KEYS = {(4, 4): (4, 4, 1, 4, 1), (4, 8): (4, 4, 1, 4, 1), (4, 2): (4, 4, 1, 4, 1)}
if env_flag("TT_BIO_TRANSITION_SWIGLU_DIAG", False):
    BLOCK_KEYS = {**BLOCK_KEYS, **DIAG_KEYS}

#: OFF by default. It is a new kernel on a shared module (`Transition` serves every model in the
#: repo), so it ships dark until the A/B and the parity leg have run on qb2.
ENABLED = env_flag("TT_BIO_TRANSITION_SWIGLU", False)


def set_enabled(on: bool) -> bool:
    """A/B switch for the paired harness. Returns the previous state."""
    global ENABLED
    prev, ENABLED = ENABLED, bool(on)
    return prev


def _tiles(n):
    return (int(n) + TILE - 1) // TILE


@lru_cache(maxsize=None)
def _block_for(kt, nt):
    return BLOCK_KEYS.get((kt, nt))


STATS = [0, 0]          # served, declined
REJECTS: dict = {}      # (reason, shape) -> count, so a decline is diagnosable from a fold JSON


def _reject(why, shape=""):
    STATS[1] += 1
    REJECTS[(why, shape)] = REJECTS.get((why, shape), 0) + 1
    return None


def eligible(x, w_plain, w_act):
    """None when this descriptor covers the call, else the reason it does not.

    Every clause is a real assumption of the kernel, so a decline names which one. Note what is
    NOT here: the model. The gate is shape, dtype and memory config only.
    """
    if ttnn.bfloat16 not in (x.dtype, w_plain.dtype, w_act.dtype):
        return "dtype"
    if w_plain.dtype != w_act.dtype or x.dtype != w_plain.dtype:
        return "dtype_pair"
    if len(w_plain.shape) != 2 or tuple(w_plain.shape) != tuple(w_act.shape):
        return "weight_shape_pair"
    if str(w_plain.memory_config()) != str(w_act.memory_config()):
        return "weight_memcfg_pair"
    kt, nt = _tiles(w_plain.shape[-2]), _tiles(w_plain.shape[-1])
    block = _block_for(kt, nt)
    if block is None:
        return f"block_key={kt}x{nt}"
    M, K, N, _, _ = block
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


def _build(device, x, w_plain, w_act, out, grid, ckc, block):
    defs = {"TRIMUL_TAIL_PASSES": PASSES, "TRIMUL_TAIL_ROUND": ROUND,
            "TRIMUL_TAIL_MUL_BATCH": MUL_BATCH, "TRIMUL_TAIL_MUL_MODE": MUL_MODE}
    entry = MG.build(device, x, w_plain, [out], (block, grid), ckc,
                     defines=defs, kernel_dir=KERNEL_DIR)

    gx, gy = grid
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    out_block = block[0] * block[2]
    # c_4 / c_5: the two bf16 projections, double buffered. c_6 is the gate CB `trimul_tail`
    # needed and this kernel does not, but the descriptor keeps the same CB set so the two
    # kernels' dataflow sources stay byte-identical.
    entry["cbs"] += [_cb(4, core_grid, out_block * 2),
                     _cb(5, core_grid, out_block * 2),
                     _cb(6, core_grid, 2)]

    compute = entry["kernels"][4]
    compute.kernel_source = str(KERNEL_DIR / "compute.cpp")
    compute.defines = [(k, str(v)) for k, v in defs.items()]

    _bind_b(entry, x.buffer_address(), w_act.buffer_address())
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


def fused_swiglu(x, w_plain, w_act, ckc, grid, memory_config=None):
    """``(x @ w_plain) * silu(x @ w_act)`` in one kernel. None when out of scope.

    `memory_config` defaults to L1, because the three ops this replaces all ask for L1 and a DRAM
    output would put back more traffic than the fusion deletes.
    """
    if not ENABLED:
        return _reject("disabled")
    why = eligible(x, w_plain, w_act)
    if why is not None:
        return _reject(why, "x".join(str(int(d)) for d in x.padded_shape)
                       + "@" + "x".join(str(int(d)) for d in w_plain.shape))
    device = x.device()
    mc = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    spec = lambda t: (str(t.padded_shape), str(t.dtype), str(t.memory_config()))
    key = (spec(x), spec(w_plain), tuple(grid), tuple(str(c) for c in ckc), ROUND, MUL_BATCH, MUL_MODE, str(mc))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([int(d) for d in x.shape][:-1] + [int(w_plain.shape[-1])]),
        ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)

    entry = _CACHE.get(key)
    if entry is None:
        entry = _CACHE[key] = _build(device, x, w_plain, w_act, out, grid, ckc,
                                     _block_for(_tiles(w_plain.shape[-2]),
                                                _tiles(w_plain.shape[-1])))
    else:
        addrs = (x.buffer_address(), w_plain.buffer_address(), (out.buffer_address(),))
        b = (x.buffer_address(), w_act.buffer_address())
        stale_b = b != entry["b_addrs"]
        if stale_b:
            _bind_b(entry, *b)
        if addrs != entry["addrs"]:
            MG.rebind(entry, *addrs)
        elif stale_b:
            _repack(entry)

    ttnn.generic_op([x, w_plain, x, w_act, out], entry["pd"])
    STATS[0] += 1
    return out
