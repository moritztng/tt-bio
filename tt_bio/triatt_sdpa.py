"""Triangle attention's SDPA with the bias held in a permanently fronted CB.

The reader re-reads the whole triangle bias once per batch row. At 512 aa that is 4.19 MB read 512
times, 2048 MiB/call against the 4 MiB the maths needs, and it is 84.2 % of the op's read traffic.
Nothing about the mask depends on the batch: it is `[1, n_heads, S, S]`, so `mask_batch_offset` is 0
and every batch reads identical tiles.

So the work split is made head-contiguous (one head per core), the reader fills the head's whole
mask once before the batch loop, and the compute path indexes that fronted CB instead of popping it.
Driven through :mod:`tt_bio.sdpa_generic`, a transcription of `sdpa_program_factory.cpp` at the
`v0.68.0` tag, with the two kernel edits guarded on `PERSISTENT_MASK` in
``tt_bio/kernels/triatt_sdpa/``.

MEASURED on qb2 card 1 at 512 aa (`perf/triatt_fused/s6_gate.json`), `torch.equal` throughout:

    native SDPA                             6.521 ms
    transcription, head-contiguous split    6.498 ms
    + persistent mask                       2.673 ms    2.431x

The mask CB grows with the k chunk count: the persistent form needs `k_num_chunks * Sq_chunk_t *
Sk_chunk_t` tiles against the stock `Sq_chunk_t * Sk_chunk_t * 2` for double buffering, so it is
`k_num_chunks / 2` times the stock CB. That is a wash at two k chunks (512 aa, 256 tiles either
way), 1.5x at three (768 aa, 288 against 192, and it fits) and 2x at four (1024 aa, 512 against 256,
and L1 refuses it). It does NOT hold flat, and a refusal is handled below rather than predicted.

The gate is narrow on purpose. It needs one head and one q chunk per core, a batch-broadcast mask,
no padded mask, and bf16 interleaved DRAM throughout; anything else falls through to the stock op.

RE-MEASURED on qb2 card 3 (Blackhole p300c, 11x10) at the same 512 aa shape, 40 interleaved blocks,
own-session A/A 1.2 % (`perf/roof_triatt_levers/sdpa_bh_b40.json`):

    stock op, batch-1 bias        3.6273 ms
    stock op, no mask at all      1.3682 ms      the bias costs the stock op 2.65x
    this kernel, mask on          1.3684 ms      at the no-mask arm, inside the A/A floor
    this kernel, mask add ablated 1.2089 ms      1.132x, and that add is the model's own maths

So the whole of the stock op's mask cost is already gone here, and what is left is the bias add
itself. `roof-tri-close` sized that cost at 1.88x on Wormhole and named it as an unbuilt lever; it
was already built. The Boltz-2 512 aa fold serves 560 of 560 triangle-attention calls through this
path on this part (`perf/sizegate/baseline/census_boltz2_512_p300c.json`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import ttnn

from . import sdpa_generic as SG
from .envflags import env_flag, env_int

# Five files, not the fifteen the stock kernel directory holds. Three carry our edits:
# `compute/sdpa.cpp`, `compute/compute_common.hpp`, `dataflow/reader_interleaved.cpp`. The other
# two, `compute/compute_streaming.hpp` and `dataflow/dataflow_common.hpp`, are byte-identical to
# the wheel and stay only because those two entry points include them by bare name, which the
# kernel compiler resolves against the directory the including kernel lives in. The rest of the
# original copy -- the joint and ring-joint kernels with their headers, and
# `dataflow/writer_interleaved.cpp`, since `sdpa_generic.sdpa` takes the writer from the wheel --
# was reachable from nothing and was deleted.
KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "triatt_sdpa"

# (calls served, calls declined)
STATS = [0, 0]
REJECTS: dict = {}

TRIATT_PERSISTENT_MASK = True
_ENABLED = os.environ.get(
    "TT_BIO_TRIATT_PERSISTENT_MASK", "1" if TRIATT_PERSISTENT_MASK else "0") == "1"

# The q-split above, ON by default up to _Q_SPLIT_MAX_S padded tokens. Verified at 768 aa with the
# _PM_OVER_L1 fix in place: -17.670 s (6.4 %) on the fold, byte-identical CIF and plDDT, 7x the
# 2.543 s A/A floor (perf/sizes/qsplitfix_768.json, qb1 card 2, benchlock). Raised to 1024 on the
# boltz2 evidence: 1.0777x at 768 aa and 1.0893x at 1024 aa, CIF byte-identical, and three
# back-to-back 1024 aa folds with the lever on showed no OOM and flat VmHWM (perf/b2sizes/). At
# 1024 the q1024/q512 persistent-mask configs refuse at compile, land in _PM_OVER_L1, and all 560
# calls serve at q256 (55.6 % of the per-core budget). Above 1024 the mask CB growth is untested,
# so the shipped split stays there; an L1 refusal at any size is caught and falls back to the
# stock op. "0" forces off.
_Q_SPLIT = env_flag("TT_BIO_TRIATT_MASK_Q_SPLIT", True)
# 1024 is where the persistent mask CB stopped being MEASURED, not where it stops fitting:
# `sdpa_generic.cb_bytes` prices it exactly now (10 refusals, to the byte), so above the cap the
# budget can decide instead of a number. Raising it is release-gated -- it changes which kernel a
# path shared by five models takes at every length above 1024 -- so it ships at 1024 and
# `TT_BIO_TRIATT_MASK_Q_SPLIT_MAX` is how an A/B lifts it. Nothing above the cap serves fused
# today: `perf/bgsdpa/fused_reach.py` counts 0 of the 50 lengths from 1024 to 2592.
_Q_SPLIT_MAX_S = env_int("TT_BIO_TRIATT_MASK_Q_SPLIT_MAX", 1024)

# q_chunks whose PERSISTENT mask CB does not fit. Deliberately not `_SDPA_Q_CHUNK_OVER_L1`: that set
# is the wide-q ladder memo of q_chunks the STOCK op cannot fit, and `_tri_att_sdpa_at` filters its
# candidate list with it. This kernel allocates a strictly larger mask CB -- `k_num_chunks *
# Sq_chunk_t * Sk_chunk_t` tiles against the stock `2 * Sq_chunk_t * Sk_chunk_t` -- so a refusal here
# says nothing about what the stock op fits. Writing it into the shared set retires a q_chunk the
# stock op runs perfectly well, and the fold loses the wide-q win (1.08-1.81x) on every later call at
# that shape. MEASURED at 512 aa: the 995-token refiner fell from q_chunk 512 to 256 after one such
# throw and the fold lost 3.129 s against an A/A floor of 0.056 s.
_PM_OVER_L1: set = set()

# The device's own message for the refusal that retired each key. The exception is caught below and
# turned into a decline, so without this the byte figures it carries -- allocated, budget, and the
# shortfall -- are lost, and an L1 screen has to re-run the call outside the gate to read them.
PM_L1_ERRORS: dict = {}


# Compute kernel config for the fused SDPA when the caller does not pass one. None means the op
# default below, which is what every call took until RF3's triangle attention started passing its
# own: `(HiFi2, approx, no fp32_dest_acc)` is a LOW-PRECISION config, and reading the fused path as
# "bf16 softmax" conflated the storage with it.
#
# MEASURED on one captured RF3 triangle-attention call, all thirteen arms against an fp64 evaluation
# of the SAME bf16 operands, so only the kernel's own error is left (perf/rf3/triatt_ckc_sweep.py,
# qb2 card 0, `--sweep ckc`; rel_rms, 512 aa then 128 aa):
#
#     bf16 ceiling (torch bf16 storage)        0.00163   0.00165
#     HiFi4, approx off, fp32_dest_acc  <- **0.00470**   0.00512     1.819 ms   0.163 ms
#     HiFi2, approx off, fp32_dest_acc       0.00612    0.00686     1.516      0.164
#     _fp32_softmax_attention (shipped)      0.00883    0.00977    36.727      0.569
#     HiFi2, approx on, no acc (the default) 0.01293    0.01293     1.432      0.169
#     LoFi, any                              0.043-0.048           1.35-1.40
#
# CORRECTED (perf/fused_sdpa/errstruct_rf3_512.json). The 1.88x above is real but it is a norm, and
# reading it as "the fused kernel is more accurate" conflated two arms and hid which component of the
# error moved. On 14 calls captured across a real 512 aa recycler pass, the error split into a
# per-row gain along the fp64 reference (par) and a per-channel direction error (perp):
#
#     arm                             rel_total vs materialised   rel_perp vs materialised
#     op default (HiFi2, approx, -)   1.08 - 1.50x WORSE          1.12 - 2.38x WORSE
#     HiFi4, approx off, acc          1.4 - 4.2x BETTER           0.97 - 1.13x, a wash
#
# So the op default -- which is what four of the six models ship -- is NOT more accurate per op, and
# the HiFi4 arm's whole win sits in `par`, the component a residual+LayerNorm trunk tolerates. That
# is why adopting it moved no fold. The knobs own different components: math_approx owns par and
# does nothing to perp, fp32_dest_acc owns perp (1.51x at the deepest captured call), and
# HiFi2->HiFi4 reaches perp only once the accumulator is already wide.
#
# Fidelity is free on time either way: this op is bandwidth-bound, and every arm above is within
# 27% on time while spanning 10x on error.
#
# What does NOT work is lifting the kernel's intermediate CBs to fp32 (scores / attn@v accumulator /
# running max+sum, `sdpa_program_factory.cpp:651-653`). Tried, all eight combinations: any mix of
# fp32 and bf16 among the three groups returns NaN, and all three fp32 together returns finite but
# wrong values (pcc 0.893). The scores CB is the second matmul's in0, so fp32 there is a mixed-format
# matmul against a bf16 v, and the statistics CBs meet a bf16 scalar in the reduce. The plumbing was
# removed again rather than left as a dark knob -- there is nothing to gain from it, since the fp32
# DST already carries the reduction and the arm above beats the materialised path outright.
# INSTRUMENT, never a shipped knob. `TT_BIO_TRIATT_ABLATE=EXP,MASKADD` compiles the kernel with
# the named stage REMOVED, so the arm's output is wrong on purpose and only its time means anything.
# It exists because neither roof this campaign uses -- bytes moved and matmul FLOPs -- prices the
# SFPU, and at head_dim 32 the score matrix this op exponentiates is 16x larger than the operands
# the roofs count. An ablation is the only way to read that cost: a standalone `ttnn.exp` is
# bandwidth-bound at any size that fits, so it measures DRAM, not the SFPU.
_ABLATE = tuple(x.strip().upper() for x in os.environ.get("TT_BIO_TRIATT_ABLATE", "").split(",")
                if x.strip())
if _ABLATE:
    import warnings
    warnings.warn(f"TT_BIO_TRIATT_ABLATE={','.join(_ABLATE)}: triangle attention is computing "
                  "WRONG VALUES on purpose. Never set this outside a perf instrument.",
                  stacklevel=2)

_FIDELITY = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2,
             "HiFi4": ttnn.MathFidelity.HiFi4}


def ckc_from_env(spec=None):
    """`TT_BIO_TRIATT_SDPA_CKC=<LoFi|HiFi2|HiFi4>,<math_approx 0|1>,<fp32_dest_acc 0|1>`, or None.

    An A/B on this path has to flip the three knobs INDEPENDENTLY -- the op default bundles them and
    a previous pass could not say which one carried the damage. Unset means today's op default, so
    this is dead unless a leg asks for it.
    """
    spec = os.environ.get("TT_BIO_TRIATT_SDPA_CKC", "") if spec is None else spec
    if not spec:
        return None
    parts = [x.strip() for x in spec.split(",")]
    if len(parts) != 3 or parts[0] not in _FIDELITY:
        raise SystemExit(f"TT_BIO_TRIATT_SDPA_CKC={spec!r}: want "
                         f"<{'|'.join(_FIDELITY)}>,<0|1>,<0|1>")
    return (_FIDELITY[parts[0]], parts[1] == "1", parts[2] == "1", False)


_CKC_OVERRIDE = ckc_from_env()


def q_parallel_factor(S: int, H: int, q_chunk: int, cores: int, cap: int = -1) -> int:
    """The q-chunk split `fill_preconditions` needs, or 1 when there is none.

    The hoisted fill wants one q chunk per core, and the factory's own choice of `q_pf = 1` hands
    every core all of them -- so above the size where the widest q_chunk still spans the whole
    sequence, `q_per_core > 1` and this kernel declines every call. Pure, so
    `perf/bgsdpa/fused_reach.py` can ask which sizes it reaches without a device.

    `cap` defaults to the shipped `_Q_SPLIT_MAX_S`. `_tri_att_sdpa_at`'s above-cap route passes
    `cap=0` for the uncapped answer: the cap is what keeps the STOCK ladder off the fused kernel
    above 1024, and that route is the one thing allowed past it.
    """
    if cap < 0:
        cap = _Q_SPLIT_MAX_S
    if not _Q_SPLIT or (cap and S > cap):
        return 1
    qnc = -(-S // q_chunk)
    return qnc if qnc > 1 and cores // (H * qnc) >= 1 else 1


def per_core_cost(p, q_chunk: int, seq: int) -> int:
    """What one core pays for this (q_chunk, k_chunk), in tensor elements, ranked not absolute.

    Three terms, and only the first is fixed by the shape:

      * the two matmuls, `work * 2 * seq` where `work = batch_per_core * q_chunk` is the q rows
        this core owns. Constant across the surface only when the split lights up the same number
        of cores; it is the term that punishes a q_chunk whose `q_pf` leaves cores idle.
      * the online-softmax rescale, `work * (k_num_chunks - 1)`: the accumulator is rescaled once
        per k chunk after the first, so one k chunk pays nothing and a 32-wide k pays `seq/32`.
      * K and V, `batch_per_core * 2 * seq`: the whole of both is re-read from DRAM once per
        (batch row, q chunk) a core owns, so a wider q_chunk reads them fewer times.

    Everything is divided through by head_dim, which multiplies all three. Ranked against the
    measured surface in `fused_pairs`; its top pick is the measured optimum at padded 1920 and
    2208 and 4.4 % off it at 1536.
    """
    return (p["batch_per_core"] * q_chunk * (2 * seq + p["k_num_chunks"] - 1)
            + p["batch_per_core"] * 2 * seq)


@lru_cache(maxsize=None)
def fused_pairs(seq: int, heads: int, head_dim: int, cores: int, mask_dtype=None) -> tuple:
    """(q_chunk, k_chunk) pairs this kernel can serve at padded length `seq`, best first.

    Its two constraints pull in opposite directions, and neither is on the stock op's ladder:

      * `fill_preconditions` wants one q chunk per core, which `q_parallel_factor` supplies only
        while `seq // q_chunk <= cores // heads`.
      * the persistent mask CB holds `k_num_chunks * Sq_chunk_t * Sk_chunk_t` tiles, which works
        out to `seq * q_chunk / 1024` and carries NO k_chunk term. So a wide q is what breaks L1;
        a wide k costs only the k and v CBs, 4 tiles per k tile-row.

    So the pair is a narrow q against a wide k, which is exactly what the stock ladder never
    offers. Ordered by `per_core_cost`, which is where widest-k-first stops being the answer.

    K5 measured widest-k-first at padded 864 and it was right there: every candidate had the same
    core utilisation, so only the rescale count separated them. Above 1024 the candidates no
    longer tie, and the widest k forces a q narrow enough to cost more cores than the rescale
    saves. MEASURED interleaved against the incumbent ladder at padded 1536 / 1920 / 2208
    (`perf/ttx_a3/ab1536_qb2c2.json` on qb2 card 2 p300c 11x10, `perf/bgsdpa/ab1920.json` and
    `ab2208.json`): widest-k-first picks 2.061x / 1.518x / 1.865x, this order picks
    2.737x / 2.678x / 1.865x, and the surface's own best over all 47 / 44 / 3 configs that fit is
    2.862x / 2.678x / 1.865x. Exact at two of the three, 4.4 % off at 1536.

    Empty when nothing fits, which is the answer at 1184, 1312, 1856 and every other padded
    length whose only 32-aligned divisors are 32 and itself.
    """
    out = []
    for kc in SG.chunk_divisors(seq):
        for qc in SG.chunk_divisors(seq):
            q_pf = q_parallel_factor(seq, heads, qc, cores, cap=0)
            # `plan` reads the grid only as a core count here, and its split assert is against
            # that count -- so the grid has to carry the caller's `cores`, not the module default.
            # On a 13x10 p150a (130 cores) the default (11, 10) made every candidate assert
            # instead of answering, so `TT_BIO_SDPA_FUSED_LARGE_S=1` raised AssertionError out of
            # the fold rather than falling through to the stock ladder.
            p = SG.plan_for_shape(seq, heads, head_dim, qc, kc, grid=(cores, 1), split=(
                max(cores // (heads * q_pf), 1), heads, q_pf), dtype=mask_dtype)
            if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
                continue
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            if SG.cb_fits_l1(p, mask_cb_tiles=pers,
                             **({} if mask_dtype is None else {"mask_dtype": mask_dtype})):
                out.append((per_core_cost(p, qc, seq), qc, kc))
    return tuple((qc, kc) for _c, qc, kc in sorted(out))


def _reject(reason, shape):
    key = (reason, tuple(shape))
    REJECTS[key] = REJECTS.get(key, 0) + 1
    STATS[1] += 1
    return None


def sdpa(q, k, v, bias, scale, q_chunk, k_chunk, ckc_default=None, kv_buffer_factor=2,
         q_split_cap: int = -1):
    """The fold's SDPA with the mask read once per head, or `None` to leave the call alone.

    `q_split_cap=0` lifts `_Q_SPLIT_MAX_S` for this call. Only `_tri_att_sdpa_at`'s above-cap
    route passes it, and it passes a pair `fused_pairs` already priced against the same L1 model.
    """
    if not _ENABLED or bias is None:
        return None
    shape = [int(d) for d in q.shape]
    if len(shape) != 4 or len(bias.shape) != 4:
        return _reject("rank", shape)
    if any(t.dtype != ttnn.bfloat16 for t in (q, k, v, bias)):
        return _reject("dtype", shape)
    if any(t.layout != ttnn.TILE_LAYOUT for t in (q, k, v, bias)):
        return _reject("layout", shape)
    for t in (q, k, v, bias):
        mc = t.memory_config()
        if (mc.buffer_type != ttnn.BufferType.DRAM
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return _reject("memory_config", shape)

    from .tenstorrent import COMPUTE_GRID_MAIN, _SDPA_Q_CHUNK_OVER_L1
    grid = tuple(COMPUTE_GRID_MAIN)
    l1_key = (int(q.shape[2]), int(k.shape[2]), q_chunk)
    # `_PM_OVER_L1` is keyed on the FULL config, not on `l1_key`. This kernel's L1 cost moves with
    # k_chunk and with the k/v buffer factor as well as with q_chunk -- the wide-k ladder in
    # `_tri_att_sdpa_at` calls here with several k_chunks at one q_chunk -- so a three-term key lets
    # one refusal at (q, wide k) retire that q_chunk against every k the ladder still has to try.
    # Same all-or-nothing retirement as `rf3-latching-l1-gate-all-or-nothing-retirement`: a refusal
    # must narrow the shape class it retires, not the whole class.
    pm_key = (l1_key[0], l1_key[1], q_chunk, k_chunk, kv_buffer_factor)
    if l1_key in _SDPA_Q_CHUNK_OVER_L1:
        return _reject("q_chunk_over_l1", shape)
    if pm_key in _PM_OVER_L1:
        return _reject("pm_over_l1", shape)
    H = shape[1]
    cores = grid[0] * grid[1]
    if cores // H < 1:
        return _reject("grid_too_small", shape)
    # One q chunk per core is a precondition of the hoisted fill, and `q_pf = 1` hands a core every
    # q chunk there is. That is free while the widest q_chunk spans the sequence, which is true up to
    # 512 aa and false above it -- L1 refuses a full-S chunk at 768, the ladder drops to S/2, and the
    # gate then declines the whole fold (0 of 2424 calls served at 768, 0 of 2528 at 1024, all
    # `fill_preconditions`, all on this one term). Give the q chunks their own factor instead. Tiles
    # per core are unchanged: the batch factor shrinks by exactly the amount the q factor grows.
    q_pf = q_parallel_factor(shape[2], H, q_chunk, cores, cap=q_split_cap)
    split = (cores // (H * q_pf), H, q_pf)

    dev = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    # The op's own default compute kernel config, not the trunk's -- see perf/triatt_fused/s4_gate.py
    ckc = ckc_default or _CKC_OVERRIDE or (ttnn.MathFidelity.HiFi2, True, False, False)

    p = SG.plan(q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, scale, split)
    # everything the hoisted fill assumes
    if not (p["nh_per_core"] == 1 and p["q_per_core"] == 1 and p["bcast_batch"]
            and not p["use_padded_mask"] and p["NKH"] == H and p["NVH"] == H):
        ttnn.deallocate(out)
        return _reject("fill_preconditions", shape)

    persistent = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    try:
        SG.sdpa(dev, q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, scale, split=split,
                kernel_dir=KERNEL_DIR, mask_cb_tiles=persistent,
                kv_buffer_factor=kv_buffer_factor,
                defines_extra={"PERSISTENT_MASK": p["k_num_chunks"],
                               **{f"ABLATE_{a}": 1 for a in _ABLATE}})
    except Exception as exc:  # noqa: BLE001 -- an L1 refusal must reach the stock op, not the caller
        ttnn.deallocate(out)
        if "circular buffers" not in str(exc):
            raise
        # Remember it here only, so the next call declines instead of re-throwing while the stock
        # ladder keeps the q_chunk it fits.
        _PM_OVER_L1.add(pm_key)
        PM_L1_ERRORS[pm_key] = str(exc)
        return _reject("l1_budget", shape)
    STATS[0] += 1
    return out


# ROOF Phase A: the qkv projection moved inside this kernel.
#
# The pair it replaces is `generic_minimal_matmul(x, w) -> q, k, v` followed by `sdpa(q, k, v)`.
# Folding deletes the projection program outright, and the SDPA's own read GROWS -- it reads
# x[b] (Sqt*Ct tiles) where it used to read q, k and v (3*Sqt*DHt tiles). At boltz-2's 512 aa
# triangle attention that is 201.3 -> 268.4 MB on the consumer against 268.4 MB deleted from the
# producer, so the pair moves 603.9 -> 402.7 MB. See `state/roof-qkv-sdpa-build.md`.
#
# Off by default. This is a release-gated arm: it changes which arithmetic a shipped call reaches.
_FUSE_QKV = env_flag("TT_BIO_TRIATT_FUSE_QKV", False)
FUSE_REJECTS: dict = {}


def _fuse_reject(reason, shape):
    FUSE_REJECTS[reason] = FUSE_REJECTS.get(reason, 0) + 1
    return None


def sdpa_fused_qkv(x, w, bias, scale, n_heads, head_dim, q_chunk, k_chunk, ckc_default=None,
                   x_buffer_factor=2, force=False):
    """Triangle attention from the PRE-projection pair tensor, or `None` to decline.

    `x` is `[B, S, C]` and `w` its `[C, 3*n_heads*head_dim]` qkv weight, laid out exactly as
    `TriangleAttention.qkv_weight` already lays it (q heads, then k heads, then v heads). Returns
    the attention output `[B, n_heads, S, head_dim]`; the caller still owns the gate and the output
    projection.
    """
    if not (_FUSE_QKV or force) or bias is None:
        return None
    if not _ENABLED:
        return _fuse_reject("persistent_mask_off", [])
    shape = [int(d) for d in x.shape]
    if len(shape) != 3 or len(bias.shape) != 4:
        return _fuse_reject("rank", shape)
    B, S, C = shape
    S_pad = int(x.padded_shape[-2])
    if C != n_heads * head_dim or head_dim != 32:
        # The reader's K read is a tile-ORDER transpose, which is the identity only at DHt == 1.
        return _fuse_reject("head_dim", shape)
    if any(t.dtype != ttnn.bfloat16 for t in (x, w, bias)):
        return _fuse_reject("dtype", shape)
    if any(t.layout != ttnn.TILE_LAYOUT for t in (x, w, bias)):
        return _fuse_reject("layout", shape)
    for t in (x, w, bias):
        mc = t.memory_config()
        if (mc.buffer_type != ttnn.BufferType.DRAM
                or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED):
            return _fuse_reject("memory_config", shape)

    from .tenstorrent import COMPUTE_GRID_MAIN
    grid = tuple(COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    if cores // n_heads < 1:
        return _fuse_reject("grid_too_small", shape)
    # One q chunk per core AND the whole sequence in that chunk: the fold contracts x[b] once and
    # makes q, k and v from it, so a split q would re-read x per chunk and put the byte delta back
    # on the wrong side. q_pf stays 1 here for that reason, unlike `sdpa` above.
    if q_chunk != S_pad:
        return _fuse_reject("q_chunk_not_full_S", shape)
    split = (cores // n_heads, n_heads, 1)

    dev = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, n_heads, S, head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG)
    ckc = ckc_default or _CKC_OVERRIDE or (ttnn.MathFidelity.HiFi2, True, False, False)
    p = SG.plan(out, out, out, bias, out, q_chunk, k_chunk, grid, ckc, scale, split)
    if not (p["nh_per_core"] == 1 and p["q_per_core"] == 1 and p["bcast_batch"]
            and not p["use_padded_mask"] and p["Skt"] == p["Sq_chunk_t"]):
        ttnn.deallocate(out)
        return _fuse_reject("fill_preconditions", shape)

    # cb_k_in and cb_v_in have to hold the whole sequence, not one chunk: one pass over the
    # resident x produces k and v for every k chunk at once.
    kvbf = p["k_num_chunks"]
    p = SG.plan(out, out, out, bias, out, q_chunk, k_chunk, grid, ckc, scale, split, kvbf)
    persistent = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    Ct = C // 32
    # Drop to a single x buffer rather than refuse. The double buffer is what lets the reader
    # fetch the next batch row while the compute kernel is still contracting this one.
    for xbf in (x_buffer_factor, 1):
        extra = SG.fuse_qkv_cbs(p, Ct, xbf, x.dtype)
        if SG.cb_fits_l1(p, mask_cb_tiles=persistent, extra_cbs=extra, mask_dtype=bias.dtype):
            break
    else:
        ttnn.deallocate(out)
        return _fuse_reject("l1_budget", shape)
    try:
        SG.sdpa(dev, out, out, out, bias, out, q_chunk, k_chunk, grid, ckc, scale, split=split,
                kernel_dir=KERNEL_DIR, mask_cb_tiles=persistent, kv_buffer_factor=kvbf,
                fuse_qkv=(x, w, xbf),
                defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]})
    except Exception as exc:  # noqa: BLE001
        ttnn.deallocate(out)
        if "circular buffers" not in str(exc):
            raise
        SG.note_l1_refusal(str(exc))
        PM_L1_ERRORS[("fuse", S, q_chunk, k_chunk)] = str(exc)
        return _fuse_reject("l1_budget_device", shape)
    return out
