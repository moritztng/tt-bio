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
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import sdpa_generic as SG
from .envflags import env_flag

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
_Q_SPLIT_MAX_S = 1024

# q_chunks whose PERSISTENT mask CB does not fit. Deliberately not `_SDPA_Q_CHUNK_OVER_L1`: that set
# is the wide-q ladder memo of q_chunks the STOCK op cannot fit, and `_tri_att_sdpa_at` filters its
# candidate list with it. This kernel allocates a strictly larger mask CB -- `k_num_chunks *
# Sq_chunk_t * Sk_chunk_t` tiles against the stock `2 * Sq_chunk_t * Sk_chunk_t` -- so a refusal here
# says nothing about what the stock op fits. Writing it into the shared set retires a q_chunk the
# stock op runs perfectly well, and the fold loses the wide-q win (1.08-1.81x) on every later call at
# that shape. MEASURED at 512 aa: the 995-token refiner fell from q_chunk 512 to 256 after one such
# throw and the fold lost 3.129 s against an A/A floor of 0.056 s.
_PM_OVER_L1: set = set()


# Compute kernel config for the fused SDPA when the caller does not pass one. None means the op
# default below, which is what every call took until RF3's triangle attention started passing its
# own: `(HiFi2, approx, no fp32_dest_acc)` is a LOW-PRECISION config, and reading the fused path as
# "bf16 softmax" conflated the storage with it.
#
# MEASURED on one captured RF3 triangle-attention call, all thirteen arms against an fp64 evaluation
# of the SAME bf16 operands, so only the kernel's own error is left (perf/rf3/triatt_fused_fp32.py,
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


def _reject(reason, shape):
    key = (reason, tuple(shape))
    REJECTS[key] = REJECTS.get(key, 0) + 1
    STATS[1] += 1
    return None


def sdpa(q, k, v, bias, scale, q_chunk, k_chunk, ckc_default=None):
    """The fold's SDPA with the mask read once per head, or `None` to leave the call alone."""
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
    if l1_key in _SDPA_Q_CHUNK_OVER_L1:
        return _reject("q_chunk_over_l1", shape)
    if l1_key in _PM_OVER_L1:
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
    q_pf = 1
    if _Q_SPLIT and shape[2] <= _Q_SPLIT_MAX_S:
        qnc = -(-shape[2] // q_chunk)
        if qnc > 1 and cores // (H * qnc) >= 1:
            q_pf = qnc
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
                defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]})
    except Exception as exc:  # noqa: BLE001 -- an L1 refusal must reach the stock op, not the caller
        ttnn.deallocate(out)
        if "circular buffers" not in str(exc):
            raise
        # Remember it here only, so the next call declines instead of re-throwing while the stock
        # ladder keeps the q_chunk it fits.
        _PM_OVER_L1.add(l1_key)
        return _reject("l1_budget", shape)
    STATS[0] += 1
    return out
