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

The gate is narrow on purpose. It needs one head per core, a batch-broadcast mask, no padded mask,
and bf16 interleaved DRAM throughout; anything else falls through to the stock op. A core may own
several q chunks -- `_q_split` picks how many and the CB holds one mask block per (q chunk, k chunk)
pair it owns.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from . import sdpa_generic as SG

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
_Q_SPLIT = os.environ.get("TT_BIO_TRIATT_MASK_Q_SPLIT", "1") == "1"
_Q_SPLIT_MAX_S = 1024

# `(Sq, Sk, q_chunk, q_per_core)` whose PERSISTENT mask CB does not fit. Deliberately not
# `_SDPA_Q_CHUNK_OVER_L1`: that set is the wide-q ladder memo of q_chunks the STOCK op cannot fit,
# and `_tri_att_sdpa_at` filters its candidate list with it. This kernel allocates a strictly larger
# mask CB -- `q_per_core * k_num_chunks * Sq_chunk_t * Sk_chunk_t` tiles against the stock
# `2 * Sq_chunk_t * Sk_chunk_t` -- so a refusal here says nothing about what the stock op fits.
# `q_per_core` is in the key because the widest work split is tried first and a refusal there must
# fall back to a narrower one, not retire the shape. Writing it into the shared set retires a q_chunk the
# stock op runs perfectly well, and the fold loses the wide-q win (1.08-1.81x) on every later call at
# that shape. MEASURED at 512 aa: the 995-token refiner fell from q_chunk 512 to 256 after one such
# throw and the fold lost 3.129 s against an A/A floor of 0.056 s.
_PM_OVER_L1: set = set()


# Compute kernel config for the fused SDPA when the caller does not pass one. None means the
# op default below. Set it to raise the fused path precision -- openfold3 triangle attention runs
# _fp32_softmax_attention instead of SDPA precisely because bf16 softmax costs it 0.108 plDDT, and
# the fused kernel already threads fp32_dest_acc through dst_size and every subblock, so the fp32
# reduction is a config and not a kernel edit. Inert by default: nothing reads it unless it is set.
_CKC_OVERRIDE = None


# Let a core own more than one q chunk when the one-chunk-per-core split leaves the grid idle.
# "0" forces the pre-lever split back. Default set from the fold A/B below.
_Q_PER_CORE = os.environ.get("TT_BIO_TRIATT_MASK_Q_PER_CORE", "0") == "1"


def _q_split(qnc, H, cores, B):
    """`(q_pf, q_per_core)` candidates, cheapest makespan first, in q-chunk units.

    A core serves one (head, q chunk) pair per batch row, so the split granularity is `H * q_pf`
    cores and the makespan is `ceil(B / (cores // (H * q_pf))) * q_per_core` q-chunk units. Giving
    every q chunk its own factor (`q_per_core = 1`) is the widest split and usually the best, but
    `cores // (H * q_pf)` floors, and where it floors hard the grid goes idle: at 1024 aa on a
    130-core grid, `H = 12` and `qnc = 4` make the granularity 48 cores, so 96 cores get work and
    each owns 512 rows. Two q chunks per core drop the granularity to 24, use 120 cores at 205 rows,
    and 205 * 2 = 410 < 512.

    VALIDITY: searched per call, never gated on a size, because it is arithmetic and not a tuned
    constant. It can only bite where `cores // (H * qnc)` wastes a slice of the grid, and ties go to
    the smallest `q_per_core`, i.e. to the pre-lever split. So: `qnc = 1` (up to 640 aa here, and up
    to 512 aa on any grid) has no `c > 1` at all; `qnc = 2` at 130 cores (768 aa) ties at 154 units
    and keeps `c = 1`. 1024 aa on 130 cores is the only shipped point it moves. Other grids are
    covered by the ranking rather than by a threshold: on 110 cores it does prefer `c = 4` at
    1024 aa, whose mask CB refuses L1, and the caller then walks down this list instead of retiring
    the shape.
    """
    if not _Q_PER_CORE:
        return [(qnc, 1)]
    cand = []
    for c in range(1, qnc + 1):
        if qnc % c:
            continue
        pf = qnc // c
        b_pf = cores // (H * pf)
        if b_pf < 1:
            continue
        cand.append((-(-B // b_pf) * c, c, pf))
    cand.sort()
    return [(pf, c) for _, c, pf in cand]


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
    H = shape[1]
    cores = grid[0] * grid[1]
    if cores // H < 1:
        return _reject("grid_too_small", shape)
    # `q_pf = 1` hands a core every q chunk there is. That is free while the widest q_chunk spans
    # the sequence, which is true up to 512 aa and false above it -- L1 refuses a full-S chunk at
    # 768, the ladder drops to S/2, and the gate then declined the whole fold (0 of 2424 calls
    # served at 768, 0 of 2528 at 1024, all `fill_preconditions`, all on this one term). So the q
    # chunks get their own factor. How many of them one core owns is then a makespan search
    # (`_q_split`), widest split first, narrowing on an L1 refusal.
    cands = [(1, 1)]
    if _Q_SPLIT and shape[2] <= _Q_SPLIT_MAX_S:
        qnc = -(-shape[2] // q_chunk)
        if qnc > 1 and cores // (H * qnc) >= 1:
            cands = _q_split(qnc, H, cores, shape[0])
    cands = [c for c in cands if l1_key + (c[1],) not in _PM_OVER_L1]
    if not cands:
        return _reject("pm_over_l1", shape)

    dev = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    # The op's own default compute kernel config, not the trunk's -- see perf/triatt_fused/s4_gate.py
    ckc = ckc_default or _CKC_OVERRIDE or (ttnn.MathFidelity.HiFi2, True, False, False)

    for q_pf, q_per_core in cands:
        split = (cores // (H * q_pf), H, q_pf)
        p = SG.plan(q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, scale, split)
        # everything the hoisted fill assumes
        if not (p["nh_per_core"] == 1 and p["q_per_core"] == q_per_core and p["bcast_batch"]
                and not p["use_padded_mask"] and p["NKH"] == H and p["NVH"] == H):
            ttnn.deallocate(out)
            return _reject("fill_preconditions", shape)

        persistent = p["q_per_core"] * p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
        try:
            SG.sdpa(dev, q, k, v, bias, out, q_chunk, k_chunk, grid, ckc, scale, split=split,
                    kernel_dir=KERNEL_DIR, mask_cb_tiles=persistent,
                    defines_extra={"PERSISTENT_MASK": p["k_num_chunks"]})
        except Exception as exc:  # noqa: BLE001 -- an L1 refusal must reach the stock op
            if "circular buffers" not in str(exc):
                ttnn.deallocate(out)
                raise
            # Remember it here only, so the next call goes straight to the next-narrowest split
            # while the stock ladder keeps the q_chunk it fits.
            _PM_OVER_L1.add(l1_key + (q_per_core,))
            continue
        STATS[0] += 1
        return out
    ttnn.deallocate(out)
    return _reject("l1_budget", shape)
