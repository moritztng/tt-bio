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

The mask CB does not grow: the persistent form needs `k_num_chunks * Sq_chunk_t * Sk_chunk_t` tiles
and the stock one already allocates `Sq_chunk_t * Sk_chunk_t * 2` for double buffering, which is the
same 256 tiles whenever there are two k chunks.

The gate is narrow on purpose. It needs one head and one q chunk per core, a batch-broadcast mask,
no padded mask, and bf16 interleaved DRAM throughout; anything else falls through to the stock op.
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


# Compute kernel config for the fused SDPA when the caller does not pass one. None means the
# op default below. Set it to raise the fused path precision -- openfold3 triangle attention runs
# _fp32_softmax_attention instead of SDPA precisely because bf16 softmax costs it 0.108 plDDT, and
# the fused kernel already threads fp32_dest_acc through dst_size and every subblock, so the fp32
# reduction is a config and not a kernel edit. Inert by default: nothing reads it unless it is set.
_CKC_OVERRIDE = None


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
    if grid[0] * grid[1] // H < 1:
        return _reject("grid_too_small", shape)
    split = (grid[0] * grid[1] // H, H, 1)

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
        # Remember it, so the next call skips straight to the next q_chunk instead of re-throwing.
        _SDPA_Q_CHUNK_OVER_L1.add(l1_key)
        return _reject("l1_budget", shape)
    STATS[0] += 1
    return out
