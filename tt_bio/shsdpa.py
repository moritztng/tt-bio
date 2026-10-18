"""Key-padding SDPA for the language models on Wormhole (`esmc.lm_sdpa` with a `KeyPad` mask).

ESMC's language-model attention masks padded keys with an additive [B, 1, S, S] mask whose rows
are all the same: mask[b, :, j] = -inf for j >= len_b (`esmc._batch_tokens`). The stock op re-reads
that whole mask for every (batch, head), ~62 MB a call at the page batch, and adds it to every score
tile. Here the mask is a per-batch key length instead:

  * the mask is one [32, S] tile row per batch, read only for the key chunk holding the boundary;
  * every other key chunk skips the mask add;
  * key chunks wholly past len_b are not read or computed (exp(-inf) adds exactly 0 to the sum).

The per-batch (key chunk count, boundary chunk) table is a small device tensor, not runtime args, so a
captured trace follows the lengths the caller writes into it. Driven through `sdpa_generic` with the
`KEYPAD` edits in `tt_bio/kernels/shsdpa/`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from . import sdpa_generic as SG
from .envflags import env_flag

KERNEL_DIR = Path(__file__).parent / "kernels" / "shsdpa"
HIFI2 = (ttnn.MathFidelity.HiFi2, True, False, False)
K_CHUNK = 192
# The key-padding mask is the same row for every query, so the reader fetches one row of tiles a k chunk and the compute
# adds it to each q row (KEYPAD_ROW), instead of fetching Sq_chunk copies. Same adds of the same values. perf/shsdpa
# bench, Galaxy chip at 1000 MHz, page length 514, ms a call: 300M B7H15 0.454 -> 0.361, 600M B7H18 0.550 -> 0.431,
# 6B B1H40 0.230 -> 0.166, rel_rms vs float64 unchanged to 4 digits.
_KEYPAD_ROW = env_flag("TT_BIO_SHSDPA_KEYPAD_ROW", True)
# 288-key chunks (9 tiles) instead of 192: with KEYPAD_ROW the same bench reads 300M 0.361 -> 0.348, 600M 0.431 -> 0.417,
# 6B 0.166 -> 0.162 ms a call, rel_rms vs float64 0.0269 -> 0.0264. Changes the output (another softmax blocking), so it
# is graded on its own; off until then.
_K_CHUNK_WIDE = env_flag("TT_BIO_SHSDPA_K288", False)


def k_chunk_default() -> int:
    return 288 if _K_CHUNK_WIDE else K_CHUNK


def keypad_args(lens, k_chunk: int, k_num_chunks: int):
    """Per batch (key chunks to run, chunk that carries the boundary or k_num_chunks for none)."""
    out = []
    for n in lens:
        end = -(-n // k_chunk)
        out.append((end, end - 1 if end * k_chunk > n else k_num_chunks))
    return out


def keypad_table(lens, k_chunk: int, k_num_chunks: int) -> torch.Tensor:
    """keypad_args as the [1, 1, 1, W] int32 page the kernel reads (W a multiple of 8: 32-byte DRAM reads)."""
    flat = [x for pair in keypad_args(lens, k_chunk, k_num_chunks) for x in pair]
    w = -(-len(flat) // 8) * 8
    return torch.tensor(flat + [0] * (w - len(flat)), dtype=torch.int32).reshape(1, 1, 1, w)


def key_pad(lens, S: int, device=None, k_chunk: int | None = None):
    """(table, row) for key lengths `lens` at padded length S: the [1, 1, 1, W] uint32 chunk table and the
    [B, 1, 32, S] bf16 mask row. Host tensors without `device`; a traced forward keeps one device pair and
    rewrites it per call with `copy_host_to_device_tensor`."""
    k_chunk = k_chunk or k_chunk_default()
    m = torch.zeros(len(lens), 1, 32, S)
    for b, n in enumerate(lens):
        m[b, :, :, n:] = float("-inf")
    on = {"device": device, "memory_config": ttnn.DRAM_MEMORY_CONFIG} if device is not None else {}
    return (ttnn.from_torch(keypad_table(lens, k_chunk, -(-S // k_chunk)), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, **on),
            ttnn.from_torch(m.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **on))


# A unit is one (batch, head, q chunk). Device time on WH (B7 H15 S544 D64, perf/shsdpa a1/t1) tracks
# units-per-busiest-core x (q tiles per unit + 2): the 2 is the per-unit fixed cost a small q chunk pays.
UNIT_COST_TILES = 2


def plan(B, H, S, cores, k_chunk=None, max_qt=None):
    """(q_chunk, (batch_pf, head_pf, q_pf)) minimising the busiest core's cost, padding included, q chunks of at most
    `max_qt` tiles."""
    k_chunk = k_chunk or k_chunk_default()
    St = -(-S // 32)
    best = None
    for qt in range(1, min(St + 1, max_qt or St + 1) + 1):
        nq = -(-St // qt)
        if nq * qt - St > 1:        # at most one padded q tile
            continue
        for bpf in range(1, B + 1):
            for hpf in range(1, H + 1):
                qpf = min(nq, cores // (bpf * hpf))
                if qpf < 1:
                    break
                units = -(-B // bpf) * -(-H // hpf) * -(-nq // qpf)
                cost = (units * (qt + UNIT_COST_TILES), bpf * hpf * qpf)
                if best is None or cost < best[0]:
                    best = (cost, qt * 32, (bpf, hpf, qpf))
    return best[1], best[2]


def sdpa(q, k, v, scale, pad, q_chunk=None, k_chunk=None, split=None, ckc=HIFI2, grid=None, defines=None,
         merged=False):
    """softmax(q k^T * scale, padded keys masked) v for bf16 [B, H, S, D] interleaved tensors (DRAM or L1); `pad` is the device
    `key_pad` pair built at the same k_chunk. `merged` writes the heads side by side, [B, 1, S, H * D], as
    nlp_concat_heads would (bit-identical)."""
    from . import tenstorrent as T
    k_chunk = k_chunk or k_chunk_default()
    B, H, S, D = (int(d) for d in q.shape)
    dev = q.device()
    grid = grid or tuple(T.COMPUTE_GRID_MAIN)
    auto = q_chunk is None and split is None
    if _KEYPAD_ROW:
        defines = {"KEYPAD_ROW": 1, **(defines or {})}
    key = (B, H, S, D, grid, tuple(t.memory_config().buffer_type == ttnn.BufferType.L1 for t in (q, k, v)))
    table, row = pad
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, 1, S, H * D] if merged else [B, H, S, D]), q.dtype,
                                         ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    while True:
        if auto:
            q_chunk, split = plan(B, H, S, grid[0] * grid[1], k_chunk, _Q_TILES_CAP.get(key))
        q_chunk = q_chunk or -(-S // 32) * 32
        try:
            return SG.sdpa(dev, q, k, v, row, out, q_chunk, k_chunk, grid, ckc, scale, split=split,
                           kernel_dir=KERNEL_DIR, keypad=table, defines_extra=defines,
                           merged_out=merged)
        except Exception as exc:
            if not auto or q_chunk == 32:
                raise
            T.absorb_l1_refusal("shsdpa q chunk", exc)
            _Q_TILES_CAP[key] = q_chunk // 32 - 1


# Largest q chunk (tiles) per shape and operand placement that the device accepted. L1 operands (SaProt's RoPE leaves
# q, k there) sit at the top of L1, below which the circular buffers must fit: the 544-row chunk the plan picks for
# 7 x 20 heads fell 61 KB short, so a refused plan steps down to the next smaller q chunk for that shape from then on.
_Q_TILES_CAP: dict = {}
