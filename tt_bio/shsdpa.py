"""Small-head SDPA for Wormhole (spd-bet-shsdpa). Prototype, not wired into any model yet.

ESMC's language-model attention masks padded keys with an additive [B, 1, S, S] mask whose rows
are all the same: mask[b, :, j] = -inf for j >= len_b (`esmc._batch_tokens`). The stock op re-reads
that whole mask for every (batch, head), ~62 MB a call at the page batch, and adds it to every score
tile. Here the mask is a per-batch key length instead:

  * the mask is one [32, S] tile row per batch, read only for the key chunk holding the boundary;
  * every other key chunk skips the mask add;
  * key chunks wholly past len_b are not read or computed (exp(-inf) adds exactly 0 to the sum).

Driven through `sdpa_generic` with the `KEYPAD` edits in `tt_bio/kernels/shsdpa/`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

from . import sdpa_generic as SG

KERNEL_DIR = Path(__file__).parent / "kernels" / "shsdpa"
HIFI2 = (ttnn.MathFidelity.HiFi2, True, False, False)
_MASKS: dict = {}


def keypad_args(lens, k_chunk: int, k_num_chunks: int):
    """Per batch (key chunks to run, chunk that carries the boundary or k_num_chunks for none)."""
    out = []
    for n in lens:
        end = -(-n // k_chunk)
        out.append((end, end - 1 if end * k_chunk > n else k_num_chunks))
    return out


def _mask_row(dev, lens, S):
    key = (id(dev), tuple(lens), S)
    hit = _MASKS.get(key)
    if hit is None:
        m = torch.zeros(len(lens), 1, 32, S)
        for b, n in enumerate(lens):
            m[b, :, :, n:] = float("-inf")
        hit = _MASKS[key] = (dev, ttnn.from_torch(m.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                                   memory_config=ttnn.DRAM_MEMORY_CONFIG))
    return hit[1]


def sdpa(q, k, v, scale, key_lens, q_chunk=None, k_chunk=192, split=None, ckc=HIFI2, grid=None, defines=None):
    """softmax(q k^T * scale, keys j >= key_lens[b] masked) v for bf16 [B, H, S, D] DRAM tensors."""
    from . import tenstorrent as T
    B, H, S, D = (int(d) for d in q.shape)
    dev = q.device()
    grid = grid or tuple(T.COMPUTE_GRID_MAIN)
    q_chunk = q_chunk or -(-S // 32) * 32
    mask = _mask_row(dev, key_lens, S)
    k_num = -(-S // k_chunk)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([B, H, S, D]), q.dtype, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    return SG.sdpa(dev, q, k, v, mask, out, q_chunk, k_chunk, grid, ckc, scale, split=split,
                   kernel_dir=KERNEL_DIR, keypad=keypad_args(key_lens, k_chunk, k_num),
                   defines_extra=defines)
