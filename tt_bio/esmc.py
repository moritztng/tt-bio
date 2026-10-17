"""ESMC protein language model on Tenstorrent (ttnn).

A from-scratch ttnn implementation of EvolutionaryScale / Biohub's ESMC
(Evolutionary Scale Modeling Cambrian) sequence-only protein language model,
built on the tt-bio ttnn framework (``tenstorrent.Module`` / ``WeightScope`` /
``get_device``). We start with the smallest variant, ESMC-300M.

Reference (PyTorch): ``/home/ttuser/esm`` — esm/models/esmc.py, esm/layers/*.
The reference forward (use_flash_attn=False) is:

    x = embed(tokens)                       # [B, L, d_model]
    x = transformer(x)                      # 30 x UnifiedTransformerBlock + final LayerNorm
    logits = sequence_head(x)               # [B, L, 64]

Built bottom-up, one tested component at a time. This module currently
implements: token embedding.
"""

from __future__ import annotations

import collections
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tt_bio.envflags import env_flag, env_int

import numpy as np
import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    TorchWrapper,
    Weights,
    WeightScope,
    _dtype,
    _PAIR_FFN_FC1_BLOCK_W,
    _PAIR_FFN_FC1_BW,
    _capped_sdpa_chunk_size,
    _pair_proj_linear,
    _sdpa_program_config,
    fused_sdpa,
    get_device,
    trace_region_size,
)

import time as _time

_TIMING = os.environ.get("TT_BIO_TIMING")

def _tlog(msg):
    if not _TIMING:
        return
    line = f"[timing pid={os.getpid()} t={_time.perf_counter():.2f}] {msg}"
    if "/" in _TIMING:
        with open(_TIMING, "a") as _f:
            _f.write(line + "\n")
    else:
        print(line, file=sys.stderr, flush=True)


VOCAB_SIZE = 64
ROPE_BASE = 10000.0

# Sequence vocab (esm.utils.constants.esm3.SEQUENCE_VOCAB): token id = index here.
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T",
    "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U",
    "Z", "O", ".", "-", "|", "<mask>",
]
BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN = 0, 2, 3, 32
PAD_TOKEN = 1  # SEQUENCE_VOCAB index of <pad>
from .token_axis import bucket_multiple as _bucket_multiple

# pad the LM length to a multiple of this to avoid per-length recompilation; fleet value.
BUCKET = _bucket_multiple("esmc-300m")
# Per-batch token budget (rows x bucketed length) for the batched embed path:
# short sequences pack a full batch_size, long ones shrink the batch toward 1 so
# a mixed FASTA never OOMs. Scaled by batch_size so raising the knob raises headroom.
_MAX_BATCH_TOKENS_PER_SEQ = 512
_AA_TO_ID = {a: i for i, a in enumerate(SEQUENCE_VOCAB)}

# name -> (config, hf repo id, weights path within repo). Both ship as a single
# esm-repo-format .pth (identical key layout, just wider/deeper), so one loader
# covers them; the 6B is a separate sharded-safetensors path (see below).
CONFIGS = {
    "esmc-300m": (
        dict(d_model=960, n_heads=15, n_layers=30),
        "biohub/esmc-300m-2024-12",
        "data/weights/esmc_300m_2024_12_v0.pth",
    ),
    "esmc-600m": (
        dict(d_model=1152, n_heads=18, n_layers=36),
        "biohub/esmc-600m-2024-12",
        "data/weights/esmc_600m_2024_12_v0.pth",
    ),
}

# Architecture configs for the larger variants. ESMC-6B is the LM backbone of
# ESMFold2; the ttnn ESMC architecture supports it via config (identical to
# 300M, just larger), validated by the 300M parity. Real-weight loading for 6B
# needs a sharded-safetensors + key-remap loader (transformers format, ~12GB)
# and block-fp8 to fit one Blackhole — separate from the single-.pth 300M path.
ARCH_CONFIGS = {
    "esmc-300m": dict(d_model=960, n_heads=15, n_layers=30),
    "esmc-600m": dict(d_model=1152, n_heads=18, n_layers=36),
    "esmc-6b": dict(d_model=2560, n_heads=40, n_layers=80),  # ESMFold2 LM backbone
}


def tokenize(sequence: str) -> "torch.Tensor":
    """Protein string -> token ids [1, L+2] with <cls>/<eos> (matches esm)."""
    ids = [BOS_TOKEN] + [_AA_TO_ID.get(c, UNK_TOKEN) for c in sequence.upper()] + [EOS_TOKEN]
    return torch.tensor([ids], dtype=torch.long)


def rope_tables(seq_len: int, head_dim: int, base: float = ROPE_BASE, device=None):
    """Precompute NeoX-style RoPE cos/sin tables, shaped [1, 1, L, head_dim].

    Mirrors esm.layers.rotary.RotaryEmbedding (scale_base=None, interleaved=False):
    inv_freq = 1 / base**(arange(0,d,2)/d); freqs = outer(arange(L), inv_freq);
    cos/sin duplicated along the last dim ([c0..c_{d/2-1}, c0..c_{d/2-1}]).
    """
    device = device or get_device()
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [L, d/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).view(1, 1, seq_len, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).view(1, 1, seq_len, head_dim)
    to_tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return to_tt(cos), to_tt(sin)


def apply_rotary(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Apply RoPE to x [B, H, L, head_dim]; cos/sin broadcast as [1, 1, L, head_dim].

    out = x * cos + rotate_half(x) * sin, rotate_half(x) = cat([-x2, x1]).
    """
    x1, x2 = ttnn.chunk(x, 2, dim=-1)
    rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    out = ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    ttnn.deallocate(rot)
    return out


def _rope(q: ttnn.Tensor, k: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, l1: bool = True):
    """RoPE for per-head q, k [B, H, L, head_dim].

    When L is tile-aligned (the bucketed LM path — ``BUCKET`` is 32, and the 6B
    backbone always pads to it) the fused ``ttnn.experimental.rotary_embedding``
    kernel replaces ``apply_rotary``'s six-op rotate-half pile with one dispatch
    per tensor. This is the largest single share of ESMC attention (a dispatch-
    bound elementwise stack, not a matmul), so collapsing it is a real per-layer
    win, largest on the smaller models. Matches the reference within bf16 noise;
    the ragged fallback keeps arbitrary single-sequence lengths exact.
    """
    if q.shape[2] % 32 == 0:
        key = tuple(q.padded_shape)
        if _ROPE_L1 and l1 and key not in _ROPE_L1_REFUSED:
            l1 = ttnn.L1_MEMORY_CONFIG
            out = []
            try:
                out.append(ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=l1))
                out.append(ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=l1))
            except Exception as exc:
                if not _is_alloc_refusal(exc):
                    from tt_bio import tenstorrent
                    tenstorrent.absorb_l1_refusal("esmc rope L1", exc)
                _ROPE_L1_REFUSED.add(key)
                for t in out:
                    ttnn.deallocate(t)
            else:
                ROPE_L1_STATS[0] += 1
                return out[0], out[1]
        ROPE_L1_STATS[1] += _ROPE_L1 and l1
        return (ttnn.experimental.rotary_embedding(q, cos, sin),
                ttnn.experimental.rotary_embedding(k, cos, sin))
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


# The rotated q and k written to L1: SDPA reads them straight back. Same kernel, same values (a memory
# placement), refusal falls back to DRAM for that shape for good. p150a page cell (esmc_ab.py, qb1 card 2, 1350 MHz,
# 2 rounds each, output identical to the DRAM path), batch of 7: 300M 81.1 -> 79.5 ms, 300M --fast 87.4 -> 81.1,
# 600M 139.9 -> 135.1, 600M --fast 119.2 -> 114.9. Same model gate as QK_L1 (Attention.qk_l1): the 6B is not measured.
_ROPE_L1 = env_flag("TT_BIO_ESMC_ROPE_L1", True)
_ROPE_L1_REFUSED: set = set()
#: [rope calls served from L1, declined], read by the A/B harness.
ROPE_L1_STATS = [0, 0]


# The protein LMs' (ESMC, SaProt) SDPA chunks. The shared cap (256 x 256) was 1.285 ms a call at the page shape (batch 7 x 18 heads x 544
# tokens x d 64, p150a, 1350 MHz); q 128 x k 64 is 0.629 ms, the fastest of the 25 pairs swept, at the same error to a
# float64 reference (PCC 0.99970 vs 0.99969, perf/wh-embed/sdpa_sweep.py). Chunking only reorders the online softmax.
ESMC_SDPA_Q_CHUNK = 128
# q, k and v as three projections from the packed weight's column blocks instead of one projection and a
# 3-way `ttnn.chunk` copy of its output. Embeddings identical to the chunk path (same vs_fp32 to every digit). Page
# cell, batch of 7, perf/spd_gap_esmf/esmc_ab.py: Galaxy chip (1000 MHz) 300M 176.9 -> 172.4 ms, --fast 168.2 ->
# 161.2, 600M 254.5 -> 252.0, --fast 245.1 -> 242.2; p150a (1350 MHz) 300M 74.5 -> 70.9. The slices are a second
# copy of the qkv weight, so the 6B (Attention.qk_l1 off, 3 GB of it) keeps the chunk.
_ESMC_QKV_SPLIT = env_flag("TT_BIO_ESMC_QKV_SPLIT", True)
# The attention's input layer norm, the q/k layer norms and the qkv re-pack written to L1 instead of
# DRAM: the qkv projection and the head split read them straight back, so DRAM never carries them. Same ops, same values (a memory
# placement). An L1 refusal at a shape falls back to DRAM for that shape for good, decided in the
# warm passes before a trace capture. p150a page cell (esmc_ab.py, 2 rounds each, output identical to the DRAM path):
# 600M 136.0/122.4 -> 113.9/115.4 ms, 300M --fast 85.4/88.1 -> 79.8/80.4 ms per batch of 7. Inside ESMFold2 the
# ESMC-6B gains nothing (p300c page cell, identical digests: ESMFold2 +0.76 s on a 0.89 s spread, ESMFold2-Fast
# 16.48 -> 16.55 s), so the Block of a model past _LM_LARGE_MODEL_BYTES keeps the DRAM path (Attention.qk_l1).
_ESMC_QK_L1 = env_flag("TT_BIO_ESMC_QK_L1", True)

# Replay batched forwards from the trace too (ESMC.trace gated on B == 1). A 7 x 544 Blackhole batch spreads 110-127 ms
# within one process, the signature of a host-dispatch-bound forward. Same captured program, so output is bit-identical
# to eager. The first try (tb-esmc-600m) replayed garbage, but beside several live traces, the fault TRACE_CACHE_MAX=1
# fixed later; retried on one live trace (esmc_trace_check.py --batch). Off until measured.
_TRACE_BATCH = env_flag("TT_BIO_ESMC_TRACE_BATCH", False)
_QK_L1_REFUSED: set = set()
#: [attention calls served from L1, declined], read by the A/B harness.
QK_L1_STATS = [0, 0]
ESMC_SDPA_K_CHUNK = 64


# q in one chunk up to ESMC_SDPA_Q_WHOLE rows. The page bucket, 544 rows, is 17 tiles, a prime, so a 128-row chunk pads
# q to 640. Galaxy chip at 1000 MHz, batch 7 x 15 heads x 544 x d 64 (sdpa_sweep.py), ms per call: q128/k64 1.167 ->
# q544/k64 0.792, error against float64 unchanged (max abs 0.0467, pcc 0.99970 both); 6B shape (1 x 40 heads) 0.572 ->
# 0.413. Page cell (esmc_ab.py, 2 rounds, 1000 MHz), embeddings bit-identical to the 128-row chunks: 300M 161.7 -> 151.6 ms,
# --fast 160.1 -> 150.1.
_ESMC_SDPA_Q_WHOLE = env_flag("TT_BIO_ESMC_SDPA_Q_WHOLE", True)
ESMC_SDPA_Q_WHOLE = 544


# With q whole, keys in the fewest chunks that keep the padded key length minimal, up to 192 rows (q544 x k544 does not
# fit L1). At 544 keys that is 3 x 192 (576 padded, as 9 x 64): same chip and clock, ms per call, k64 -> k192: 300M shape
# 0.777 -> 0.600, SaProt 650M (b7 x 20 heads) 0.811 -> 0.722, 6B (b1 x 40) 0.413 -> 0.328; k128/k160 pad to 640 and are
# slower than k64. Error against float64 within the k64 spread (max abs 0.0467 -> 0.0467 / 0.0323 -> 0.0343 / 0.0349 ->
# 0.0356). Its circular buffers are larger, so where q, k and v sit in L1 (600M, SaProt) the device can refuse the
# program; lm_sdpa then runs that shape at the k64 config. Page cell on a Galaxy chip at 1000 MHz, 3 rounds, k64 -> few:
# 300M 149.7 -> 143.8 ms, --fast 144.2 -> 137.0, 6B 302.5 -> 297.2, --fast 293.9 -> 287.1; 600M and SaProt flat (several
# shapes fall back); pcc vs fp32 moves <= 0.00008. On by default on Wormhole; Blackhole keeps k64 until its own A/B says.
# None = by arch, the switch overrides.
_ESMC_SDPA_K_FEW = (env_flag("TT_BIO_ESMC_SDPA_K_FEW", False) if os.environ.get("TT_BIO_ESMC_SDPA_K_FEW", "").strip()
                    else None)
ESMC_SDPA_K_FEW_MAX = 192
_SDPA_K_FEW_REFUSED: set = set()


def _lm_sdpa_chunks(q_len: int, k_len: int, k_few: bool) -> tuple[int, int]:
    q_pad = -(-q_len // 32) * 32
    if _ESMC_SDPA_Q_WHOLE and q_pad <= ESMC_SDPA_Q_WHOLE:
        k_chunk = min(ESMC_SDPA_K_CHUNK, _capped_sdpa_chunk_size(k_len))
        k_pad = -(-k_len // 32) * 32
        if k_few and k_pad > ESMC_SDPA_K_CHUNK:
            k_chunk = min(range(ESMC_SDPA_K_CHUNK, min(ESMC_SDPA_K_FEW_MAX, k_pad) + 1, 32),
                          key=lambda c: (-(-k_pad // c) * c, -c))
        return q_pad, k_chunk
    return (min(ESMC_SDPA_Q_CHUNK, _capped_sdpa_chunk_size(q_len)),
            min(ESMC_SDPA_K_CHUNK, _capped_sdpa_chunk_size(k_len)))


def _sdpa_k_few() -> bool:
    if _ESMC_SDPA_K_FEW is not None:
        return _ESMC_SDPA_K_FEW
    from tt_bio import tenstorrent
    return tenstorrent.is_wormhole()


def lm_sdpa(q, k, v, attn_mask, scale: float):
    """The language models' fused SDPA at the _lm_sdpa_chunks config. A shape whose few-k-chunk program the device
    refuses (its circular buffers clash with L1 tensors) runs at the k64 config from then on."""
    q_len, k_len = int(q.shape[2]), int(k.shape[2])
    key = (tuple(q.padded_shape), k_len, q.memory_config().buffer_type == ttnn.BufferType.L1)
    base = _lm_sdpa_chunks(q_len, k_len, False)
    if _sdpa_k_few() and key not in _SDPA_K_FEW_REFUSED:
        few = _lm_sdpa_chunks(q_len, k_len, True)
        if few != base:
            try:
                return fused_sdpa(q, k, v, attn_mask=attn_mask, scale=scale, program_config=_sdpa_program_config(*few))
            except Exception as exc:
                from tt_bio import tenstorrent
                tenstorrent.absorb_l1_refusal("esmc sdpa k chunk", exc)
                _SDPA_K_FEW_REFUSED.add(key)
    return fused_sdpa(q, k, v, attn_mask=attn_mask, scale=scale, program_config=_sdpa_program_config(*base))


class Embedding(Module):
    """Token embedding lookup (mirrors nn.Embedding(64, d_model)).

    Weight key: ``<scope>.weight`` of shape [vocab, d_model] (no transpose).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        # Embedding table is indexed, not matmul'd: keep [vocab, d_model] as-is.
        self.weight = self.torch_to_tt("weight", transform=lambda x: x)

    def __call__(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        # tokens: ROW_MAJOR uint32 [B, L]; output [B, L, d_model] in TILE layout.
        return ttnn.embedding(
            tokens,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# ESMC's projections on `minimal_matmul` instead of `ttnn.linear(core_grid=...)`. WH Galaxy chip, page shape (batch
# 7 x 544 rows), perf/spd_gap_esmf/lin_bench.py against float64 of the same operands, equal or lower error:
# 600M fc2 0.719 -> 0.466 ms, out 0.350 -> 0.263, qkv 0.851 -> 0.710; 300M qkv 0.651 -> 0.537, fc2 0.485 -> 0.416.
# At the 6B's 544 rows it gains nothing (out 0.200 -> 0.242), so only a projection over ESMC_MM_MIN_ROWS takes it.
# Page cell on a Galaxy chip (esmc_ab.py, 2 rounds, batch of 7): 300M 177.8 -> 171.8 ms, 600M 272.0 -> 253.3 ms, with
# embeddings equal to the linear's. On by default on Wormhole;
# Blackhole keeps the linear until its own A/B says. None = by arch, the switch overrides.
_ESMC_MM = (env_flag("TT_BIO_ESMC_MM", False) if os.environ.get("TT_BIO_ESMC_MM", "").strip() else None)
ESMC_MM_MIN_ROWS = 2048
#: [projections served by minimal_matmul, declined], read by the A/B harness.
MM_STATS = [0, 0]
# minimal_matmul's block config for those projections: 16 x 4 output tiles a block, K in blocks of 4 tiles (2 where K
# does not divide). perf/spd_gap_whlm/mm_cfg_sweep.py on a Galaxy chip at 1000 MHz, 3808 rows, ms per call against the
# op's default: 300M proj 0.233 -> 0.197, fc1 half 0.495 -> 0.461, fc2 0.392 -> 0.382; 600M proj 0.268 -> 0.226, fc1
# half 0.645 -> 0.587, fc2 0.470 -> 0.452; error against float64 equal to the default's. Off = the op's default.
_ESMC_MM_CFG = env_flag("TT_BIO_ESMC_MM_CFG", True)
# --fast's block-fp8 projection weights on the same minimal_matmul instead of ttnn.linear. Page cell on a Galaxy chip at
# 1000 MHz, 2 rounds: 300M --fast 173.2/166.5 -> 162.8/157.0 ms, 600M 264.6/244.6 -> 237.5/237.0, embeddings equal to the
# linear's (pcc 1.0). The same 16 x 4 block config is within 3 % of the best per-shape one with these weights.
_ESMC_MM_BFP8 = env_flag("TT_BIO_ESMC_MM_BFP8", True)


# q, k and v from ONE projection over the packed weight, split by the matmul as it writes them
# (ttnn.experimental.minimal_matmul_split), instead of three projections over the host-cut weights. Holds the packed
# weight beside the cut ones (300M 166 MB, 600M 287 MB), which still serve batches under ESMC_MM_MIN_ROWS. Page cell on
# a Galaxy chip at 1000 MHz, 3 rounds, median ms: 300M 143.3 -> 138.9, --fast 142.6 -> 139.7, 600M 228.6 -> 224.5,
# --fast flat; embeddings identical (pcc 1.0 against the three projections). Rides on minimal_matmul, so Wormhole only.
_ESMC_QKV_MM_SPLIT = env_flag("TT_BIO_ESMC_QKV_MM_SPLIT", True)
# The block's two residual adds inside out_proj's and fc2's matmuls (dit_minimal_matmul_addcmul_fused: x + 1 * (o W) *
# ones), so neither add re-reads a projection from DRAM. bf16 weights only: the op refuses --fast's block-fp8 ones.
# Same page cell: 300M 143.6 -> 141.3, 600M 229.9 -> 223.4, embeddings identical (pcc 1.0). Wormhole only, as above.
_ESMC_MM_RESID = env_flag("TT_BIO_ESMC_MM_RESID", True)
#: [qkv split matmuls, residual-fused projections] served, read by the A/B harness.
MM_FUSE_STATS = [0, 0]
LIN6B_STATS = [0]  # projections served by _lin6b_program


# minimal_matmul accumulating in bf16 dest registers instead of fp32, which frees room for 2 x 4 output subblocks.
# mm_cfg_sweep (grid ma) on a Galaxy chip at 1000 MHz, best config each: 300M q/k/v/out 0.195 -> 0.178 ms, fc1 half
# 0.461 -> 0.425, fc2 0.382 -> 0.350; 600M 0.226 -> 0.208, 0.587 -> 0.544, 0.452 -> 0.417. Error against float64 per
# matmul rel_l2 0.0017 -> 0.007-0.009. Graded on the page cell (batch of 7, minimal_matmul forced on, p150a, 2 rounds):
# pcc vs fp32 300M 0.99967 -> 0.99956, --fast 0.99961 -> 0.99952; 600M 0.99972 -> 0.99962, --fast 0.99964 -> 0.99954;
# pooled cos >= 0.99996 everywhere. Inside the fast bar, not the normal one, so None = on for block-fp8 weights only,
# which only --fast loads. Galaxy chip at 1000 MHz, best of 3: 300M --fast 139.5 -> 133.6 ms, 600M 203.6 -> 197.4.
_ESMC_MM_BF16_ACC = env_flag("TT_BIO_ESMC_MM_BF16_ACC", None)
# The LM FFN's fc1 halves and SwiGLU product stored as block-fp8 where the weights already are (--fast), so the gate
# multiply and fc2 read half the bytes: at the 300M page shape that chain moves ~117 MB a layer through DRAM in bf16.
# Wormhole minimal_matmul path only (fc2 takes the bfp8 operand there). Off: an A/B candidate for --fast.
_ESMC_ACT_BFP8 = env_flag("TT_BIO_ESMC_ACT_BFP8", False)


# ESMC-6B's projections at up to 544 rows on Wormhole (the page shape is one 544-row sequence) on explicit multicast
# programs instead of ttnn.linear's own pick for the grid. lin6b_sweep on a Galaxy chip at 1000 MHz, ms a call at 544 rows
# (grid -> program): bf16 qkv 0.585 -> 0.556 and fc1 half 0.508 -> 0.455 (1D, N across every core), fc2 0.452 -> 0.431
# (2D); block-fp8 qkv 0.590 -> 0.474 and fc1 half 0.482 -> 0.415 (2D), fc2 and out_proj stay on the linear's program.
_ESMC_LIN6B_PROG = env_flag("TT_BIO_ESMC_LIN6B_PROG", False)
_LIN6B_PROG_MAX_ROWS = 544


def _subblock(h_tiles: int, w_tiles: int, cap: int = 4) -> tuple[int, int]:
    """Largest out subblock (h, w) with h | h_tiles, w | w_tiles and h * w <= cap, wide first."""
    return max(((h, w) for h in range(1, h_tiles + 1) for w in range(1, w_tiles + 1)
                if h_tiles % h == 0 and w_tiles % w == 0 and h * w <= cap), key=lambda hw: (hw[0] * hw[1], hw[1]))


def _lin6b_program(x, w):
    """The measured multicast program for this large-LM projection, or None where ttnn.linear's own pick serves."""
    k, n = int(w.shape[-2]), int(w.shape[-1])
    rows = x.volume() // k
    if rows > _LIN6B_PROG_MAX_ROWS or n == k:
        return None
    bfp8 = w.dtype == ttnn.bfloat8_b
    if bfp8 and n < k:
        return None
    g = x.device().compute_with_storage_grid_size()
    mt, kt, nt = -(-rows // 32), k // 32, n // 32
    kb = next(b for b in (8, 4, 2, 1) if kt % b == 0)
    if n > k and not bfp8:
        pn = -(-nt // (g.x * g.y))
        sh, sw = _subblock(mt, pn)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), in0_block_w=kb, out_subblock_h=sh,
            out_subblock_w=sw, per_core_M=mt, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)
    pm, pn = -(-mt // g.y), -(-nt // g.x)
    sh, sw = _subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y), in0_block_w=kb, out_subblock_h=sh, out_subblock_w=sw,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True)


def _mm_bf16_acc(w) -> bool:
    return w.dtype == ttnn.bfloat8_b if _ESMC_MM_BF16_ACC is None else _ESMC_MM_BF16_ACC


def _mm_config(x, w):
    if not _ESMC_MM_CFG:
        return {}
    g = x.device().compute_with_storage_grid_size()
    kt = int(w.shape[-2]) // 32
    return {"config": ttnn.MinimalMatmulConfig(
        M_block_size=16, K_block_size=4 if kt % 4 == 0 else 2, N_block_size=4, subblock_h=2,
        subblock_w=4 if _mm_bf16_acc(w) else 2, compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))}


def _esmc_mm() -> bool:
    if _ESMC_MM is not None:
        return _ESMC_MM
    from tt_bio import tenstorrent
    return tenstorrent.is_wormhole()


class _MMLin:
    """`Module._lin`, served by `minimal_matmul` for a plain bf16 projection over ESMC_MM_MIN_ROWS rows. A SiLU
    (the SwiGLU's fc1a) rides along as the matmul's fused activation, applied per tile at pack as the linear does."""

    _ACT = {"silu": ttnn.UnaryOpType.SILU, "gelu": ttnn.UnaryOpType.GELU}

    def _mm_on(self) -> bool:
        return _esmc_mm()

    def _mm_ck(self, w):
        ck = self.compute_kernel_config
        if not _mm_bf16_acc(w):
            return ck
        if getattr(self, "_mm_ck_bf16", None) is None:
            self._mm_ck_bf16 = type(ck)(math_fidelity=ck.math_fidelity, math_approx_mode=ck.math_approx_mode,
                                        fp32_dest_acc_en=False, packer_l1_acc=ck.packer_l1_acc)
        return self._mm_ck_bf16

    def _mm_ok(self, x, w) -> bool:
        return ((w.dtype == ttnn.bfloat16 or (_ESMC_MM_BFP8 and w.dtype == ttnn.bfloat8_b))
                and (x.dtype == ttnn.bfloat16 or (_ESMC_ACT_BFP8 and x.dtype == ttnn.bfloat8_b == w.dtype))
                and x.volume() // int(x.shape[-1]) >= ESMC_MM_MIN_ROWS and self._mm_on())

    def _lin_split(self, x, w, chunks: int):
        """x @ w split into `chunks` outputs along the last dim by the matmul itself, or None where it does not serve."""
        if not (_ESMC_QKV_MM_SPLIT and self._mm_ok(x, w)):
            return None
        try:
            out = ttnn.experimental.minimal_matmul_split(
                x, w, chunks=chunks, compute_kernel_config=self._mm_ck(w), dtype=_dtype(ttnn.bfloat16),
                memory_config=ttnn.DRAM_MEMORY_CONFIG, **_mm_config(x, w))
        except RuntimeError:
            return None
        MM_FUSE_STATS[0] += 1
        return tuple(out)

    def _lin_resid(self, x, w, resid):
        """resid + x @ w in one matmul, or None where it does not serve (the caller adds)."""
        if not (_ESMC_MM_RESID and w.dtype == ttnn.bfloat16 and self._mm_ok(x, w)):
            return None
        n = int(w.shape[-1])
        ones = getattr(self, "_ones_row", None)
        if ones is None or int(ones.shape[-1]) != n:
            ones = self._ones_row = ttnn.from_torch(torch.ones(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                   device=x.device())
        try:
            out = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, w, 1.0, resid, ones, compute_kernel_config=self._mm_ck(w),
                dtype=_dtype(ttnn.bfloat16), memory_config=ttnn.DRAM_MEMORY_CONFIG, **_mm_config(x, w))
        except RuntimeError:
            return None
        MM_FUSE_STATS[1] += 1
        return out

    def _lin(self, x, w, bias=None, dtype=None, **kw):
        mm = self._mm_on()
        act = kw.get("activation")
        if (mm and not set(kw) - {"memory_config", "activation"} and (act is None or act in self._ACT)
                and self._mm_ok(x, w)):
            try:
                out = ttnn.experimental.minimal_matmul(
                    x, w, bias_tensor=bias, compute_kernel_config=self._mm_ck(w),
                    dtype=_dtype(ttnn.bfloat16) if dtype is None else dtype,
                    memory_config=kw.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG, **_mm_config(x, w),
                    **({"fused_activation": (self._ACT[act], False)} if act else {}))
                MM_STATS[0] += 1
                return out
            except RuntimeError:  # an L1 destination it cannot place: the linear serves it
                pass
        MM_STATS[1] += mm
        if _ESMC_LIN6B_PROG and getattr(self, "lm_large", False) and not kw and bias is None:
            from tt_bio import tenstorrent
            cfg = _lin6b_program(x, w) if tenstorrent.is_wormhole() else None
            if cfg is not None:
                LIN6B_STATS[0] += 1
                return ttnn.linear(x, w, program_config=cfg, compute_kernel_config=self.compute_kernel_config,
                                   dtype=_dtype(ttnn.bfloat16) if dtype is None else dtype)
        return super()._lin(x, w, bias=bias, dtype=dtype, **kw)


class Attention(_MMLin, Module):
    """Multi-head self-attention with QK-LayerNorm + RoPE (no biases on projections).

    Mirrors esm.layers.attention.MultiHeadAttention (qk_layernorm=True, bias=False):
      qkv = Linear(LayerNorm(x)); q,k,v = chunk(qkv,3)
      q = LayerNorm(q); k = LayerNorm(k)            # over full d_model, then per-head RoPE
      o = SDPA(rope(q), rope(k), v, scale=d_head**-0.5); out_proj(o)
    """

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config, out_scale: float = 1.0,
                 qk_l1: bool = True, d_model: int = 0):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.qk_l1 = qk_l1  # this model may take the L1 path; Block clears it for the 6B
        self.in_norm_weight = self.torch_to_tt("layernorm_qkv.0.weight")
        self.in_norm_bias = self.torch_to_tt("layernorm_qkv.0.bias")
        # The two big projection weights (qkv, out_proj) carry the bulk of the
        # ESMC-6B's parameters; in fast mode they load as block-fp8 (bfloat8_b),
        # halving their weight-read bandwidth and resident size. _dtype() is bf16
        # otherwise (full precision, the default).
        # q, k and v as three weights cut from the packed one on the host, so the device never holds both layouts
        # (a device-side slice of the 6B's bf16 qkv OOMs: the weights leave a Wormhole chip ~7 MB of DRAM a bank).
        if _ESMC_QKV_SPLIT and qk_l1 and d_model:
            d = d_model  # from the caller: a weight-cache load carries no state dict to read it from
            self.qkv_weight = None
            self._qkv_w = tuple(self.torch_to_tt("layernorm_qkv.1.weight", transform=lambda w, i=i: w[i * d:(i + 1) * d].t(),
                                                 dtype=_dtype()) for i in range(3))
            if _ESMC_QKV_MM_SPLIT and _esmc_mm():
                self.qkv_weight = self.torch_to_tt("layernorm_qkv.1.weight", dtype=_dtype())
        else:
            self.qkv_weight = self.torch_to_tt("layernorm_qkv.1.weight", dtype=_dtype())
        self.q_ln_weight = self.torch_to_tt("q_ln.weight")
        self.k_ln_weight = self.torch_to_tt("k_ln.weight")
        self.out_weight = self.torch_to_tt("out_proj.weight", transform=_scaled_t(out_scale), dtype=_dtype())

    def _qkv(self, x, ln_mc=None):
        """q, k, v projections of LayerNorm(x), the layer norm written to `ln_mc` (DRAM if None)."""
        x_norm = ttnn.layer_norm(x, weight=self.in_norm_weight, bias=self.in_norm_bias, epsilon=1e-5,
                                 compute_kernel_config=self.compute_kernel_config, memory_config=ln_mc)
        try:
            if self.qkv_weight is not None:
                split = self._lin_split(x_norm, self.qkv_weight, 3)
                if split is not None:
                    return split
            if getattr(self, "_qkv_w", None) is not None:
                return tuple(self._lin(x_norm, w) for w in self._qkv_w)
            qkv = self._lin(x_norm, self.qkv_weight)
            q, k, v = ttnn.chunk(qkv, 3, dim=-1)
            ttnn.deallocate(qkv)
            return q, k, v
        finally:
            ttnn.deallocate(x_norm)

    def _qk_heads(self, q, k, v):
        """LayerNorm(q), LayerNorm(k) and v re-packed and split into per-head [B, H, L, d_head]
        (the tile-aware head split wants one packed tensor); frees q, k and v."""
        ck = self.compute_kernel_config
        key = tuple(v.padded_shape)
        if _ESMC_QK_L1 and self.qk_l1 and key not in _QK_L1_REFUSED:
            l1 = ttnn.L1_MEMORY_CONFIG
            live = []
            try:
                live.append(ttnn.layer_norm(q, weight=self.q_ln_weight, epsilon=1e-5, compute_kernel_config=ck, memory_config=l1))
                live.append(ttnn.layer_norm(k, weight=self.k_ln_weight, epsilon=1e-5, compute_kernel_config=ck, memory_config=l1))
                live.append(ttnn.concat([live[0], live[1], v], dim=-1, memory_config=l1))
                heads = self._split_heads(live[2], self.n_heads)
            except Exception as exc:
                if not _is_alloc_refusal(exc):
                    from tt_bio import tenstorrent
                    tenstorrent.absorb_l1_refusal("esmc qk L1", exc)
                _QK_L1_REFUSED.add(key)
            else:
                QK_L1_STATS[0] += 1
                for t in (live[0], live[1], q, k, v):
                    ttnn.deallocate(t)
                return heads
            for t in live:
                ttnn.deallocate(t)
        QK_L1_STATS[1] += _ESMC_QK_L1 and self.qk_l1
        # q and k go as soon as their norms exist: with the 6B's bf16 weights resident, the 1968 aa re-pack on a
        # Wormhole chip only fits without them (30 MB asked, 42 MB free in 1.8 MB blocks per bank with them alive).
        qn = ttnn.layer_norm(q, weight=self.q_ln_weight, epsilon=1e-5, compute_kernel_config=ck)
        ttnn.deallocate(q)
        kn = ttnn.layer_norm(k, weight=self.k_ln_weight, epsilon=1e-5, compute_kernel_config=ck)
        ttnn.deallocate(k)
        qkv = ttnn.concat([qn, kn, v], dim=-1)
        for t in (v, qn, kn):
            ttnn.deallocate(t)
        return self._split_heads(qkv, self.n_heads)

    def __call__(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor,
                 attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None, resid: bool = False):
        """out_proj(attention(x)); with `resid`, (x + that, True) when the matmul fused the add, else (that, False)."""
        ck = self.compute_kernel_config
        d_model = x.shape[-1]
        head_dim = d_model // self.n_heads

        key = tuple(x.padded_shape)
        if _ESMC_QK_L1 and self.qk_l1 and key not in _QK_L1_REFUSED:
            try:
                q, k, v = self._qkv(x, ttnn.L1_MEMORY_CONFIG)
            except Exception as exc:
                if not _is_alloc_refusal(exc):
                    from tt_bio import tenstorrent
                    tenstorrent.absorb_l1_refusal("esmc qkv LN L1", exc)
                _QK_L1_REFUSED.add(key)
                q, k, v = self._qkv(x)
        else:
            q, k, v = self._qkv(x)
        # Re-pack and use the tile-aware head split, then apply per-head RoPE.
        q, k, v = self._qk_heads(q, k, v)
        q, k = _rope(q, k, cos, sin, l1=self.qk_l1)
        if key_valid is not None and attn_mask is None:
            # Zero padded keys/values so their attention contribution is exactly 0. With the -inf
            # additive mask present this is redundant: the masked softmax weight is already exactly
            # 0, and dropping the two multiplies leaves every output byte unchanged at every chunk
            # config swept (perf/wh-embed/sdpa_sweep.py), 0.22 ms a layer at the page shape.
            k = ttnn.multiply(k, key_valid)
            v = ttnn.multiply(v, key_valid)

        o = lm_sdpa(q, k, v, attn_mask, scale=head_dim**-0.5)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = self._merge_heads(o)  # [B, L, d_model]
        out = self._lin_resid(o, self.out_weight, x) if resid else None
        added = out is not None
        if not added:
            out = self._lin(o, self.out_weight)
        ttnn.deallocate(o)
        return (out, added) if resid else out


# ESMFold2 asks its pair transition for a fused SwiGLU kernel that no shipped ttnn wheel has:
# `ttnn.experimental.minimal_matmul` takes no `fuse_swiglu` kwarg on 0.68.0, so the request fell
# through to `_lin -> chunk -> silu -> multiply`. That chain moves 4.832 GB per call at
# [1,512,512,256] to compute an expression whose information content is 1.611 GB, because
# `ttnn.chunk` copies the whole fc1 output and `silu` and `multiply` each re-read what the op
# before them wrote.
#
# Splitting fc1 into its two halves on the host removes the chunk, and running the SiLU as the
# multiply's operand-A activation removes the silu. MEASURED on qb2, median of 5, c_z=256,
# d_ff=1024 (perf/esm2land/): 9.564 -> 6.860 ms/call at 298 aa, 10.196 -> 7.314 at 320,
# 29.442 -> 21.166 at 512, `torch.equal` against the chain it replaces at every size.
#
# Two dead ends, measured, so they are not retried: putting the SiLU in the fc1 epilogue instead
# is 1.50x SLOWER (8.621 vs 5.747 ms), and `ttnn.swiglu` is both slower and not bit-exact.
# See state/esmfold2-512aa-deep-perf.md.
SPLIT_SWIGLU = True
_SPLIT_SWIGLU = env_flag("TT_BIO_SPLIT_SWIGLU", SPLIT_SWIGLU)

# Blocking the same FFN over rows puts the SwiGLU product in L1, so fc2 reads its operand from L1
# instead of streaming 33.55 MB back out of DRAM. The row block is not the win and does not pay for
# itself: with the product left in DRAM the identical block is a LOSS at every size (21.407 vs
# 21.166 ms at 512 aa). It is what shrinks the product enough for L1 to serve it, and the L1 term
# is worth 3.27 ms/call at 512 aa: 21.327 -> 18.057 (perf/esm2land/probe_ladder_c0.json).
#
# 32 rows is pinned by parity, not tuned. 16 and 8 are faster still (16.144 ms at 512 aa) and are
# not torch.equal at any size, so they are a release-gate decision and not this one; 64 is refused
# outright by a circular-buffer clash.
PAIR_FFN_ROW_BLOCK = 32
_PAIR_FFN_ROW_BLOCK = int(os.environ.get("TT_BIO_PAIR_FFN_ROW_BLOCK", PAIR_FFN_ROW_BLOCK))

# Both levers carry a measured sequence-length window, and the window is not a formula.
# `ttnn.linear(core_grid=...)` derives its own matmul program from (M, N, K, grid), so halving
# fc1's N -- or blocking the rows, which changes the batch extent -- can land on a different
# K-accumulation blocking and round differently in the last bf16 bit. Which L that happens at is
# not predictable from L: the row block is torch.equal at every one of the 23 sizes measured from
# 320 to 768 aa whether or not 32 divides them, and differs at every size below 320 except 96 and
# 128; the split fc1 is torch.equal at all 25 sizes from 144 to 768 and differs at 96 and 128.
# The row-block window was extended to 1024 on the same evidence: torch.equal at the op
# (perf/esm2sizes/screen_qb1c3.json) and a byte-identical CIF at the fold
# (perf/esm2sizes/fold_1024_window_qb1c3.json), measured on the qb1 13x10 grid, ttnn 0.67.4.
# The differences are one bf16 ulp (max_abs/peak 4.7e-3 to 5.7e-3 against a bf16 relative eps of
# 3.9e-3), so they are an accumulation order rather than a defect -- but the bar for landing this
# was byte-identical output, so each lever fires only inside the window where that was measured.
# Ladders: perf/esm2land/probe_stage2_c0.json decomposes it per op over 14 sizes (the fused SiLU
# is exact at every size on its own; the fc1 split and the row block are what move), and
# probe_ladder_c0.json / probe_ladder2_c0.json cover 28 sizes end to end.
#
# Both are also restricted to the 11x10 Blackhole grid they were measured on. A smaller grid
# derives different matmul programs and does its own row blocking through `pair_row_tile`, so the
# parity above says nothing there and the shipped unsplit chain keeps running.
SPLIT_SWIGLU_MIN_SEQ = 144

# ...and the small grid gets its own opt-in, because "no evidence there" is not "measured slower
# there". The Wormhole Galaxy JapanFold runs on is 8x9 = 72 cores, so `_IS_SMALL_GRID` is True and
# the whole family above -- split fc1, the 32-row block, the L1-resident fc1 -- is dead on the one
# machine that serves users. This flag is read ONLY when `_IS_SMALL_GRID` is True, so on any grid
# >= 110 cores the expression below is byte-for-byte the one that shipped before it existed.
# Measured on 8x9, --fast, bit-exact at every size (one CIF sha256 per size across both arms):
# 256 aa 28.727 -> 26.597 s, 298 aa 41.083 -> 36.556, 512 aa 93.512 -> 83.843, 640 aa
# 155.701 -> 141.117. Inert below SPLIT_SWIGLU_MIN_SEQ, as 128 aa confirms at 0.1 %. The 13x10
# Blackhole A/A is -0.54 % inside a 0.6 % spread with an identical digest. See
# state/wh-perf-esmfold2.md.
SPLIT_SWIGLU_SMALL_GRID = True
_SPLIT_SWIGLU_SMALL_GRID = os.environ.get(
    "TT_BIO_SPLIT_SWIGLU_SMALL_GRID", "1" if SPLIT_SWIGLU_SMALL_GRID else "0") == "1"
PAIR_FFN_ROW_BLOCK_SEQ = (320, 1024)

# Inside the row block, fc1's two halves can also write their OUTPUT to L1, so fc2's operand never
# leaves the chip. That is a second, independent lever on top of the row block: the block alone
# only shrinks the SwiGLU product enough for L1 to serve it, while fc1 itself still round-trips
# 2.15 GB per call through DRAM. `_pair_proj_program_config` hardcoded `out_block_w = n_tiles`,
# which at this operand class ([B*rows, N, 256] x [256, 1024], k_tiles 8, n_tiles 32) spends
# 1,212,416 B of a 1,461,760 B bank on the in0/in1 buffers alone, so the gate refused an L1
# destination at every row height and both halves fell back to a DRAM output.
#
# MEASURED on qb2 card 0, ttnn 0.68.0, 11x10 grid, whole `SwiGLUFFN` at [1,512,512,256],
# median of 5: 18.095 -> 14.657 ms per call, 1.235x, `torch.equal`
# (perf/esm3p4/accept_l2_c0.json). At the 512 aa fold it is worth -1.926 s.
#
# Ships ON, and checked end to end at 298 / 512 / 768 / 1024 aa on qb2, ttnn 0.68.0, 11x10:
# byte-identical CIF and plDDT against the OFF arm at every size. The fold deltas are 1.0234x at
# 298, 1.0879x at 512, and 0.9998x / 0.9999x at 768 and 1024, where the block's other L1 residents
# leave no room, the device refuses the first call and the class retires to DRAM
# (perf/ttx_b3/fold_ab_esm512_c0.json, fold_ab_esm768_c0.json, perf/esm3p4close/fold_ab_*_c1.json).
#
# `PAIR_FFN_ROW_BLOCK_SEQ` reads the PADDED token axis, so 298 aa is INSIDE the window, not below
# it: it runs 320/32 = 10 row blocks and the gated-call census is 538 x 10 x 2 = 10760, exactly
# 10/16 of the 17216 at 512 aa. An earlier version of this note called it inert at 298, which its
# own artifact contradicts. Still gated, so an A/B stays one call away. 13x10 is unchecked with
# this config, the same exposure the row block already ships with.
PAIR_FFN_L1_FC1 = True
_PAIR_FFN_L1_FC1 = os.environ.get(
    "TT_BIO_PAIR_FFN_L1_FC1", "1" if PAIR_FFN_L1_FC1 else "0") == "1"

# [served, declined] `_pair_proj_linear` calls, so a consumer census counts what executed rather
# than what greps. Same idiom as `reblock_permute.STATS_GATED`.
L1_FC1_STATS = [0, 0]

# Third lever on the same block, and the one that pays best. The block layer_norm wrote its result
# to DRAM and fc1's two halves each read all 8.39 MB of it back. Giving the norm an L1 destination
# removes that round trip, but the win is not the removed bytes: fc1's in0 stops stalling on DRAM
# and the matmul goes from 31 % to ~52 % of the compute roof. So this is not the killed
# "faster layer_norm config" arm (state/esmfold2-to-3p4x.md 11.15, best bit-exact 1.008x) -- that
# one changed the reduction's blocking to speed up the norm itself. The kernel and its reduction
# are untouched here; only the destination moves, and the gain lands in fc1.
#
# MEASURED on qb1 card 0, ttnn 0.68.0, 13x10 grid, the whole block body (norm, fc1 x2, SwiGLU,
# fc2) at c_z=256, d_ff=1024, rows=32, median of 5, batched 4 calls per synchronize:
#   298 aa  5.4985 -> 4.9290 ms/call     512 aa  12.7100 -> 10.2619
#   640 aa 18.3031 -> 14.6634            768 aa  31.0049 -> 27.2667
# `torch.equal` at the BODY output (not just the norm) with max abs diff 0.0 at every size
# (perf/esmbeat/p3_s_lnl1_c0.json). At 512 aa that is -2.448 ms x 538 calls = -1.317 s of fold.
#
# Rides inside `l1_gated`, so it inherits the row block's 320-1024 aa window and cannot reach
# ESMC's 3-D LM FFN. The refusal cache is load-bearing rather than defensive: at the top of the
# window the block's L1 residents outgrow a 110-core grid, and `ttnn.layer_norm` RAISES on an L1
# refusal instead of falling back the way `_pair_proj_linear` does. Keyed on `padded_shape`, so a
# size that declines costs one exception per fold, not one per block.
PAIR_FFN_L1_LN = True
# The gate silu(h1) * h2 as ttnn.multiply with an input activation runs binary_ng's per-tile preprocess: copy h1
# into dest, silu, pack to a scratch CB, re-init the binary op, unpack both, multiply, pack, one tile at a time.
# On the 512 aa row block that is ~0.28 ms for 16k tiles, ~40 % of the FFN (ffn-split, qb2 p300c). With the silu in
# fc1a's matmul epilogue (on the fp32 accumulator, before the bf16 pack) the gate is a plain FPU multiply:
# gate_bench at a 16-row block, 0.144 ms for the activated multiply vs 0.058 ms plain. One switch for every
# split-fc1 SwiGLU: the pair transition (row-blocked) and the ESMC LM blocks, which are ESMFold2's LM backbone.
_SWIGLU_SILU_FC1 = env_flag("TT_BIO_SWIGLU_SILU_FC1", False)
# The same activated multiply on ttnn's legacy binary op: same SFPU silu, no per-tile re-init, 0.101 vs 0.144 ms
# (gate_bench, same block). Pair transition row blocks only (`l1_gated`): folds -1.1 s ESMFold2 / -0.7 s -Fast on
# p150a, while ESMC 300M's LM blocks measured no faster with it and further from fp32 (cos_min 0.9971 vs 0.9987).
# Graded PASS on the normal bar for both models on Blackhole (11 complexes x 4 seeds, docs/tuning-flags.md), so on by
# default there. Wormhole is not graded and keeps its multiply (and its digests) unless the switch is set: None = by arch.
_SWIGLU_GATE_LEGACY = (env_flag("TT_BIO_SWIGLU_GATE_LEGACY", False)
                       if os.environ.get("TT_BIO_SWIGLU_GATE_LEGACY", "").strip() else None)


def _swiglu_gate_legacy() -> bool:
    if _SWIGLU_GATE_LEGACY is not None:
        return _SWIGLU_GATE_LEGACY
    from tt_bio import tenstorrent
    return tenstorrent.arch_name() == "blackhole"

# ESMC's language-model FFN runs over all rows in one pass on every arch. Wormhole used to tile it at
# SMALL_GRID_SEQ_TILE rows up front so the 6B's fc1 activation never asked a 12 GB chip that already holds the 6B for
# one large block; that cut a 544-row page cell into three blocks plus a chunk and a concat, and each block re-streamed
# the layer's FFN weights. One pass, Galaxy chip at 1000 MHz, 6B page cell: 445 -> 315 ms, --fast 426 -> 304 ms, same
# error against float32 (pcc 0.99904 / 0.99894 either way; earlier the same for 300M/600M, 213 -> 190 / 337 -> 276 ms).
# A model past _LM_LARGE_MODEL_BYTES of resident weights (the 6B) still tiles above _LM_FFN_WHOLE_ROWS rows: in bf16
# it leaves ~30 MB a DRAM bank beside the weights, and at its 1968 aa ceiling the one-pass FFN fit but left DRAM too
# fragmented for a later layer's attention re-pack (2.5 MB asked a bank, 3.5 MB free, 1.8 MB the largest block).
# 1056 rows is the 1024 aa bucket, measured to embed through the CLI in one pass; --fast embeds 8192 aa either way.
# A shape the allocator refuses in one piece below the bound is tiled from then on.
# A model past _LM_LARGE_MODEL_BYTES also keeps the attention's DRAM path (Attention.qk_l1).
_LM_LARGE_MODEL_BYTES = env_int("TT_BIO_LM_LARGE_MODEL_BYTES", 4 << 30)
# The 6B on the same attention path as the small models: q, k and v from the split weight (cut on the host, no second
# copy) and the norms, re-pack and rope in L1, each refusal falling back to DRAM. Its first try
# kept both weights: --fast 304.8 -> 286.2 ms at 544 aa on a Galaxy chip, normal out of DRAM at load. None = on for
# Wormhole with block-fp8 weights (--fast): 6B page cell 284.1 -> 266.0 ms at 1000 MHz, pcc vs fp32 0.99888 -> 0.99890,
# and ESMFold2-Fast on Wormhole grades PASS against the fast bar (11-set x 4 seeds, 0 failed folds, 24/44 docked
# both arms, median same-seed pose deviation 0.97 A against a 0.85 A seed floor).
_ESMC_QK_L1_LARGE = (env_flag("TT_BIO_ESMC_QK_L1_LARGE", False)
                     if os.environ.get("TT_BIO_ESMC_QK_L1_LARGE", "").strip() else None)


def _qk_l1_large(wdtype) -> bool:
    if _ESMC_QK_L1_LARGE is not None:
        return _ESMC_QK_L1_LARGE
    from tt_bio import tenstorrent
    return wdtype == ttnn.bfloat8_b and tenstorrent.is_wormhole()
_LM_FFN_WHOLE_ROWS = env_int("TT_BIO_LM_FFN_WHOLE_ROWS", 1056)
# The per-token LM FFN's SwiGLU gate silu(x fc1_a) * (x fc1_b) written to L1, where fc2 reads it back. Same op, same
# values (a memory placement); a refusal falls back to DRAM for that shape for good. Off until the page-cell A/B says.
_ESMC_GATE_L1 = env_flag("TT_BIO_ESMC_GATE_L1", False)
_GATE_L1_REFUSED: set = set()
#: [gates served from L1, declined], read by the A/B harness.
GATE_L1_STATS = [0, 0]

_PAIR_FFN_L1_LN = os.environ.get(
    "TT_BIO_PAIR_FFN_L1_LN", "1" if PAIR_FFN_L1_LN else "0") == "1"

# [served, declined] block layer_norms, same census idiom as `L1_FC1_STATS`: an A/B arm reading
# [0, n] is vacuous, and one reading [0, 0] never reached the gate at all.
L1_LN_STATS = [0, 0]
_L1_LN_REFUSED = set()

# Fourth lever (C-in). The row block cut all 16 blocks into DRAM up front and lever E's
# `layer_norm` read each one straight back out. Slicing a block lazily INTO L1 leaves the copy
# in place and moves only its destination, removing one 8.39 MB DRAM write and one 8.39 MB DRAM
# read per block at 512 aa.
#
# The control is what makes this a mechanism rather than a coincidence: the same lazy per-block
# slice landing in DRAM returns -0.026 s, i.e. nothing (`cin_dram` in
# perf/esmbeat/p3_s_cin_512_c0.json). The L1 destination is the whole lever.
#
# MEASURED on qb1 card 0, ttnn 0.68.0, 13x10 grid, the shipped chain at [1,512,512,256],
# 4 calls per synchronize, median of 5: 11.6856 -> 11.1378 ms/call = -0.295 s of fold.
# `torch.equal` on the assembled output, max abs diff 0.0, and the sliced block is `torch.equal`
# to the chunked block as well, so a future failure is attributable.
#
# Rides inside `l1_gated`, so it inherits the row block's 320-1024 aa window.
PAIR_FFN_L1_SLICE = True
_PAIR_FFN_L1_SLICE = os.environ.get(
    "TT_BIO_PAIR_FFN_L1_SLICE", "1" if PAIR_FFN_L1_SLICE else "0") == "1"

# [served, declined] row-blocked calls, per call and not per block. Same census idiom as
# `L1_LN_STATS`.
L1_SLICE_STATS = [0, 0]
_L1_SLICE_REFUSED = set()

# Fifth lever (F). `PairUpdateBlock` owned `z = z + transition(z)`, a full-tensor add sitting
# outside the row-block loop that read the FFN output back out of the DRAM the `concat` had just
# written it to. Folding the add into the block lets fc2's output stay in L1 and never round-trip:
#
#   before  slice 134 read | fc2 134 write | concat 134 read + write | add 268 read + 134 write
#   after   slice 134 read | fc2 -> L1, 0  | add -> 134 write        | concat 134 read + write
#
# 402 MB/call of the 938 removed at 512 aa. The add is elementwise and row-independent, so
# per-block and full-tensor are the same arithmetic in the same order -- measured, not assumed.
#
# MEASURED on qb1 card 0, ttnn 0.68.0, 13x10 grid, same protocol as C-in:
# 12.0471 -> 11.0393 ms/call = -0.542 s of fold, `torch.equal`, max abs diff 0.0
# (perf/esmbeat/p3_s_resid_512_c0.json). With C-in it is -0.842 s together, slightly better than
# the sum because F's per-block add reads its first operand out of the L1 slice C-in put there.
#
# Why fc2 can write L1 here when holding all 16 block outputs in L1 cannot (that one is measured
# dead, `TT_THROW @ program.cpp:1052`, perf/esmbeat/p3_s_split_512_c0.json): F consumes each
# block's output immediately, so L1 holds one 8.39 MB block rather than 134 MB.
PAIR_FFN_FUSED_RESIDUAL = True
_PAIR_FFN_FUSED_RESIDUAL = os.environ.get(
    "TT_BIO_PAIR_FFN_FUSED_RESIDUAL", "1" if PAIR_FFN_FUSED_RESIDUAL else "0") == "1"

# [served, declined] `SwiGLUFFN.residual` calls. A caller that never reaches the row block counts
# declined, which is what separates "the gate refused" from "the gate was never asked".
FUSED_RESID_STATS = [0, 0]
_FUSED_RESID_REFUSED = set()
# padded_shape -> the block height that stopped throwing for it. Read back on the next
# call for the same shape so a fold pays the exception once, not once per block, which is
# the same contract the three *_REFUSED sets above keep.
_ROW_BLOCK_SHRUNK: dict[tuple, int] = {}
# Smallest block the shrink rung will go to. Below this the per-block dispatch and CB fill
# dominate what the L1 residency buys, and the honest answer is to drop G and F instead.
PAIR_FFN_ROW_BLOCK_MIN = 8

# Sixth lever (G). With F in place the block loop still assembled its result with
# `ttnn.concat`, which reads 134 MB and writes 134 MB per call at 512 aa purely to lay 16 blocks
# side by side. `ttnn.fill_cache(cache, input, i)` writes `input` into `cache[i]` in place, and
# under `ttnn.experimental.view` the pair tensor's row block IS a dim-0 index: view the
# [1, L, L, C] output as [L/rows, rows, L, C] and block `i` is exactly `cache[i]`. So the block's
# residual add keeps its output in L1 and is written straight into a pre-allocated output, and
# the concat disappears along with the block's own DRAM write.
#
# The output buffer comes from `ttnn.allocate_tensor_on_device` -- a bare allocation, no host
# copy and no zero fill, which is correct because every one of the nblk blocks is written.
#
# MEASURED on qb1 card 2, ttnn 0.68.0, 13x10 grid, 16 blocks of [1,32,512,256], allocation timed
# INSIDE the arm: concat 1.8005 -> fill 1.3745 ms = -0.230 s of fold, `torch.equal` on the
# assembled output, max abs diff 0.0 (perf/esmbeat/p3_s_fillcache_512_c2.json). The view was
# checked to alias (`buffer_address()` equal), and `fill_cache` was checked to accept an L1
# input -- with a DRAM input it would pay the traffic the concat already pays and return nothing.
#
# On the residual path it rides on F, which makes the block's output the final content of its
# rows; without a residual (the MSA encoder's pair transition) fc2's output already is, and the
# concat it replaces needed a second whole-pair buffer: 2 GiB at 2048 aa, the --fast OOM on a
# p300c chip (spd-trifuse 10-12). Needs
# `nblk * rows == L` for the view's volume to match, so a length that does not divide by the row
# block keeps the concat.
PAIR_FFN_FILL_ASSEMBLY = True
_PAIR_FFN_FILL_ASSEMBLY = os.environ.get(
    "TT_BIO_PAIR_FFN_FILL_ASSEMBLY", "1" if PAIR_FFN_FILL_ASSEMBLY else "0") == "1"

# [served, declined] row-blocked calls, same idiom as `FUSED_RESID_STATS`.
FILL_ASSEMBLY_STATS = [0, 0]
_FILL_ASSEMBLY_REFUSED = set()

# [split, unsplit] `SwiGLUFFN.__call__` invocations. An A/B arm on the small-grid opt-in needs both
# `_SPLIT_SWIGLU` and `_SPLIT_SWIGLU_SMALL_GRID`, and if either is missed the arm is a silent A/A;
# this counter is what makes that visible instead of inferred.
SPLIT_STATS = [0, 0]

# A pair FFN shape the DRAM allocator refused UNBLOCKED, so the row block is taken straight
# away on every later call at that shape instead of paying one refusal per call (48 pair
# transitions x 10 trunk recycles on ESMFold2).
_UNBLOCKED_REFUSED: set[tuple] = set()
#: [served by the out-of-window fallback, ran unblocked]. Census, so "the fallback fired" is
#: something a run can be asked rather than assumed.
WINDOW_FALLBACK_STATS = [0, 0]


def _is_alloc_refusal(exc: BaseException) -> bool:
    """See ``size_limits.is_alloc_refusal``: one definition, shared with esmfold2's
    OuterProductMean. Imported lazily; size_limits is a leaf module and this keeps esmc's
    import graph unchanged."""
    from tt_bio.size_limits import is_alloc_refusal

    return is_alloc_refusal(exc)


def set_split_swiglu(on: bool) -> bool:
    """A/B switch for the split-fc1 SwiGLU path. Returns the previous state."""
    global _SPLIT_SWIGLU
    prev, _SPLIT_SWIGLU = _SPLIT_SWIGLU, bool(on)
    return prev


def set_pair_ffn_l1_fc1(on: bool) -> bool:
    """A/B switch for the L1-resident fc1 inside the row-blocked pair FFN. Returns the previous state."""
    global _PAIR_FFN_L1_FC1
    prev, _PAIR_FFN_L1_FC1 = _PAIR_FFN_L1_FC1, bool(on)
    return prev


def set_pair_ffn_l1_ln(on: bool) -> bool:
    """A/B switch for the L1-resident block layer_norm inside the row-blocked pair FFN. Returns
    the previous state."""
    global _PAIR_FFN_L1_LN
    prev, _PAIR_FFN_L1_LN = _PAIR_FFN_L1_LN, bool(on)
    return prev


def set_pair_ffn_l1_slice(on: bool) -> bool:
    """A/B switch for the lazy L1 row slice feeding the blocked pair FFN (lever C-in). Returns
    the previous state."""
    global _PAIR_FFN_L1_SLICE
    prev, _PAIR_FFN_L1_SLICE = _PAIR_FFN_L1_SLICE, bool(on)
    return prev


def set_pair_ffn_fused_residual(on: bool) -> bool:
    """A/B switch for the per-block residual add inside the blocked pair FFN (lever F). Returns
    the previous state."""
    global _PAIR_FFN_FUSED_RESIDUAL
    prev, _PAIR_FFN_FUSED_RESIDUAL = _PAIR_FFN_FUSED_RESIDUAL, bool(on)
    return prev


def set_pair_ffn_fill_assembly(on: bool) -> bool:
    """A/B switch for writing each row block into a pre-allocated output instead of concatenating
    the blocks (lever G). Returns the previous state."""
    global _PAIR_FFN_FILL_ASSEMBLY
    prev, _PAIR_FFN_FILL_ASSEMBLY = _PAIR_FFN_FILL_ASSEMBLY, bool(on)
    return prev


def set_pair_ffn_row_block(rows: int) -> int:
    """Row height for the blocked pair FFN, 0 to switch it off. Returns the previous value."""
    global _PAIR_FFN_ROW_BLOCK
    prev, _PAIR_FFN_ROW_BLOCK = _PAIR_FFN_ROW_BLOCK, int(rows)
    return prev


def _scaled_t(scale: float):
    """Weight transform: transpose, times `scale` in fp32 when it is not 1 (then byte-identical)."""
    if scale == 1.0:
        return lambda w: w.t()
    return lambda w: (w.float() * scale).t()


def _pack_swiglu_weight(weight: torch.Tensor) -> torch.Tensor:
    packed = weight.t()
    rows, two_n = packed.shape
    return packed.reshape(rows, 2, -1, 32).permute(0, 2, 1, 3).reshape(rows, two_n)


class SwiGLUFFN(_MMLin, Module):
    """SwiGLU feed-forward (mirrors esm.layers.blocks.swiglu_ln_ffn, bias=False):
      h = Linear(LayerNorm(x)); x1,x2 = chunk(h,2); Linear(silu(x1) * x2).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config, fuse_swiglu: bool = False,
                 out_scale: float = 1.0):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("0.weight")
        self.norm_bias = self.torch_to_tt("0.bias")
        minimal_matmul = getattr(ttnn.experimental, "minimal_matmul", None)
        self.fuse_swiglu = bool(
            fuse_swiglu
            and minimal_matmul is not None
            and "fuse_swiglu" in (minimal_matmul.__doc__ or "")
        )
        # fc1/fc2 are the FFN's big matmuls (and the bulk of the ESMC-6B FLOPs);
        # block-fp8 in fast mode, bf16 otherwise. Shared with the folding trunk's
        # pair-transition, so fast mode bf8's that too.
        # With no fused kernel in the wheel, `fuse_swiglu=True` really selects the split-fc1
        # path (see SPLIT_SWIGLU above). Build only the fc1 layout the live path reads.
        self.split_swiglu = bool(fuse_swiglu and not self.fuse_swiglu)
        self.model_bytes = float("inf")  # resident weights of the model this FFN is in; ESMC's Block sets it
        self.out_scale = out_scale
        if self.split_swiglu:
            half = self.weights["1.weight"].shape[0] // 2
            self.fc1_weight = None
            self.fc1_a_weight = self.torch_to_tt(
                "1.weight", transform=lambda w: w[:half].t(), dtype=_dtype()
            )
            self.fc1_b_weight = self.torch_to_tt(
                "1.weight", transform=lambda w: w[half:].t(), dtype=_dtype()
            )
        else:
            transform = _pack_swiglu_weight if self.fuse_swiglu else lambda weight: weight.t()
            self.fc1_weight = self.torch_to_tt("1.weight", transform=transform, dtype=_dtype())
        self.fc2_weight = self.torch_to_tt("3.weight", transform=_scaled_t(out_scale), dtype=_dtype())

    def _fc1_full(self) -> ttnn.Tensor:
        """The unsplit fc1 weight, built on demand for the A/B control arm.

        `concat` along the output dim is the two halves' own values in their own order, so the
        control arm stays byte-identical to the shipped path without a second copy of the weight
        resident in production: 1.05 MB per block, 50 MB over ESMFold2's 48 pair transitions.
        """
        if self.fc1_weight is None:
            self.fc1_weight = ttnn.concat([self.fc1_a_weight, self.fc1_b_weight], dim=-1)
        return self.fc1_weight

    def _ffn(self, x: ttnn.Tensor, split: bool = False, l1_gated: bool = False,
             out_mc=None, resid: bool = False) -> ttnn.Tensor:
        """`out_mc` is fc2's output memory config; `None` -- every caller but lever F -- leaves
        the op byte-identical to the call that had no such keyword.

        `split` is a request, not a promise: only a module that built the two fc1 halves
        (`split_swiglu`) can serve it. `_row_blocked` asks for it unconditionally, which was
        right while the only caller reaching it was ESMFold2's trunk transition, built with
        `fuse_swiglu=True`. The DRAM-refusal fallback then routed the msa and pair transitions
        down the same path; those are built unsplit, and they died on
        `AttributeError: fc1_a_weight` the moment a refusal fired. The branch therefore reads
        the layout this module actually holds instead of trusting the caller to know it.
        """
        ck = self.compute_kernel_config
        ln = dict(weight=self.norm_weight, bias=self.norm_bias,
                  epsilon=1e-5, compute_kernel_config=ck)
        x_norm = None
        if l1_gated and _PAIR_FFN_L1_LN:
            key = tuple(x.padded_shape)
            if key not in _L1_LN_REFUSED:
                try:
                    x_norm = ttnn.layer_norm(x, memory_config=ttnn.L1_MEMORY_CONFIG, **ln)
                    L1_LN_STATS[0] += 1
                except Exception:
                    _L1_LN_REFUSED.add(key)
        if x_norm is None:
            L1_LN_STATS[1] += bool(l1_gated and _PAIR_FFN_L1_LN)
            x_norm = ttnn.layer_norm(x, **ln)
        if self.fuse_swiglu:
            gated = ttnn.experimental.minimal_matmul(
                input_tensor=x_norm,
                weight_tensor=self.fc1_weight,
                compute_kernel_config=ck,
                dtype=self.fc1_weight.dtype,
                fuse_swiglu=True,
            )
            ttnn.deallocate(x_norm)
        elif split and self.split_swiglu:
            from tt_bio import tenstorrent as _T
            out = self._sharded(x_norm, out_mc) if _T.lever("transition_shard") else None
            if out is not None:
                ttnn.deallocate(x_norm)
                return out
            if l1_gated and _PAIR_FFN_L1_FC1:
                L1_FC1_STATS[0] += 2
                l1 = dict(l1_out=True, l1_bw=_PAIR_FFN_FC1_BW,
                          l1_block_w=_PAIR_FFN_FC1_BLOCK_W)
                # The unsplit control resolves `_dtype(ttnn.bfloat16)`, so a bare `_dtype()`
                # here makes turning the split on a PRECISION change as well as a traffic one
                # whenever fast mode is set: MEASURED max_abs/peak 2.5e-2 at 512 aa on the
                # 72-core Galaxy, where --fast is forced. The small-grid path is new, so it
                # takes the control's dtype and stays comparable; Blackhole keeps the dtype
                # its shipped parity was measured with.
                from tt_bio import tenstorrent as _T
                dt = (_dtype(ttnn.bfloat16)
                      if (_SPLIT_SWIGLU_SMALL_GRID
                          and getattr(_T, "_IS_SMALL_GRID", False))
                      else _dtype())
                h1 = _pair_proj_linear(x_norm, self.fc1_a_weight, ck, dt, silu=_SWIGLU_SILU_FC1, **l1)
                h2 = _pair_proj_linear(x_norm, self.fc1_b_weight, ck, dt, **l1)
                silu_done = _SWIGLU_SILU_FC1
            else:
                L1_FC1_STATS[1] += 2 * bool(l1_gated)
                dt = (ttnn.bfloat8_b if _ESMC_ACT_BFP8 and len(x_norm.shape) == 3
                      and self.fc1_a_weight.dtype == ttnn.bfloat8_b and self._mm_ok(x_norm, self.fc1_a_weight) else None)
                h1 = self._lin(x_norm, self.fc1_a_weight, dtype=dt,
                               activation="silu" if _SWIGLU_SILU_FC1 else None)
                h2 = self._lin(x_norm, self.fc1_b_weight, dtype=dt)
                silu_done = _SWIGLU_SILU_FC1
            ttnn.deallocate(x_norm)
            act = dict(input_tensor_a_activations=[] if silu_done else [ttnn.UnaryOpType.SILU])
            gated = None
            key = tuple(h1.padded_shape)
            if not l1_gated and _ESMC_GATE_L1 and len(h1.shape) == 3 and key not in _GATE_L1_REFUSED:
                try:
                    gated = ttnn.multiply(h1, h2, memory_config=ttnn.L1_MEMORY_CONFIG, **act)
                    GATE_L1_STATS[0] += 1
                except Exception as exc:
                    if not _is_alloc_refusal(exc):
                        from tt_bio import tenstorrent as _T
                        _T.absorb_l1_refusal("esmc gate L1", exc)
                    _GATE_L1_REFUSED.add(key)
            if gated is None:
                GATE_L1_STATS[1] += bool(not l1_gated and _ESMC_GATE_L1 and len(h1.shape) == 3)
                gated = ttnn.multiply(
                    h1, h2, **act,
                    **({"use_legacy": True} if l1_gated and _swiglu_gate_legacy() else {}),
                    **({"memory_config": ttnn.L1_MEMORY_CONFIG} if l1_gated else {}),
                )
            ttnn.deallocate(h1)
            ttnn.deallocate(h2)
        else:
            if self.split_swiglu and _SPLIT_SWIGLU:
                # Below SPLIT_SWIGLU_MIN_SEQ a split-built module still multiplies by its own halves. The
                # concatenated fc1 is for the TT_BIO_SPLIT_SWIGLU=0 control arm only: built here it stayed
                # resident in every block, 70.8 MB a layer for the ESMC-6B, which OOMed it on a 12 GB
                # Wormhole chip for any sequence under 144 tokens.
                x1 = self._lin(x_norm, self.fc1_a_weight)
                x2 = self._lin(x_norm, self.fc1_b_weight)
            else:
                h = self._lin(x_norm, self._fc1_full())
                x1, x2 = ttnn.chunk(h, 2, dim=-1)
                ttnn.deallocate(h)
            ttnn.deallocate(x_norm)
            gated = ttnn.multiply(ttnn.silu(x1), x2)
            ttnn.deallocate(x1); ttnn.deallocate(x2)
        out = self._lin_resid(gated, self.fc2_weight, x) if resid and out_mc is None else None
        if out is None:
            out = self._lin(gated, self.fc2_weight,
                            **({"memory_config": out_mc} if out_mc is not None else {}))
        ttnn.deallocate(gated)
        return out

    def _sharded(self, x_norm: ttnn.Tensor, out_mc=None):
        """fc2(silu(x fc1_a) * (x fc1_b)) on the pair Transition's block-sharded swiglu
        (`transition_shard`; with `swiglu_fc12g` and `silu_f32` the fused fc12g kernel, no fc1 tensors
        and no multiply), None when the block does not shard. fc2 writes to `out_mc`, so the row-blocked
        residual add reads it from L1 as on the unsharded path. Same structure as tenstorrent.Transition's
        fc1/fc2/fc3, so ESMFold2's trunk FFN takes the shared path and levers.
        """
        from tt_bio import tenstorrent as T
        w12 = None
        if T.lever("swiglu_fc12g") and T.lever("silu_f32"):
            if getattr(self, "_w12", None) is None:
                w1, w2 = (ttnn.to_torch(w).float() for w in (self.fc1_a_weight, self.fc1_b_weight))
                self._w12 = ttnn.from_torch(T._fc12_interleave(w1, w2, 8), layout=ttnn.TILE_LAYOUT,
                                            device=self.fc1_a_weight.device(), dtype=self.fc1_a_weight.dtype)
            w12 = self._w12
        ck = self.compute_kernel_config
        return T._transition_swiglu_sharded(x_norm, self.fc1_a_weight, self.fc1_b_weight, self.fc2_weight,
                                            ck, T.silu_ckc(ck), _dtype(), x_norm.dtype, w12=w12, out_mc=out_mc)

    def _split_plan(self, x: ttnn.Tensor) -> tuple[bool, int]:
        """(split fc1?, rows per block) for this input, 0 rows if the 4-D row-blocked pair path
        does not apply. Carries no census; the caller owns `SPLIT_STATS`."""
        from tt_bio import tenstorrent
        split = bool(
            self.split_swiglu and _SPLIT_SWIGLU
            and (_SPLIT_SWIGLU_SMALL_GRID
                 or not getattr(tenstorrent, "_IS_SMALL_GRID", False))
            and x.shape[-2] >= SPLIT_SWIGLU_MIN_SEQ
        )
        lo, hi = PAIR_FFN_ROW_BLOCK_SEQ
        rows = _PAIR_FFN_ROW_BLOCK
        blocked = split and len(x.shape) == 4 and rows and lo <= x.shape[1] <= hi
        if blocked and tenstorrent.lever("transition_shard"):
            # the sharded swiglu needs a block of at most _TRANSITION_SHARD_PM row tiles per core row
            rows = tenstorrent._transition_shard_rows(x.shape[2], x.shape[3],
                                                      int(self.fc1_a_weight.shape[-1])) or rows
        return split, (rows if blocked else 0)

    def _rows_after_refusal(self, x: ttnn.Tensor) -> int:
        """Rows per block for a pair shape the allocator already refused unblocked, else 0.

        Only ever non-zero after a real refusal at this exact padded shape, so it cannot pull a
        size inside the parity window off the path its byte-identical output was measured on.
        """
        if len(x.shape) != 4 or not _PAIR_FFN_ROW_BLOCK:
            return 0
        return _PAIR_FFN_ROW_BLOCK if tuple(x.padded_shape) in _UNBLOCKED_REFUSED else 0

    def _row_blocked(self, x: ttnn.Tensor, rows: int, residual: bool) -> ttnn.Tensor:
        """The 4-D pair FFN over `rows`-row blocks, optionally with `x +` folded into each block.

        Three levers ride here and all three only move a destination, so all three are
        bit-exact: C-in slices each block lazily into L1 instead of cutting all of them into DRAM
        up front, F (`residual=True`) keeps fc2's output in L1 so the per-block add reads it on
        chip, and G writes each block straight into a pre-allocated output with `fill_cache`
        instead of concatenating the blocks afterwards.

        The try/except is load-bearing, not defensive. Block-sized L1 residents do run out on a
        smaller grid -- 16 of them is a measured `TT_THROW @ program.cpp:1052` even on qb1's
        larger one -- and F keeps the sliced block alive through fc1 on top of that. On a refusal
        this HALVES the block first, and only when the block cannot shrink further drops G, then
        F, then C-in, re-running the loop on the weaker rung each time; every refusal is cached
        per `padded_shape`, so a size that declines costs one exception per fold rather than one
        per block. Each rung is also reachable on its own through its env gate:
        both -0.842 s/fold at 512 aa on qb1, F alone -0.392, C-in alone -0.295.

        The block is `rows` rows of [1, rows, L, C], so at a fixed `rows` it grows with L and the
        L1 residents stop fitting at some length. Shrinking it is therefore the rung that fits the
        actual cause, and it keeps both destination levers; dropping G and F does not shrink
        anything, it just stops using L1. Measured on the 8x9 Wormhole Galaxy at 1024 aa: with
        `rows` pinned at 32 the size-ladder census reads F and G at 0 served / 538 declined, and
        with `rows` at 16 both read 538/0 and the fold goes 333.1/333.4 -> 325.6 s (-2.3 %) with
        an identical CIF sha256 (`131382c1247c0ff8`). ESMFold2 at 512 and 768 aa is untouched:
        `rows` stays 32 there because nothing throws. Bit-exact by the same argument as the row
        blocking itself -- the FFN is row-independent over dim 1, so regrouping rows cannot change
        a value.
        """
        L, key = x.shape[1], tuple(x.padded_shape)
        rows = min(rows, _ROW_BLOCK_SHRUNK.get(key, rows))
        nblk = -(-L // rows)
        lazy = _PAIR_FFN_L1_SLICE and key not in _L1_SLICE_REFUSED
        fused = residual and _PAIR_FFN_FUSED_RESIDUAL and key not in _FUSED_RESID_REFUSED
        # G needs each block's output to be the final content of its rows: fc2's output when
        # there is no residual, the folded add's under F. Either way that output sits in L1.
        filled = ((fused or not residual) and _PAIR_FFN_FILL_ASSEMBLY and nblk * rows == L
                  and key not in _FILL_ASSEMBLY_REFUSED)
        while True:
            parts = None if lazy else ttnn.chunk(x, nblk, dim=1)
            outs: list[ttnn.Tensor] = []
            dst = None
            try:
                if filled:
                    dst = ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, L, x.shape[2], x.shape[3]]), x.dtype, x.layout,
                        x.device(), ttnn.DRAM_MEMORY_CONFIG)
                    view = ttnn.experimental.view(dst, [nblk, rows, x.shape[2], x.shape[3]])
                for i in range(nblk):
                    part = (ttnn.slice(
                        x, [0, i * rows, 0, 0],
                        [1, min((i + 1) * rows, L), x.shape[2], x.shape[3]],
                        memory_config=ttnn.L1_MEMORY_CONFIG) if lazy else parts[i])
                    out = self._ffn(part, split=True, l1_gated=True,
                                    out_mc=ttnn.L1_MEMORY_CONFIG if fused or filled else None)
                    if fused:
                        out, ffn_out = ttnn.add(
                            part, out,
                            memory_config=(ttnn.L1_MEMORY_CONFIG if filled
                                           else ttnn.DRAM_MEMORY_CONFIG)), out
                        ttnn.deallocate(ffn_out)
                    ttnn.deallocate(part)
                    if filled:
                        ttnn.fill_cache(view, out, i)
                        ttnn.deallocate(out)
                    else:
                        outs.append(out)
            except Exception:
                for tensor in outs:
                    ttnn.deallocate(tensor)
                if dst is not None:
                    ttnn.deallocate(dst)
                if rows > PAIR_FFN_ROW_BLOCK_MIN and rows % 2 == 0:
                    # Shrink before giving up a lever: the block not fitting L1 is a size problem,
                    # and G and F are destinations, not sizes. Recompute everything derived from
                    # `rows`; `filled` additionally needs the blocks to tile L exactly.
                    rows //= 2
                    _ROW_BLOCK_SHRUNK[key] = rows
                    nblk = -(-L // rows)
                    filled = filled and nblk * rows == L
                elif filled:
                    _FILL_ASSEMBLY_REFUSED.add(key)
                    filled = False
                elif fused:
                    _FUSED_RESID_REFUSED.add(key)
                    fused = False
                elif lazy:
                    _L1_SLICE_REFUSED.add(key)
                    lazy = False
                else:
                    raise
                continue
            break
        L1_SLICE_STATS[0 if lazy else 1] += 1
        if residual:
            FUSED_RESID_STATS[0 if fused else 1] += 1
        FILL_ASSEMBLY_STATS[0 if filled else 1] += 1
        if filled:
            return dst
        out = ttnn.concat(outs, dim=1)
        for tensor in outs:
            ttnn.deallocate(tensor)
        if residual and not fused:
            out, ffn_out = ttnn.add(x, out), out
            ttnn.deallocate(ffn_out)
        return out

    def residual(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`x + ffn(x)`, with the add folded into the row block where that path applies (lever F).

        A separate entry point rather than a change to `__call__`, because `SwiGLUFFN` is shared
        with ESMC's LM FFN, Boltz-2, Protenix-v2, OpenFold3 and OpenDDE and none of them wants
        `x + ffn(x)`. Does not free `x`; the caller does, as it did around the add it used to own,
        except under `ws_resid`, where the kernel adds into `x` and returns `x` itself.
        """
        from tt_bio.tenstorrent import lever
        if lever("ws_resid"):
            out = self._weight_stationary(x, resid=x)
            if out is not None:
                return out
        out = self._weight_stationary(x)
        if out is not None:
            return ttnn.add_(out, x)  # x is the caller's to free
        split, rows = self._split_plan(x)
        rows = rows or self._rows_after_refusal(x)
        if rows:
            SPLIT_STATS[0] += 1
            return self._row_blocked(x, rows, residual=True)
        FUSED_RESID_STATS[1] += 1
        return ttnn.add(x, self(x))

    def _weight_stationary(self, x: ttnn.Tensor, resid: ttnn.Tensor | None = None):
        """`transition_ws` on ESMFold2's split pair FFN: the shared `tenstorrent.ws_swiglu` kernel over this
        module's weights in bfp8 on both archs, or None. bfp8 is what the lever was graded with (Wormhole's fast
        weights) and fits c_z 256 / d_ff 1024 in one chunk; Blackhole's fast keeps bf16 module weights, which
        took no kernel at all (b9: on and off wrote the same CIF), and two bf16 chunks carry wd1's chunk-0 fault."""
        from tt_bio import tenstorrent as T
        if not (self.split_swiglu and T._ws_applies(x, self.compute_kernel_config)):
            return None
        if getattr(self, "_ws", None) is None:
            w1, half = self.weights["1.weight"], self.weights["1.weight"].shape[0] // 2
            w3 = _scaled_t(self.out_scale)(self.weights["3.weight"])
            self._ws = T._pair_ws.weights(w1[:half].t(), w1[half:].t(), w3, self.device, ttnn.bfloat8_b)
            if not all(T._pair_ws.eligible(x, *p) for p in self._ws):  # the weights decide, not x: drop for good
                for p in self._ws:
                    for t in p:
                        ttnn.deallocate(t)
                self._ws = False
        if not self._ws:
            return None
        return T.ws_swiglu(x, self.norm_weight, self.norm_bias, self.compute_kernel_config, self._ws, resid=resid)

    def _lm_whole(self, x, L: int) -> bool:
        """Whether the per-token LM FFN runs over all L rows in one pass (see _LM_FFN_WHOLE_ROWS)."""
        fits = self.model_bytes <= _LM_LARGE_MODEL_BYTES or (self.model_bytes != float("inf") and L <= _LM_FFN_WHOLE_ROWS)
        return fits and tuple(x.padded_shape) not in _UNBLOCKED_REFUSED

    def __call__(self, x: ttnn.Tensor, resid: bool = False):
        """The FFN of x; with `resid`, (x + FFN(x), True) when fc2's matmul fused the add, else (FFN(x), False)."""
        if resid:
            served = MM_FUSE_STATS[1]
            out = self._call(x, resid=True)
            return out, MM_FUSE_STATS[1] != served
        return self._call(x)

    def _call(self, x: ttnn.Tensor, resid: bool = False) -> ttnn.Tensor:
        # The fc1 activation (2*d_ff wide) is several GB at long L and, on top of
        # the resident 6B weights, OOMs the 12 GB/chip Wormhole DRAM. The FFN is
        # row-independent over dim=1, so tiling it is bit-exact. 4D pair input
        # (ESMFold2 MSA-encoder pair_transition, [B,L,L,c]) has transient ~ rows*L
        # -> area-bounded tile; 3D per-token (ESMC LM FFN, [B,L,d]) -> fixed row
        # tile. See tenstorrent.pair_row_tile: Blackhole is single-pass to 1024 and
        # row-blocked above it, on its own budget.
        from tt_bio import tenstorrent
        L = x.shape[1]
        split, rows = self._split_plan(x)
        SPLIT_STATS[0 if split else 1] += 1
        rows = rows or self._rows_after_refusal(x)
        if rows:
            return self._row_blocked(x, rows, residual=False)
        if len(x.shape) == 4:
            chunk = tenstorrent.pair_row_tile(L)
        else:
            t = tenstorrent.SMALL_GRID_SEQ_TILE
            chunk = t if (t and L > t and not self._lm_whole(x, L)) else 0
        try:
            if chunk:
                parts = ttnn.chunk(x, -(-L // chunk), dim=1)
                return ttnn.concat([self._ffn(p, split=split) for p in parts], dim=1)
            WINDOW_FALLBACK_STATS[1] += 1
            return self._ffn(x, split=split, resid=resid)
        except Exception as exc:
            # Above PAIR_FFN_ROW_BLOCK_SEQ neither lever applies -- the row block is outside its
            # measured byte-identical window and pair_row_tile returns 0 on a big grid -- so fc1
            # allocates its whole 2*d_ff activation in one piece. On ESMFold2's pair transition
            # at 1536 tokens that is [1,1536,1536,1024] bf16, 4831838208 B, which DRAM refused on
            # a p150a with 26.67 of 31.875 GiB resident: room enough, no single block big enough.
            # The window exists to stop a lever changing the last bf16 bit of a fold that ALREADY
            # works. It cannot be protecting this one: there is no unblocked output here to be
            # byte-identical to. So take the row block after the refusal, never before it, which
            # leaves every size inside the window running exactly the path its parity was
            # measured on and turns a dead fold above the window into a live one.
            # Same shape as OpenFold3's refusal-narrowed OuterProductMean route.
            if len(x.shape) == 3 and not chunk and tenstorrent.SMALL_GRID_SEQ_TILE and L > tenstorrent.SMALL_GRID_SEQ_TILE \
                    and _is_alloc_refusal(exc):  # the LM FFN in one pass did not fit: tile this shape from now on
                _UNBLOCKED_REFUSED.add(tuple(x.padded_shape))
                WINDOW_FALLBACK_STATS[1] -= 1
                print(f"[lm-ffn] DRAM refused the one-pass LM FFN at {tuple(x.shape)}; re-running it in "
                      f"{tenstorrent.SMALL_GRID_SEQ_TILE}-row blocks", flush=True)
                return self(x)
            if len(x.shape) != 4 or not _PAIR_FFN_ROW_BLOCK or not _is_alloc_refusal(exc):
                raise
            _UNBLOCKED_REFUSED.add(tuple(x.padded_shape))
            WINDOW_FALLBACK_STATS[0] += 1
            WINDOW_FALLBACK_STATS[1] -= 1
            # Say it once per shape, on stdout, which `predict --debug` leaves connected. A
            # fold that only completed because the allocator refused first is a DIFFERENT
            # result from one that fit, and fragmentation is stateful enough that re-running
            # the same rung can pass by luck -- so without this line a PASS here cannot be
            # attributed to the fallback rather than to a tidier chip.
            print(f"[pair-ffn] DRAM refused the unblocked pair FFN at "
                  f"{tuple(x.shape)}; re-running it in {_PAIR_FFN_ROW_BLOCK}-row blocks",
                  flush=True)
            return self._row_blocked(x, _PAIR_FFN_ROW_BLOCK, residual=False)


class Block(Module):
    """UnifiedTransformerBlock, plain path (mirrors esm.layers.blocks):
      x = x + attn(x) / s ; x = x + ffn(x) / s,  s = sqrt(n_layers / 36).

    1/s is folded into the out_proj and fc2 weights in fp32 before their one cast to the device
    dtype, so each residual is a single add instead of a multiply and an add over the whole
    activation. The FFN takes the split-fc1 SwiGLU (two half matmuls, SiLU as the multiply's
    operand activation), which drops the chunk copy and the separate silu.
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        inv_scale = 1.0 / (n_layers / 36) ** 0.5
        self.ffn = SwiGLUFFN(self.scope("ffn"), compute_kernel_config, fuse_swiglu=True,
                             out_scale=inv_scale)
        shape = self.ffn.fc2_weight.shape  # ttnn.Shape takes no slice
        d_ff, d = int(shape[-2]), int(shape[-1])
        self.ffn.model_bytes = n_layers * (4 * d * d + 3 * d * d_ff) * 2
        self.attn = Attention(n_heads, self.scope("attn"), compute_kernel_config, out_scale=inv_scale,
                              qk_l1=(self.ffn.model_bytes <= _LM_LARGE_MODEL_BYTES
                                     or _qk_l1_large(self.ffn.fc2_weight.dtype)), d_model=d)
        self.attn.lm_large = self.ffn.lm_large = self.ffn.model_bytes > _LM_LARGE_MODEL_BYTES

    def __call__(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor,
                 attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None) -> ttnn.Tensor:
        r1, added = self.attn(x, cos, sin, attn_mask, key_valid, resid=True)
        if added:
            x = r1
        else:
            x = ttnn.add(x, r1)
            ttnn.deallocate(r1)
        r3, added = self.ffn(x, resid=True)
        if added:
            ttnn.deallocate(x)
            return r3
        x = ttnn.add(x, r3)
        ttnn.deallocate(r3)
        return x


class RegressionHead(Module):
    """Sequence head MLP (mirrors esm.layers.regression_head.RegressionHead, biases on):
      Linear -> GELU -> LayerNorm -> Linear.
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        row = lambda x: x.reshape(1, -1)
        self.fc1_weight = self.torch_to_tt("0.weight")
        self.fc1_bias = self.torch_to_tt("0.bias", transform=row)
        self.norm_weight = self.torch_to_tt("2.weight")
        self.norm_bias = self.torch_to_tt("2.bias")
        self.fc2_weight = self.torch_to_tt("3.weight")
        self.fc2_bias = self.torch_to_tt("3.bias", transform=row)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        a = self._lin(x, self.fc1_weight, bias=self.fc1_bias)
        a = ttnn.gelu(a)
        a = ttnn.layer_norm(
            a, weight=self.norm_weight, bias=self.norm_bias,
            epsilon=1e-5, compute_kernel_config=ck,
        )
        logits = self._lin(a, self.fc2_weight, bias=self.fc2_bias)
        ttnn.deallocate(a)
        return logits


class ESMCModel(Module):
    """Full ESMC stack: embed -> N blocks -> final LayerNorm (-> head).

    __call__ returns (logits[B,L,64], embeddings[B,L,d_model]); embeddings are
    the post-final-norm hidden states (matches esm.models.esmc.ESMC).
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.embed = Embedding(self.scope("embed"), compute_kernel_config)
        self.blocks = [
            Block(n_heads, n_layers, self.scope(f"transformer.blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.norm_weight = self.torch_to_tt("transformer.norm.weight")
        self.head = RegressionHead(self.scope("sequence_head"), compute_kernel_config)

    def __call__(self, tokens: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None, _rope=None, want_logits: bool = True):
        """With ``want_logits=False`` the sequence head is not run and ``logits`` is None."""
        seq_len = tokens.shape[-1]
        head_dim = self.norm_weight.shape[-1] // self.n_heads
        # _rope: precomputed (cos, sin) device tensors. Trace capture passes these
        # in because rope_tables uploads from the host, which a captured graph
        # cannot replay.
        cos, sin = _rope if _rope is not None else rope_tables(seq_len, head_dim, device=self.device)

        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask, key_valid)
        emb = ttnn.layer_norm(
            x, weight=self.norm_weight, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)
        logits = self.head(emb) if want_logits else None
        return logits, emb


class ESMC(TorchWrapper):
    """Top-level ESMC model (torch in / torch out). Mirrors esm.models.esmc.ESMC.

    Usage: m = ESMC(d_model, n_heads, n_layers); m.load_state_dict(sd); m(tokens).
    forward(tokens[int B,L]) -> (logits[B,L,64], embeddings[B,L,d_model]).

    ``trace`` (default on) replays single-sequence forwards as a captured ttnn
    device graph: the ~1300-op forward is host-dispatch-bound, so replaying one
    captured program instead of re-dispatching every op is ~1.5x faster per call
    (measured 17.7 -> 12.1 ms on ESMC-300M ubiquitin, p150a, ttnn 0.68). The win
    is specific to B=1 — batched forwards already amortize dispatch (measured
    ~1.02x at B=4), so batches always run eager. Tracing activates on the SECOND
    forward of a repeated input shape (one capture per bucketed shape, LRU-capped),
    so one-shot calls never pay the capture cost; output is bit-identical to
    eager (same captured program, fresh input contents). Needs the device opened
    with a trace region — ``load_esmc`` reserves one automatically; if the device
    was already open without one (e.g. a fleet worker that opened it at startup)
    the model simply stays eager.
    """

    # One captured trace per (bucketed length, mask layout) key. 8 concurrent
    # traces fit the reserved region with 2x headroom (one ESMC-300M trace at
    # 1534 residues is 1.25 MB on each of Wormhole's 12 banks) and cover the working set
    # of a length-sorted single-sequence stream. A capture that does not fit
    # raises, and _dispatch falls back to eager.
    #
    # ONE live trace, not 8: a replay writes its intermediates to the addresses they had at capture, which the
    # allocator freed afterwards. A second trace's inputs, rope tables and outputs, allocated after the first
    # capture, can land there, so replaying the first corrupts the second (esmc_trace_check.py, p150a, 600M:
    # 544 then 300 residues interleaved, the 300 replay read rel_l2 1.61 vs eager). Evicting before each capture
    # keeps every replayed trace the only one alive; a length switch costs a recapture.
    _TRACE_CACHE_MAX = env_int("TT_BIO_ESMC_TRACE_CACHE_MAX", 1)
    MATH_FIDELITY = "HiFi3"

    def __init__(self, d_model: int, n_heads: int, n_layers: int, *, trace: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.trace = trace
        self._trace_cache: collections.OrderedDict = collections.OrderedDict()
        self._trace_seen: collections.Counter = collections.Counter()
        self._trace_broken = False
        # HiFi3 with fp32 accumulation, every op. On a Blackhole batch of 7 (qb2 p300c, 1350 MHz) it cuts
        # 300M 88.5 -> 84.1 ms and 600M 135.0 -> 123.1 ms, with the error against the fp32 reference
        # unchanged (rel_l2 0.0252 vs 0.0256, 0.0250 vs 0.0236); on Wormhole it saves 1-5 %.
        # The ESMC-6B (ESMFold2's LM) is ESMCLanguageModel and keeps HiFi4.
        self.compute_kernel_config = type(self.compute_kernel_config)(
            math_fidelity=getattr(ttnn.MathFidelity, self.MATH_FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_pretrained(cls, name: str = "esmc-300m", *, trace: bool = True) -> "ESMC":
        """Download + load trained weights from HuggingFace (e.g. 'esmc-300m')."""
        from tt_bio import weights as _w

        config, repo_id, weights_path = CONFIGS[name]
        path = _w.fetch(name)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        model = cls(**config, trace=trace)
        model.load_state_dict(sd, strict=False)
        return model

    def _create_module(self, weights: WeightScope) -> ESMCModel:
        return ESMCModel(self.n_heads, self.n_layers, weights, self.compute_kernel_config)

    def _release_traces(self):
        for tr in self._trace_cache.values():
            try:
                ttnn.release_trace(self.tt_device, tr["tid"])
            except Exception:
                pass
            # Inputs/rope are ordinary device buffers allocated before capture;
            # the outputs live in the trace's pinned buffers and are freed by
            # release_trace itself, so they must not be deallocated here.
            for k in ("tokens", "mask", "kv", "cos", "sin"):
                self._deallocate_tensor_like(tr.get(k))
        self._trace_cache.clear()

    def reset_static_cache(self):
        super().reset_static_cache()
        self._release_traces()
        self._trace_seen.clear()

    def _capture_esmc_trace(self, tokens, attn_mask, key_valid, key):
        dev = self.tt_device
        # Evict BEFORE capturing: the new trace needs free space in the trace
        # region while the live ones still occupy it.
        while len(self._trace_cache) >= self._TRACE_CACHE_MAX:
            _, old = self._trace_cache.popitem(last=False)
            try:
                ttnn.release_trace(dev, old["tid"])
            except Exception:
                pass
            for k in ("tokens", "mask", "kv", "cos", "sin"):
                self._deallocate_tensor_like(old.get(k))
        tok_d = ttnn.from_torch(tokens.to(torch.int32), device=dev,
                                layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
        mask_d = None if attn_mask is None else ttnn.from_torch(
            attn_mask.unsqueeze(1).to(torch.bfloat16), device=dev,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        kv_d = None if key_valid is None else ttnn.from_torch(
            key_valid.to(torch.bfloat16), device=dev,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        rope = rope_tables(tokens.shape[-1], self.d_model // self.n_heads, device=dev)
        # warm compile + program cache outside the capture; also leaves fused_sdpa's zero mask
        # on the device for an unpadded forward, so the capture does not upload one
        for _ in range(2):
            wl, we = self.module(tok_d, mask_d, kv_d, _rope=rope)
            ttnn.deallocate(wl)
            ttnn.deallocate(we)
        ttnn.synchronize_device(dev)
        from .tenstorrent import capture_trace
        tid, (lg, em) = capture_trace(dev, lambda: self.module(tok_d, mask_d, kv_d, _rope=rope))
        tr = {"tid": tid, "tokens": tok_d, "mask": mask_d, "kv": kv_d,
              "cos": rope[0], "sin": rope[1], "logits": lg, "emb": em}
        self._trace_cache[key] = tr
        return tr

    @staticmethod
    def _host_tokens(tokens: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(tokens.to(torch.int32),
                               layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)

    def _forward_traced(self, tokens, attn_mask, key_valid):
        """Replay the captured device graph for this exact input shape.

        Caller (``forward``) gates on B==1 and shape repetition; this captures
        on first sight of ``key``. Bit-identical to ``_forward_eager``: the
        replayed graph is the exact captured program with fresh input buffer
        contents, and ``_to_torch`` copies out of the replay-owned output
        buffers before returning (never hands back a live trace buffer).
        """
        key = (tuple(tokens.shape), attn_mask is not None, key_valid is not None)
        tr = self._trace_cache.get(key)
        if tr is None:
            tr = self._capture_esmc_trace(tokens, attn_mask, key_valid, key)
        else:
            self._trace_cache.move_to_end(key)
        ttnn.copy_host_to_device_tensor(self._host_tokens(tokens), tr["tokens"])
        if attn_mask is not None:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(attn_mask.unsqueeze(1).to(torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                tr["mask"])
        if key_valid is not None:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(key_valid.to(torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
                tr["kv"])
        ttnn.execute_trace(self.tt_device, tr["tid"], cq_id=0, blocking=False)
        return self._to_torch(tr["logits"]), self._to_torch(tr["emb"])

    def _forward_eager(self, tokens, attn_mask, key_valid, want_logits=True):
        tokens_tt = ttnn.from_torch(
            tokens.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        mask_tt = None if attn_mask is None else ttnn.from_torch(
            attn_mask.unsqueeze(1).to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        kv_tt = None if key_valid is None else ttnn.from_torch(
            key_valid.to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        logits, emb = self.module(tokens_tt, mask_tt, kv_tt, want_logits=want_logits)
        return (None if logits is None else self._to_torch(logits)), self._to_torch(emb)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_valid: torch.Tensor | None = None, want_logits: bool = True):
        """tokens[B,L] -> (logits[B,L,64] or None without ``want_logits``, emb[B,L,d]). Optional padding masks
        (built by ``_batch_tokens``) let a batch of unequal-length sequences share
        one padded, bucketed forward: ``attn_mask`` [B,L,L] additive removes padded
        keys from the softmax denominator; ``key_valid`` [B,1,L,1] zeros padded
        keys/values so their contribution is exactly 0.

        The token axis is bucketed to a multiple of BUCKET HERE rather than only in
        ``_batch_tokens``, so a direct API call at a ragged L cannot reach the SDPA ragged.
        ``_batch_tokens`` has already bucketed on every CLI path, so there it is a no-op."""
        tokens, attn_mask, key_valid, _, L = bucket_token_axis(tokens, attn_mask, key_valid)
        logits, emb = self._dispatch(tokens, attn_mask, key_valid, want_logits)
        if int(tokens.shape[1]) == L:
            return logits, emb
        return (None if logits is None else logits[:, :L]), emb[:, :L]

    def _dispatch(self, tokens, attn_mask, key_valid, want_logits=True):
        if self.trace and not self._trace_broken and (tokens.shape[0] == 1 or _TRACE_BATCH):
            key = (tuple(tokens.shape), attn_mask is not None, key_valid is not None)
            if key in self._trace_cache:
                return self._forward_traced(tokens, attn_mask, key_valid)
            if trace_region_size() <= 0:
                if not getattr(self, "_trace_note_shown", False):
                    self._trace_note_shown = True
                    print("ESMC trace disabled: device was opened without a trace "
                          "region; running eager. Open with get_device("
                          "trace=\"esmc\") before load_esmc to enable.",
                          file=sys.stderr)
            else:
                # Capture on the SECOND sighting of a shape: tracing pays only
                # when a shape repeats (serving / repeated buckets); a one-shot
                # call stays pure eager and never pays the capture cost.
                self._trace_seen[key] += 1
                if len(self._trace_seen) > 4096:
                    self._trace_seen.clear()  # bound memory on huge varied inputs
                if self._trace_seen[key] >= 2:
                    try:
                        return self._forward_traced(tokens, attn_mask, key_valid)
                    except Exception as exc:
                        self._trace_broken = True
                        self._release_traces()
                        print(f"ESMC trace capture failed ({exc!r}); falling back "
                              f"to eager for the rest of this process", file=sys.stderr)
        return self._forward_eager(tokens, attn_mask, key_valid, want_logits)


# ===========================================================================
# ESMC-6B language-model backbone for ESMFold2
# ===========================================================================
#
# The 6B checkpoint ships in HuggingFace transformers / TransformerEngine
# layout (sharded safetensors, fused LayerNormLinear / LayerNormMLP modules),
# so its weight keys differ from the esm-repo names the ttnn blocks expect.
# This remap renames TE keys to the esm-repo `nn.Sequential`-index names, after
# which the existing `Block` / `Embedding` modules load unchanged.

_TE_KEY_REMAP = (
    ("attn.layernorm_qkv.layer_norm_weight", "attn.layernorm_qkv.0.weight"),
    ("attn.layernorm_qkv.layer_norm_bias", "attn.layernorm_qkv.0.bias"),
    ("attn.layernorm_qkv.weight", "attn.layernorm_qkv.1.weight"),
    ("ffn.layer_norm_weight", "ffn.0.weight"),
    ("ffn.layer_norm_bias", "ffn.0.bias"),
    ("ffn.fc1_weight", "ffn.1.weight"),
    ("ffn.fc2_weight", "ffn.3.weight"),
)


def load_esmc6b_state_dict(snapshot_dir: str, dtype: torch.dtype | None = None) -> dict:
    """Read the sharded 6B safetensors and remap TE keys to esm-repo names.

    Keeps only weights the ttnn stack consumes (embed, transformer blocks,
    final norm); drops `_extra_state`, the LM head and any classifier heads.
    ``dtype`` overrides the device-driven load dtype; the fp32 parity reference passes
    ``torch.float32`` so it reads the same snapshot as the device, not a second copy.
    """
    import glob
    import json
    import os

    from safetensors import safe_open

    import tt_bio.tenstorrent as _tt

    # Load straight to bf16 (the device dtype) so the upload moves/tiles half the
    # data — ~2.6x faster ESMC-6B load, bit-identical (fp32->bf16 rounding just
    # happens once, here vs in from_torch). In fast mode the big matmul weights
    # become block-fp8, whose quantization is sensitive to the fp32 mantissa, so
    # keep fp32 there to preserve exact fast-mode numerics.
    load_dtype = dtype or (torch.float32 if _tt._FAST_MODE else torch.bfloat16)
    idx_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(k)

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(snapshot_dir, shard), "pt") as f:
            for k in keys:
                if k.endswith("_extra_state") or k.startswith("lm_head"):
                    continue
                if not k.startswith("esmc."):
                    continue
                nk = k[len("esmc."):]  # drop the "esmc." prefix
                for src, dst in _TE_KEY_REMAP:
                    nk = nk.replace(src, dst)
                sd[nk] = f.get_tensor(k).to(load_dtype)
    _ = glob  # (kept for symmetry with other loaders)
    return sd


def load_esmc6b_shared(cache_dir: str, *, name: str = "esmc-6b", fast: bool = False):
    """Load ESMC-6B for data-parallel fanout via a shared /dev/shm tile cache.

    Data-parallel fanout ran O(N) redundant work: every one of the N card-workers
    independently read the 24 GB checkpoint, converted it, and tiled it on host --
    all bandwidth-bound, so per-worker load grew ~linearly with N and 6B fanout
    regressed past 2 cards. Here the first worker to arrive (the builder) does that
    work exactly once, publishing each tiled weight to ``cache_dir``; peers block on
    the build lock, then load the pre-tiled weights straight to their own card (no
    checkpoint read, no re-tiling) and pay only the per-card DMA -- which runs in
    parallel across the independent PCIe links. Bit-exact vs the single-card path:
    a loaded tile is exactly what from_torch would have produced.
    """
    import fcntl

    from tt_bio import weights as _w

    import tt_bio.tenstorrent as _tt

    _tt.set_fast_mode(fast)
    snap = _w.fetch("esmc-6b")
    os.makedirs(cache_dir, exist_ok=True)
    done = os.path.join(cache_dir, ".done")
    lockf = open(os.path.join(cache_dir, ".lock"), "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)  # one builder; peers wait here until .done
    try:
        if not os.path.exists(done):
            sd = load_esmc6b_state_dict(snap)
            _t = _time.perf_counter()
            with _tt.weight_cache(cache_dir, "dump"):
                model = ESMCLanguageModel(name=name)
                model.load_state_dict(sd, strict=False)
            open(done, "w").close()
            _tlog(f"cache_build {_time.perf_counter()-_t:.2f}s")
            return model
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()
    _t = _time.perf_counter()
    with _tt.weight_cache(cache_dir, "load"):
        model = ESMCLanguageModel(name=name)
        model.load_state_dict({}, strict=False)  # weights come from the tile cache
    _tlog(f"cache_load {_time.perf_counter()-_t:.2f}s")
    return model


class ESMCHiddenStatesModel(Module):
    """ESMC stack returning all `n_layers + 1` hidden states (ESMFold2 LM input).

    Matches `EsmcTransformerStack` collection semantics:
    `hs[0]` = embedding output, `hs[i]` = input to block `i` (= output of block
    `i-1`) for `1 <= i < n_layers`, and `hs[n_layers]` = final-LayerNorm output.
    Single-sequence / single-chain only (full attention, no padding) — which is
    how `compute_lm_hidden_states` feeds one wrapped chain at a time.
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embed = Embedding(self.scope("embed"), compute_kernel_config)
        self.blocks = [
            Block(n_heads, n_layers, self.scope(f"transformer.blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.norm_weight = self.torch_to_tt("transformer.norm.weight")

    def __call__(self, tokens: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None, last_hidden_only: bool = False,
                 combine=None):
        """Return the hidden states. With ``last_hidden_only`` only the final-norm one.

        ``combine = (w, ln_w, ln_b)`` returns ONE state instead, sum_k w[k] * LayerNorm(hs[k]) accumulated
        in fp32 on the device: ESMFold2's LanguageModelShim takes LayerNorm, a bias-free linear and a
        weighted sum over the layers, so the sum moves ahead of the linear and the 80 per-layer copies
        (each a pipeline drain) become one.

        The intermediate ``_to_host`` calls exist for ESMFold2, whose LanguageModelShim
        consumes all ``n_layers + 1`` states. The standalone embed path uses only the last
        (``_trunk_forward`` takes ``[-1, 0]``), and for the 6B that means 80 blocking
        device->host copies per forward whose results are thrown away. Each one is also a
        pipeline drain -- the host cannot dispatch block ``i+1`` until block ``i``'s copy has
        returned -- so skipping them recovers the dispatch/compute overlap as well as the
        traffic. Default is unchanged, so ESMFold2 is untouched by construction.
        """
        seq_len = tokens.shape[-1]
        head_dim = self.norm_weight.shape[-1] // self.n_heads
        cos, sin = rope_tables(seq_len, head_dim, device=self.device)

        x = self.embed(tokens)
        if combine is not None:
            w, ln_w, ln_b = combine
            acc = [None]

            def keep(t, k):
                y = ttnn.layer_norm(t, weight=ln_w, bias=ln_b, epsilon=1e-5,
                                    compute_kernel_config=self.compute_kernel_config)
                y = ttnn.multiply(ttnn.typecast(y, ttnn.float32), float(w[k]))
                acc[0] = y if acc[0] is None else ttnn.add(acc[0], y)
        else:
            hidden = []

            def keep(t, k):
                hidden.append(self._to_host(t))
        if not last_hidden_only:
            keep(x, 0)  # hs[0] = embedding output
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, key_valid)
            if not last_hidden_only and i < self.n_layers - 1:
                keep(x, i + 1)  # hs[i+1] = block i output
        norm_x = ttnn.layer_norm(
            x, weight=self.norm_weight, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)
        if combine is None:
            hidden.append(self._to_host(norm_x))  # hs[n_layers] = post-norm output
            return hidden
        keep(norm_x, self.n_layers)
        return [self._to_host(acc[0])]

    @staticmethod
    def _to_host(t: ttnn.Tensor) -> torch.Tensor:
        return torch.Tensor(ttnn.to_torch(t)).float()


class ESMCLanguageModel(TorchWrapper):
    """ESMC-6B backbone (torch in / torch out) producing ESMFold2 LM hidden states.

    `forward(input_ids[B,L])` -> hidden states `[n_layers+1, B, L, d_model]`,
    matching `transformers` ESMC `output_hidden_states=True` (the stacked input
    consumed by ESMFold2's `LanguageModelShim`).
    """

    def __init__(self, name: str = "esmc-6b"):
        super().__init__()
        cfg = ARCH_CONFIGS[name]
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_layers = cfg["n_layers"]

    @classmethod
    def from_pretrained(cls, repo_id: str = "biohub/ESMC-6B", name: str = "esmc-6b") -> "ESMCLanguageModel":
        from tt_bio import weights as _w

        snap = _w.fetch(name) if name in _w.ARTIFACTS else _w.fetch_hf_repo(repo_id)
        model = cls(name=name)
        model.load_state_dict(load_esmc6b_state_dict(snap), strict=False)
        return model

    def _create_module(self, weights: WeightScope) -> ESMCHiddenStatesModel:
        return ESMCHiddenStatesModel(self.n_heads, self.n_layers, weights, self.compute_kernel_config)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None,
                last_hidden_only: bool = False, combine=None) -> torch.Tensor:
        """``[n_layers+1, B, L, d]``, or ``[1, B, L, d]`` with ``last_hidden_only`` or ``combine``
        (the layer-combined state, see ESMCHiddenStatesModel).

        Either way ``[-1]`` is the final-norm state, which is the whole contract
        ``_trunk_forward`` depends on.
        """
        B, Lm = input_ids.shape
        # Bucket the LM length to a multiple of 64 so the 80-layer ESMC kernels
        # are shared across nearby lengths instead of recompiling per length.
        # Padded tokens are masked out of attention (additive -inf, seq_id-style
        # mask like the reference) and sliced off — the residual numerical effect
        # is within the diffusion's seed-to-seed noise floor.
        Lb = ((Lm + BUCKET - 1) // BUCKET) * BUCKET
        if Lb != Lm:
            pad = Lb - Lm
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=PAD_TOKEN)
            if attn_mask is None:
                attn_mask = torch.zeros(B, Lb, Lb, dtype=torch.float32)
            else:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad, 0, pad), value=0.0)
            attn_mask[:, :, Lm:] = float("-inf")  # no token attends to padded keys
        tokens_tt = ttnn.from_torch(
            input_ids.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        mask_tt = key_valid_tt = None
        if attn_mask is not None:
            # [B,L,L] additive mask -> [B,1,L,L] bf16 for SDPA
            mask_tt = ttnn.from_torch(
                attn_mask.unsqueeze(1).to(torch.bfloat16), device=self.tt_device,
                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            )
        if Lb != Lm:
            kv = torch.ones(1, 1, Lb, 1); kv[:, :, Lm:, :] = 0.0  # zero padded keys/values
            key_valid_tt = ttnn.from_torch(
                kv.to(torch.bfloat16), device=self.tt_device,
                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            )
        hidden = self.module(tokens_tt, mask_tt, key_valid_tt,
                             last_hidden_only=last_hidden_only, combine=combine)  # list of [B, Lb, d_model]
        return torch.stack(hidden, dim=0)[:, :, :Lm, :]  # slice padding -> [n+1, B, Lm, d_model]

    def release(self):
        """Free all ttnn device weights (≈12.8 GB for the 6B). Call after the
        single LM forward so the folding trunk reclaims DRAM on long sequences.
        Hidden states are already on host, so only weights are released."""
        if self.module is not None:
            _free_ttnn_tensors(self.module)
            self.module = None


def _free_ttnn_tensors(obj, seen=None):
    """Recursively ttnn.deallocate every device tensor reachable from `obj`."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, ttnn.Tensor):
        try:
            ttnn.deallocate(obj)
        except Exception:
            pass
        return
    if isinstance(obj, (list, tuple, set)):
        for x in obj:
            _free_ttnn_tensors(x, seen)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            _free_ttnn_tensors(x, seen)
        return
    d = getattr(obj, "__dict__", None)
    if d:
        for x in list(d.values()):
            _free_ttnn_tensors(x, seen)


# ===========================================================================
# Standalone embedding API (sequence -> per-residue + pooled embeddings)
# ===========================================================================
#
# The LM trunk alone — no folding head, no MSA: a protein string in, its
# per-residue and pooled final-layer hidden-state embeddings out (plus the
# sequence-head logits on request). Thin wrappers over the ESMC / ESMC-6B
# forwards above: tokenize, run, strip the <cls>/<eos> special tokens so rows
# align 1:1 with residues, then pool.

MODELS = tuple(CONFIGS) + ("esmc-6b",)

_POOLERS = {
    "mean": lambda e: e.mean(axis=0),
    "max": lambda e: e.max(axis=0),
    "cls": None,  # uses the <cls> summary token; handled before stripping
}


@dataclass
class ESMCEmbedding:
    """One sequence's embeddings from the ESMC language-model trunk.

    ``per_residue`` has one row per amino acid — the <cls>/<eos> special tokens
    are stripped, so ``per_residue[i]`` is residue ``sequence[i]``. ``pooled`` is
    a single fixed-size vector (see the ``pool`` argument). ``logits`` are the
    per-residue sequence-head logits ([L, 64]) when requested — ESMC-300M/600M
    only, since the 6B port carries no sequence head.
    """

    id: str
    sequence: str
    per_residue: np.ndarray            # [L, d_model] float32
    pooled: np.ndarray                 # [d_model] float32
    logits: Optional[np.ndarray]       # [L, 64] float32 or None


def read_fasta(path) -> dict[str, str]:
    """Parse a FASTA file into an ordered {id: sequence} dict (uppercased).

    Colliding record ids are disambiguated with a numeric suffix so no sequence
    is silently dropped.
    """
    seqs: dict[str, str] = {}
    sid, buf = None, []

    def flush():
        if sid is None:
            return
        seq = "".join(buf).upper()
        name = sid
        n = 2
        while name in seqs:
            name = f"{sid}_{n}"
            n += 1
        seqs[name] = seq

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            sid = line[1:].split()[0] if line[1:].split() else f"seq{len(seqs)}"
            buf = []
        else:
            buf.append(line)
    flush()
    return seqs


def read_yaml(path) -> dict[str, str]:
    """Parse a YAML {id: sequence} mapping into an ordered dict (uppercased)."""
    import yaml

    doc = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(doc, dict) or not doc or not all(isinstance(v, str) for v in doc.values()):
        raise ValueError(f"{path}: expected a YAML mapping of {{id: sequence}}, got {doc!r}")
    return {str(k): "".join(v.split()).upper() for k, v in doc.items()}


def load_sequences(data) -> dict[str, str]:
    """Load {id: sequence} from a path or a bare sequence string.

    ``data`` may be: a FASTA file, a directory of FASTA files (merged, id
    collisions disambiguated), a YAML file (a flat ``{id: sequence}`` mapping),
    or — if it isn't an existing path — a single raw protein sequence (given
    the id ``seq0``).
    """
    path = Path(data).expanduser()
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".fa", ".fasta", ".fas"))
        if not files:
            raise ValueError(f"no FASTA files (.fa/.fasta/.fas) found in {path}")
        seqs: dict[str, str] = {}
        for fp in files:
            for sid, seq in read_fasta(fp).items():
                key, n = sid, 2
                while key in seqs:
                    key, n = f"{sid}_{n}", n + 1
                seqs[key] = seq
    elif path.is_file():
        suffix = path.suffix.lower()
        if suffix in (".fa", ".fasta", ".fas"):
            seqs = read_fasta(path)
        elif suffix in (".yml", ".yaml"):
            seqs = read_yaml(path)
        else:
            raise ValueError(f"unsupported file type {suffix!r} for {path} "
                             "(use .fasta/.fa/.fas or .yml/.yaml)")
    else:
        seq = str(data).strip()
        if not (seq and seq.replace(" ", "").isalpha()):
            raise ValueError(f"{data!r} is not an existing file/directory, and not a bare "
                             "protein sequence (letters only)")
        seqs = {"seq0": "".join(seq.split()).upper()}
    if not seqs:
        raise ValueError(f"no sequences found in {data}")
    # The tokenizer maps anything it does not know to <unk>, so a digit or a stray symbol in a
    # file would embed as a residue that is not there. The bare-string path already refused it.
    for sid, seq in seqs.items():
        bad = [f"{i + 1}{c}" for i, c in enumerate(seq) if not c.isalpha()]
        if bad:
            raise ValueError(f"sequence {sid!r} has non-letter character(s) at "
                             f"{', '.join(bad[:8])}: a protein sequence is letters only")
    return seqs


# The trace region an ESMC-300M/600M load reserves (tenstorrent.TRACE_REGIONS["esmc"]) is not
# free: it comes off EVERY DRAM bank, so on a 12-bank Wormhole chip every MiB of it costs 12 MiB.
# Measured on j10glx02 off the allocator's own refusal: "bank size is 805306336 B" with a 256 MiB
# region against "1073741792 B" without. On the sequence-length axis that was the difference
# between esmc-300m refusing 65537 residues and saprot-35m, same code path and no region,
# embedding 73728 on the same part. The region is sized to the 8 live traces (221 MiB on
# Wormhole, since ttnn checks their total against it), so trace_pays still matters.


def trace_pays(sequences, bucket: int = BUCKET) -> bool:
    """Whether reserving a trace region can pay for itself on this workload.

    ``_dispatch`` captures a trace only on the SECOND sighting of a bucketed shape, on purpose:
    "tracing pays only when a shape repeats ... a one-shot call stays pure eager and never pays
    the capture cost". So on a workload where no bucketed width repeats, the region is reserved,
    never captured into, and never replayed -- while still costing its bytes on every DRAM bank
    (221 MiB per bank, 2.6 GiB of a Wormhole chip, and 1.18x of the sequence ceiling at 256 MiB).

    This is the same condition, asked one step earlier, where it can still be acted on: the
    reservation has to happen at device OPEN and cannot be taken back once a capture turns out to
    be worth it. Deliberately an OVER-approximation -- it ignores how ``_batch_tokens`` will group
    rows, so it can say yes where the runtime would not capture. Saying yes is the old behaviour
    and costs only memory; saying no where a trace WOULD have been replayed would cost speed, so
    the error is pointed at the side that was already being paid.
    """
    seqs = ([sequences] if isinstance(sequences, str)
            else list(sequences.values()) if isinstance(sequences, dict) else list(sequences))
    widths = collections.Counter(
        ((len(q) + 2 + bucket - 1) // bucket) * bucket for q in seqs)
    return any(n >= 2 for n in widths.values())


def load_esmc(name: str = "esmc-300m", *, fast: bool = False, trace: bool = True):
    """Load an ESMC model onto the TT device. ``name`` is one of ``MODELS``.

    300M/600M load from a single esm-repo .pth (with a sequence head, so logits
    are available); 6B loads from the sharded TransformerEngine safetensors
    (embeddings only). ``fast`` selects the block-fp8 weight path and must be set
    before the weights are materialized, hence here rather than at call time.
    ``trace`` (300M/600M only) enables traced single-sequence forwards — see
    ``ESMC``; it reserves a trace region when this call opens the device, and is
    a no-op for esmc-6b (the hidden-states shim reads every layer back to the
    host, so its forward cannot be captured).
    """
    from tt_bio import tenstorrent

    tenstorrent.set_fast_mode(fast)
    if name == "esmc-6b":
        return ESMCLanguageModel.from_pretrained(name=name)
    if name not in CONFIGS:
        raise ValueError(f"unknown ESMC model {name!r}; choose from {list(MODELS)}")
    if trace:
        # Reserve the trace region up front. If the device is already open this
        # returns it unchanged and forward() simply stays eager (it checks
        # trace_region_size() per call).
        get_device(trace="esmc")
    return ESMC.from_pretrained(name, trace=trace)


def _trunk_forward(model, seq: str, return_logits: bool):
    """Run the LM trunk on one sequence (used for the 6B backbone).

    Returns (per_residue[L, d], cls[d], logits[L, 64] | None) as float32 numpy,
    with the <cls>/<eos> special tokens stripped from per_residue/logits.
    """
    tokens = tokenize(seq)  # [1, len(seq)+2] with <cls> … <eos>
    logits = None
    if isinstance(model, ESMCLanguageModel):
        # Only the final-norm state is used, so do not pay for the other n_layers
        # readbacks (80 for the 6B). See ESMCHiddenStatesModel.__call__.
        emb = model(tokens, last_hidden_only=True)[-1, 0]   # final-norm hidden state [L+2, d]
    else:
        lg, em = model(tokens)              # [1, L+2, 64], [1, L+2, d]
        emb = em[0]
        if return_logits:
            logits = lg[0][1:-1].numpy().astype(np.float32)
    emb = emb.numpy().astype(np.float32)
    return emb[1:-1], emb[0], logits


def bucket_token_axis(tokens, attn_mask=None, key_valid=None, embed_mask=None,
                      bucket: int = BUCKET, pad_token: int = PAD_TOKEN):
    """Pad a forward's token axis to a multiple of *bucket* and extend the padding masks.

    ``_batch_tokens`` already buckets, so on every shipped CLI path ``Lb == L`` and this returns
    its arguments unchanged. It exists for the direct API caller, who reaches ``Model.forward``
    with whatever length they have: a ragged L used to go straight into the SDPA and pick up its
    padded key columns at a bias of zero. See ``tt_bio/token_axis.py`` and PLAYBOOKS.md §MODEL 2b.
    An unpadded call keeps ``attn_mask=None``; ``tenstorrent.fused_sdpa`` masks it.

    Returns ``(tokens, attn_mask, key_valid, embed_mask, L)``; slice the outputs back to ``L``.
    """
    L = int(tokens.shape[1])
    Lb = ((L + bucket - 1) // bucket) * bucket
    if Lb == L:
        return tokens, attn_mask, key_valid, embed_mask, L
    B, pad = int(tokens.shape[0]), Lb - L
    tokens = torch.nn.functional.pad(tokens, (0, pad), value=pad_token)
    # Same construction as _batch_tokens: additive -inf takes padded keys out of the softmax
    # denominator, key_valid zeroes their value contribution, embed_mask zeroes their embedding.
    attn_mask = (torch.zeros(B, Lb, Lb, dtype=torch.float32) if attn_mask is None
                 else torch.nn.functional.pad(attn_mask, (0, pad, 0, pad), value=0.0))
    attn_mask[:, :, L:] = float("-inf")
    key_valid = (torch.ones(B, 1, Lb, 1, dtype=torch.float32) if key_valid is None
                 else torch.nn.functional.pad(key_valid, (0, 0, 0, pad), value=0.0))
    key_valid[:, :, L:, :] = 0.0
    if embed_mask is not None:
        embed_mask = torch.nn.functional.pad(embed_mask, (0, 0, 0, pad), value=0.0)
    return tokens, attn_mask, key_valid, embed_mask, L


def _batch_tokens(seqs: list[str], bucket: int = BUCKET):
    """Pad a batch of sequences to a common bucketed length and build padding masks.

    Each sequence is tokenized to ``[<cls> … <eos>]`` (length ``len(seq)+2``); the
    batch is right-padded with ``<pad>`` to ``Lb`` = the smallest multiple of
    ``bucket`` covering the longest row. Bucketing means nearby lengths share one
    compiled program (the per-length JIT compile — not device exec — is the CLI
    embed bottleneck). Returns ``(input_ids[B,Lb], lens, attn_mask[B,Lb,Lb] | None,
    key_valid[B,1,Lb,1] | None)`` where ``lens[i]`` is row ``i``'s real token count.
    The masks are ``None`` only when no row is padded (all equal length == Lb)."""
    tok = [tokenize(s)[0] for s in seqs]         # list of 1D LongTensors
    lens = [int(t.numel()) for t in tok]
    Lb = ((max(lens) + bucket - 1) // bucket) * bucket
    B = len(seqs)
    input_ids = torch.full((B, Lb), PAD_TOKEN, dtype=torch.long)
    for i, t in enumerate(tok):
        input_ids[i, :lens[i]] = t
    if all(li == Lb for li in lens):
        return input_ids, lens, None, None
    attn_mask = torch.zeros(B, Lb, Lb, dtype=torch.float32)
    key_valid = torch.ones(B, 1, Lb, 1, dtype=torch.float32)
    for i, li in enumerate(lens):
        attn_mask[i, :, li:] = float("-inf")     # no query attends to padded keys
        key_valid[i, :, li:, :] = 0.0            # padded keys/values contribute 0
    return input_ids, lens, attn_mask, key_valid


def embed_sequences(model, sequences: dict[str, str], *, return_logits: bool = False,
                    pool: str = "mean", batch_size: int = 8) -> list[ESMCEmbedding]:
    """Embed each {id: sequence} with an already-loaded ESMC ``model``.

    For the 300M/600M models, sequences are grouped (sorted by length to minimise
    padding) into batches of up to ``batch_size`` and run through a single padded,
    length-bucketed device forward per batch — padded positions are masked out of
    attention so each row's embeddings are identical to running it alone. This
    amortises the per-length kernel compile and host dispatch that dominate the
    one-at-a-time path. The 6B backbone stays one-sequence-at-a-time (its forward
    already buckets, and its ~13 GB of resident weights leave no room to widen the
    batch). ``pool`` in {"mean", "max", "cls"} selects the pooled vector.
    """
    if pool not in _POOLERS:
        raise ValueError(f"unknown pool {pool!r}; choose from {sorted(_POOLERS)}")
    for sid, seq in sequences.items():
        if not seq:
            raise ValueError(f"sequence {sid!r} is empty")

    # 6B backbone: no cross-sequence batching (already bucketed, weight-bound).
    if isinstance(model, ESMCLanguageModel):
        results = []
        for sid, seq in sequences.items():
            model.reset_static_cache()
            per_residue, cls, logits = _trunk_forward(model, seq, return_logits)
            pooled = cls if pool == "cls" else _POOLERS[pool](per_residue)
            results.append(ESMCEmbedding(sid, seq, per_residue,
                                         pooled.astype(np.float32), logits))
        return results

    items = list(sequences.items())
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))  # short→long
    # Sorting keeps each batch's lengths close (little padding waste). A token
    # budget caps rows*bucketed_len so batches auto-shrink toward 1 for long
    # sequences — full batch_size for short seqs, no OOM on a long-protein FASTA.
    budget = batch_size * _MAX_BATCH_TOKENS_PER_SEQ
    batches, cur, cur_max = [], [], 0
    for i in order:
        tok = len(items[i][1]) + 2
        nxt_max = max(cur_max, ((tok + BUCKET - 1) // BUCKET) * BUCKET)
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * nxt_max > budget):
            batches.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, ((tok + BUCKET - 1) // BUCKET) * BUCKET)
    if cur:
        batches.append(cur)

    by_id: dict[str, ESMCEmbedding] = {}
    for idx in batches:
        batch = [items[i] for i in idx]
        input_ids, lens, attn_mask, key_valid = _batch_tokens([s for _, s in batch])
        logits_b, emb_b = model(input_ids, attn_mask, key_valid, want_logits=return_logits)  # [B,Lb,64] | None, [B,Lb,d]
        for row, (sid, seq) in enumerate(batch):
            li = lens[row]
            emb = emb_b[row, :li].numpy().astype(np.float32)
            per_residue, cls = emb[1:-1], emb[0]
            logits = (logits_b[row, 1:li - 1].numpy().astype(np.float32)
                      if return_logits else None)
            pooled = cls if pool == "cls" else _POOLERS[pool](per_residue)
            by_id[sid] = ESMCEmbedding(sid, seq, per_residue,
                                       pooled.astype(np.float32), logits)
    return [by_id[sid] for sid, _ in items]  # restore input order


def _shard_by_length(items: list, n: int, *,
                     key=lambda it: len(it[1])) -> list[list]:
    """Split ``(id, payload)`` pairs into ``n`` balanced shards for data-parallel embedding.

    Pairs are length-sorted (by ``key``, default the character length of the pair's
    second element) and striped round-robin across shards, so every shard gets a
    similar length distribution (tight length-bucketing, little padding waste) and a
    balanced total workload — long sequences don't all land on one card. Input
    ordering is irrelevant here; results are reassembled by id afterwards. ``key`` is
    overridden by callers whose payload isn't a bare string (e.g. SaProt's
    ``(aa, struc)`` tuple sorts on the AA length).
    """
    shards: list[list] = [[] for _ in range(n)]
    order = sorted(range(len(items)), key=lambda i: key(items[i]))
    for rank, i in enumerate(order):
        shards[rank % n].append(items[i])
    return shards


def _reassemble(items: list[tuple[str, object]],
                shard_results: list[list]) -> list:
    """Flatten per-shard embeddings and restore the original ``items`` order.

    Duck-typed on the per-sequence result's ``.id`` (both ESMCEmbedding and
    SaprotEmbedding expose one), so the same helper reassembles ESMC and SaProt
    fanout results.
    """
    by_id: dict[str, object] = {}
    for res in shard_results:
        for emb in res:
            by_id[emb.id] = emb
    return [by_id[sid] for sid, _ in items]


def _run_embed_shard(in_path: str, out_path: str) -> None:
    """Subprocess entry point: embed one shard on the pinned card, pickle results.

    Invoked as a fresh interpreter with ``TT_VISIBLE_DEVICES`` already set in the
    environment (so the assigned physical chip is logical device 0 and ttnn, imported
    at module load, binds to it). Reads a pickled request
    ``{model, sequences, fast, return_logits, pool, batch_size}`` and writes the
    resulting ``list[ESMCEmbedding]``.
    """
    with open(in_path, "rb") as f:
        req = pickle.load(f)
    _t = _time.perf_counter()
    if req.get("cache_dir"):
        model = load_esmc6b_shared(req["cache_dir"], name=req["model"], fast=req["fast"])
    else:
        model = load_esmc(req["model"], fast=req["fast"],
                          trace=trace_pays(req["sequences"]))
    _tlog(f"load_total {_time.perf_counter()-_t:.2f}s")
    results = embed_sequences(model, req["sequences"], return_logits=req["return_logits"],
                              pool=req["pool"], batch_size=req["batch_size"])
    with open(out_path, "wb") as f:
        pickle.dump(results, f)


def _thread_cap_env(n_workers: int) -> dict:
    """Cap each shard's torch/OMP/BLAS host thread pools to cores/n_workers.

    Each subprocess's numpy/torch pools otherwise default to ALL host cores, so N
    co-resident shards spawn N*cores threads that thrash the host CPU -- confirmed
    via `ps -eLo pcpu` during a 4-card esmc-6b run (each shard bursts to 200-380%
    CPU, host loadavg > 2x core count) as the residual fanout regression left after
    fixing the weight-load contention. Mirrors
    the identical fix already applied to the fleet worker pool in
    ``main._cap_worker_threads``; an operator-set value wins.
    """
    from . import runtime

    return runtime.host_thread_cap_env(n_workers)


def _spawn_shard(idx: int, device: int, shard: list[tuple[str, str]], workdir: str, *,
                 model: str, fast: bool, return_logits: bool, pool: str, batch_size: int,
                 cache_dir: str | None = None, thread_cap_env: dict | None = None):
    """Launch a pinned subprocess embedding ``shard`` on physical card ``device``.

    Returns ``(proc, out_path, device, log_path, logf)``. The child sets
    ``TT_VISIBLE_DEVICES=<device>`` via its environment so the chip is logical device 0
    (see ``get_device``). stdout/stderr go to a per-shard log file rather than a pipe,
    so a chatty ttnn child never deadlocks on a full pipe buffer.
    """
    in_path = os.path.join(workdir, f"shard{idx}.in.pkl")
    out_path = os.path.join(workdir, f"shard{idx}.out.pkl")
    log_path = os.path.join(workdir, f"shard{idx}.log")
    with open(in_path, "wb") as f:
        pickle.dump(dict(model=model, sequences=dict(shard), fast=fast,
                         return_logits=return_logits, pool=pool, batch_size=batch_size,
                         cache_dir=cache_dir), f)
    env = {**os.environ, **(thread_cap_env or {}),
           "TT_VISIBLE_DEVICES": str(device), "TT_BIO_LOGICAL_DEVICE_ID": "0"}
    # P300 boards are a custom topology; like the `predict` and `boltzgen gen`
    # fanout paths, a single-chip worker needs the 1x1 Blackhole mesh-graph
    # descriptor or ttnn.open_device aborts with "Custom fabric mesh graph
    # descriptor path must be specified".
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(env, device)
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from tt_bio.esmc import _run_embed_shard; "
         "_run_embed_shard(sys.argv[1], sys.argv[2])",
         in_path, out_path],
        env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc, out_path, device, log_path, logf


def _read_log_tail(path: str, n: int) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _await_shard(proc, out_path: str, device: int, log_path: str, logf) -> list[ESMCEmbedding]:
    """Wait for a shard subprocess and return its embeddings (raises with log tail)."""
    proc.wait()
    logf.close()
    if proc.returncode != 0:
        raise RuntimeError(f"embed shard on device {device} failed "
                           f"(exit {proc.returncode}):\n{_read_log_tail(log_path, 25)}")
    with open(out_path, "rb") as f:
        return pickle.load(f)


def _shm_dir() -> str:
    """RAM-backed scratch dir for the shared tile cache; falls back to $TMPDIR."""
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


def embed_multicard(sequences: dict[str, str], *, model: str, devices: list[int],
                    fast: bool = False, return_logits: bool = False, pool: str = "mean",
                    batch_size: int = 8) -> list[ESMCEmbedding]:
    """Data-parallel ESMC embedding across multiple physical TT cards.

    Shards ``sequences`` across ``devices`` (one pinned subprocess per card), runs the
    single-card :func:`embed_sequences` in each, then gathers and reassembles the
    embeddings in original input order. Embarrassingly parallel: ESMC embeddings are
    row-independent (no cross-sequence state), so a sequence's output is identical to
    running it on one card — sharding changes only which chip computes which row.

    More cards than sequences is harmless: extra cards simply get no shard.
    """
    items = list(sequences.items())
    devices = list(devices)[:max(1, len(items))]
    shards = _shard_by_length(items, len(devices))
    workdir = tempfile.mkdtemp(prefix="tt-bio-embed-fanout-")
    # ESMC-6B weights (~24 GB) dominate fanout wall-clock and are identical across
    # workers, so share one host-tiled copy via /dev/shm instead of each worker
    # re-reading+re-tiling the checkpoint (which regressed past 2 cards).
    cache_dir = (tempfile.mkdtemp(prefix="esmc6b-tiles-", dir=_shm_dir())
                 if model == "esmc-6b" else None)
    thread_cap_env = _thread_cap_env(len(devices))
    try:
        handles = [
            _spawn_shard(idx, dev, shard, workdir, model=model, fast=fast,
                         return_logits=return_logits, pool=pool, batch_size=batch_size,
                         cache_dir=cache_dir, thread_cap_env=thread_cap_env)
            for idx, (dev, shard) in enumerate(zip(devices, shards)) if shard
        ]
        results = [_await_shard(*h) for h in handles]
    finally:
        if not os.environ.get("TT_BIO_KEEP_FANOUT_WORKDIR"):
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"fanout workdir kept: {workdir}")
        if cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)
    return _reassemble(items, results)


def embed(sequences, model: str = "esmc-300m", *, fast: bool = False,
          return_logits: bool = False, pool: str = "mean", batch_size: int = 8,
          devices: list[int] | None = None) -> list[ESMCEmbedding]:
    """One-shot embedding: load ``model`` and embed ``sequences``.

    ``sequences`` may be a single string, a list of strings (auto-named seq0…),
    or an {id: sequence} dict. Returns one ESMCEmbedding per input sequence.

    ``devices`` shards the input across multiple physical TT cards (one pinned
    subprocess each, data-parallel); with 0 or 1 device the model is loaded in-process
    on the single card this process already sees.
    """
    if isinstance(sequences, str):
        sequences = {"seq0": sequences}
    elif isinstance(sequences, (list, tuple)):
        sequences = {f"seq{i}": s for i, s in enumerate(sequences)}
    if devices and len(devices) > 1:
        return embed_multicard(sequences, model=model, devices=devices, fast=fast,
                               return_logits=return_logits, pool=pool, batch_size=batch_size)
    m = load_esmc(model, fast=fast, trace=trace_pays(sequences))
    return embed_sequences(m, sequences, return_logits=return_logits, pool=pool,
                           batch_size=batch_size)


def write_npz(emb: ESMCEmbedding, path) -> None:
    """Write one sequence's full embeddings to a compressed .npz."""
    arrays = dict(per_residue=emb.per_residue, pooled=emb.pooled,
                  sequence=np.array(emb.sequence))
    if emb.logits is not None:
        arrays["logits"] = emb.logits
    np.savez_compressed(path, **arrays)


def write_npz_many(embeddings, out_dir, max_workers: int | None = None) -> None:
    """Write one npz per embedding, parallel across host threads.

    np.savez_compressed spends its time in zlib's C compress loop, which
    releases the GIL on multi-KB buffers, so threads give a near-linear
    speedup. The serial loop otherwise dominates multicard embed wall-clock:
    measured 83 ms/seq (72 s of an 83 s 864-sequence 4-card run sat in the
    parent's write phase while the shards' device work took ~7 s).
    """
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, min(32, os.cpu_count() or 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda e: write_npz(e, out_dir / f"{e.id}.npz"), embeddings))


def write_parquet(embeddings: list[ESMCEmbedding], path) -> None:
    """Write the pooled embedding matrix (one row per sequence) to Parquet.

    Per-residue embeddings are ragged (per-length), so the tabular artifact
    holds the fixed-size pooled vector; use ``write_npz`` for per-residue output.
    """
    import pandas as pd

    df = pd.DataFrame({
        "id": [e.id for e in embeddings],
        "sequence": [e.sequence for e in embeddings],
        "length": [len(e.sequence) for e in embeddings],
        "pooled": [e.pooled.tolist() for e in embeddings],
    })
    df.to_parquet(path)


# What a run's logits array is, per model family. SaProt's head is the fused AA+3Di
# vocabulary, ESMC's is the 64-token sequence head; the manifest has to say which, and
# saying it here rather than patching the written JSON afterwards is what keeps a run
# that produced no logits from advertising a shape for them.
LOGITS_SHAPE = "[length, 64] float32 (per-residue sequence-head logits)"
SAPROT_LOGITS_SHAPE = ("[length, 446] float32 (per-residue MLM logits over the fused "
                       "AA+3Di vocab)")


def write_manifest(embeddings: list[ESMCEmbedding], path, *, model: str, pool: str,
                   fast: bool, out_format: str, return_logits: bool,
                   logits_shape: str = LOGITS_SHAPE) -> None:
    """Write a manifest.json documenting a run's outputs — shapes, dtype, ordering,
    and which file holds each sequence — so a downstream consumer never has to
    read the code to know what it's looking at.
    """
    id_lengths = [(e.id, len(e.sequence)) for e in embeddings]
    d_model = int(embeddings[0].pooled.shape[0])
    write_manifest_for(id_lengths, d_model, path, model=model, pool=pool, fast=fast,
                       out_format=out_format, return_logits=return_logits,
                       logits_shape=logits_shape)


def write_manifest_for(id_lengths: list[tuple[str, int]], d_model: int, path, *, model: str,
                       pool: str, fast: bool, out_format: str, return_logits: bool,
                       logits_shape: str = LOGITS_SHAPE) -> None:
    """Core of :func:`write_manifest`, taking ``(id, length)`` pairs directly.

    Lets a caller that only has per-sequence id/length (e.g. reassembled from
    several --controller shard results, never materialized as ESMCEmbedding
    objects) still emit the same manifest shape.
    """
    import json

    manifest = {
        "model": model, "pool": pool, "fast": fast, "format": out_format,
        "d_model": d_model, "dtype": "float32", "logits": bool(return_logits),
        "shapes": {
            "per_residue": "[length, d_model] float32, one row per residue, <cls>/<eos> stripped",
            "pooled": "[d_model] float32",
            "logits": logits_shape if return_logits else None,
        },
        "sequences": [
            {"id": sid, "length": length,
             "file": f"{sid}.npz" if out_format == "npz" else "embeddings.parquet"}
            for sid, length in id_lengths
        ],
    }
    Path(path).write_text(json.dumps(manifest, indent=2))
