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

from tt_bio.envflags import env_flag

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
    _pair_proj_linear,
    _sdpa_program_config_for_lengths,
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
BUCKET = 64    # pad the LM length to a multiple of this to avoid per-length recompilation
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


def _rope(q: ttnn.Tensor, k: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor):
    """RoPE for per-head q, k [B, H, L, head_dim].

    When L is tile-aligned (the bucketed LM path — ``BUCKET`` is 64, and the 6B
    backbone always pads to it) the fused ``ttnn.experimental.rotary_embedding``
    kernel replaces ``apply_rotary``'s six-op rotate-half pile with one dispatch
    per tensor. This is the largest single share of ESMC attention (a dispatch-
    bound elementwise stack, not a matmul), so collapsing it is a real per-layer
    win, largest on the smaller models. Matches the reference within bf16 noise;
    the ragged fallback keeps arbitrary single-sequence lengths exact.
    """
    if q.shape[2] % 32 == 0:
        return (ttnn.experimental.rotary_embedding(q, cos, sin),
                ttnn.experimental.rotary_embedding(k, cos, sin))
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


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


class Attention(Module):
    """Multi-head self-attention with QK-LayerNorm + RoPE (no biases on projections).

    Mirrors esm.layers.attention.MultiHeadAttention (qk_layernorm=True, bias=False):
      qkv = Linear(LayerNorm(x)); q,k,v = chunk(qkv,3)
      q = LayerNorm(q); k = LayerNorm(k)            # over full d_model, then per-head RoPE
      o = SDPA(rope(q), rope(k), v, scale=d_head**-0.5); out_proj(o)
    """

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.in_norm_weight = self.torch_to_tt("layernorm_qkv.0.weight")
        self.in_norm_bias = self.torch_to_tt("layernorm_qkv.0.bias")
        # The two big projection weights (qkv, out_proj) carry the bulk of the
        # ESMC-6B's parameters; in fast mode they load as block-fp8 (bfloat8_b),
        # halving their weight-read bandwidth and resident size. _dtype() is bf16
        # otherwise (full precision, the default).
        self.qkv_weight = self.torch_to_tt("layernorm_qkv.1.weight", dtype=_dtype())
        self.q_ln_weight = self.torch_to_tt("q_ln.weight")
        self.k_ln_weight = self.torch_to_tt("k_ln.weight")
        self.out_weight = self.torch_to_tt("out_proj.weight", dtype=_dtype())

    def __call__(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor,
                 attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        d_model = x.shape[-1]
        head_dim = d_model // self.n_heads

        x_norm = ttnn.layer_norm(
            x, weight=self.in_norm_weight, bias=self.in_norm_bias,
            epsilon=1e-5, compute_kernel_config=ck,
        )
        qkv = self._lin(x_norm, self.qkv_weight)
        ttnn.deallocate(x_norm)

        q, k, v = ttnn.chunk(qkv, 3, dim=-1)
        ttnn.deallocate(qkv)
        q = ttnn.layer_norm(q, weight=self.q_ln_weight, epsilon=1e-5, compute_kernel_config=ck)
        k = ttnn.layer_norm(k, weight=self.k_ln_weight, epsilon=1e-5, compute_kernel_config=ck)

        # Re-pack and use the tile-aware head split, then apply per-head RoPE.
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        q, k, v = self._split_heads(qkv, self.n_heads)
        q, k = _rope(q, k, cos, sin)
        if key_valid is not None:
            # Zero padded keys/values so their attention contribution is exactly
            # 0 (weight x 0) — exact masking, not reliant on bf16 exp(-inf).
            k = ttnn.multiply(k, key_valid)
            v = ttnn.multiply(v, key_valid)

        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=head_dim**-0.5,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2]),
        )
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        o = self._merge_heads(o)  # [B, L, d_model]
        out = self._lin(o, self.out_weight)
        ttnn.deallocate(o)
        return out


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
# Ships ON. The 320-1024 aa window it rides inside is now checked end to end at 298 / 512 / 768 /
# 1024 aa on qb2 card 1, ttnn 0.68.0, 11x10: byte-identical CIF and plDDT against the OFF arm at
# every size, and inert at 298 because that sits below the window
# (perf/esm3p4close/fold_ab_*_c1.json). Still gated, so an A/B stays one call away. 13x10 is
# unchecked with this config, the same exposure the row block already ships with.
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
# Rides on F: it needs the block's output to be the final content of its rows. Needs
# `nblk * rows == L` for the view's volume to match, so a length that does not divide by the row
# block keeps the concat.
PAIR_FFN_FILL_ASSEMBLY = True
_PAIR_FFN_FILL_ASSEMBLY = os.environ.get(
    "TT_BIO_PAIR_FFN_FILL_ASSEMBLY", "1" if PAIR_FFN_FILL_ASSEMBLY else "0") == "1"

# [served, declined] residual row-blocked calls, same idiom as `FUSED_RESID_STATS`.
FILL_ASSEMBLY_STATS = [0, 0]
_FILL_ASSEMBLY_REFUSED = set()

# [split, unsplit] `SwiGLUFFN.__call__` invocations. An A/B arm on the small-grid opt-in needs both
# `_SPLIT_SWIGLU` and `_SPLIT_SWIGLU_SMALL_GRID`, and if either is missed the arm is a silent A/A;
# this counter is what makes that visible instead of inferred.
SPLIT_STATS = [0, 0]


def set_split_swiglu(on: bool) -> bool:
    """A/B switch for the split-fc1 SwiGLU path. Returns the previous state."""
    global _SPLIT_SWIGLU
    prev, _SPLIT_SWIGLU = _SPLIT_SWIGLU, bool(on)
    return prev


def set_split_swiglu_small_grid(on: bool) -> bool:
    """A/B switch for the split-fc1 SwiGLU family on a small (< 110 core) grid. Returns the
    previous state. Inert on Blackhole: the flag is only consulted when `_IS_SMALL_GRID`."""
    global _SPLIT_SWIGLU_SMALL_GRID
    prev, _SPLIT_SWIGLU_SMALL_GRID = _SPLIT_SWIGLU_SMALL_GRID, bool(on)
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


def _pack_swiglu_weight(weight: torch.Tensor) -> torch.Tensor:
    packed = weight.t()
    rows, two_n = packed.shape
    return packed.reshape(rows, 2, -1, 32).permute(0, 2, 1, 3).reshape(rows, two_n)


class SwiGLUFFN(Module):
    """SwiGLU feed-forward (mirrors esm.layers.blocks.swiglu_ln_ffn, bias=False):
      h = Linear(LayerNorm(x)); x1,x2 = chunk(h,2); Linear(silu(x1) * x2).
    """

    def __init__(self, state_dict: Weights, compute_kernel_config, fuse_swiglu: bool = False):
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
        self.fc2_weight = self.torch_to_tt("3.weight", dtype=_dtype())

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
             out_mc=None) -> ttnn.Tensor:
        """`out_mc` is fc2's output memory config; `None` -- every caller but lever F -- leaves
        the op byte-identical to the call that had no such keyword."""
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
        elif split:
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
                h1 = _pair_proj_linear(x_norm, self.fc1_a_weight, ck, dt, **l1)
                h2 = _pair_proj_linear(x_norm, self.fc1_b_weight, ck, dt, **l1)
            else:
                L1_FC1_STATS[1] += 2 * bool(l1_gated)
                h1 = self._lin(x_norm, self.fc1_a_weight)
                h2 = self._lin(x_norm, self.fc1_b_weight)
            ttnn.deallocate(x_norm)
            gated = ttnn.multiply(
                h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                **({"memory_config": ttnn.L1_MEMORY_CONFIG} if l1_gated else {}),
            )
            ttnn.deallocate(h1)
            ttnn.deallocate(h2)
        else:
            h = self._lin(x_norm, self._fc1_full())
            ttnn.deallocate(x_norm)
            x1, x2 = ttnn.chunk(h, 2, dim=-1)
            ttnn.deallocate(h)
            gated = ttnn.multiply(ttnn.silu(x1), x2)
            ttnn.deallocate(x1); ttnn.deallocate(x2)
        out = self._lin(gated, self.fc2_weight,
                        **({"memory_config": out_mc} if out_mc is not None else {}))
        ttnn.deallocate(gated)
        return out

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
        return split, (rows if blocked else 0)

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
        this drops G first, then F, then C-in, and re-runs the loop on the weaker rung; the
        refusal is
        cached per `padded_shape`, so a size that declines costs one exception per fold rather
        than one per block. Each rung is also reachable on its own through its env gate:
        both -0.842 s/fold at 512 aa on qb1, F alone -0.392, C-in alone -0.295.
        """
        L, key = x.shape[1], tuple(x.padded_shape)
        nblk = -(-L // rows)
        lazy = _PAIR_FFN_L1_SLICE and key not in _L1_SLICE_REFUSED
        fused = residual and _PAIR_FFN_FUSED_RESIDUAL and key not in _FUSED_RESID_REFUSED
        filled = (fused and _PAIR_FFN_FILL_ASSEMBLY and nblk * rows == L
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
                                    out_mc=ttnn.L1_MEMORY_CONFIG if fused else None)
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
                if filled:
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
        `x + ffn(x)`. Does not free `x`; the caller does, as it did around the add it used to own.
        """
        split, rows = self._split_plan(x)
        if rows:
            SPLIT_STATS[0] += 1
            return self._row_blocked(x, rows, residual=True)
        FUSED_RESID_STATS[1] += 1
        return ttnn.add(x, self(x))

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # The fc1 activation (2*d_ff wide) is several GB at long L and, on top of
        # the resident 6B weights, OOMs the 12 GB/chip Wormhole DRAM. The FFN is
        # row-independent over dim=1, so tiling it is bit-exact. 4D pair input
        # (ESMFold2 MSA-encoder pair_transition, [B,L,L,c]) has transient ~ rows*L
        # -> area-bounded tile; 3D per-token (ESMC LM FFN, [B,L,d]) -> fixed row
        # tile. Single pass on Blackhole. See tenstorrent._apply_grid_thresholds.
        from tt_bio import tenstorrent
        L = x.shape[1]
        split, rows = self._split_plan(x)
        SPLIT_STATS[0 if split else 1] += 1
        if rows:
            return self._row_blocked(x, rows, residual=False)
        if len(x.shape) == 4:
            chunk = tenstorrent.pair_row_tile(L)
        else:
            t = tenstorrent.SMALL_GRID_SEQ_TILE
            chunk = t if (t and L > t) else 0
        if chunk:
            parts = ttnn.chunk(x, -(-L // chunk), dim=1)
            return ttnn.concat([self._ffn(p, split=split) for p in parts], dim=1)
        return self._ffn(x, split=split)


class Block(Module):
    """UnifiedTransformerBlock, plain path (mirrors esm.layers.blocks):
      x = x + attn(x) / s ; x = x + ffn(x) / s,  s = sqrt(n_layers / 36).
    """

    def __init__(self, n_heads: int, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = Attention(n_heads, self.scope("attn"), compute_kernel_config)
        self.ffn = SwiGLUFFN(self.scope("ffn"), compute_kernel_config)
        self.inv_scale = 1.0 / (n_layers / 36) ** 0.5

    def __call__(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor,
                 attn_mask: ttnn.Tensor | None = None,
                 key_valid: ttnn.Tensor | None = None) -> ttnn.Tensor:
        r1 = self.attn(x, cos, sin, attn_mask, key_valid)
        x = ttnn.add(x, ttnn.multiply(r1, self.inv_scale))
        ttnn.deallocate(r1)
        r3 = self.ffn(x)
        x = ttnn.add(x, ttnn.multiply(r3, self.inv_scale))
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
                 key_valid: ttnn.Tensor | None = None, _rope=None):
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
        logits = self.head(emb)
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
    # traces fit the reserved region with headroom (one ESMC-300M trace is a
    # few MB of trace buffer) and cover the working set of a length-sorted
    # single-sequence stream.
    _TRACE_CACHE_MAX = 8

    def __init__(self, d_model: int, n_heads: int, n_layers: int, *, trace: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.trace = trace
        self._trace_cache: collections.OrderedDict = collections.OrderedDict()
        self._trace_seen: collections.Counter = collections.Counter()
        self._trace_broken = False

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
        for _ in range(2):  # warm compile + program cache outside the capture
            wl, we = self.module(tok_d, mask_d, kv_d, _rope=rope)
            ttnn.deallocate(wl)
            ttnn.deallocate(we)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        lg, em = self.module(tok_d, mask_d, kv_d, _rope=rope)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
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

    def _forward_eager(self, tokens, attn_mask, key_valid):
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
        logits, emb = self.module(tokens_tt, mask_tt, kv_tt)
        return self._to_torch(logits), self._to_torch(emb)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_valid: torch.Tensor | None = None):
        """tokens[B,L] -> (logits[B,L,64], emb[B,L,d]). Optional padding masks
        (built by ``_batch_tokens``) let a batch of unequal-length sequences share
        one padded, bucketed forward: ``attn_mask`` [B,L,L] additive removes padded
        keys from the softmax denominator; ``key_valid`` [B,1,L,1] zeros padded
        keys/values so their contribution is exactly 0.

        The token axis is bucketed to a multiple of BUCKET HERE rather than only in
        ``_batch_tokens``, so a direct API call at a ragged L cannot reach the SDPA ragged.
        ``_batch_tokens`` has already bucketed on every CLI path, so there it is a no-op."""
        tokens, attn_mask, key_valid, _, L = bucket_token_axis(tokens, attn_mask, key_valid)
        logits, emb = self._dispatch(tokens, attn_mask, key_valid)
        if int(tokens.shape[1]) == L:
            return logits, emb
        return logits[:, :L], emb[:, :L]

    def _dispatch(self, tokens, attn_mask, key_valid):
        if self.trace and not self._trace_broken and tokens.shape[0] == 1:
            key = (tuple(tokens.shape), attn_mask is not None, key_valid is not None)
            if key in self._trace_cache:
                return self._forward_traced(tokens, attn_mask, key_valid)
            if trace_region_size() <= 0:
                if not getattr(self, "_trace_note_shown", False):
                    self._trace_note_shown = True
                    print("ESMC trace disabled: device was opened without a trace "
                          "region; running eager. Open with get_device("
                          "trace_region_size=...) before load_esmc to enable.",
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
        return self._forward_eager(tokens, attn_mask, key_valid)


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


def load_esmc6b_state_dict(snapshot_dir: str) -> dict:
    """Read the sharded 6B safetensors and remap TE keys to esm-repo names.

    Keeps only weights the ttnn stack consumes (embed, transformer blocks,
    final norm); drops `_extra_state`, the LM head and any classifier heads.
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
    load_dtype = torch.float32 if _tt._FAST_MODE else torch.bfloat16
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
                 key_valid: ttnn.Tensor | None = None, last_hidden_only: bool = False):
        """Return the hidden states. With ``last_hidden_only`` only the final-norm one.

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
        hidden = [] if last_hidden_only else [self._to_host(x)]  # hs[0] = embedding output
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, key_valid)
            if not last_hidden_only and i < self.n_layers - 1:
                hidden.append(self._to_host(x))  # hs[i+1] = block i output
        norm_x = ttnn.layer_norm(
            x, weight=self.norm_weight, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x)
        hidden.append(self._to_host(norm_x))  # hs[n_layers] = post-norm output
        return hidden

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
                last_hidden_only: bool = False) -> torch.Tensor:
        """``[n_layers+1, B, L, d]``, or ``[1, B, L, d]`` with ``last_hidden_only``.

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
                             last_hidden_only=last_hidden_only)  # list of [B, Lb, d_model]
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
    return {str(k): v.upper() for k, v in doc.items()}


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
        seqs = {"seq0": seq.upper()}
    if not seqs:
        raise ValueError(f"no sequences found in {data}")
    return seqs


# DRAM reserved for ttnn trace capture when an ESMC-300M/600M model is loaded
# with tracing on (the default). Sized for _TRACE_CACHE_MAX concurrent captured
# forwards; 256 MB leaves the device layout otherwise unchanged.
_ESMC_TRACE_REGION_SIZE = 1 << 28


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
        get_device(trace_region_size=_ESMC_TRACE_REGION_SIZE)
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
    its arguments unchanged -- byte for byte the old call, which is what makes the change
    bit-exact there. It exists for the direct API caller, who reaches ``Model.forward`` with
    whatever length they have: the bucket used to live in the caller, so a ragged L went straight
    into the SDPA and picked up its padded key columns at a bias of zero. See
    ``tt_bio/token_axis.py`` and PLAYBOOKS.md §MODEL 2b.

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
        logits_b, emb_b = model(input_ids, attn_mask, key_valid)  # [B,Lb,64], [B,Lb,d]
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
        model = load_esmc(req["model"], fast=req["fast"])
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
    m = load_esmc(model, fast=fast)
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


def write_manifest(embeddings: list[ESMCEmbedding], path, *, model: str, pool: str,
                   fast: bool, out_format: str, return_logits: bool) -> None:
    """Write a manifest.json documenting a run's outputs — shapes, dtype, ordering,
    and which file holds each sequence — so a downstream consumer never has to
    read the code to know what it's looking at.
    """
    id_lengths = [(e.id, len(e.sequence)) for e in embeddings]
    d_model = int(embeddings[0].pooled.shape[0])
    write_manifest_for(id_lengths, d_model, path, model=model, pool=pool, fast=fast,
                       out_format=out_format, return_logits=return_logits)


def write_manifest_for(id_lengths: list[tuple[str, int]], d_model: int, path, *, model: str,
                       pool: str, fast: bool, out_format: str, return_logits: bool) -> None:
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
            "logits": "[length, 64] float32 (per-residue sequence-head logits)" if return_logits else None,
        },
        "sequences": [
            {"id": sid, "length": length,
             "file": f"{sid}.npz" if out_format == "npz" else "embeddings.parquet"}
            for sid, length in id_lengths
        ],
    }
    Path(path).write_text(json.dumps(manifest, indent=2))
