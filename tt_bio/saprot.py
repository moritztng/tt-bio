"""SaProt structure-aware protein language model on Tenstorrent (ttnn).

SaProt (westlake-repl) is an ESM-2 masked-LM encoder over a fused
structure-aware vocabulary: 20 amino acids x 21 Foldseek 3Di states plus 5
special tokens = 446. Architecturally it is an ESM-2 transformer (pre-LN,
rotary embeddings, GELU feed-forward, attention projections with bias, a
final ``emb_layer_norm_after``), distinct from ESMC (QK-LayerNorm + SwiGLU +
residual scaling). The rotary embedding, the ttnn framework
(``Module``/``WeightScope``/``get_device``), the token-embedding lookup, the
length bucketing and the multi-card fanout are reused from the ESMC port; the
ESM-2 block and the 446-token embedding table are the new work.

Reference: HuggingFace ``EsmForMaskedLM`` (westlake-repl/SaProt_*_AF2),
``transformers.models.esm.modeling_esm``. The 3Di tokenization front-end runs
on host (Foldseek) and is not a ttnn concern -- structure in, fused tokens out.
"""

from __future__ import annotations

import itertools
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import ttnn

from tt_bio.envflags import env_flag
from tt_bio.qk_rope_heads import STATS as QKRH_STATS, rope_heads_packed  # noqa: F401 (QKRH_STATS: read by esmc_ab)

from tt_bio import ops
from tt_bio.tenstorrent import (
    CORE_GRID_MAIN,
    Module,
    TorchWrapper,
    Weights,
    WeightScope,
    _dtype,
)
from tt_bio.esmc import (
    _MMLin,
    _key_lens,
    _lm_mask,
    lm_sdpa,
    rope_tables,
    _rope,
    BUCKET,
    bucket_token_axis,
    _shard_by_length,
    _reassemble,
    _thread_cap_env,
    _await_shard,
    _subblock,
)

ROPE_BASE = 10000.0
# token_dropout compensation baked into embeddings at inference (no mask tokens):
# embeddings *= (1 - 0.15*0.8) / (1 - 0) = 0.88. See EsmEmbeddings.forward.
_EMBED_SCALE = 0.88

# Foldseek vocabularies (utils.constants): 20 AA + "#" pad, 21 3Di states + "#".
FOLDSEEK_SEQ_VOCAB = "ACDEFGHIKLMNPQRSTVWY#"
FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"
_SPECIALS = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
VOCAB = _SPECIALS + [a + b for a, b in itertools.product(FOLDSEEK_SEQ_VOCAB, FOLDSEEK_STRUC_VOCAB)]
assert len(VOCAB) == 446, len(VOCAB)
_TOK_TO_IDX = {t: i for i, t in enumerate(VOCAB)}
CLS, PAD, EOS, UNK, MASK = 0, 1, 2, 3, 4

# name -> (arch config, HF repo). ESM-2 variants; vocab=446 for all.
CONFIGS = {
    "saprot-35m": (
        dict(hidden=480, n_heads=20, n_layers=12, intermediate=1920),
        "westlake-repl/SaProt_35M_AF2",
    ),
    "saprot-650m": (
        dict(hidden=1280, n_heads=20, n_layers=33, intermediate=5120),
        "westlake-repl/SaProt_650M_AF2",
    ),
    "saprot-1.3b": (
        dict(hidden=1280, n_heads=20, n_layers=66, intermediate=5120),
        "westlake-repl/SaProt_1.3B_AF2",
    ),
}
MODELS = tuple(CONFIGS)


def tokenize(aa_seq: str, struc_seq: str) -> "torch.Tensor":
    """Pair an amino-acid sequence with a Foldseek 3Di sequence into fused
    SaProt tokens: ``[<cls>, a0+s0, a1+s1, ..., <eos>]`` -> [1, L+2] long."""
    aa_seq = aa_seq.upper()
    if len(struc_seq) < len(aa_seq):  # right-pad 3Di with the unknown-structure "#"
        struc_seq = struc_seq + "#" * (len(aa_seq) - len(struc_seq))
    ids = [CLS]
    for a, s in zip(aa_seq, struc_seq):
        ids.append(_TOK_TO_IDX.get(a + s, UNK))
    ids.append(EOS)
    return torch.tensor([ids], dtype=torch.long)


class SaprotEmbeddingLayer(Module):
    """Token embedding + the ESM-2 token-dropout compensation (0.88 at inference).

    Rotary position embeddings -> no learned position embeddings are added.
    Weight key: ``embed.weight`` of shape [446, hidden].
    """

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.weight = self.torch_to_tt("weight", transform=lambda x: x)

    def __call__(self, tokens: ttnn.Tensor, embed_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x = ttnn.embedding(
            tokens, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.multiply(x, _EMBED_SCALE)
        if embed_mask is not None:
            x = ttnn.multiply(x, embed_mask)
        return x


# SaProt's projections (bias, fc1's GELU as the fused activation) on ESMC's minimal_matmul path instead of
# ttnn.linear, for projections over esmc.ESMC_MM_MIN_ROWS rows. None = Wormhole and a model no wider than
# _MM_MAX_WIDTH: minimal_matmul wins on the narrow LMs (ESMC 300M/600M, SaProt 35M: 44.7 -> 42.8 ms a page batch)
# and loses on SaProt 650M/1.3B (d 1280: +7/+8 %).
_SAPROT_MM = (env_flag("TT_BIO_SAPROT_MM", False) if os.environ.get("TT_BIO_SAPROT_MM", "").strip() else None)
_MM_MAX_WIDTH = 1152


def _saprot_mm(width: int) -> bool:
    if _SAPROT_MM is not None:
        return _SAPROT_MM
    from tt_bio import tenstorrent
    return tenstorrent.is_wormhole() and width <= _MM_MAX_WIDTH


# RoPE + head split of the packed qkv as one program (qk_rope_heads.rope_heads_packed, no LayerNorm) in place of
# nlp_create_qkv_heads + two rotary_embedding calls, head dim 64 (650M / 1.3B; 35M folds its RoPE into the qkv).
# Against float64 the program is the more accurate one (q rel_l2 0.0027 vs 0.0036, perf/spd_gap_whlm/rope_packed_bench.py).
# Page cell on a Galaxy chip at 1000 MHz, best of 3: 650M 217.5 -> 200.1 ms (--fast 205.4 -> 190.4), 1.3B 414.0 -> 381.8
# (400.2 -> 366.4); pcc vs fp32 moves -0.000126 / -0.000109 (650M) and -0.000096 / -0.000025 (1.3B). That passes the fast
# bar (0.0002) and missed the normal one (0.0001) on 650M hPCSK9 (-0.000133). That miss was the program's FPU add
# truncating both RoPE products (qk_rope_heads.SFPU); with the sums on the SFPU, normal mode, Galaxy chip at 1000 MHz,
# 2 rounds x 3, pcc vs fp32 against the chain on ten cells: 650M hPCSK9 -0.000003, hEGFR +0.000105, hHSA +0.000077,
# hTF +0.000049, hTNFa +0.000046; 1.3B +0.000020..+0.000126; time 650M -6..-8 %, 1.3B -6..-7 %.
# None = on with block-fp8 weights (--fast), and in normal mode where the SFPU sums run (Wormhole until Blackhole grades).
# Blackhole --fast 650M on a p300c chip at 1350 MHz: 98.5 -> 91.2 ms, pcc -0.000111.
_SAPROT_QKRH = (env_flag("TT_BIO_SAPROT_QKRH", False) if os.environ.get("TT_BIO_SAPROT_QKRH", "").strip() else None)


# Diagnostic only (off): scale QKRH's q by this factor. rope_packed_bench reads the packed program's |q|, |k| 0.07 % low
# and the rotary_embedding chain's 0.19 % high against float64; this tests whether the end-to-end pcc follows the logit scale.
_QKRH_QSCALE = None


# The attention behind the packed RoPE program runs at HiFi4 with fp32 dest instead of shsdpa's HiFi2 bf16 dest. Its
# truncated q.k deflates the logits, which the rotary_embedding chain's +0.19 % on |q| and |k| happened to cancel and the
# packed program (closer to float64) does not: 650M --fast hPCSK9 read pcc -0.000345 vs QKRH off, past the fast bar 0.0002.
# HiFi4 needs the fp32 dest (HiFi4 at bf16 dest was worse than HiFi2, HiFi2 at fp32 dest no better). Galaxy chip, 1000 MHz,
# 2 rounds, --fast, pcc vs fp32 against HiFi2: 650M hPCSK9 +0.000265, hEGFR +0.000093, hTF +0.000035, hTNFa +0.000012,
# 1.3B hPCSK9 +0.000080, hEGFR +0.000128, at +0.9 to +1.5 % time. None = with the packed RoPE program only.
_SAPROT_SDPA = None
_SDPA_FP32 = (ttnn.MathFidelity.HiFi4, False, True, False)


# SaProt 35M (head_dim 24, padded to 32): q / k dims ordered in NeoX pairs so rotary_embedding_llama's 32 x 32 pair
# rotation runs RoPE on the padded heads, in place of the folded companion projection (rot_weight), two multiplies and
# an add over [B, L, 3*H*32] (WH 35M page census: 14.9 of 41.5 ms). Galaxy chip, 1000 MHz, 2 processes per arm x 3, ms:
# normal hHSA 39.1 -> 31.7, hPCSK9 38.8 -> 31.7; --fast hHSA 39.0 -> 31.1, hPCSK9 37.9 -> 31.3 (-19 %). pcc vs fp32: normal
# hPCSK9 -0.000014, hHSA +0.000003, --fast +0.000021 / +0.000036. p150a 1350 MHz, hHSA: normal 23.6 -> 22.0, --fast
# 22.7 -> 18.7, pcc +0.00001..+0.00005 in every cell. False keeps the folded projection.
_SAPROT_RL = env_flag("TT_BIO_SAPROT_RL", True)
_RL_CK = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=False)


def _sdpa_ckc(qkrh: bool):
    """shsdpa's kernel config for a block whose heads came from the packed RoPE program (`qkrh`); None = its HiFi2."""
    return _SDPA_FP32 if (qkrh if _SAPROT_SDPA is None else _SAPROT_SDPA) else None


def _saprot_qkrh(w) -> bool:
    if _SAPROT_QKRH is not None:
        return _SAPROT_QKRH
    from tt_bio import qk_rope_heads
    return w.dtype == ttnn.bfloat8_b or qk_rope_heads._sfpu()


# The measured program for each SaProt projection at the page shape (7 x 544 rows), per arch, keyed (wormhole, K, N).
# perf/spd_m_saprot/lin_sweep.py on a Galaxy chip at 1000 MHz and a p300c chip at 1350 MHz, against ttnn.linear's own
# pick at equal float64 error (ms, normal mode; --fast lands within 3 % of the same programs):
#   WH 650M/1.3B qkv 1.026 -> 0.780 (minimal_matmul), out 0.400 -> 0.346 (mm), fc2 1.145 -> 0.979 (2D multicast)
#   BH 650M/1.3B qkv 0.424 -> 0.377, out 0.155 -> 0.135, fc2 0.480 -> 0.430 (mm), fc1 0.733 -> 0.546 (2D, GELU fused)
#   WH 35M out 0.121 -> 0.103, fc2 0.278 -> 0.245 (against esmc._mm_config's blocks)
#   BH 35M out 0.054 -> 0.048, fc2 0.113 -> 0.095 (mm), fc1 0.196 -> 0.124 (2D)
# "mm" = minimal_matmul blocks (M, K, N, subblock h, w) on the model's own kernel config (fp32 accumulation, as the sweep
# ran it); "2d" = (transpose_mcast, in0_block_w[, out blocks along M[, along N]]). Out blocks: the 2D writer shares
# BRISC with the weight reader and writes an out block only after its K loop, so with one out block per core all of fc1's 39 MB go
# out after the last K pass. Two out blocks re-read the weight once more and write the first half under the second's
# compute. WH 650M fc1 (GELU fused, gelu_f32), device profiler, Galaxy chip at 1000 MHz, us a call against the linear's
# own pick: --fast 1075 -> 986, normal 1369 -> 1329 (ls57; one out block at in0_block_w 4 does not fit L1). The --fast
# page holds the block LN's output and the residual in L1, where that program no longer places; with both resident
# (mm_prof --l1x) 7 x 10 out blocks at in0_block_w 8 read 1082 -> 1021 (ls59/60), the --fast entry below.
_SAPROT_PROG = (env_flag("TT_BIO_SAPROT_PROG", False) if os.environ.get("TT_BIO_SAPROT_PROG", "").strip() else None)
_PROGS = {
    (True, 1280, 3840): ("mm", (8, 8, 8, 1, 4)),
    (True, 1280, 1280): ("mm", (16, 4, 4, 1, 4)),
    (True, 5120, 1280): ("2d", (False, 10)),
    (True, 1280, 5120): ("2d", (False, 4, 2)),
    (True, 640, 480): ("mm", (8, 4, 4, 1, 4)),
    (True, 1920, 480): ("mm", (8, 4, 4, 1, 2)),
    (False, 1280, 3840): ("mm", (8, 2, 4, 1, 4)),
    (False, 1280, 1280): ("mm", (16, 4, 4, 1, 4)),
    (False, 5120, 1280): ("mm", (8, 4, 4, 1, 4)),
    (False, 1280, 5120): ("2d", (True, 2)),   # kb 4 does not fit L1 at fp32 accumulation (ls2 pk: fell back)
    (False, 640, 480): ("mm", (16, 4, 4, 1, 4)),
    (False, 1920, 480): ("mm", (16, 4, 4, 1, 4)),
    (False, 480, 1920): ("2d", (True, 5)),
}
# --fast at HiFi2 (_SAPROT_HIFI2) runs a different best program on some shapes; lin_sweep --fast --fidelity HiFi2, Galaxy
# chip at 1000 MHz, 650M (ms, against the HiFi3 pick above): fc2 1.006 -> 0.680. Shapes not listed keep the table above.
_PROGS_HIFI2 = {
    (True, 5120, 1280): ("mm", (8, 8, 8, 1, 2)),
    (True, 1280, 5120): ("2d", (False, 8, 2, 2)),
}
# Off drops the out-block entries (WH fc1) back to the linear's own pick.
_SAPROT_OUT_SPLIT = env_flag("TT_BIO_SAPROT_OUT_SPLIT", True)
_PROG_ROWS = (3072, 4096)   # page batch 7 x 544 = 3808 rows: the shape the table was measured at
PROG_STATS = [0, 0, 0]   # projections served by a measured program: minimal_matmul, 2D multicast, minimal_matmul + residual
# The block's residual add inside the out projection's and fc2's matmul (dit_minimal_matmul_addcmul_fused:
# x + 1 * (o W + b) * ones) wherever _PROGS serves that projection with minimal_matmul and the weight is bf16, so the
# add does not re-read the projection from DRAM (ls1 census: 66 adds = 13.4 ms of the WH 650M page batch, 6.6 on BH).
# bf16 weights only: with --fast's block-fp8 weights the op needs a bf16 copy, whose doubled weight stream cost more than
# the add (WH 650M +2.9 %, 1.3B +3.5 %).
RESID_REFUSED: set = set()
_SAPROT_RESID = env_flag("TT_BIO_SAPROT_RESID", True)


# The unfused residual adds write the residual stream to L1, so the next add and the block layer_norm read it from L1
# instead of DRAM. Page cell on a Galaxy chip at 1000 MHz, --fast, 2 rounds x 3, output bit-identical: 35M 39.3 -> 38.6 ms,
# 650M 155.2 -> 151.7 (hPCSK9 156.2 -> 152.1), 1.3B 297.5 -> 290.0. None = on with block-fp8 weights (--fast) only, as
# _SAPROT_LN_L1; normal mode's out/fc2 adds are fused into the matmul (_SAPROT_RESID) where it serves.
_SAPROT_X_L1 = (env_flag("TT_BIO_SAPROT_X_L1", False) if os.environ.get("TT_BIO_SAPROT_X_L1", "").strip() else None)
_X_L1_REFUSED: set = set()
#: [residual sums written to L1, declined]
X_L1_STATS = [0, 0]


# A projection small enough for L1 writes there through the linear instead of DRAM through minimal_matmul, and the
# minimal_matmul pick declines an L1 input it would have to share (both measured slower there). A Galaxy chip at
# 1000 MHz, 35M --fast page batch (3808 rows), ms per call: qkv linear 0.342 -> 0.206 with L1 out; fc2 reading fc1's
# L1 output 0.245 (mm pick) -> 0.187. 650M fc1 (39 MB) does not place in L1 and reads the same at 952 rows, so the
# budget keeps the large sizes as they are. Page, 35M at 1000 MHz: --fast 30.9 -> 28.2 ms, normal 31.9 -> 30.0 ms with
# the same output; 650M and 1.3B have no projection on this path.
_SAPROT_OUT_L1 = env_flag("TT_BIO_SAPROT_OUT_L1", True)
_OUT_L1_BYTES = 16 << 20
OUT_L1_STATS = [0, 0]   # projections written to L1, refused


def _out_l1(x, w) -> bool:
    return _SAPROT_OUT_L1 and x.volume() // int(w.shape[-2]) * int(w.shape[-1]) * 2 <= _OUT_L1_BYTES


def _saprot_prog(x, w, act, hifi2=False):
    """("mm", MinimalMatmulConfig) or ("2d", program config) for this projection, or None where the default serves."""
    if _SAPROT_PROG is False:
        return None
    from tt_bio import tenstorrent
    k, n = int(w.shape[-2]), int(w.shape[-1])
    rows = x.volume() // k
    key = (tenstorrent.is_wormhole(), k, n)
    hit = (_PROGS_HIFI2.get(key) if hifi2 else None) or _PROGS.get(key)
    if hit is None or not _PROG_ROWS[0] <= rows <= _PROG_ROWS[1] or act not in (None, "gelu", "gelu_approx"):
        return None
    kind, p = hit
    if kind == "2d" and len(p) > 2 and not _SAPROT_OUT_SPLIT:
        return None
    g = x.device().compute_with_storage_grid_size()
    grid = ttnn.CoreCoord(g.x, g.y)
    if kind == "mm":
        mb, kb, nb, sh, sw = p
        return kind, ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh,
                                              subblock_w=sw, compute_with_storage_grid_size=grid)
    tm, kb, m_splits, n_splits = (*p, 1, 1)[:4]
    rows_ax, cols_ax = (g.x, g.y) if tm else (g.y, g.x)
    mt = -(-rows // 32)
    pm, pn = -(-mt // rows_ax), -(-(n // 32) // cols_ax)
    oh, ow = (max(b for b in range(1, t + 1) if t % b == 0 and b <= -(-t // splits))
              for t, splits in ((pm, m_splits), (pn, n_splits)))
    sh, sw = _subblock(oh, ow)
    fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, float(act == "gelu_approx")) if act else None
    return kind, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=kb, out_subblock_h=sh, out_subblock_w=sw, out_block_h=oh,
        out_block_w=ow, per_core_M=pm, per_core_N=pn, transpose_mcast=tm, fused_activation=fa, fuse_batch=True)


# The block-fp8 projections of --fast at HiFi2 instead of the model's HiFi3, fp32 accumulation kept: a block-fp8 weight's
# 7-bit mantissa is consumed whole in two fidelity phases (ESMC ships the same on its minimal_matmuls). None = on for
# models at least _HIFI2_MIN_WIDTH wide: SaProt 35M's pooled cos vs fp32 fell to 0.999879 at HiFi2, under the 0.9999 bar.
# Blackhole p300c at 1350 MHz, 2 rounds, fc1 + fc2: 650M 104.9 -> 94.6 ms (pcc +0.000006), 1.3B 191.0 -> 161.7 (-0.000154).
_SAPROT_HIFI2 = env_flag("TT_BIO_SAPROT_HIFI2", None)
_HIFI2_MIN_WIDTH = 1280
# Which projections _SAPROT_HIFI2 moves to HiFi2: the FFN's. Page cell --fast, Galaxy chip at 1000 MHz, 2 rounds, pcc vs
# fp32 HF: fc1 + fc2 650M 176.9 -> 156.7 ms (pcc +0.000052), 1.3B 337.7 -> 298.5 (-0.000003). qkv + out add little
# speed and most of the error (1.3B: 342.5 -> 342.4 ms, pcc -0.000709), so all four failed the fast bar.
_SAPROT_HIFI2_ROLES = ("fc1", "fc2")


def _role(d, k, n):
    """The block projection a (k, n) weight serves in a model of width d."""
    return "qkv" if n == 3 * d else "out" if n == k else "fc1" if n > k else "fc2"


class _SaprotLin(_MMLin):
    MM_BFP8_HIFI2 = False   # SaProt 35M --fast at HiFi2: pooled cos vs fp32 0.999915 -> 0.999879, under the 0.9999 bar

    def _mm_on(self) -> bool:
        return _saprot_mm(int(self.norm_weight.shape[-1]))

    def _ck(self, w):
        """The model's kernel config, at HiFi2 for a block-fp8 weight where _SAPROT_HIFI2 serves."""
        ck = self.compute_kernel_config
        d = int(self.norm_weight.shape[-1])
        on = d >= _HIFI2_MIN_WIDTH if _SAPROT_HIFI2 is None else _SAPROT_HIFI2
        if not (on and w.dtype == ttnn.bfloat8_b and _role(d, int(w.shape[-2]), int(w.shape[-1])) in _SAPROT_HIFI2_ROLES
                and ck.math_fidelity in (ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4)):
            return ck
        if getattr(self, "_ck_hifi2", None) is None:
            self._ck_hifi2 = type(ck)(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=ck.math_approx_mode,
                                      fp32_dest_acc_en=ck.fp32_dest_acc_en, packer_l1_acc=ck.packer_l1_acc)
        return self._ck_hifi2

    def _lin(self, x, w, bias=None, dtype=None, **kw):
        act = kw.get("activation")
        ck = self._ck(w)
        hifi2 = ck.math_fidelity == ttnn.MathFidelity.HiFi2
        prog = None if set(kw) - {"activation"} else _saprot_prog(x, w, act, hifi2)
        l1 = not set(kw) - {"activation"} and _out_l1(x, w)
        if prog is not None and prog[0] == "mm" and l1 and x.memory_config().buffer_type == ttnn.BufferType.L1:
            prog = None
        if prog is not None:
            kind, cfg = prog
            dt = _dtype(ttnn.bfloat16) if dtype is None else dtype
            try:
                if kind == "mm":
                    out = ttnn.experimental.minimal_matmul(
                        x, w, bias_tensor=bias, compute_kernel_config=ck, dtype=dt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        config=cfg, **({"fused_activation": (ttnn.UnaryOpType.GELU, act == "gelu_approx")} if act else {}))
                else:
                    out = ttnn.linear(x, w, bias=bias, program_config=cfg, compute_kernel_config=ck, dtype=dt,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
                PROG_STATS[kind == "2d"] += 1
                return out
            except RuntimeError:   # a program this shape or grid cannot place: the default serves
                pass
        elif l1:
            try:
                out = ops.linear(x, w, bias=bias, compute_kernel_config=ck, dtype=_dtype(ttnn.bfloat16) if dtype is None
                                 else dtype, core_grid=CORE_GRID_MAIN, memory_config=ttnn.L1_MEMORY_CONFIG, **kw)
                OUT_L1_STATS[0] += 1
                return out
            except RuntimeError:   # L1 too full at this point of the block: DRAM serves
                OUT_L1_STATS[1] += 1
        if ck is not self.compute_kernel_config and not self._mm_on():
            return ops.linear(x, w, bias=bias, compute_kernel_config=ck, dtype=_dtype(ttnn.bfloat16) if dtype is None
                              else dtype, core_grid=CORE_GRID_MAIN, **kw)
        return super()._lin(x, w, bias=bias, dtype=dtype, **kw)

    def _lin_add(self, x, w, bias, resid):
        """resid + (x @ w + bias), in one matmul where _SAPROT_RESID serves, else as the projection and an add."""
        x_l1 = w.dtype == ttnn.bfloat8_b if _SAPROT_X_L1 is None else _SAPROT_X_L1
        prog = _saprot_prog(x, w, None) if _SAPROT_RESID and w.dtype == ttnn.bfloat16 else None
        if prog is not None and prog[0] == "mm":
            n = int(w.shape[-1])
            if getattr(self, "_ones_row", None) is None:
                self._ones_row = ttnn.from_torch(torch.ones(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                 device=x.device())
            try:
                out = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                    x, w, 1.0, resid, self._ones_row, bias, config=prog[1], compute_kernel_config=self.compute_kernel_config,
                    dtype=_dtype(ttnn.bfloat16), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                PROG_STATS[2] += 1
                return out
            except RuntimeError as exc:   # a shape the fused op refuses: say so once, then project + add
                if not RESID_REFUSED:
                    warnings.warn(f"saprot: fused residual matmul refused {tuple(w.shape)} {w.dtype}: {str(exc)[:400]}")
                RESID_REFUSED.add((tuple(w.shape), w.dtype))
        r = self._lin(x, w, bias=bias)
        out = None
        if x_l1 and tuple(resid.padded_shape) not in _X_L1_REFUSED:
            try:
                out = ttnn.add(resid, r, memory_config=ttnn.L1_MEMORY_CONFIG)
                X_L1_STATS[0] += 1
                ttnn.deallocate(resid)   # the block rebinds x to the sum; the old stream would pin L1 until GC
            except RuntimeError:
                _X_L1_REFUSED.add(tuple(resid.padded_shape))
                X_L1_STATS[1] += 1
        if out is None:
            out = ttnn.add(resid, r)
        ttnn.deallocate(r)
        return out


class ESM2Attention(_SaprotLin, Module):
    """ESM-2 self-attention: pre-LN, fused q/k/v (with bias), per-head RoPE, SDPA,
    output projection (with bias), plus the block's residual: returns ``x + attn(x)``.
    No QK-LayerNorm (that is ESMC-specific).

    For ``head_dim`` not aligned to 64 (the 35M variant, ``head_dim = 24``) the fused
    ``rotary_embedding`` kernel and the tile-split rotate-half both break, so RoPE is
    folded into the weights instead. Every head is zero-padded 24 -> 32 in the q/k/v
    projection, and a second projection carries ``rotate_half`` as a signed column
    permutation of the q/k weights (exact in any dtype). RoPE is then
    ``qkv * cos + qkv_rot * sin`` on the packed ``[B, L, 3*H*32]`` tensor, with tables
    that are 1/0 over v, before the usual head split. The 8 zero dims add 0 to every
    score, and the output projection gets matching zero rows, so nothing is sliced."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        d_model = self.weights["qkv.weight"].shape[-1]  # torch Linear [3*hidden, hidden]
        self.head_dim = d_model // n_heads
        self.fold_rope = self.head_dim % 64 != 0
        self.rope_pairs = self.fold_rope and _SAPROT_RL
        if self.fold_rope:
            w, b, wr, br = _fold_rope_qkv(self.weights["qkv.weight"], self.weights["qkv.bias"],
                                          n_heads, self.head_dim, pairs=self.rope_pairs)
            ow = _pad_heads_in(self.weights["out.weight"], n_heads, self.head_dim)
            dev = lambda x, dt=None: ttnn.from_torch(x.contiguous(), layout=ttnn.TILE_LAYOUT,
                                                     device=self.device,
                                                     dtype=_dtype(ttnn.bfloat16) if dt is None else dt)
            self.qkv_weight, self.qkv_bias = dev(w.t(), _dtype()), dev(b)
            if self.rope_pairs:
                self.trans_mat = dev(_pair_rotation()[None, None], ttnn.bfloat16)
            else:
                self.rot_weight, self.rot_bias = dev(wr.t(), _dtype()), dev(br)
            self.out_weight = dev(ow.t(), _dtype())
        else:
            self.qkv_weight = self.torch_to_tt("qkv.weight", dtype=_dtype())
            self.qkv_bias = self.torch_to_tt("qkv.bias")
            self.out_weight = self.torch_to_tt("out.weight", dtype=_dtype())
        self.out_bias = self.torch_to_tt("out.bias")

    def _rope_pairs(self, t, cos, sin):
        """rotary_embedding_llama on [B, H, L, 32] heads, as one batch of B * H heads (its prefill takes batch 1)."""
        b, h, length, w = (int(d) for d in t.shape)
        out = ttnn.experimental.rotary_embedding_llama(ttnn.reshape(t, (1, b * h, length, w)), cos, sin, self.trans_mat,
                                                       compute_kernel_config=_RL_CK)
        ttnn.deallocate(t)
        return ttnn.reshape(out, (b, h, length, w))

    def __call__(self, x, cos, sin, attn_mask=None, key_valid=None):
        ck = self.compute_kernel_config
        x_norm = _block_ln(x, self.norm_weight, self.norm_bias, ck, self.qkv_weight)
        qkv = self._lin(x_norm, self.qkv_weight, bias=self.qkv_bias)
        heads = None
        if self.rope_pairs:   # q / k dims in adjacent pairs (_fold_rope_qkv pairs=True), cos / sin [1, 1, L, 32]
            ttnn.deallocate(x_norm)
            q, k, v = self._split_heads(qkv, self.n_heads)
            q, k = (self._rope_pairs(t, cos, sin) for t in (q, k))
        elif self.fold_rope:  # cos/sin are the packed [1, L, 3*H*32] tables from fold_rope_tables
            rot = self._lin(x_norm, self.rot_weight, bias=self.rot_bias)
            ttnn.deallocate(x_norm)
            a = ttnn.multiply(qkv, cos)
            ttnn.deallocate(qkv)
            r = ttnn.multiply(rot, sin)
            ttnn.deallocate(rot)
            qkv = ttnn.add(a, r)
            ttnn.deallocate(a); ttnn.deallocate(r)
            q, k, v = self._split_heads(qkv, self.n_heads)
        else:
            ttnn.deallocate(x_norm)
            heads = rope_heads_packed(qkv, cos, sin) if self.head_dim == 64 and _saprot_qkrh(self.qkv_weight) else None
            if heads is not None:
                ttnn.deallocate(qkv)
                q, k, v = heads
                if _QKRH_QSCALE:
                    q = ttnn.multiply(q, _QKRH_QSCALE)
            else:
                q, k, v = self._split_heads(qkv, self.n_heads)
                q, k = _rope(q, k, cos, sin)
        if key_valid is not None and attn_mask is None:  # the -inf mask already zeroes them, see esmc.Attention
            k = ttnn.multiply(k, key_valid)
            v = ttnn.multiply(v, key_valid)
        o = lm_sdpa(q, k, v, attn_mask, scale=self.head_dim ** -0.5, merge=True, ckc=_sdpa_ckc(heads is not None))  # [B, L, H*32] when fold_rope; out_weight has the zero rows
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        out = self._lin_add(o, self.out_weight, self.out_bias, x)
        ttnn.deallocate(o)
        return out


def _pad_heads(w: torch.Tensor, n_heads: int, d: int) -> torch.Tensor:
    """[H*d, ...] -> [H*32, ...], each head's d rows followed by 32-d zero rows."""
    w = w.reshape(n_heads, d, *w.shape[1:])
    return torch.nn.functional.pad(w, (0, 0) * (w.dim() - 2) + (0, 32 - d)).reshape(n_heads * 32, *w.shape[2:])


def _pair_rotation() -> torch.Tensor:
    """[32, 32] T with x @ T = (-x1, x0, -x3, x2, ...): rotate_half over adjacent pairs, as rotary_embedding_llama wants."""
    t = torch.zeros(32, 32)
    t[torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1.0
    t[torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1.0
    return t


def _pair_order(d: int) -> torch.Tensor:
    """Head dims in pair order: NeoX's (i, i + d/2) become positions (2i, 2i + 1)."""
    return torch.stack([torch.arange(d // 2), torch.arange(d // 2, d)], dim=1).flatten()


def _pad_heads_in(w: torch.Tensor, n_heads: int, d: int) -> torch.Tensor:
    """Output projection [out, H*d] -> [out, H*32] with zero columns for the pad dims."""
    return _pad_heads(w.t(), n_heads, d).t()


def _fold_rope_qkv(w, b, n_heads: int, d: int, pairs: bool = False):
    """Head-padded q/k/v projection plus its RoPE companion.

    ``w`` is torch Linear ``[3*H*d, hidden]`` (q, k, v stacked), ``b`` is ``[3*H*d]``.
    Returns ``(w, b)`` padded to ``[3*H*32, ...]`` and ``(w_rot, b_rot)`` of the same
    shape whose q/k rows are ``rotate_half`` of the q/k rows (``cat(-x2, x1)`` per head)
    and whose v rows are zero, so ``x @ w_rot.T + b_rot == rotate_half(x @ w.T + b)``
    over q and k exactly: a row permutation with a sign flip, no arithmetic.
    ``pairs`` instead orders each q / k head's rows in NeoX pairs (``_pair_order``) for rotary_embedding_llama; q . k
    is unchanged by the common permutation, v keeps its order (the output projection reads it).
    """
    h = d // 2
    ws, bs = w.reshape(3, n_heads, d, -1).clone(), b.reshape(3, n_heads, d).clone()
    if pairs:
        o = _pair_order(d)
        ws[:2], bs[:2] = ws[:2, :, o], bs[:2, :, o]
    wr = torch.cat([-ws[:, :, h:], ws[:, :, :h]], dim=2)
    br = torch.cat([-bs[:, :, h:], bs[:, :, :h]], dim=2)
    wr[2] = 0
    br[2] = 0
    pad = lambda t: torch.cat([_pad_heads(t[i].reshape(n_heads * d, *t.shape[3:]), n_heads, d)
                               for i in range(3)])
    return pad(ws), pad(bs), pad(wr), pad(br)


def _fold_rope_tables_torch(seq_len: int, n_heads: int, d: int, dtype=torch.float32):
    """Packed ``[L, 3*H*32]`` cos/sin for ``_fold_rope_qkv``: the head_dim-``d`` NeoX
    tables tiled per head over q and k (zero-padded to 32), cos = 1 / sin = 0 over v."""
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, d, 2, dtype=dtype) / d))
    freqs = torch.outer(torch.arange(seq_len, dtype=dtype), inv_freq)
    head = lambda f: torch.nn.functional.pad(torch.cat([f, f], dim=-1), (0, 32 - d))  # [L, 32]
    qk = lambda f: head(f).repeat(1, 2 * n_heads)
    cos = torch.cat([qk(freqs.cos()), torch.ones(seq_len, n_heads * 32, dtype=dtype)], dim=-1)
    sin = torch.cat([qk(freqs.sin()), torch.zeros(seq_len, n_heads * 32, dtype=dtype)], dim=-1)
    return cos, sin


def pair_rope_tables_torch(seq_len: int, d: int, dtype=torch.float32):
    """[L, 32] cos / sin for head dims in pair order (``_pair_order``), zero over the pad dims."""
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, d, 2, dtype=dtype) / d))
    f = torch.outer(torch.arange(seq_len, dtype=dtype), inv_freq).repeat_interleave(2, dim=-1)
    pad = lambda t: torch.nn.functional.pad(t, (0, 32 - d))
    return pad(f.cos()), pad(f.sin())


def fold_rope_tables(seq_len: int, n_heads: int, d: int, device):
    """``_fold_rope_tables_torch`` on device as ``[1, L, 3*H*32]`` bf16."""
    cos, sin = _fold_rope_tables_torch(seq_len, n_heads, d)
    to_tt = lambda t: ttnn.from_torch(t[None], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    return to_tt(cos), to_tt(sin)


# The block layer_norms write to L1, where the next projection (qkv, fc1) reads its input back, instead of a DRAM round
# trip: the same placement that moved ESMC's pair-FFN fc1 from 31 % to ~52 % of its roof (esmc.PAIR_FFN_L1_LN). Same op,
# same values (output bit-identical in every A/B). A shape whose L1 output the allocator refuses keeps DRAM from then on.
# Page cell on a Galaxy chip at 1000 MHz, 3 rounds, --fast: 35M 40.0 -> 38.6 ms, 650M 160.6 -> 153.8, 1.3B 297.6 -> 294.2
# (1022 aa: 782.6 -> 749.6). Normal mode reads 35M +1.6 %, 650M +3.5 %, 1.3B -1.2 %, so None = on with block-fp8
# weights (--fast) only.
_SAPROT_LN_L1 = (env_flag("TT_BIO_SAPROT_LN_L1", False) if os.environ.get("TT_BIO_SAPROT_LN_L1", "").strip() else None)
_LN_L1_REFUSED: set = set()
#: [block layer_norms written to L1, declined]
LN_L1_STATS = [0, 0]


def _block_ln(x, w, b, ck, proj_w):
    """The block's pre-LN, in L1 where _SAPROT_LN_L1 serves (None: when the next projection's weight proj_w is
    block-fp8) and the shape fits."""
    key = tuple(x.padded_shape)
    l1 = proj_w.dtype == ttnn.bfloat8_b if _SAPROT_LN_L1 is None else _SAPROT_LN_L1
    if l1 and key not in _LN_L1_REFUSED:
        try:
            out = ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck,
                                  memory_config=ttnn.L1_MEMORY_CONFIG)
            LN_L1_STATS[0] += 1
            return out
        except RuntimeError:
            _LN_L1_REFUSED.add(key)
            LN_L1_STATS[1] += 1
    return ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck)


# fc1's GELU applied by the matmul at pack instead of as its own op over the [rows, 4*d] activation. Page cell, batch of
# 7, Galaxy chip at 1000 MHz: 650M 250.0 -> 234.2 ms (--fast 244.8 -> 228.5), 1.3B 482.4 -> 450.1 (472.7 -> 441.3),
# 35M 55.4 -> 52.5 (54.8 -> 53.0); error against float32 HF unchanged (1.3B pcc 0.99916 -> 0.99929, 650M 0.99976 both).
_GELU_FUSED = env_flag("TT_BIO_SAPROT_GELU_FUSED", True)
# fc1's fused GELU as calculate_gelu_f32 (metal_overlay gelu_f32, metal's approx-mode gelu): 3e-4 of float64 against
# 4e-3 for the wheel's accurate gelu (rel_l2 5.8e-5 vs 3.1e-4 on N(0, 1.5)) at a third of its math-thread time: WH 650M
# --fast fc1 1203 -> 1079 us a call. Page cell, Galaxy chip at 1000 MHz, 2 rounds x 3, off -> on, pcc vs fp32 HF:
# 650M --fast hHSA 151.0 -> 146.1 ms (+0.000045), hPCSK9 155.2 -> 146.2 (+0.000074); normal hHSA 175.4 -> 170.2
# (+0.000019), hPCSK9 175.3 -> 171.4 (+0.000064); 1.3B --fast hPCSK9 284.9 -> 275.7 (-0.000071), normal hHSA 337.3 ->
# 328.2 (-0.000014); 35M --fast hHSA 29.1 -> 26.3 (-0.000011), normal hPCSK9 29.6 -> 27.6 (+0.000010).
# None = Wormhole until Blackhole is graded.
_SAPROT_GELU_F32 = env_flag("TT_BIO_SAPROT_GELU_F32", None)


def _fc1_act() -> str:
    if _SAPROT_GELU_F32 is not None:
        return "gelu_approx" if _SAPROT_GELU_F32 else "gelu"
    from tt_bio import tenstorrent
    return "gelu_approx" if tenstorrent.is_wormhole() else "gelu"


class ESM2FFN(_SaprotLin, Module):
    """ESM-2 feed-forward: pre-LN, Linear(intermediate)+GELU, Linear(hidden), residual: returns ``x + ffn(x)``."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight", dtype=_dtype())
        self.fc1_bias = self.torch_to_tt("fc1.bias")
        self.fc2_weight = self.torch_to_tt("fc2.weight", dtype=_dtype())
        self.fc2_bias = self.torch_to_tt("fc2.bias")

    def __call__(self, x):
        ck = self.compute_kernel_config
        x_norm = _block_ln(x, self.norm_weight, self.norm_bias, ck, self.fc1_weight)
        h = self._lin(x_norm, self.fc1_weight, bias=self.fc1_bias, **({"activation": _fc1_act()} if _GELU_FUSED else {}))
        ttnn.deallocate(x_norm)
        if not _GELU_FUSED:
            h = ttnn.gelu(h)
        out = self._lin_add(h, self.fc2_weight, self.fc2_bias, x)
        ttnn.deallocate(h)
        return out


class ESM2Block(Module):
    """ESM-2 pre-LN block: x = x + attn(x); x = x + ffn(x). No residual scaling."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = ESM2Attention(n_heads, self.scope("attn"), compute_kernel_config)
        self.ffn = ESM2FFN(self.scope("ffn"), compute_kernel_config)

    def __call__(self, x, cos, sin, attn_mask=None, key_valid=None):
        x = self.attn(x, cos, sin, attn_mask, key_valid)
        return self.ffn(x)


class SaprotLMHead(Module):
    """ESM-2 MLM head: Linear -> GELU -> LayerNorm -> decoder (tied to the token
    embedding) + bias. The decoder weight is the embedding table, passed in at
    call time (weight tying)."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.dense_weight = self.torch_to_tt("dense.weight")
        self.dense_bias = self.torch_to_tt("dense.bias")
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.decoder_weight = self.torch_to_tt("decoder_weight")
        self.decoder_bias = self.torch_to_tt("decoder_bias")

    def __call__(self, x):
        ck = self.compute_kernel_config
        x = self._lin(x, self.dense_weight, bias=self.dense_bias)
        x = ttnn.gelu(x)
        x = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias,
                            epsilon=1e-5, compute_kernel_config=ck)
        logits = self._lin(x, self.decoder_weight, bias=self.decoder_bias)
        ttnn.deallocate(x)
        return logits


class SaprotModel(Module):
    """Full SaProt stack: embed -> N ESM-2 blocks -> emb_layer_norm_after -> LM head.

    __call__ returns (logits[B,L,446], embeddings[B,L,hidden]); embeddings are the
    post-``emb_layer_norm_after`` hidden states (the structure-aware per-residue
    representation)."""

    def __init__(self, n_heads, n_layers, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.embed = SaprotEmbeddingLayer(self.scope("embed"), compute_kernel_config)
        self.blocks = [
            ESM2Block(n_heads, self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.head = SaprotLMHead(self.scope("head"), compute_kernel_config)
        self._rope = {}

    def __call__(self, tokens, attn_mask=None, key_valid=None, embed_mask=None,
                 want_logits: bool = True):
        """With ``want_logits=False`` the MLM head is not run and ``logits`` is None.

        The embedding path is finished before the head is reached, so dropping the head
        cannot move ``emb`` by a bit. What it drops is the head's own work plus a
        ``[B, L, 446]`` float copy back to the host, which the embed CLI throws away:
        ``embed_sequences`` touches ``logits`` only when the caller asked for it. Same
        shape of fix as ``ESMCHiddenStatesModel``'s ``last_hidden_only``.
        """
        seq_len = tokens.shape[-1]
        rope = self._rope.get(seq_len)
        if rope is None:  # built and uploaded once per bucket length, then resident
            attn = self.blocks[0].attn
            if attn.rope_pairs:
                rope = tuple(ttnn.from_torch(t[None, None], layout=ttnn.TILE_LAYOUT, device=self.device,
                                             dtype=ttnn.bfloat16) for t in pair_rope_tables_torch(seq_len, attn.head_dim))
            elif attn.fold_rope:
                rope = fold_rope_tables(seq_len, self.n_heads, attn.head_dim, self.device)
            else:
                rope = rope_tables(seq_len, attn.head_dim, device=self.device)
            self._rope[seq_len] = rope
        cos, sin = rope
        x = self.embed(tokens, embed_mask)
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask, key_valid)
        emb = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias,
                              epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(x)
        logits = self.head(emb) if want_logits else None
        return logits, emb


def _remap_state_dict(sd: dict, n_layers: int) -> dict:
    """Remap HuggingFace EsmForMaskedLM keys to the SaprotModel scheme."""
    out = {}
    out["embed.weight"] = sd["esm.embeddings.word_embeddings.weight"]
    for i in range(n_layers):
        p = f"esm.encoder.layer.{i}."
        out[f"blocks.{i}.attn.norm.weight"] = sd[p + "attention.LayerNorm.weight"]
        out[f"blocks.{i}.attn.norm.bias"] = sd[p + "attention.LayerNorm.bias"]
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat(
            [sd[p + "attention.self.query.weight"], sd[p + "attention.self.key.weight"],
             sd[p + "attention.self.value.weight"]], dim=0).contiguous()
        out[f"blocks.{i}.attn.qkv.bias"] = torch.cat(
            [sd[p + "attention.self.query.bias"], sd[p + "attention.self.key.bias"],
             sd[p + "attention.self.value.bias"]], dim=0).contiguous()
        out[f"blocks.{i}.attn.out.weight"] = sd[p + "attention.output.dense.weight"]
        out[f"blocks.{i}.attn.out.bias"] = sd[p + "attention.output.dense.bias"]
        out[f"blocks.{i}.ffn.norm.weight"] = sd[p + "LayerNorm.weight"]
        out[f"blocks.{i}.ffn.norm.bias"] = sd[p + "LayerNorm.bias"]
        out[f"blocks.{i}.ffn.fc1.weight"] = sd[p + "intermediate.dense.weight"]
        out[f"blocks.{i}.ffn.fc1.bias"] = sd[p + "intermediate.dense.bias"]
        out[f"blocks.{i}.ffn.fc2.weight"] = sd[p + "output.dense.weight"]
        out[f"blocks.{i}.ffn.fc2.bias"] = sd[p + "output.dense.bias"]
    out["norm.weight"] = sd["esm.encoder.emb_layer_norm_after.weight"]
    out["norm.bias"] = sd["esm.encoder.emb_layer_norm_after.bias"]
    out["head.dense.weight"] = sd["lm_head.dense.weight"]
    out["head.dense.bias"] = sd["lm_head.dense.bias"]
    out["head.norm.weight"] = sd["lm_head.layer_norm.weight"]
    out["head.norm.bias"] = sd["lm_head.layer_norm.bias"]
    out["head.decoder_bias"] = sd["lm_head.bias"]
    # lm_head.decoder.weight is tied to the token embedding; store a dedicated
    # linear-weight copy (transposed for ttnn.linear) rather than reusing the
    # embedding-table tensor in the decoder matmul.
    out["head.decoder_weight"] = sd["esm.embeddings.word_embeddings.weight"]
    return out


class Saprot(TorchWrapper):
    """Top-level SaProt model (torch in / torch out). Mirrors EsmForMaskedLM.

    forward(tokens[B,L]) -> (logits[B,L,446], emb[B,L,hidden]). Optional padding
    masks (built by ``esmc._batch_tokens``) let a batch of unequal-length sequences
    share one padded, bucketed forward.
    """

    def __init__(self, hidden: int, n_heads: int, n_layers: int, intermediate: int):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.intermediate = intermediate

    @classmethod
    def from_pretrained(cls, name: str = "saprot-650m") -> "Saprot":
        from tt_bio import weights as _w
        config, repo = CONFIGS[name]
        p = _w.fetch(name)
        # Fail loudly on a wrong CONFIGS entry: the checkpoint's own config.json is
        # the source of truth for the architecture. A mismatch here means the
        # CONFIGS arch dict is wrong, and load_state_dict(strict=False) would
        # otherwise silently mask it (wrong head split / truncated layers /
        # dropped tensors) and run a structurally-wrong model. See
        # tt-bio-saprot-1.3b-config-fix for the bug this guards against.
        import json
        with open(os.path.join(p, "config.json")) as f:
            hf_cfg = json.load(f)
        declared = dict(
            hidden=hf_cfg["hidden_size"],
            n_heads=hf_cfg["num_attention_heads"],
            n_layers=hf_cfg["num_hidden_layers"],
            intermediate=hf_cfg["intermediate_size"],
        )
        if declared != config:
            raise ValueError(
                f"CONFIGS[{name!r}] arch {config} does not match checkpoint "
                f"{repo} config.json {declared}; refusing to load "
                f"(strict=False would otherwise silently mask the shape mismatch).")
        sd = torch.load(os.path.join(p, "pytorch_model.bin"), map_location="cpu")
        remapped = _remap_state_dict(sd, config["n_layers"])
        model = cls(**config)
        model.load_state_dict(remapped, strict=False)
        return model

    def _create_module(self, weights: WeightScope) -> SaprotModel:
        return SaprotModel(self.n_heads, self.n_layers, weights, self.compute_kernel_config)

    def forward(self, tokens, attn_mask=None, key_valid=None, embed_mask=None,
                want_logits: bool = True):
        """Bucketed at the op boundary, not only in the batcher -- see esmc.bucket_token_axis.
        `_batch_saprot` has already bucketed on the CLI path, so there this is a no-op."""
        tokens, attn_mask, key_valid, embed_mask, L = bucket_token_axis(
            tokens, attn_mask, key_valid, embed_mask, pad_token=PAD)
        logits, emb = self._dispatch(tokens, attn_mask, key_valid, embed_mask, want_logits)
        if int(tokens.shape[1]) == L:
            return logits, emb
        return (None if logits is None else logits[:, :L]), emb[:, :L]

    def _dispatch(self, tokens, attn_mask, key_valid, embed_mask, want_logits=True):
        tokens_tt = ttnn.from_torch(
            tokens.to(torch.int32), device=self.tt_device,
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        )
        mask_tt = _lm_mask(attn_mask, _key_lens(attn_mask), self.tt_device)
        kv_tt = None if key_valid is None else ttnn.from_torch(
            key_valid.to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        em_tt = None if embed_mask is None else ttnn.from_torch(
            embed_mask.to(torch.bfloat16), device=self.tt_device,
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
        logits, emb = self.module(tokens_tt, mask_tt, kv_tt, em_tt, want_logits)
        return (None if logits is None else self._to_torch(logits)), self._to_torch(emb)


def load_saprot(name: str = "saprot-650m", *, fast: bool = False):
    from tt_bio import tenstorrent
    tenstorrent.set_fast_mode(fast)
    if name not in CONFIGS:
        raise ValueError(f"unknown SaProt model {name!r}; choose from {list(MODELS)}")
    return Saprot.from_pretrained(name)


@dataclass
class SaprotEmbedding:
    """One sequence's structure-aware embeddings from the SaProt trunk.

    ``per_residue`` has one row per residue (the <cls>/<eos> special tokens are
    stripped). ``pooled`` is a single fixed-size vector. ``logits`` are the
    per-residue MLM logits ([L, 446]) when requested.
    """

    id: str
    sequence: str
    per_residue: np.ndarray
    pooled: np.ndarray
    logits: Optional[np.ndarray]


_POOLERS = {"mean": lambda e: e.mean(axis=0), "max": lambda e: e.max(axis=0),
            "cls": None}


def embed_sequences(model, sequences, *, return_logits=False, pool="mean", batch_size=8):
    """Embed each {id: (aa, struc)} pair with an already-loaded SaProt ``model``.

    ``sequences`` maps id -> (aa_seq, struc_seq) or id -> aa_seq (3Di taken as
    all-"#", i.e. sequence-only). Sequences are length-bucketed and run through a
    single padded, bucketed device forward per batch.
    """
    if pool not in _POOLERS:
        raise ValueError(f"unknown pool {pool!r}; choose from {sorted(_POOLERS)}")
    norm = {}
    for sid, v in sequences.items():
        if isinstance(v, (list, tuple)):
            aa, struc = v[0], (v[1] if len(v) > 1 else "")
        else:
            aa, struc = v, ""
        if not aa:
            raise ValueError(f"sequence {sid!r} is empty")
        norm[sid] = (aa.upper(), struc)

    items = list(norm.items())
    order = sorted(range(len(items)), key=lambda i: len(items[i][1][0]))
    budget = batch_size * 512
    batches, cur, cur_max = [], [], 0
    for i in order:
        tok = len(items[i][1][0]) + 2
        nxt_max = max(cur_max, ((tok + BUCKET - 1) // BUCKET) * BUCKET)
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * nxt_max > budget):
            batches.append(cur); cur, cur_max = [], 0
        cur.append(i); cur_max = max(cur_max, ((tok + BUCKET - 1) // BUCKET) * BUCKET)
    if cur:
        batches.append(cur)

    by_id = {}
    for idx in batches:
        batch = [items[i] for i in idx]
        # token id lists (already fused SaProt tokens), then pad to a common bucket.
        toks = [tokenize(aa, struc)[0].tolist() for _, (aa, struc) in batch]
        lens = [len(t) for t in toks]
        Lb = ((max(lens) + BUCKET - 1) // BUCKET) * BUCKET
        B = len(batch)
        input_ids = torch.full((B, Lb), PAD, dtype=torch.long)
        for i, t in enumerate(toks):
            input_ids[i, : lens[i]] = torch.tensor(t, dtype=torch.long)
        if all(li == Lb for li in lens):
            attn_mask = key_valid = embed_mask = None
        else:
            attn_mask = torch.zeros(B, Lb, Lb, dtype=torch.float32)
            key_valid = torch.ones(B, 1, Lb, 1, dtype=torch.float32)
            embed_mask = torch.ones(B, Lb, 1, dtype=torch.float32)
            for i, li in enumerate(lens):
                attn_mask[i, :, li:] = float("-inf")
                key_valid[i, :, li:, :] = 0.0
                embed_mask[i, li:, :] = 0.0
        logits_b, emb_b = model(input_ids, attn_mask, key_valid, embed_mask, return_logits)
        for row, (sid, (aa, struc)) in enumerate(batch):
            li = lens[row]
            emb = emb_b[row, :li].numpy().astype(np.float32)
            per_residue, cls = emb[1:-1], emb[0]
            lg = None
            if return_logits:
                lg = logits_b[row, 1:li - 1].numpy().astype(np.float32)
            pooled = cls if pool == "cls" else _POOLERS[pool](per_residue)
            by_id[sid] = SaprotEmbedding(sid, aa, per_residue, pooled.astype(np.float32), lg)
    return [by_id[k] for k in sequences]


def _run_saprot_shard(in_path: str, out_path: str) -> None:
    """Subprocess entry point: embed one SaProt shard on the pinned card, pickle results.

    Invoked as a fresh interpreter with ``TT_VISIBLE_DEVICES`` already set in the
    environment (so the assigned physical chip is logical device 0 and ttnn, imported
    at module load, binds to it). Reads a pickled request
    ``{model, sequences, fast, return_logits, pool, batch_size}`` and writes the
    resulting ``list[SaprotEmbedding]``. Mirrors ``esmc._run_embed_shard``; the only
    difference is the model loader and the (aa, struc) payload shape.
    """
    with open(in_path, "rb") as f:
        req = pickle.load(f)
    model = load_saprot(req["model"], fast=req["fast"])
    results = embed_sequences(model, req["sequences"], return_logits=req["return_logits"],
                              pool=req["pool"], batch_size=req["batch_size"])
    with open(out_path, "wb") as f:
        pickle.dump(results, f)


def _spawn_saprot_shard(idx: int, device: int, shard: list, workdir: str, *,
                        model: str, fast: bool, return_logits: bool, pool: str,
                        batch_size: int, thread_cap_env: dict | None = None):
    """Launch a pinned subprocess embedding ``shard`` on physical card ``device``.

    Returns ``(proc, out_path, device, log_path, logf)``. The child sets
    ``TT_VISIBLE_DEVICES=<device>`` so the chip is logical device 0 (see
    ``get_device``). stdout/stderr go to a per-shard log file rather than a pipe,
    so a chatty ttnn child never deadlocks on a full pipe buffer. Mirrors
    ``esmc._spawn_shard`` minus the esmc-6B shared-cache path (SaProt has no 6B-style
    shared weight cache — its largest resident checkpoint, 1.3B, is small enough
    that per-shard reload isn't the fanout bottleneck).
    """
    in_path = os.path.join(workdir, f"shard{idx}.in.pkl")
    out_path = os.path.join(workdir, f"shard{idx}.out.pkl")
    log_path = os.path.join(workdir, f"shard{idx}.log")
    with open(in_path, "wb") as f:
        pickle.dump(dict(model=model, sequences=dict(shard), fast=fast,
                         return_logits=return_logits, pool=pool, batch_size=batch_size), f)
    env = {**os.environ, **(thread_cap_env or {}),
           "TT_VISIBLE_DEVICES": str(device), "TT_BIO_LOGICAL_DEVICE_ID": "0"}
    # P300 boards are a custom topology; like the `embed`/`predict`/`boltzgen` fanout
    # paths, a single-chip worker needs the 1x1 Blackhole mesh-graph descriptor or
    # ttnn.open_device aborts with "Custom fabric mesh graph descriptor path must be
    # specified".
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(env, device)
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from tt_bio.saprot import _run_saprot_shard; "
         "_run_saprot_shard(sys.argv[1], sys.argv[2])",
         in_path, out_path],
        env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc, out_path, device, log_path, logf


def embed_multicard(sequences: dict, *, model: str, devices: list[int],
                    fast: bool = False, return_logits: bool = False, pool: str = "mean",
                    batch_size: int = 8) -> list:
    """Data-parallel SaProt embedding across multiple physical TT cards.

    Shards ``sequences`` across ``devices`` (one pinned subprocess per card), runs the
    single-card :func:`embed_sequences` in each, then gathers and reassembles the
    embeddings in original input order. Embarrassingly parallel: SaProt embeddings are
    row-independent (no cross-sequence state), so a sequence's output is identical to
    running it on one card — sharding changes only which chip computes which row.
    Reuses the ESMC fanout primitives (``_shard_by_length`` with an aa-length key,
    ``_reassemble``, ``_await_shard``, ``_thread_cap_env``); only the shard payload
    and model loader are SaProt-specific.

    More cards than sequences is harmless: extra cards simply get no shard.
    """
    items = list(sequences.items())
    devices = list(devices)[:max(1, len(items))]
    shards = _shard_by_length(items, len(devices), key=lambda it: len(it[1][0]))
    workdir = tempfile.mkdtemp(prefix="tt-bio-saprot-fanout-")
    thread_cap_env = _thread_cap_env(len(devices))
    try:
        handles = [
            _spawn_saprot_shard(idx, dev, shard, workdir, model=model, fast=fast,
                               return_logits=return_logits, pool=pool, batch_size=batch_size,
                               thread_cap_env=thread_cap_env)
            for idx, (dev, shard) in enumerate(zip(devices, shards)) if shard
        ]
        results = [_await_shard(*h) for h in handles]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return _reassemble(items, results)


def embed(sequences, model: str = "saprot-650m", *, fast=False, return_logits=False,
          pool="mean", batch_size=8, devices: list[int] | None = None):
    """One-shot embedding: load ``model`` and embed ``sequences``.

    ``sequences`` may be a single (aa, struc) pair, a dict of {id: (aa, struc)},
    or {id: aa} (sequence-only, 3Di = "#"). Returns one SaprotEmbedding per input.

    ``devices`` shards the input across multiple physical TT cards (one pinned
    subprocess each, data-parallel); with 0 or 1 device the model is loaded in-process
    on the single card this process already sees.
    """
    if isinstance(sequences, str):
        sequences = {"seq0": sequences}
    elif isinstance(sequences, (list, tuple)) and len(sequences) == 2 and isinstance(sequences[0], str):
        sequences = {"seq0": sequences}
    if devices and len(devices) > 1:
        return embed_multicard(sequences, model=model, devices=devices, fast=fast,
                               return_logits=return_logits, pool=pool, batch_size=batch_size)
    m = load_saprot(model, fast=fast)
    return embed_sequences(m, sequences, return_logits=return_logits, pool=pool, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Foldseek 3Di front-end (host-side preprocessing -- structure in, 3Di tokens out)
# ---------------------------------------------------------------------------
#
# foldseek is an external dependency (GPL-3.0, not vendored): the user installs
# the binary (e.g. `conda install -c bioconda foldseek`) and tt-bio shells out to
# its `structureto3didescriptor` subcommand. It runs on CPU, off-device, and is
# not a ttnn concern. The struc vocab is the 20 Foldseek 3Di states + "#" (unknown).

def find_foldseek(foldseek_bin: str | None = None) -> str:
    """Locate the foldseek binary: ``--foldseek``, then ``FOLDSEEK_BIN``, then PATH.

    A path given explicitly has to be there. Falling back to PATH after a bad
    ``--foldseek`` would run a different binary than the one asked for.
    """
    import shutil

    explicit = [(foldseek_bin, "--foldseek"),
                (os.environ.get("FOLDSEEK_BIN") or None, "FOLDSEEK_BIN")]
    for cand, where in explicit:
        if not cand:
            continue
        p = os.path.expanduser(cand)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        raise ValueError(
            f"{where}={cand} is not an executable file. SaProt needs foldseek to turn "
            "a structure into 3Di tokens.")
    found = shutil.which("foldseek")
    if found:
        return found
    raise ValueError(
        "foldseek not found on PATH. Install it (e.g. `conda install -c bioconda "
        "foldseek`), or point --foldseek / FOLDSEEK_BIN at the binary. SaProt needs "
        "3Di tokens from a structure.")


def foldseek_3di(pdb_path: str, foldseek_bin: str | None = None, chains: list | None = None) -> dict:
    """Run foldseek ``structureto3didescriptor`` on one PDB/cif -> {chain: (aa, struc)}.

    Returns the amino-acid sequence and the Foldseek 3Di structural sequence per
    chain, exactly what SaProt's fused tokenizer needs. ``struc`` uses the 20
    Foldseek 3Di states (lower-cased) plus ``#`` for masked/unknown residues.
    """
    import subprocess, tempfile
    bin_ = find_foldseek(foldseek_bin)
    if not os.path.isfile(pdb_path):
        raise ValueError(f"structure file not found: {pdb_path}")
    out = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False).name
    try:
        cmd = [bin_, "structureto3didescriptor", "-v", "0", "--threads", "1",
               "--chain-name-mode", "1", pdb_path, out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise ValueError(f"foldseek failed on {pdb_path}:\n{r.stderr[-800:]}")
        seqs: dict[str, tuple[str, str]] = {}
        with open(out) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")[:3]
                if len(parts) < 3:
                    continue
                desc, aa, struc = parts
                chain = desc.split(" ")[0].split("_")[-1]
                if chains is None or chain in chains:
                    seqs[chain] = (aa.upper(), struc.lower())
        return seqs
    finally:
        for ext in ("", ".dbtype"):
            try:
                os.remove(out + ext)
            except OSError:
                pass


def read_shard_yaml(path) -> dict:
    """Parse a controller shard's YAML ``{id: [aa, 3di]}`` mapping for SaProt.

    ``esmc.read_yaml`` deliberately rejects non-string values, which is the right
    validation for ESMC and the wrong one here: a SaProt shard carries the 3Di
    string alongside the amino-acid string. A bare string value is accepted too
    and read as sequence-only (3Di = "#"), so a shard written by an ESMC-shaped
    producer still loads.
    """
    import yaml

    doc = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{path}: expected a YAML mapping of {{id: [aa, 3di]}}, got {doc!r}")
    out: dict[str, tuple[str, str]] = {}
    for k, v in doc.items():
        if isinstance(v, str):
            out[str(k)] = (v.upper(), "#" * len(v))
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            out[str(k)] = (str(v[0]).upper(), str(v[1]))
        else:
            raise ValueError(f"{path}: entry {k!r} is neither a sequence nor an [aa, 3di] pair")
    return out


def load_sequences_with_structure(data, structure=None, foldseek_bin=None) -> dict:
    """Load {id: (aa, struc)} for the SaProt tokenizer.

    ``data`` is a FASTA file/dir/bare AA sequence (reuses ``esmc.load_sequences``).
    ``structure`` is a single PDB/cif (its first chain is paired with a single
    input sequence), a directory of PDB/cif files named ``<id>.pdb``/``<id>.cif``
    matched to the FASTA ids, or None (sequence-only: 3Di taken as all ``#``).

    The 3Di string is placed on the sequence by matching the structure's own residues,
    not by length: foldseek reports 3Di only for residues the structure resolves, so a
    deposited structure with an unresolved loop returns a shorter string and lining the
    two up by length shifts every token after the gap. See ``_map_3di``.
    """
    from tt_bio import esmc
    aas = esmc.load_sequences(data)  # {id: aa}
    out: dict[str, tuple[str, str]] = {}
    if structure is None:
        out = {sid: (aa, "#" * len(aa)) for sid, aa in aas.items()}
        return out
    spath = Path(structure).expanduser()
    if spath.is_file():
        seqs = foldseek_3di(str(spath), foldseek_bin)
        first = next(iter(seqs.values()))
        if len(aas) == 1:
            sid = next(iter(aas))
            out[sid] = (aas[sid], _map_3di(aas[sid], first[0], first[1], spath))
        else:
            raise ValueError(
                "a single --structure pairs with a single input sequence; pass a "
                "directory of <id>.pdb files for multiple sequences")
        return out
    if spath.is_dir():
        for sid, aa in aas.items():
            cand = None
            for ext in (".pdb", ".cif", ".PDB", ".CIF"):
                p = spath / f"{sid}{ext}"
                if p.is_file():
                    cand = p; break
            if cand is None:
                raise ValueError(
                    f"no structure file '{sid}.pdb/.cif' in {spath}; --structure dir "
                    "must contain one per FASTA id")
            ch = foldseek_3di(str(cand), foldseek_bin)
            first = next(iter(ch.values()))
            out[sid] = (aa, _map_3di(aa, first[0], first[1], cand))
        return out
    raise ValueError(f"--structure must be a PDB/cif file or a directory: {structure}")


def _map_3di(aa: str, struct_aa: str, struct_3di: str, path) -> str:
    """Put each structural token on the residue it belongs to, or refuse.

    Foldseek returns the structure's own amino-acid sequence alongside the 3Di string,
    and that is what says which residues the structure actually resolved. Matching the
    two sequences and filling the unresolved positions with ``#`` (SaProt's own token for
    unknown structure) is the whole fix. Reconciling by length instead put another
    residue's token on 82 of 117 positions for a structure missing three residues, and
    the run reported success.

    Every residue the structure resolves has to land somewhere in the sequence. If some
    do not, the structure is not of this sequence and pairing them would invent a fused
    AA+3Di token for each mismatch, so this refuses rather than guess.
    """
    from difflib import SequenceMatcher

    if struct_aa == aa:
        return struct_3di
    tokens = ["#"] * len(aa)
    placed = 0
    for i, j, n in SequenceMatcher(a=aa, b=struct_aa, autojunk=False).get_matching_blocks():
        tokens[i:i + n] = list(struct_3di[j:j + n])
        placed += n
    if placed < len(struct_aa):
        raise ValueError(
            f"{path}: the structure resolves {len(struct_aa)} residues but only {placed} of "
            f"them appear in the {len(aa)}-residue input sequence, so this structure is not "
            "of this sequence. Pass the matching structure, or drop --structure to run "
            "sequence-only.")
    return "".join(tokens)


